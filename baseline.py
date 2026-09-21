"""The comparison agent: an ordinary DQN, and what it cannot be told.

A DQN learns one thing — given what it sees, which shove tends to work out
better — and bakes the answer into its weights. It solves CartPole outright, so
on plain balancing we tie at best and we say so plainly.

The point of this file is the other measurement. Faced with a goal nobody
mentioned during training, the DQN has no mechanism to be *told* about it; the
only way to change what it wants is to write a new reward function and train a
new agent. We measure that cost in environment steps, against the planner's
zero.

Both agents are scored through the same evaluate() in plan.py, on the same
seeds, so neither gets a friendlier measurement path than the other.

Output: goal-by-goal comparison numbers and figures. Written to figures/
"""

import json
import os
import random
import time
from collections import deque

import gymnasium as gym
import matplotlib
import numpy as np
import torch
import torch.nn as nn

# Draw to a file rather than a window, so this runs the same over SSH.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from goals import (CORRIDOR_HALF_WIDTH, PARK_TARGET,  # noqa: E402
                   REPORT_UNITS, WAYPOINT_FIRST, WAYPOINT_HOLD_STEPS,
                   WAYPOINT_SECOND)
from plan import (DEVICE, MAX_STEPS, RESULTS_PATH,  # noqa: E402
                  TRACES_PATH, evaluate)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = "checkpoints/dqn.pt"
RESULTS_OUT = "checkpoints/baseline_results.json"
COMPARISON_PLOT = "figures/goal_comparison.png"
PARK_TRACE_PLOT = "figures/goal_park_traces.png"

# Figures for the first half of the article, where the DQN is the only agent in
# the room and world models have not been mentioned yet.
LEARNING_PLOT = "figures/dqn_learning.png"
DQN_PARK_PLOT = "figures/dqn_park_only.png"

# The balance-trained DQN being asked for a goal it was never told about.
UNTOLD_PARK_PLOT = "figures/dqn_untold_park.png"

# The DQN sees all four numbers, not the two the world model gets. This makes
# the comparison generous to the baseline on purpose: we are not trying to win
# by handicapping it.
STATE_SIZE = 4
N_ACTIONS = 2

# Two hidden layers of this width. Small, because CartPole is small, but not as
# skinny as a single layer: two layers trained noticeably more reliably here,
# and an unreliable baseline would be an unfair one.
HIDDEN_SIZE = 128

# How many episodes to train for, and the most steps that can buy. This is the
# number the article quotes as the DQN's cost for learning one goal.
TRAIN_EPISODES = 300

# Standard DQN machinery. None of it is the point of the article, so it is kept
# to the textbook defaults rather than tuned.
LEARNING_RATE = 5e-4
DISCOUNT = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = 50_000

# How often the slowly-updated copy of the network is refreshed, in steps.
# Refreshing it too rarely was the single biggest cause of training falling
# apart here, so this is deliberately frequent.
TARGET_UPDATE = 200

# Exploration: start by acting almost at random, end by trusting the network.
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_EPISODES = 150

# How many independently trained DQNs to average over. Reinforcement learning
# is noisy enough that a single training run is not evidence of anything.
N_SEEDS = 5

# Where CartPole ends the episode for leaving the rail, in metres. Used only to
# label what a training run did: fell over, or drove off the end.
RAIL_LIMIT = 2.4

# How many recent episodes to judge a snapshot on, when keeping the best one.
#
# DQN training is genuinely unstable: left to run, the same agent will solve
# CartPole outright at episode 240 and then forget how at episode 260. Reporting
# whatever happened to be in the weights at the final episode would understate
# the baseline and make our comparison look better than it is, so we keep the
# best version it ever reached. This costs no extra environment steps — it is
# judged on the episodes it was already playing.
SNAPSHOT_WINDOW = 20

SEED = 0


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Maps a state to a score for each of the two actions."""

    def __init__(self):
        super().__init__()

        # Two small layers. The output is one number per action: roughly "how
        # much total reward do I expect if I take this action from here".
        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, N_ACTIONS),
        )

    def forward(self, state):
        return self.net(state)


# ---------------------------------------------------------------------------
# Reward functions: the only way to tell a DQN what you want
# ---------------------------------------------------------------------------

def goal_reward(goal_name, observation, step, terminated):
    """The reward the DQN is trained on for a given goal.

    This function is the whole argument in miniature. The planner receives its
    objective at run time, as a scoring function over imagined futures. The DQN
    can only receive its objective *here*, before training, baked into the
    reward it learns from. Changing this line means throwing the agent away and
    training a new one.
    """
    # Falling over is always bad, whatever the goal.
    if terminated:
        return -1.0

    # Where the cart currently is, which every goal below cares about.
    position = observation[0]

    if goal_name == "balance":
        # The standard CartPole reward: one point per step survived.
        return 1.0

    if goal_name == "park":
        # Alive, minus how far the cart is from the target.
        return 1.0 - abs(position - PARK_TARGET)

    if goal_name == "corridor":
        # Alive, minus a penalty for straying outside the corridor.
        excess = max(abs(position) - CORRIDOR_HALF_WIDTH, 0.0)
        return 1.0 - 10.0 * excess

    # waypoint: the target moves partway through the episode.
    target = WAYPOINT_FIRST if step < WAYPOINT_HOLD_STEPS else WAYPOINT_SECOND
    return 1.0 - abs(position - target)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_dqn(goal_name, seed, episodes=TRAIN_EPISODES, verbose=False):
    """Train one DQN from scratch on one goal's reward.

    Returns the trained network, the number of environment steps it consumed
    getting there (the cost the article reports), and how long each training
    episode lasted, which is what learning progress looks like from outside.
    """
    # Make this run reproducible.
    torch.manual_seed(seed)
    random.seed(seed)

    # Unlike the world model, this agent learns by playing: every step below is
    # a real interaction, and counting them is the point of the exercise.
    env = gym.make("CartPole-v1")

    # The network being trained, and a slowly-updated copy of it used to
    # compute targets. Without the copy, the network chases its own moving
    # predictions and training tends to fall apart.
    online = QNetwork().to(DEVICE)
    target = QNetwork().to(DEVICE)

    # Start the copy off identical to the original.
    target.load_state_dict(online.state_dict())

    # Adam, at the usual learning rate. Nothing here is tuned.
    optimizer = torch.optim.Adam(online.parameters(), lr=LEARNING_RATE)

    # Past experience, sampled from at random so consecutive, highly similar
    # steps do not dominate each update.
    buffer = deque(maxlen=BUFFER_SIZE)

    # The number we are here to measure.
    env_steps = 0

    # Recent episode returns, and the best weights seen so far. See the comment
    # on SNAPSHOT_WINDOW for why keeping the best version matters.
    recent = deque(maxlen=SNAPSHOT_WINDOW)
    best_score, best_weights = -float("inf"), None

    # How long each episode lasted, purely so the article can show the agent
    # getting better (and, honestly, occasionally getting worse again).
    episode_lengths = []

    for episode in range(episodes):
        observation, _ = env.reset(seed=seed * 10_000 + episode)

        # Explore a lot early, then increasingly trust what has been learned.
        epsilon = max(
            EPSILON_END,
            EPSILON_START - (EPSILON_START - EPSILON_END)
            * episode / EPSILON_DECAY_EPISODES,
        )

        # What this episode earned, used only to decide whether the current
        # weights are the best so far.
        episode_return = 0.0

        for step in range(MAX_STEPS):
            # All four numbers, including the velocities the world model is
            # not allowed to see.
            state = torch.tensor(observation, dtype=torch.float32,
                                 device=DEVICE)

            if random.random() < epsilon:
                # Explore: try something and see what happens.
                action = random.randrange(N_ACTIONS)
            else:
                # Exploit: take whichever action the network rates higher.
                with torch.no_grad():
                    action = int(online(state).argmax())

            # A real interaction with the real environment.
            next_observation, _, terminated, truncated, _ = env.step(action)

            # The cost the article reports, ticking up one step at a time.
            env_steps += 1

            # The environment's own reward is discarded. The goal's reward
            # replaces it, which is the only channel through which a DQN can
            # be told what to want.
            reward = goal_reward(goal_name, next_observation, step, terminated)
            episode_return += reward

            buffer.append((observation, action, reward, next_observation,
                           float(terminated)))

            # Move on; next loop this becomes the current state.
            observation = next_observation

            # Learn from a random slice of past experience.
            if len(buffer) >= BATCH_SIZE:
                learn(online, target, optimizer, buffer)

            # Refresh the slowly-updated copy every so often.
            if env_steps % TARGET_UPDATE == 0:
                target.load_state_dict(online.state_dict())

            if terminated or truncated:
                break

        recent.append(episode_return)
        episode_lengths.append(step + 1)

        # Once there is a full window to judge, keep a copy of the weights
        # whenever the agent is playing better than it ever has.
        if len(recent) == SNAPSHOT_WINDOW and np.mean(recent) > best_score:
            best_score = float(np.mean(recent))
            best_weights = {k: v.clone()
                            for k, v in online.state_dict().items()}

        if verbose and (episode + 1) % 100 == 0:
            print(f"    episode {episode + 1:3d}: lasted {step + 1:3d} steps")

    env.close()

    # Hand back the best version it reached rather than whatever it happened to
    # be doing at the final episode.
    if best_weights is not None:
        online.load_state_dict(best_weights)

    # The trained agent, what it cost to get one, and the story of getting it.
    return online, env_steps, episode_lengths


def learn(online, target, optimizer, buffer):
    """One gradient step of ordinary DQN learning."""
    # A random slice of past experience, unpacked into columns.
    batch = random.sample(buffer, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    # Onto the GPU as tensors. The np.array() calls around the tuples are there
    # because building a tensor straight from a tuple of arrays is slow enough
    # for PyTorch to complain about it.
    states = torch.tensor(np.array(states), dtype=torch.float32, device=DEVICE)
    actions = torch.tensor(actions, device=DEVICE)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32,
                               device=DEVICE)

    # Stored as 1.0/0.0 rather than True/False so it can be used in arithmetic.
    dones = torch.tensor(dones, dtype=torch.float32, device=DEVICE)

    # What the network currently predicts for the actions actually taken.
    predicted = online(states).gather(1, actions[:, None]).squeeze(1)

    with torch.no_grad():
        # The best the slow copy thinks is available from the next state. The
        # (1 - dones) term zeroes this out where the episode ended: there is no
        # future to be had after falling over.
        best_next = target(next_states).max(dim=1).values
        wanted = rewards + DISCOUNT * best_next * (1.0 - dones)

    # Pull the prediction towards what actually happened. Huber loss rather
    # than squared error, because it shrugs off the occasional wild target.
    loss = nn.functional.smooth_l1_loss(predicted, wanted)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# ---------------------------------------------------------------------------
# The trained DQN, wrapped to look like every other agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """A trained DQN, exposing the same interface as the planner.

    Wrapping it this way is what lets both agents go through the identical
    evaluate() in plan.py, on identical seeds.
    """

    def __init__(self, network):
        self.network = network

        # Nothing here trains; we only read its opinions.
        self.network.eval()

    def reset(self):
        # Stateless: it looks only at the current observation, with no memory
        # of how it got here. That is also why it needs all four numbers.
        pass

    @torch.no_grad()
    def act(self, observation, step):
        # Timed on the same clock as the planner, so the two costs in the
        # article's table are measured the same way rather than one being
        # measured and the other waved at.
        start_time = time.perf_counter()

        # All four numbers, as during training.
        state = torch.tensor(observation, dtype=torch.float32, device=DEVICE)

        # One pass through a small network. This is why it is so much faster
        # than the planner: it is recalling an answer, not working one out.
        action = int(self.network(state).argmax())

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return action, elapsed_ms


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_learning(histories):
    """How long the DQN keeps the pole up as training goes on.

    Several seeds overlaid, because a single run would hide the thing this
    figure is honest about: two of these agents solve the game and the rest
    have a much worse time of it.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    for seed, lengths in enumerate(histories):
        # Raw episode lengths are extremely noisy, so smooth over a short
        # window to make the trend visible without hiding the instability.
        smoothed = np.convolve(lengths, np.ones(20) / 20, mode="valid")
        ax.plot(smoothed, alpha=0.8, label=f"training run {seed + 1}")

    # The best score the game allows, for scale.
    ax.axhline(MAX_STEPS, color="grey", ls="--", lw=0.8, label="perfect score")

    ax.set_xlabel("training episode")
    ax.set_ylabel("steps survived (smoothed)")
    ax.set_title("A DQN learning to balance, five separate times")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(LEARNING_PLOT, dpi=150)
    print(f"saved {LEARNING_PLOT}")


def plot_untold_park(traces):
    """A DQN trained only to balance, asked to park at PARK_TARGET.

    Nobody told it about the target, and there was no way to. This figure is
    what "the goal is set in concrete" looks like from the outside: a
    perfectly competent agent, cheerfully doing the wrong job.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    for i, trace in enumerate(traces[:5]):
        ax.plot(trace, color="tab:orange", alpha=0.8,
                label="DQN trained to balance" if i == 0 else None)

    # The thing we asked for and had no way to ask for.
    ax.axhline(PARK_TARGET, color="black", ls="--", lw=1.0,
               label=f"where we wanted it (x={PARK_TARGET})")

    ax.set_xlabel("step")
    ax.set_ylabel("cart position (m)")
    ax.set_title('"Park at x = +1.0", said nobody it could hear')
    ax.legend()
    fig.tight_layout()
    fig.savefig(UNTOLD_PARK_PLOT, dpi=150)
    print(f"saved {UNTOLD_PARK_PLOT}")


def plot_dqn_park(traces):
    """Where the retrained DQN takes the cart when told to park at the target.

    One trace per training run, not per episode. Runs of the same code differ
    far more from each other than episodes of the same run do, and that spread
    is the thing worth showing: some runs park, some drive off the rail.

    Deliberately shows the DQN on its own. In the article this figure appears
    before world models have been mentioned, so it must not give away what the
    alternative does.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    for i, trace in enumerate(traces):
        ax.plot(trace, alpha=0.85, label=f"training run {i + 1}")

    # Where it was asked to stop.
    ax.axhline(PARK_TARGET, color="black", ls="--", lw=1.0,
               label=f"target x={PARK_TARGET}")

    # Where the rail ends and the episode fails.
    ax.axhline(2.4, color="tab:red", ls=":", lw=1.0,
               label="end of the rail (x=2.4)")

    ax.set_xlabel("step")
    ax.set_ylabel("cart position (m)")
    ax.set_title("Five DQNs, same code, same reward, told to stop at x = +1.0")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(DQN_PARK_PLOT, dpi=150)
    print(f"saved {DQN_PARK_PLOT}")


def plot_comparison(rows):
    """One panel per goal, planner against DQN on that goal's own metric."""
    # One panel per goal, side by side, because the goals are measured in
    # different units and cannot share a y-axis.
    fig, axes = plt.subplots(1, len(rows), figsize=(4 * len(rows), 4))

    for ax, (goal_name, planner, dqn, untold) in zip(axes, rows):
        # Three bars, because there are three ways to be handed a goal: told at
        # run time, retrained for it, or never informed at all. The third bar
        # is the one that says what being *told* was worth.
        labels = ["world model\n(told)", "DQN\n(retrained)",
                  "DQN\n(never told)"]
        values = [planner["goal_score_mean"], dqn["goal_score_mean"],
                  untold["goal_score_mean"]]
        errors = [planner["goal_score_std"], dqn["goal_score_std"],
                  untold["goal_score_std"]]

        bars = ax.bar(labels, values, yerr=errors, capsize=4,
                      color=["tab:blue", "tab:orange", "tab:grey"])

        # Print how long each agent stayed alive on top of its bar. Without
        # this the waypoint panel is actively misleading: the DQN posts the
        # lower tracking error only because it falls over before the target
        # moves, so it is never scored on the hard half of the task.
        survivals = [planner["survival_mean"], dqn["survival_mean"],
                     untold["survival_mean"]]

        for bar, survived, spread in zip(bars, survivals, errors):
            # Sit the label clear of the top of the error bar, not the bar.
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + spread,
                    f"survived\n{survived:.0f}",
                    ha="center", va="bottom", fontsize=8)

        # Leave room above the bars for that annotation.
        ax.margins(y=0.25)

        ax.set_title(goal_name)

        # Each goal is judged in its own units, spelled out on the axis.
        ax.set_ylabel(REPORT_UNITS[goal_name])

    fig.suptitle("Same goals, three ways of being given them "
                 "(lower is better)")

    # Leave headroom for the suptitle rather than letting tight_layout fight it.
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(COMPARISON_PLOT, dpi=150)
    print(f"saved {COMPARISON_PLOT}")


def plot_park_traces(planner_traces, dqn_traces):
    """Where the cart actually goes when told to park at PARK_TARGET."""
    # A single panel: both agents belong on the same axes here, because the
    # comparison is exactly "does the cart end up on the line or not".
    fig, ax = plt.subplots(figsize=(7, 4))

    # A handful of episodes each, enough to show the pattern without turning
    # the figure into spaghetti.
    for i, trace in enumerate(planner_traces[:5]):
        ax.plot(trace, color="tab:blue", alpha=0.7,
                label="world model" if i == 0 else None)

    for i, trace in enumerate(dqn_traces[:5]):
        ax.plot(trace, color="tab:orange", alpha=0.7,
                label="DQN (retrained)" if i == 0 else None)

    # The line both agents are being asked to sit on.
    ax.axhline(PARK_TARGET, color="black", ls="--", lw=1.0,
               label=f"target x={PARK_TARGET}")

    ax.set_xlabel("step")
    ax.set_ylabel("cart position (m)")
    ax.set_title('"Balance the pole, and park at x = +1.0"')
    ax.legend()
    fig.tight_layout()
    fig.savefig(PARK_TRACE_PLOT, dpi=150)
    print(f"saved {PARK_TRACE_PLOT}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print(f"device: {DEVICE}")

    # The planner's side of the comparison, measured in plan.py. Reading it
    # back rather than recomputing keeps both halves of the comparison from
    # drifting apart between runs.
    with open(RESULTS_PATH) as handle:
        planner_results = json.load(handle)

    # Its cart traces, kept so both agents can be drawn on the same axes.
    planner_traces = np.load(TRACES_PATH)

    # First the concession. On plain CartPole the DQN is not worse than us, and
    # the article says so before claiming anything else.
    print("\nplain CartPole (the contest we do not win):")

    networks, costs, histories = [], [], []
    for seed in range(N_SEEDS):
        network, steps, lengths = train_dqn("balance", seed)
        networks.append(network)
        costs.append(steps)
        histories.append(lengths)

    # What learning looks like from the outside, before any of this is
    # compared against anything.
    plot_learning(histories)

    # Every DQN seed goes through the same evaluate() the planner used.
    balance_scores = [evaluate(DQNAgent(n), "balance")["survival_mean"]
                      for n in networks]

    print(f"  DQN          {np.mean(balance_scores):6.1f} +/- "
          f"{np.std(balance_scores):5.1f} steps  "
          f"({np.mean(costs):.0f} env steps to train)")
    print(f"  world model  "
          f"{planner_results['balance']['world model']['survival_mean']:6.1f} "
          f"+/- {planner_results['balance']['world model']['survival_std']:5.1f}"
          f" steps  (0 env steps)")

    # Now the measurement that makes the argument: take those same agents,
    # trained on nothing but "stay alive", and ask them for the new goals.
    #
    # This is the honest version of "you cannot tell a DQN what you want". It
    # is not that the agent breaks — it plays on perfectly well. It simply
    # carries on doing the only job it was ever given, because there is no
    # channel through which to mention the new one.
    # Everything worth quoting ends up in here and then in the JSON.
    summary = {}

    # Part 2 of the article quotes all three of these. Printing them was not
    # enough: a later rerun would move them and the draft would quietly lie.
    summary["balance_dqn"] = {
        "survival_mean": float(np.mean(balance_scores)),
        "survival_std": float(np.std(balance_scores)),
        "env_steps_mean": float(np.mean(costs)),
        "env_steps_std": float(np.std(costs)),
        "per_seed_survival": [float(s) for s in balance_scores],
        "per_seed_env_steps": [int(c) for c in costs],
        # The best episode each run ever reached while training, which is what
        # the article means by "two of our five runs did hit the perfect 500".
        "per_seed_best_episode": [int(max(h)) for h in histories],
    }

    print("\nthe same DQNs, asked for a goal nobody trained them on:")

    untold = {}
    for goal_name in ["park", "corridor", "waypoint"]:
        # Same networks as above. No retraining, no reward change, nothing.
        scores = [evaluate(DQNAgent(n), goal_name) for n in networks]

        untold[goal_name] = {
            "goal_score_mean": float(np.mean(
                [s["goal_score_mean"] for s in scores])),
            "goal_score_std": float(np.std(
                [s["goal_score_mean"] for s in scores])),
            "survival_mean": float(np.mean(
                [s["survival_mean"] for s in scores])),
            "latency_ms": float(np.mean([s["latency_ms"] for s in scores])),
        }

        print(f"  {goal_name:<10} {untold[goal_name]['goal_score_mean']:7.3f}"
              f"   {REPORT_UNITS[goal_name]}"
              f"   (survived {untold[goal_name]['survival_mean']:.0f})")

        # The picture the article needs: a competent agent, still balancing
        # beautifully, completely ignoring the thing we asked for.
        if goal_name == "park":
            plot_untold_park(scores[0]["positions"])

    # Then the part that matters: goals nobody trained for.
    print("\nnew goals. the planner was told; the DQN had to be retrained:")

    rows = []

    for goal_name in ["park", "corridor", "waypoint"]:
        # Retrain from scratch, because that is the only way to change what a
        # DQN wants. Several seeds, since one run proves nothing.
        dqn_scores, dqn_survivals, dqn_costs = [], [], []

        # One trace per training run, not one run's worth of traces. The
        # article's "retraining is a lottery" claim is a claim about
        # independent trainings, so the figure has to show independent
        # trainings: same evaluation seed, so the only difference between
        # these five lines is which training run produced them.
        dqn_run_traces, dqn_fates = [], []

        for seed in range(N_SEEDS):
            network, steps, _ = train_dqn(goal_name, seed)
            outcome = evaluate(DQNAgent(network), goal_name)

            dqn_scores.append(outcome["goal_score_mean"])
            dqn_survivals.append(outcome["survival_mean"])
            dqn_costs.append(steps)

            # Episode 0 of this run's evaluation, which every run faces from
            # the same starting state.
            trace = outcome["positions"][0]
            dqn_run_traces.append(trace)

            # What that run actually did, written down so the article's table
            # of fates can be checked against a file instead of a memory.
            dqn_fates.append({
                "seed": seed,
                "episode_length": int(len(trace)),
                "reached_500": bool(len(trace) >= MAX_STEPS),
                "final_position": float(trace[-1]),
                "closest_to_target": float(np.min(np.abs(trace - PARK_TARGET))),
                "left_the_rail": bool(np.max(np.abs(trace)) >= RAIL_LIMIT),
                "goal_score_mean": float(outcome["goal_score_mean"]),
                "survival_mean": float(outcome["survival_mean"]),
                "env_steps": int(steps),
            })

        dqn = {
            "goal_score_mean": float(np.mean(dqn_scores)),
            "goal_score_std": float(np.std(dqn_scores)),
            "survival_mean": float(np.mean(dqn_survivals)),
            "env_steps": float(np.mean(dqn_costs)),
            # Per-run spread, because "one run in five parks" is the claim.
            "env_steps_std": float(np.std(dqn_costs)),
            "per_run": dqn_fates,
        }

        # The planner's numbers for the same goal, measured with zero retraining.
        planner = planner_results["goals"][goal_name]

        print(f"  {goal_name}  ({REPORT_UNITS[goal_name]})")
        print(f"    world model  {planner['goal_score_mean']:7.3f} +/- "
              f"{planner['goal_score_std']:6.3f}   0 env steps, "
              f"survived {planner['survival_mean']:.0f}")
        print(f"    DQN          {dqn['goal_score_mean']:7.3f} +/- "
              f"{dqn['goal_score_std']:6.3f}   {dqn['env_steps']:.0f} env "
              f"steps, survived {dqn['survival_mean']:.0f}")

        rows.append((goal_name, planner, dqn, untold[goal_name]))

        # Three ways of being given a goal: told at run time, retrained for it,
        # or never informed at all.
        summary[goal_name] = {"planner": planner, "dqn": dqn,
                              "dqn_untold": untold[goal_name]}

        # Keep the parking traces for the figures below.
        if goal_name == "park":
            # One line per training run for the lottery figure.
            plot_dqn_park(dqn_run_traces)

            # The side-by-side planner comparison wants a spread of episodes
            # rather than one per run, so it gets the last run's evaluation.
            park_dqn_positions = outcome["positions"]

            # Print the fates so a rerun can be checked against the article.
            for fate in dqn_fates:
                ending = ("reached 500" if fate["reached_500"]
                          else f"fell at {fate['episode_length']}")
                print(f"    run {fate['seed'] + 1}: {ending}, "
                      f"closest {fate['closest_to_target']:.2f} m, "
                      f"ended at {fate['final_position']:+.2f} m"
                      f"{', off the rail' if fate['left_the_rail'] else ''}")

    plot_comparison(rows)

    # The traces the planner produced for the parking goal, saved by plan.py
    # under keys of the form "park_0", "park_1", and so on.
    planner_park = [planner_traces[k] for k in planner_traces.files
                    if k.startswith("park_")]
    plot_park_traces(planner_park, park_dqn_positions)

    # The cost we disclose rather than hide: 200 dreams per decision is not
    # free, and the article says so in the same breath as the wins. Both sides
    # are now actually timed, on the same clock, inside the same evaluate().
    planner_latency = planner_results["balance"]["world model"]["latency_ms"]
    dqn_latency = float(np.mean([u["latency_ms"] for u in untold.values()]))

    # Not one number. A dream costs what its horizon costs, and the goals that
    # need the cart to travel dream roughly twice as deep as the one that only
    # needs the pole up, so they cost roughly twice as much (see D10).
    per_goal_latency = {name: planner_results["goals"][name]["latency_ms"]
                        for name in ["park", "corridor", "waypoint"]}
    per_goal_latency["balance"] = planner_latency

    print(f"\ndecision time: DQN {dqn_latency:.3f} ms/step")
    for name, ms in per_goal_latency.items():
        print(f"  {name:<10} {ms:.3f} ms/step "
              f"({ms / max(dqn_latency, 1e-9):.0f}x the DQN)")

    summary["latency_ms"] = {"world_model": planner_latency,
                             "dqn": dqn_latency,
                             "per_goal": per_goal_latency}

    os.makedirs(os.path.dirname(RESULTS_OUT), exist_ok=True)
    with open(RESULTS_OUT, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"saved {RESULTS_OUT}")


if __name__ == "__main__":
    main()
