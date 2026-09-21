"""Control the real CartPole using only the model's imagination.

At every real step: invent 200 random action sequences, dream them all forward
inside the world model (the simulator is not touched), score each dream with a
function from `goals.py`, then execute just the first action of the best one.

Scoring runs over the whole imagined future, so the planner will accept a worse
next step to reach a better future further out. Swapping in a different goal is
the entire mechanism behind the article's main claim: a new objective at run
time, with no retraining and no new environment steps.

The planning horizon comes from the drift measurement in `imagine.py` — plan no
further ahead than the dream can be trusted.

Also records decision-time latency per action, which we report as a cost: 200
dreams per step is far slower than the baseline's single forward pass.

Output: per-goal performance against the DQN baseline, a survival-vs-horizon
sweep, and the latency numbers. Written to figures/
"""

import json
import os
import time

import gymnasium as gym
import matplotlib
import numpy as np
import torch

# Draw to a file rather than a window, so this runs the same over SSH.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from collect_data import (ANGULAR_VELOCITY_GAIN, POLE_ANGLE,  # noqa: E402
                          POLE_ANGULAR_VELOCITY)
from goals import (GOAL_HORIZON, GOALS, REPORT_UNITS,  # noqa: E402
                   report)
from model import VISIBLE_DIMS, WorldModel  # noqa: E402

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = "checkpoints/world_model.pt"

# Written by imagine.py. Reading it means the planning horizon is a measured
# number rather than one we picked and hoped about.
HORIZON_PATH = "checkpoints/horizon.json"

RESULTS_PATH = "checkpoints/plan_results.json"

# The cart-position traces from the goal episodes, kept so baseline.py can draw
# the planner and the DQN on the same axes without rerunning the planner.
TRACES_PATH = "checkpoints/plan_traces.npz"
SURVIVAL_PLOT = "figures/planner_survival.png"
SWEEP_PLOT = "figures/horizon_sweep.png"

# How many random action sequences to dream at each real step. More candidates
# means better plans and slower decisions. 200 is where the returns flatten out
# here: the goals that require steering the cart genuinely need the extra
# samples, because most random sequences drop the pole and get thrown away.
N_CANDIDATES = 200

# How many real steps to warm the memory up on before planning begins. The
# model cannot infer velocity from a single frame, so its first few decisions
# would otherwise be made blind.
WARMUP = 8

# During warm-up the planner has no usable memory yet, so it follows the same
# hand-written controller that collected the training data.
# (Imported from collect_data so there is exactly one copy of that rule.)

# Episodes per measurement, and the seed they start from. Fixed so every agent
# in the comparison faces exactly the same set of starting positions. A hundred
# is enough that the spread between agents is bigger than the noise within one.
N_EVAL_EPISODES = 100
EVAL_SEED = 50_000

# Horizons to try in the sweep. Spans well below and well above the measured
# trustworthy horizon, which is the point: we want to see it break at both ends.
HORIZON_SWEEP = [1, 3, 5, 10, 20, 29, 40, 60, 80]

# CartPole's own ceiling. No episode can run longer than this.
MAX_STEPS = 500

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def load_model():
    """Rebuild the trained world model from its checkpoint."""
    # Weights and normalization statistics travel together in one file.
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    # An empty model of the same shape, to pour the saved weights into.
    model = WorldModel().to(DEVICE)

    # Restores weights and normalization statistics in a single call.
    model.load_state_dict(checkpoint["model"])

    # Nothing here trains; we only ever read predictions out of the model.
    model.eval()
    return model


def load_horizon():
    """Read the planning horizon that imagine.py measured."""
    # Falling back to a hard-coded value would quietly hide a missing step in
    # the pipeline, so fail loudly instead.
    with open(HORIZON_PATH) as handle:
        return json.load(handle)["horizon"]


def horizon_for(goal_name, measured):
    """How deep this particular goal should dream.

    Most goals use a shorter, more reliable horizon than the measured one; the
    goals that have to steer the cart somewhere use the full measured depth.
    See the comment on GOAL_HORIZON for the reasoning.
    """
    # A None entry means "use everything the drift measurement allows".
    return GOAL_HORIZON[goal_name] or measured


def controller_action(observation):
    """The hand-written rule, used during warm-up and as a reference agent."""
    # Identical to the one that collected the dataset: lean the way the pole is
    # leaning, accounting for how fast it is already tipping.
    lean = (observation[POLE_ANGLE]
            + ANGULAR_VELOCITY_GAIN * observation[POLE_ANGULAR_VELOCITY])
    return 1 if lean > 0 else 0


# ---------------------------------------------------------------------------
# The planner
# ---------------------------------------------------------------------------

@torch.no_grad()
def choose_action(model, observation_norm, hidden, goal, horizon, step, rng):
    """Dream N_CANDIDATES futures and return the first action of the best one.

    observation_norm: (1, 2) the current real observation, normalized
    hidden:           (1, hidden_size) the memory, warmed up on real steps
    goal:             a scoring function from goals.py
    horizon:          how many steps to dream ahead
    step:             how many real steps have been taken, for moving targets

    Returns the chosen action and how long the decision took, in milliseconds.
    """
    # Start the clock. Decision latency is a cost we report rather than hide.
    start_time = time.perf_counter()

    # A fresh batch of random action sequences. This is the "invent 200
    # possible futures" step, and random really is enough on a problem this
    # small. Fifty was not: see D8 in project_plan.md.
    candidates = torch.from_numpy(
        rng.integers(0, 2, size=(N_CANDIDATES, horizon))
    ).long().to(DEVICE)

    # Every candidate starts from the same real state and the same memory, so
    # the only difference between them is the actions they take.
    start = observation_norm.expand(N_CANDIDATES, -1)

    # The same memory handed to all 200 dreamers. contiguous() because the GRU
    # needs a real tensor rather than a broadcast view of one.
    memory = hidden.expand(N_CANDIDATES, -1).contiguous()

    # Dream all 200 futures at once. The real simulator is not touched here —
    # this is entirely inside the network.
    imagined_norm, termination_logits, _ = model.imagine(
        start, candidates, memory
    )

    # Back into metres and radians, because the goal functions are written in
    # real units and that is what makes them readable.
    imagined = model.denormalize(imagined_norm)

    # Score every dream over its entire length. This is where the objective
    # enters, and it is the only thing that changes between goals.
    scores = goal(imagined, termination_logits, step=step)

    # Pick the best future, then throw away everything except its opening move.
    # Next step we will reconsider all of this from scratch.
    best = int(scores.argmax())

    # Column 0 is the opening move of the winning sequence. The remaining
    # horizon - 1 steps of that plan are discarded; only this one gets played,
    # and next step the whole search runs again from the new real state.
    action = int(candidates[best, 0])

    # How long all that daydreaming took, in milliseconds.
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return action, elapsed_ms


# ---------------------------------------------------------------------------
# Agents
#
# Every agent exposes the same two methods, so the evaluation loop below never
# needs to know which one it is driving. baseline.py plugs its DQN in here too,
# which is what stops either side of the comparison getting a private, subtly
# more favourable measurement path.
# ---------------------------------------------------------------------------

class RandomAgent:
    """Coin flips. The floor that any method has to clear."""

    def __init__(self, seed=SEED):
        # Its own generator, so its choices repeat exactly across reruns.
        self.rng = np.random.default_rng(seed)

    def reset(self):
        # Nothing to forget between episodes.
        pass

    def act(self, observation, step):
        # A shove in a random direction, and no time spent deciding.
        return int(self.rng.integers(2)), 0.0


class ControllerAgent:
    """The three-line rule that collected the training data."""

    def reset(self):
        # Stateless: it looks only at the current observation.
        pass

    def act(self, observation, step):
        # Lean the way the pole is leaning. No imagination involved.
        return controller_action(observation), 0.0


class PlannerAgent:
    """Chooses actions by dreaming futures inside the world model."""

    def __init__(self, model, goal_name, horizon, seed=SEED):
        self.model = model

        # The scoring function that decides what a good future looks like.
        # Swapping this is the entire mechanism behind the article's claim.
        self.goal = GOALS[goal_name]

        self.horizon = horizon

        # Drives the random candidate action sequences.
        self.rng = np.random.default_rng(seed)

        # Filled in by reset() before each episode.
        self.hidden = None

    def reset(self):
        # A blank memory. At this point the model has seen nothing and cannot
        # know which way anything is moving.
        self.hidden = self.model.initial_hidden(1, DEVICE)

    @torch.no_grad()
    def act(self, observation, step):
        # The two numbers the model is allowed to see, shaped as a batch of 1.
        visible = torch.tensor(
            observation[list(VISIBLE_DIMS)], dtype=torch.float32, device=DEVICE
        )[None]

        # Onto the scale the model was trained on.
        observation_norm = self.model.normalize(visible)

        if step < WARMUP:
            # Still priming the memory. Fall back on the hand-written rule,
            # because planning on an empty memory is planning blind.
            action, latency = controller_action(observation), 0.0
        else:
            # From here on, every decision comes out of imagination alone.
            action, latency = choose_action(
                self.model, observation_norm, self.hidden,
                self.goal, self.horizon, step, self.rng,
            )

        # Advance the memory by the step we are actually about to take, so the
        # model's sense of velocity keeps up with the real world.
        _, _, self.hidden = self.model.step(
            observation_norm,
            torch.tensor([action], device=DEVICE),
            self.hidden,
        )

        return action, latency


# ---------------------------------------------------------------------------
# Measurement, shared by every agent
# ---------------------------------------------------------------------------

def run_episode(agent, seed):
    """Play one real episode of CartPole with whichever agent is handed in.

    Returns how long it survived, the cart positions it visited, and the mean
    time each decision took.
    """
    # A real CartPole. The planner's dreams never touch this; only the single
    # chosen action per step does.
    env = gym.make("CartPole-v1")
    observation, _ = env.reset(seed=seed)

    # Clear any memory carried over from the previous episode.
    agent.reset()

    # Where the cart went, and how long each decision took.
    positions, latencies = [], []

    for step in range(MAX_STEPS):
        # The agent's whole interface: look at the world, return a move.
        action, latency = agent.act(observation, step)

        # Only count decisions that actually involved thinking.
        if latency > 0.0:
            latencies.append(latency)

        # Take that action for real.
        observation, _, terminated, truncated, _ = env.step(action)

        # Record where the cart went, which is what the position goals are
        # judged on after the fact.
        positions.append(observation[0])

        if terminated or truncated:
            break

    env.close()

    # step + 1 because the loop counter is zero-based.
    survived = step + 1

    # The `or [0.0]` covers agents that never think, and episodes that ended
    # before the planner's warm-up finished.
    return survived, np.array(positions), float(np.mean(latencies or [0.0]))


def evaluate(agent, goal_name, n_episodes=N_EVAL_EPISODES):
    """Average one agent's performance over a fixed set of starting states.

    Every agent in the article goes through this exact function, on the exact
    same seeds, so no comparison can be an artefact of how it was measured.
    """
    # Per-episode outcomes, averaged at the end.
    survivals, goal_scores, latencies, all_positions = [], [], [], []

    for episode in range(n_episodes):
        # Same seeds for every agent and every goal, so the contest is fair.
        survived, positions, latency = run_episode(agent, EVAL_SEED + episode)

        survivals.append(survived)
        latencies.append(latency)
        all_positions.append(positions)

        # The goal's own metric, measured on what really happened rather than
        # on what the model imagined would happen.
        goal_scores.append(report(goal_name, torch.from_numpy(positions),
                                  survived))

    return {
        "survival_mean": float(np.mean(survivals)),
        "survival_std": float(np.std(survivals)),
        "goal_score_mean": float(np.mean(goal_scores)),
        "goal_score_std": float(np.std(goal_scores)),
        "latency_ms": float(np.mean(latencies)),
        "positions": all_positions,
    }


def sweep_horizon(model, goal_name="balance"):
    """Measure how survival depends on how far ahead the planner dreams.

    This is the experiment that connects back to imagine.py: too short and the
    planner is short-sighted, too long and it optimises against a dream that
    has already drifted into fiction.
    """
    # One (horizon, mean, spread) triple per horizon tried.
    results = []

    for horizon in HORIZON_SWEEP:
        # A fresh planner per horizon; everything else about it is identical.
        agent = PlannerAgent(model, goal_name, horizon)

        # Fewer episodes per point, since this sweep is about the shape of the
        # curve rather than a precise number at any one horizon.
        outcome = evaluate(agent, goal_name, n_episodes=10)

        results.append((horizon, outcome["survival_mean"],
                        outcome["survival_std"]))
        print(f"  horizon {horizon:3d}: survived "
              f"{outcome['survival_mean']:6.1f} +/- {outcome['survival_std']:5.1f} steps")

    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_survival(results):
    """Bar chart: how long each agent keeps the pole up."""
    fig, ax = plt.subplots(figsize=(6, 4))

    # One bar per agent, in the order they were measured.
    names = list(results.keys())

    # Average survival time across the evaluation episodes.
    means = [results[n]["survival_mean"] for n in names]

    # Spread across those same episodes, drawn as error bars.
    errors = [results[n]["survival_std"] for n in names]

    # Error bars show the spread across episodes, which matters here: an agent
    # that usually survives 500 steps and occasionally dies at 40 is a
    # different proposition from one that reliably survives 400.
    ax.bar(names, means, yerr=errors, capsize=4)

    ax.set_ylabel("steps survived (max 500)")
    ax.set_title("Keeping the pole up")

    # Stop the axis labels from being clipped.
    fig.tight_layout()
    fig.savefig(SURVIVAL_PLOT, dpi=150)
    print(f"saved {SURVIVAL_PLOT}")


def plot_sweep(sweep, measured_horizon):
    """How planning quality depends on how far ahead the planner looks."""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Unpack the (horizon, mean, spread) triples into three parallel lists.
    horizons = [h for h, _, _ in sweep]
    means = [m for _, m, _ in sweep]
    errors = [s for _, _, s in sweep]

    # Points with error bars rather than a bare line, because the spread across
    # episodes is part of the story at the extreme horizons.
    ax.errorbar(horizons, means, yerr=errors, marker="o", capsize=3)

    # The horizon imagine.py measured, drawn here to show whether the drift
    # measurement actually predicted where planning works best.
    ax.axvline(measured_horizon, color="red", ls="--", lw=0.8,
               label=f"measured trustworthy horizon H={measured_horizon}")

    ax.set_xlabel("planning horizon (steps dreamed ahead)")
    ax.set_ylabel("steps survived (max 500)")
    ax.set_title("How far ahead is it worth dreaming?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SWEEP_PLOT, dpi=150)
    print(f"saved {SWEEP_PLOT}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print(f"device: {DEVICE}")

    # The trained physics engine. Nothing in this file modifies it.
    model = load_model()

    # The horizon is measured, not chosen. imagine.py wrote it out.
    horizon = load_horizon()
    print(f"planning horizon from imagine.py: H = {horizon}")

    # Keyed by agent name, so the bar chart labels itself.
    results = {}

    # First the sanity check: can the planner balance at all, and how does it
    # compare to doing nothing clever?
    print("\nbalancing:")

    # Coin flips: the floor any method has to clear.
    results["random"] = evaluate(RandomAgent(), "balance")

    # The three-line rule that collected our training data.
    results["hand-written"] = evaluate(ControllerAgent(), "balance")

    # The planner, deciding purely by imagining futures.
    results["world model"] = evaluate(
        PlannerAgent(model, "balance", horizon_for("balance", horizon)),
        "balance",
    )

    for name, outcome in results.items():
        print(f"  {name:<14} {outcome['survival_mean']:6.1f} +/- "
              f"{outcome['survival_std']:5.1f} steps   "
              f"{outcome['latency_ms']:.2f} ms/decision")

    # The bar chart that answers "does this work at all?".
    plot_survival(results)

    # Then the experiment that ties back to the drift measurement.
    print("\nhorizon sweep:")
    sweep = sweep_horizon(model)

    # Drawn with the measured horizon marked, to see whether it predicted the
    # planner's sweet spot.
    plot_sweep(sweep, horizon)

    # Now the part the article is actually about: the same weights, the same
    # dreams, three objectives nobody trained for. Nothing below retrains or
    # even touches the model's parameters.
    print("\nnew goals, no retraining:")

    goal_results, goal_traces, controller_results = {}, {}, {}

    for goal_name in ["park", "corridor", "waypoint"]:
        # A planner identical to the balancing one, except for which scoring
        # function it consults. That single swap is the whole mechanism.
        agent = PlannerAgent(model, goal_name,
                             horizon_for(goal_name, horizon))
        outcome = evaluate(agent, goal_name)

        # Keep the position traces for the figures baseline.py will draw.
        goal_traces[goal_name] = outcome.pop("positions")
        goal_results[goal_name] = outcome

        # The hand-written rule scored on the same goal, for the same reason
        # the drift plot has a freeze-last-state line: a number needs a floor
        # to be read against. The controller is not trying to do any of these
        # jobs, so this is what "balances well, ignores the request" looks
        # like from something that never even had a request to ignore.
        baseline_outcome = evaluate(ControllerAgent(), goal_name)
        baseline_outcome.pop("positions")
        controller_results[goal_name] = baseline_outcome

        print(f"  {goal_name:<10} {outcome['goal_score_mean']:7.3f} +/- "
              f"{outcome['goal_score_std']:6.3f}  {REPORT_UNITS[goal_name]}"
              f"   (survived {outcome['survival_mean']:.0f} steps)"
              f"   [controller {baseline_outcome['goal_score_mean']:.3f}, "
              f"survived {baseline_outcome['survival_mean']:.0f}]")

    # Save the numbers so the article can quote them without a rerun.
    summary = {
        "horizon": horizon,
        "goals": goal_results,
        # The same goals scored by the hand-written rule, so every row of the
        # article's comparison tables can be pointed at a field in this file.
        "controller_goals": controller_results,
        # Which horizon each goal actually planned at. The article quotes
        # these, and they are not all the measured H (see D10).
        "goal_horizons": {name: horizon_for(name, horizon)
                          for name in ["balance", "park", "corridor",
                                       "waypoint"]},
        # The per-episode position arrays are dropped: they are large, and the
        # figures that needed them have already been drawn.
        "balance": {
            name: {k: v for k, v in outcome.items() if k != "positions"}
            for name, outcome in results.items()
        },
        "sweep": sweep,
    }

    # Create checkpoints/ if this is the first run.
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nsaved {RESULTS_PATH}")

    # Episodes have different lengths, so the traces cannot go in one array.
    # Saving them keyed by goal and episode keeps the file flat and simple.
    flat_traces = {
        f"{goal}_{i}": trace
        for goal, traces in goal_traces.items()
        for i, trace in enumerate(traces)
    }
    np.savez_compressed(TRACES_PATH, **flat_traces)
    print(f"saved {TRACES_PATH}")


if __name__ == "__main__":
    main()
