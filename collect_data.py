"""Generate the offline CartPole dataset.

Rolls out `CartPole-v1` under a mixed policy: a hand-written proportional
controller for stable balancing data, with epsilon-greedy noise on top for
recovery dynamics and near-failure edge cases.

Records all 4 observation dimensions (the model is fed only 2 of them; the
other 2 exist so the article can plot ground truth), the actions, and the
`terminated` / `truncated` flags kept as separate arrays.

Output: data/cartpole.npz
"""

import json
import os

import gymnasium as gym
import numpy as np

# ---------------------------------------------------------------------------
# Settings. Everything you might want to change lives here, so no script in
# this repo needs command-line arguments.
# ---------------------------------------------------------------------------

# How many episodes to play. 500 episodes of a decent controller gives us
# roughly 100k recorded steps, which trains in a couple of minutes on a CPU.
N_EPISODES = 500

# Probability of ignoring the controller and acting randomly on any given step.
# This is the single most important number in this file. At 0 the controller
# balances perfectly and we only ever see the pole near-vertical, so the model
# never learns what recovering from a lean looks like. Too high and the pole
# falls immediately and we never see stable balancing. 0.25 gives us both.
EPSILON = 0.25

# The controller's only tuning knob: how much it cares about the pole's angular
# velocity relative to its angle. Without this term the controller reacts to
# where the pole *is* and always overcorrects; with it, the controller reacts to
# where the pole is *going*.
ANGULAR_VELOCITY_GAIN = 0.5

# Each episode gets a small constant nudge added to the controller's decision,
# drawn from +/- this value. A biased controller still balances the pole, but
# it does so while sliding steadily along the rail in one direction.
#
# This exists because of a failure we ran into and had to fix. Without it, the
# controller corrects every wobble immediately and the cart never travels far,
# so the dataset contains no long runs of pushes in the same direction. A model
# trained on that has no idea what sustained pushing does, and the planner in
# plan.py — which asks exactly that question — gets confident nonsense back.
#
# The lesson mirrors the one about episode length: it is not enough for the data
# to cover the states you care about, it must also cover the *actions* you plan
# to ask about.
DRIFT_BIAS = 0.04

# Fixed seed so that re-running this script reproduces the same dataset.
SEED = 0

# Where the dataset lands. train.py reads this exact path.
OUT_PATH = "data/cartpole.npz"

# Dataset statistics the article quotes. data/ is gitignored; this is not.
STATS_PATH = "checkpoints/data_stats.json"

# CartPole's observation is 4 numbers, and these are their positions in the
# array. Naming them avoids a file full of unexplained obs[2] indexing.
CART_POSITION = 0
CART_VELOCITY = 1
POLE_ANGLE = 2
POLE_ANGULAR_VELOCITY = 3


def proportional_controller(observation, bias=0.0):
    """Decide which way to push, using a rule a human can read.

    This replaces training a reinforcement learning agent just to collect data.
    We only need *competent* play, not optimal play, and a competent CartPole
    policy fits on one line.

    `bias` tilts the rule towards one direction. At zero it simply balances in
    place; pushed away from zero it balances while drifting along the rail,
    which is how the dataset gets its long one-directional action runs.
    """
    # Read the pole's current lean. Positive means it is tipping to the right.
    angle = observation[POLE_ANGLE]

    # Read how fast it is tipping. Positive means it is rotating rightward,
    # so even a pole that is currently upright may be about to fall right.
    angular_velocity = observation[POLE_ANGULAR_VELOCITY]

    # Combine "where it is" with "where it is heading" into a single number.
    # Looking ahead like this is what stops the controller from oscillating:
    # it starts correcting before the pole has actually leaned over.
    #
    # The bias shifts where the controller thinks "upright" is. A small offset
    # is not enough to drop the pole, but it is enough to make the cart lean
    # into one direction and keep sliding that way.
    lean = angle + ANGULAR_VELOCITY_GAIN * angular_velocity + bias

    # To catch a falling pole you move the cart *underneath* it, in the same
    # direction it is falling. Action 1 pushes the cart right, action 0 left.
    return 1 if lean > 0 else 0


def collect(n_episodes=N_EPISODES, epsilon=EPSILON, seed=SEED):
    """Play `n_episodes` games and record every transition."""
    # Build the environment. "CartPole-v1" comes wrapped in a time limit that
    # cuts episodes off at 500 steps — that wrapper is the sole source of the
    # `truncated` flag we deal with below.
    env = gym.make("CartPole-v1")

    # A dedicated random number generator for the epsilon-greedy noise. Keeping
    # it separate from the environment's own seeding makes both reproducible
    # without one interfering with the other.
    rng = np.random.default_rng(seed)

    # Buffers for the whole dataset. We append one entry per *transition*, so
    # all five lists stay the same length and line up index for index.
    observations = []  # the state before acting
    actions = []  # what we did
    next_observations = []  # the state that resulted
    terminateds = []  # did the pole actually fall as a result?
    truncateds = []  # or did the 500-step clock simply run out?

    # Episodes have different lengths, so we also record where each one ends.
    # train.py needs this to avoid building a training sequence that runs off
    # the end of one episode and into the start of an unrelated one.
    episode_lengths = []

    for episode in range(n_episodes):
        # Reset to a fresh, slightly randomised starting state. Giving each
        # episode its own seed makes the entire dataset reproducible.
        observation, _ = env.reset(seed=seed + episode)

        # Pick this episode's drift direction. Some episodes balance in place,
        # some slide left, some slide right, so the dataset ends up containing
        # long runs of pushes in the same direction as well as tidy balancing.
        bias = rng.uniform(-DRIFT_BIAS, DRIFT_BIAS)

        # Counts transitions in this episode, for episode_lengths.
        steps = 0

        # Play until the pole falls or the clock runs out.
        while True:
            # Roll the dice: mostly follow the controller, occasionally do
            # something random. Note the controller is allowed to look at the
            # velocities even though the model never will — the collector is
            # not the thing we are handicapping.
            if rng.random() < epsilon:
                # A random shove. These are what push the cart into awkward
                # states the controller would never visit on its own, which is
                # exactly the data the model needs to learn recovery physics.
                action = int(rng.integers(2))
            else:
                # The controller, nudged in this episode's drift direction.
                action = proportional_controller(observation, bias)

            # Advance the simulator by one step. `terminated` and `truncated`
            # mean genuinely different things and we keep them apart all the
            # way through the project — see the note below the loop.
            next_observation, _, terminated, truncated, _ = env.step(action)

            # Record the transition. We store the full 4-dimensional
            # observation even though the model will only be shown 2 of the
            # dimensions, because the article's plots need the ground truth.
            observations.append(observation)
            actions.append(action)
            next_observations.append(next_observation)
            terminateds.append(terminated)
            truncateds.append(truncated)

            # The state we just arrived in is the state we act from next.
            observation = next_observation
            steps += 1

            # Either flag ends the episode, but they are recorded separately
            # above, so nothing is lost by treating them the same here.
            if terminated or truncated:
                break

        episode_lengths.append(steps)

    env.close()

    # Pack everything into arrays. float32 rather than float64 because that is
    # what PyTorch wants, and converting here saves doing it every batch.
    dataset = {
        "observations": np.array(observations, dtype=np.float32),
        "actions": np.array(actions, dtype=np.int64),
        "next_observations": np.array(next_observations, dtype=np.float32),
        # Kept as separate arrays, never combined into a single "done" flag.
        # `terminated` is a physical failure and is a function of the
        # observation, so the model can learn to predict it. `truncated` is the
        # step counter hitting 500, which the model cannot see and therefore
        # cannot predict. Merging them would teach the model to guess.
        "terminated": np.array(terminateds, dtype=bool),
        "truncated": np.array(truncateds, dtype=bool),
        "episode_lengths": np.array(episode_lengths, dtype=np.int64),
    }

    # Handed back rather than saved here, so imagine.py can call collect()
    # directly for fresh evaluation episodes without touching the disk.
    return dataset


def summarise(dataset):
    """Print a few numbers so you can sanity-check the dataset before training.

    Also writes them to STATS_PATH. The article quotes these, and `data/` is
    gitignored, so printing alone would leave the claims unverifiable the
    moment the terminal scrolled.
    """
    # How long each episode ran, which is the headline health check: too short
    # and the controller is not working, all at 500 and there is no failure data.
    lengths = dataset["episode_lengths"]

    print(f"episodes:            {len(lengths)}")
    print(f"total transitions:   {len(dataset['actions'])}")
    print(f"episode length:      mean {lengths.mean():.1f}, "
          f"min {lengths.min()}, max {lengths.max()}")

    # How many episodes ended each way. A healthy mixed dataset has mostly
    # terminations (the pole fell) with a handful of truncations (survived the
    # full 500 steps). All truncations would mean epsilon is too low and we
    # have no failure data; zero would mean the controller is not balancing.
    print(f"ended by falling:    {dataset['terminated'].sum()}")
    print(f"ended by time limit: {dataset['truncated'].sum()}")

    # The spread of each observation dimension. Useful because train.py
    # standardises the observations, and wildly different scales here are why
    # that step is necessary.
    observations = dataset["observations"]

    # In the same order as the four columns of the observation array.
    names = ["cart position", "cart velocity", "pole angle", "pole ang. vel."]

    for i, name in enumerate(names):
        # One dimension at a time, across every recorded step.
        column = observations[:, i]
        print(f"  {name:<15} mean {column.mean():+.3f}  std {column.std():.3f}")

    # The same numbers, written down. One JSON field per claim in the article.
    stats = {
        "episodes": int(len(lengths)),
        "total_transitions": int(len(dataset["actions"])),
        "episode_length_mean": float(lengths.mean()),
        "episode_length_min": int(lengths.min()),
        "episode_length_max": int(lengths.max()),
        "ended_by_falling": int(dataset["terminated"].sum()),
        "ended_by_time_limit": int(dataset["truncated"].sum()),
        "observation_mean": {n: float(observations[:, i].mean())
                             for i, n in enumerate(names)},
        "observation_std": {n: float(observations[:, i].std())
                            for i, n in enumerate(names)},
        # Recorded because the article explains why this exists (D1), and a
        # future reader should be able to see it was actually switched on.
        "drift_bias": DRIFT_BIAS,
        "epsilon": EPSILON,
    }

    os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
    with open(STATS_PATH, "w") as handle:
        json.dump(stats, handle, indent=2)
    print(f"saved {STATS_PATH}")


if __name__ == "__main__":
    print(f"collecting {N_EPISODES} episodes (epsilon={EPSILON})...")

    # Play the games and gather every transition into memory.
    dataset = collect()

    # Make sure the output directory exists before writing into it.
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # Save every array into one compressed file, keyed by the names above.
    np.savez_compressed(OUT_PATH, **dataset)

    # Print the sanity checks after saving, so the file exists even if the
    # numbers look wrong and you want to inspect them by hand.
    summarise(dataset)
    print(f"\nsaved to {OUT_PATH}")
