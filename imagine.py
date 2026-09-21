"""Roll the world model forward without the simulator, and measure the drift.

Warms the hidden state up on a few real steps, then feeds the model its own
predictions for the rest of the horizon. Stops when predicted termination
probability crosses 0.5, or at a manually capped horizon (the imagined world
has no notion of the 500-step time limit).

Produces the article's figures: real vs. imagined pole angle, and prediction
error as a function of rollout horizon.

The point of that error curve is the number it yields — how many steps the dream
stays trustworthy. That number becomes the planning horizon in `plan.py`.

Output: figures/
"""

import json
import os

import matplotlib
import numpy as np
import torch

# Render to files rather than a window, so this works over SSH.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from collect_data import CART_POSITION, POLE_ANGLE, collect  # noqa: E402
from model import VISIBLE_DIMS, WorldModel  # noqa: E402

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = "checkpoints/world_model.pt"

# Where the measured planning horizon gets written, so plan.py can read it
# instead of us copying a number by hand between two files.
HORIZON_PATH = "checkpoints/horizon.json"

TRAJECTORY_PLOT = "figures/imagined_vs_real.png"
DRIFT_PLOT = "figures/drift_vs_horizon.png"
TERMINATION_PLOT = "figures/termination_timing.png"

# Fresh episodes the model has never seen. Seeded well past the training range
# so there is no chance of overlap with the dataset in data/.
N_EVAL_EPISODES = 200
EVAL_SEED = 10_000

# Real steps fed in before imagination begins. The model starts with a blank
# memory and the velocities are hidden from it, so without a warm-up it cannot
# know whether the pole is falling left or right — only where it currently is.
WARMUP = 8

# How far ahead to imagine. Long enough that the model is clearly wrong by the
# end, because the point of this script is to find where that happens.
HORIZON = 100

# How much pole-angle error we are willing to tolerate before calling the dream
# untrustworthy. The pole fails at 0.2095 rad, so this is 10% of the distance
# to failure — small enough that a planner deciding on the basis of an imagined
# angle would still be deciding about roughly the right situation.
ANGLE_TOLERANCE = 0.02

# How many example episodes to draw in the trajectory figure.
N_EXAMPLE_EPISODES = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def load_model():
    """Rebuild the trained model from its checkpoint."""
    # The checkpoint holds the weights and the normalization statistics
    # together, because those statistics were saved as buffers on the model.
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    # Build an untrained model of the same shape to load the weights into.
    model = WorldModel().to(DEVICE)

    # This restores the weights *and* the normalization statistics in one go.
    model.load_state_dict(checkpoint["model"])

    # Evaluation mode, and no gradients anywhere: we are only ever reading
    # predictions out of this model from here on.
    model.eval()
    return model


def evaluation_episodes():
    """Collect fresh episodes and reshape them into fixed-length blocks.

    Returns real observations, the actions taken, and the episode lengths, with
    every episode long enough to imagine a full HORIZON into.
    """
    # Reuse the same collector that built the training set, so the evaluation
    # trajectories come from the same mixed policy and are directly comparable.
    raw = collect(n_episodes=N_EVAL_EPISODES, seed=EVAL_SEED)

    # How long each fresh episode lasted before the pole fell or time ran out.
    lengths = raw["episode_lengths"]

    # All four dimensions; the visible two get selected further down.
    observations = raw["observations"]

    # The action taken at each step, which the imagination will replay.
    actions = raw["actions"]

    # We need WARMUP steps to prime the memory, HORIZON steps to imagine, and
    # one more observation on the end to compare the final prediction against.
    needed = WARMUP + HORIZON + 1

    # Where each episode begins inside the flat arrays.
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])

    # Keep only episodes that survived long enough to fill a whole block.
    usable = [(o, l) for o, l in zip(offsets, lengths) if l >= needed]

    # Slice out one fixed-length block per usable episode and stack them, which
    # lets every episode be imagined simultaneously as one batch.
    obs_blocks = np.stack([observations[o:o + needed] for o, _ in usable])

    # The matching actions, aligned step for step with those observations.
    action_blocks = np.stack([actions[o:o + needed] for o, _ in usable])

    print(f"evaluation episodes: {len(usable)} of {N_EVAL_EPISODES} "
          f"long enough for a {HORIZON}-step rollout")

    return obs_blocks, action_blocks, raw


# ---------------------------------------------------------------------------
# The rollout
# ---------------------------------------------------------------------------

@torch.no_grad()
def imagine_batch(model, obs_blocks, action_blocks):
    """Warm up on real steps, then imagine HORIZON steps with no simulator.

    Returns the imagined observations in real units, shaped
    (episodes, HORIZON, 2), alongside the real observations to compare against.
    """
    # Only the two dimensions the model is allowed to see. The velocities in
    # the other two columns are never shown to it, here or during training.
    visible = obs_blocks[:, :, VISIBLE_DIMS]

    observations = torch.from_numpy(visible).float().to(DEVICE)

    # Actions must be integers, because the model one-hot encodes them.
    actions = torch.from_numpy(action_blocks).long().to(DEVICE)

    # Put the real observations on the scale the model was trained on.
    normalized = model.normalize(observations)

    # Phase 1 — warm-up. Run the model over real steps with teacher forcing.
    # We throw the predictions away and keep only the memory, which by now
    # encodes the velocities the model was never told about directly.
    _, _, hidden = model(normalized[:, :WARMUP], actions[:, :WARMUP])

    # Phase 2 — imagination. The only real input from here is a single starting
    # observation; everything after that is the model consuming its own output.
    start = normalized[:, WARMUP]

    # The actions to imagine taking. These are the real actions the collector
    # chose, so the only difference from reality is the model's own physics.
    action_sequence = actions[:, WARMUP:WARMUP + HORIZON]
    imagined_norm, termination_logits, _ = model.imagine(
        start, action_sequence, hidden
    )

    # Convert back to metres and radians so the errors below mean something.
    imagined = model.denormalize(imagined_norm)

    # The model predicted the states at WARMUP+1 .. WARMUP+HORIZON, so line the
    # ground truth up with exactly that span.
    truth = observations[:, WARMUP + 1:WARMUP + HORIZON + 1]

    # The last observation actually seen before imagining began. Returned so
    # the "freeze last state" baseline can be built honestly: freezing the
    # first *target* instead would hand that baseline a free correct answer at
    # step 1 and make it look better than it is.
    last_seen = observations[:, WARMUP]

    # Back to numpy on the CPU, because everything downstream is plotting.
    return (imagined.cpu().numpy(), truth.cpu().numpy(),
            termination_logits.cpu().numpy(), last_seen.cpu().numpy())


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def drift_curve(imagined, truth):
    """Root-mean-square error at each step of the rollout, in real units.

    Averaging across episodes at each horizon position is what turns a pile of
    individual trajectories into one curve showing how error grows with depth.
    """
    # Squared error per episode, per step, per dimension.
    squared = (imagined - truth) ** 2

    # Average over episodes only, leaving one number per step per dimension.
    return np.sqrt(squared.mean(axis=0))


def trustworthy_horizon(curve):
    """The last step at which the imagined pole angle is still believable.

    This is the number the whole script exists to produce: plan.py should not
    plan further ahead than the dream can be trusted.
    """
    # Column 1 of the curve is pole angle (column 0 is cart position).
    angle_error = curve[:, 1]

    # Every step whose error is still under tolerance.
    within = np.where(angle_error <= ANGLE_TOLERANCE)[0]

    # If the model blows past tolerance immediately, there is no usable
    # horizon at all; otherwise take the last step that stayed inside it.
    # +1 converts a zero-based index into a count of steps.
    return int(within[-1] + 1) if len(within) else 0


def termination_timing(model, raw):
    """Compare when the model thinks the pole falls against when it really does.

    Uses episodes that actually ended in a fall, since those are the only ones
    with a real answer to compare against.
    """
    lengths = raw["episode_lengths"]

    # Running start position of each episode within the flat arrays.
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])

    # An episode ended by falling if its final transition was a termination.
    fell = [raw["terminated"][o + l - 1] for o, l in zip(offsets, lengths)]

    # Paired answers: when it really fell, and when the model thought it would.
    real_steps, imagined_steps = [], []

    for offset, length, did_fall in zip(offsets, lengths, fell):
        # Skip episodes cut off by the 500-step clock: nothing physically
        # failed, so there is no true fall step to compare against.
        if not did_fall or length < WARMUP + 2:
            continue

        # How many steps after warm-up the pole actually fell.
        steps_to_fall = length - WARMUP

        visible = raw["observations"][offset:offset + length, VISIBLE_DIMS]

        # [None] adds a batch dimension of 1, since the model always expects
        # to be handed a batch even when there is only one episode in it.
        observations = torch.from_numpy(visible).float().to(DEVICE)[None]

        # The same slice of actions, also given a batch dimension.
        actions = torch.from_numpy(
            raw["actions"][offset:offset + length]
        ).long().to(DEVICE)[None]

        # Onto the training scale before the model sees anything.
        normalized = model.normalize(observations)

        # Same two phases as above: prime the memory on real steps, then
        # imagine the entire remainder of the episode in one go.
        with torch.no_grad():
            _, _, hidden = model(normalized[:, :WARMUP], actions[:, :WARMUP])
            _, logits, _ = model.imagine(
                normalized[:, WARMUP], actions[:, WARMUP:], hidden
            )

        # A logit above zero means a predicted probability above 0.5, so the
        # first one of those is the model's answer to "when do I fall?".
        flagged = (logits[0] > 0).nonzero()

        # If the model never predicts a fall, record the full length as a
        # censored answer rather than dropping the episode silently.
        predicted = (int(flagged[0]) + 1 if len(flagged)
                     else int(actions.shape[1] - WARMUP))

        # One paired data point per episode, for the scatter plot.
        real_steps.append(steps_to_fall)
        imagined_steps.append(predicted)

    # Arrays rather than lists, so the caller can subtract them elementwise.
    return np.array(real_steps), np.array(imagined_steps)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_trajectories(imagined, truth):
    """Real against imagined, for a handful of episodes."""
    # Two rows: pole angle on top because it is what actually kills you, cart
    # position below because it is what the section 7 goals will care about.
    fig, axes = plt.subplots(
        2, N_EXAMPLE_EPISODES, figsize=(4 * N_EXAMPLE_EPISODES, 6), sharex=True
    )

    # The horizontal axis: how many steps into the dream we are.
    steps = np.arange(1, HORIZON + 1)

    for column in range(N_EXAMPLE_EPISODES):
        # Pole angle. Solid line is what really happened.
        axes[0, column].plot(steps, truth[column, :, 1], label="real")

        # Dashed line is the dream. Where they separate is the drift.
        axes[0, column].plot(steps, imagined[column, :, 1], "--", label="imagined")
        axes[0, column].set_title(f"episode {column + 1}")

        # Label the vertical axis on the leftmost panel only.
        axes[0, column].set_ylabel("pole angle (rad)" if column == 0 else "")

        # Cart position, same two lines on the row below.
        axes[1, column].plot(steps, truth[column, :, 0], label="real")
        axes[1, column].plot(steps, imagined[column, :, 0], "--", label="imagined")
        axes[1, column].set_xlabel("imagined steps ahead")
        axes[1, column].set_ylabel("cart position (m)" if column == 0 else "")

    # One legend for the whole figure; the lines mean the same in every panel.
    axes[0, 0].legend()
    fig.suptitle("Imagined trajectories, with the simulator switched off")

    # Keep the panels from overlapping each other's labels.
    fig.tight_layout()
    fig.savefig(TRAJECTORY_PLOT, dpi=150)
    print(f"saved {TRAJECTORY_PLOT}")


def plot_drift(curve, horizon, baseline_curve):
    """Error against how far ahead we imagined."""
    # One panel per observed dimension, side by side.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Depth of imagination along the horizontal axis.
    steps = np.arange(1, HORIZON + 1)

    # Panel titles, in the same order as the columns of the curve.
    names = ["cart position (m)", "pole angle (rad)"]

    for i, (ax, name) in enumerate(zip(axes, names)):
        # How wrong the model is, as a function of how far ahead it looked.
        ax.plot(steps, curve[:, i], label="world model")

        # A model that simply freezes the last real observation. Any learned
        # model must beat this, and showing it stops the error curve from being
        # unfalsifiable — without it, "error grows" means nothing.
        ax.plot(steps, baseline_curve[:, i], ":", color="grey",
                label="freeze last state")

        ax.set_xlabel("imagined steps ahead")
        ax.set_ylabel(f"RMSE, {name}")

        # Log scale: the error spans orders of magnitude between step 1 and
        # step 100, and a linear axis would flatten the early part to nothing.
        ax.set_yscale("log")

    # Mark the tolerance and the horizon it implies on the pole-angle panel.
    axes[1].axhline(ANGLE_TOLERANCE, color="red", lw=0.8,
                    label=f"tolerance {ANGLE_TOLERANCE} rad")
    axes[1].axvline(horizon, color="red", ls="--", lw=0.8,
                    label=f"trustworthy to {horizon} steps")

    # Both panels get a legend, since the right one has two extra markers.
    axes[0].legend()
    axes[1].legend()
    fig.suptitle("Errors compound the further ahead the model imagines")
    fig.tight_layout()
    fig.savefig(DRIFT_PLOT, dpi=150)
    print(f"saved {DRIFT_PLOT}")


def plot_termination(real_steps, imagined_steps):
    """When the model thinks the pole falls, against when it really did."""
    # Square figure, because both axes are in the same units.
    fig, ax = plt.subplots(figsize=(5, 5))

    # One dot per episode. Semi-transparent so overlapping dots stay readable.
    ax.scatter(real_steps, imagined_steps, s=12, alpha=0.5)

    # The diagonal is a perfect prediction; points below it mean the model
    # calls the fall too early, above it means too late.
    limit = max(real_steps.max(), imagined_steps.max())
    ax.plot([0, limit], [0, limit], "k--", lw=0.8, label="perfect prediction")

    ax.set_xlabel("real steps until the pole fell")
    ax.set_ylabel("imagined steps until the pole fell")
    ax.set_title("Predicting how long it survives")
    ax.legend()
    fig.tight_layout()
    fig.savefig(TERMINATION_PLOT, dpi=150)
    print(f"saved {TERMINATION_PLOT}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print(f"device: {DEVICE}")

    # The trained world model, weights and normalization statistics together.
    model = load_model()

    # Fresh episodes it has never seen, sliced into fixed-length blocks.
    obs_blocks, action_blocks, raw = evaluation_episodes()

    # Warm up on real steps, then dream forward with the simulator switched off.
    imagined, truth, _, last_seen = imagine_batch(model, obs_blocks, action_blocks)

    # How wrong the model is at each depth of imagination.
    curve = drift_curve(imagined, truth)

    # The same measurement for a model that predicts no change at all, as a
    # floor to compare against.
    # Hold the last genuinely observed state constant for the whole horizon.
    frozen = np.repeat(last_seen[:, None, :], HORIZON, axis=1)

    # Scored exactly like the model, so the two curves are comparable.
    baseline_curve = drift_curve(frozen, truth)

    # The number this whole script exists to produce, handed on to plan.py.
    horizon = trustworthy_horizon(curve)

    print("\nimagined-rollout error (pole angle, radians):")
    for h in sorted({1, 5, 10, 20, 50, 100, horizon}):
        # Report a few depths rather than the whole curve, so the growth is
        # readable at a glance in the article.
        print(f"  {h:3d} steps ahead: {curve[h - 1, 1]:.5f} rad  "
              f"(cart {curve[h - 1, 0]:.4f} m)")

    print(f"\ntrustworthy horizon H = {horizon} steps "
          f"(pole angle error stays under {ANGLE_TOLERANCE} rad)")

    # Separately, check whether the model knows when the pole is about to fall.
    real_steps, imagined_steps = termination_timing(model, raw)

    # Median rather than mean, because a few wildly wrong episodes would
    # otherwise dominate the number.
    error = np.abs(real_steps - imagined_steps)
    print(f"fall-time prediction: median error {np.median(error):.0f} steps "
          f"over {len(real_steps)} episodes")

    # Three figures: what the dreams look like, how fast they go wrong, and
    # whether the model can tell when the episode is about to end.
    plot_trajectories(imagined, truth)
    plot_drift(curve, horizon, baseline_curve)
    plot_termination(real_steps, imagined_steps)

    # Hand the horizon to plan.py through a file, so the two scripts cannot
    # disagree about a number that was measured rather than chosen.
    os.makedirs(os.path.dirname(HORIZON_PATH), exist_ok=True)
    # How long the model beats a lazy "nothing ever changes" predictor. The
    # article quotes this, and without it the drift numbers have no floor.
    beats_baseline = int((curve[:, 1] < baseline_curve[:, 1]).argmin()) \
        if (curve[:, 1] >= baseline_curve[:, 1]).any() else int(HORIZON)

    # The depths the article tabulates. H itself is in the list because the
    # error *at* H is the number that justifies choosing H, and leaving it out
    # would mean the one load-bearing row could not be checked against a file.
    reported_depths = sorted({1, 5, 10, 20, 50, 100, horizon})

    with open(HORIZON_PATH, "w") as handle:
        json.dump(
            {
                "horizon": horizon,
                "angle_tolerance": ANGLE_TOLERANCE,
                "rollout_horizon": HORIZON,
                "episodes": int(truth.shape[0]),
                # The error at H, pulled out on its own because every claim
                # about the horizon being trustworthy rests on this one value
                # sitting just under the tolerance above.
                "drift_angle_rad_at_horizon": float(curve[horizon - 1, 1]),
                "drift_position_m_at_horizon": float(curve[horizon - 1, 0]),
                # The drift table in section 5.4, one entry per row.
                "drift_angle_rad": {str(h): float(curve[h - 1, 1])
                                    for h in reported_depths},
                "drift_position_m": {str(h): float(curve[h - 1, 0])
                                     for h in reported_depths},
                "freeze_baseline_angle_rad": {
                    str(h): float(baseline_curve[h - 1, 1])
                    for h in reported_depths},
                "beats_freeze_baseline_for_steps": beats_baseline,
                "fall_time_median_error": float(np.median(error)),
                "fall_time_episodes": int(len(real_steps)),
            },
            handle,
            indent=2,
        )
    print(f"saved {HORIZON_PATH}")


if __name__ == "__main__":
    main()
