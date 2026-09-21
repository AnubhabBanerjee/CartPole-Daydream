"""Train the world model with backpropagation through time.

Samples 32-step sequence chunks, carries the hidden state across the chunk, and
backpropagates through the unrolled graph. Training is teacher-forced: every
step is fed the ground-truth observation.

Loss is MSE on the observation delta plus a weighted BCE-with-logits on the
termination head.

Output: checkpoints/world_model.pt (weights plus normalization statistics)
"""

import json
import os

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

# Draw to a file rather than a window, so this runs the same over SSH.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from model import VISIBLE_DIMS, WorldModel  # noqa: E402

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

DATA_PATH = "data/cartpole.npz"
CHECKPOINT_PATH = "checkpoints/world_model.pt"
CURVE_PATH = "figures/training_curve.png"

# Held-out accuracy numbers the article quotes, saved rather than just printed.
METRICS_PATH = "checkpoints/train_metrics.json"

# How many consecutive steps go into one training sequence. This is the span
# that backpropagation through time has to travel back across. Longer means the
# model can learn longer-range effects, but the unrolled graph costs more memory
# and gradients get harder to push back through.
SEQ_LEN = 32

# The model starts each chunk with a blank memory, so for the first few steps it
# genuinely cannot know how fast anything is moving — we hid the velocities. We
# let it watch this many steps to work that out and score nothing during them.
# Without this, the model would be punished for failing at an impossible task
# and would learn to hedge.
BURN_IN = 8

BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 1e-3

# Termination is a side quest; predicting the physics is the main job. This
# keeps the failure detector from dominating the gradient.
TERMINATION_LOSS_WEIGHT = 0.1

# Caps how large a single gradient step can be. Recurrent models occasionally
# produce enormous gradients, and one of those can undo an entire epoch.
GRAD_CLIP = 1.0

# Fraction of *episodes* (not steps) held out for validation. Splitting whole
# episodes matters: splitting by step would put step 100 in training and step
# 101 in validation, and the model would be graded on data it had effectively
# already seen.
VAL_FRACTION = 0.1

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def sequence_starts(episode_lengths, seq_len):
    """List every index where a full-length sequence can begin.

    Episodes sit end to end in one flat array, so we cannot just pick random
    offsets: a sequence that ran off the end of one episode and into the next
    would ask the model to explain a discontinuity that is really just a reset.
    """
    starts = []

    # Running offset of the current episode within the flat arrays.
    offset = 0

    for length in episode_lengths:
        # Only episodes at least seq_len long can host a sequence at all. Very
        # short episodes (the pole fell almost immediately) are skipped.
        if length >= seq_len:
            # Every start from the episode's beginning up to the last point
            # that still leaves seq_len steps inside this same episode.
            starts.extend(range(offset, offset + length - seq_len + 1))
        offset += length

    return np.array(starts, dtype=np.int64)


def load_data():
    """Load the dataset, split it by episode, and move it onto the device."""
    # Everything collect_data.py wrote, still on disk in flat arrays.
    raw = np.load(DATA_PATH)

    # Keep only the dimensions the model is allowed to see: cart position and
    # pole angle. The velocities stay in the file for the article's plots, but
    # the model never gets them.
    observations = raw["observations"][:, VISIBLE_DIMS]

    # The same two dimensions of the state each action actually led to.
    next_observations = raw["next_observations"][:, VISIBLE_DIMS]

    # Which way the cart was pushed at each step, as 0 or 1.
    actions = raw["actions"]

    # Only `terminated` becomes a training target. `truncated` is left out
    # entirely: it means the 500-step clock ran out, which is invisible to the
    # model. Note we do not need to mask truncated steps out of the physics
    # loss — the transition itself is real physics, and its termination label
    # is correctly 0, because nothing actually failed.
    # Cast from bool to float because the loss function expects float targets.
    terminated = raw["terminated"].astype(np.float32)

    # How many transitions belong to each episode, used to find the boundaries.
    episode_lengths = raw["episode_lengths"]

    # Split whole episodes into train and validation.
    n_val = int(len(episode_lengths) * VAL_FRACTION)

    # The remaining episodes, taken from the front of the file, are training.
    n_train_episodes = len(episode_lengths) - n_val

    # Number of transitions belonging to the training episodes. Because
    # episodes are stored consecutively, the split is a single cut point.
    n_train_steps = int(episode_lengths[:n_train_episodes].sum())

    # Valid sequence starts within each split, computed separately so that no
    # training sequence can reach into validation data.
    train_starts = sequence_starts(episode_lengths[:n_train_episodes], SEQ_LEN)

    # Same for the held-out episodes, numbered as if they began at zero.
    val_starts = sequence_starts(episode_lengths[n_train_episodes:], SEQ_LEN)

    # The validation starts were numbered from zero, so shift them to their
    # true positions in the flat arrays.
    val_starts += n_train_steps

    # Normalization statistics come from the training split only. Using the
    # whole dataset would leak information about the held-out episodes.
    # Per-dimension average, so each input is centred near zero.
    mean = observations[:n_train_steps].mean(axis=0)

    # Per-dimension spread, so cart metres and pole radians carry equal weight.
    std = observations[:n_train_steps].std(axis=0)

    # The dataset is a few megabytes, so the entire thing fits on the GPU.
    # Keeping it there means no host-to-device copying inside the training loop.
    tensors = {
        "observations": torch.from_numpy(observations).to(DEVICE),
        "next_observations": torch.from_numpy(next_observations).to(DEVICE),
        "actions": torch.from_numpy(actions).to(DEVICE),
        "terminated": torch.from_numpy(terminated).to(DEVICE),
    }

    return tensors, train_starts, val_starts, mean, std


def make_batch(tensors, starts, model):
    """Gather a batch of sequences, given the index each one starts at.

    starts: (B,) array of starting positions in the flat arrays.
    """
    # Turn each start index into a full row of consecutive indices, so that
    # starts [10, 50] with SEQ_LEN 3 becomes [[10,11,12], [50,51,52]]. One
    # fancy-index then pulls out the whole batch at once.
    offsets = torch.arange(SEQ_LEN, device=DEVICE)

    # Broadcasting a column of starts against a row of offsets gives the full
    # (batch, sequence) grid of positions in one shot.
    index = starts[:, None] + offsets[None, :]

    # Inputs the model sees: normalized observations and the actions taken.
    obs = model.normalize(tensors["observations"][index])

    # The action taken at each of those steps.
    actions = tensors["actions"][index]

    # Targets: where the environment actually went, on the same scale as the
    # model's output, and whether that next state was a physical failure.
    next_obs = model.normalize(tensors["next_observations"][index])

    # 1.0 where the pole physically failed on that step, 0.0 otherwise.
    terminated = tensors["terminated"][index]

    return obs, actions, next_obs, terminated


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(model, batch, pos_weight):
    """Run the model over a batch and score how wrong it was."""
    obs, actions, next_obs, terminated = batch

    # Teacher-forced pass: at every step the model receives the real
    # observation, not its own last guess.
    predicted, logits, _ = model(obs, actions)

    # Throw away the burn-in steps. During those the memory was still empty, so
    # the model had no way to know any velocity and scoring them would be
    # grading it on information we deliberately withheld.
    predicted = predicted[:, BURN_IN:]

    # The same trim applied to every tensor, so predictions and targets stay
    # aligned step for step.
    logits = logits[:, BURN_IN:]
    next_obs = next_obs[:, BURN_IN:]
    terminated = terminated[:, BURN_IN:]

    # The physics objective. Both sides are normalized, so position error and
    # angle error contribute on comparable terms.
    dynamics_loss = F.mse_loss(predicted, next_obs)

    # The failure objective. Terminations are rare — roughly 1 step in 500 —
    # so pos_weight scales up the handful of positive examples. Without it the
    # head would learn to answer "no" forever and score extremely well.
    termination_loss = F.binary_cross_entropy_with_logits(
        logits, terminated, pos_weight=pos_weight
    )

    # One number to backpropagate. The weight keeps the failure detector from
    # drowning out the physics, which is the harder and more important job.
    total = dynamics_loss + TERMINATION_LOSS_WEIGHT * termination_loss

    # The two parts come back too, purely so they can be printed separately.
    return total, dynamics_loss, termination_loss


@torch.no_grad()
def evaluate(model, tensors, starts, pos_weight):
    """Average the loss over a fixed sample of validation sequences."""
    # Switch off training-time behaviour. Harmless for this model, which has no
    # dropout or batch norm, but wrong to omit out of habit.
    model.eval()

    # Running sum of the three losses: total, dynamics, termination.
    totals = np.zeros(3)

    # How many full batches the validation split can supply.
    n_batches = max(1, len(starts) // BATCH_SIZE)

    # Cap the work: a couple of dozen batches is plenty to track progress.
    n_batches = min(n_batches, 20)

    # Move the start indices to the device once, outside the loop.
    starts_tensor = torch.from_numpy(starts).to(DEVICE)

    for i in range(n_batches):
        # Walk through the validation set in order rather than sampling, so
        # the number printed each epoch is comparable to the last one.
        chunk = starts_tensor[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

        # Same batching and same loss as training, so the two numbers printed
        # each epoch are measuring exactly the same thing.
        batch = make_batch(tensors, chunk, model)
        losses = compute_loss(model, batch, pos_weight)

        # float() pulls each loss off the GPU into a plain Python number.
        totals += np.array([float(x) for x in losses])

    # Back to training mode for the caller.
    model.train()

    # Mean rather than sum, so the value does not depend on batch count.
    return totals / n_batches


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    # Fix the seed so weight initialisation and batch sampling repeat exactly.
    torch.manual_seed(SEED)

    print(f"device: {DEVICE}")

    # Read the dataset, split it by episode, and park it all on the device.
    tensors, train_starts, val_starts, mean, std = load_data()
    print(f"train sequences: {len(train_starts)}, val: {len(val_starts)}")

    # Build the model and move it to the GPU before the optimizer is created,
    # so the optimizer tracks the parameters at their final location.
    model = WorldModel().to(DEVICE)

    # Hand the model its normalization statistics. They live inside the model
    # as buffers, so they are written into the checkpoint automatically and
    # cannot drift out of sync with the weights.
    model.set_normalization(mean, std)

    # How many negative termination labels there are per positive one. This
    # exact ratio is what balances the two classes in the loss.
    positives = tensors["terminated"].sum()

    # Everything that was not a termination, which is almost everything.
    negatives = len(tensors["terminated"]) - positives

    # Telling the loss to count each rare positive this many times over is what
    # stops the head from answering "no failure" forever and scoring well.
    pos_weight = (negatives / positives).to(DEVICE)
    print(f"termination balance: 1 positive per {negatives / positives:.0f} steps")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Ease the learning rate down to zero over training. Without this the model
    # keeps taking full-sized steps right to the end and the validation error
    # bounces around by a factor of ten between epochs — it never settles into
    # a minimum, it just keeps jumping over it.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Start indices live on the device so that batch assembly never touches
    # the CPU once training is under way.
    train_starts_tensor = torch.from_numpy(train_starts).to(DEVICE)

    # One "epoch" means enough batches to have seen every sequence start once
    # on average — they are sampled with replacement, not shuffled through.
    steps_per_epoch = len(train_starts) // BATCH_SIZE

    # Per-epoch dynamics loss, kept for the training-curve figure.
    history = {"train": [], "val": []}

    for epoch in range(EPOCHS):
        # Running sum of total, dynamics and termination loss for this epoch.
        epoch_losses = np.zeros(3)

        for _ in range(steps_per_epoch):
            # Draw a random set of starting points. Sampling starts freshly
            # every batch means the model sees a different slicing of the same
            # episodes each epoch, which is a cheap form of augmentation.
            picks = torch.randint(
                len(train_starts_tensor), (BATCH_SIZE,), device=DEVICE
            )

            # Turn those start indices into actual sequences of observations,
            # actions and targets.
            batch = make_batch(tensors, train_starts_tensor[picks], model)

            # Run the model and score it. Only `total` is backpropagated; the
            # other two are kept for reporting.
            total, dynamics, termination = compute_loss(model, batch, pos_weight)

            # Clear last step's gradients before computing this step's.
            optimizer.zero_grad()

            # This is backpropagation through time. The graph runs back through
            # all 32 GRUCell calls, so the gradient reaching step 0 has passed
            # through 32 applications of the same weights.
            total.backward()

            # Rein in occasional huge gradients before they wreck the weights.
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            # Apply the (now clipped) gradients to the weights.
            optimizer.step()

            # Accumulate the three losses for this epoch's average.
            epoch_losses += np.array(
                [float(total), float(dynamics), float(termination)]
            )

        # One learning-rate step per epoch, not per batch.
        scheduler.step()

        # Turn the epoch's running sums into averages.
        train_losses = epoch_losses / steps_per_epoch

        # Measure on episodes the model has never trained on.
        val_losses = evaluate(model, tensors, val_starts, pos_weight)

        # Index 1 is the dynamics loss, which is the one worth plotting: the
        # termination loss is on a different scale and would swamp the curve.
        history["train"].append(train_losses[1])
        history["val"].append(val_losses[1])

        print(
            f"epoch {epoch + 1:2d}/{EPOCHS}  "
            f"train dyn {train_losses[1]:.6f}  term {train_losses[2]:.4f}  |  "
            f"val dyn {val_losses[1]:.6f}  term {val_losses[2]:.4f}"
        )

    # Restate the final error in metres and radians, which are interpretable
    # in a way that a normalized MSE is not.
    report_real_units(model, tensors, val_starts, pos_weight)

    # Write the checkpoint and the training-curve figure.
    save(model, history)


@torch.no_grad()
def report_real_units(model, tensors, starts, pos_weight):
    """Restate the final error in metres and radians, which mean something."""
    # No dropout or batch norm here, but switch modes anyway for correctness.
    model.eval()

    # Every validation sequence start, moved to the device in one go.
    starts_tensor = torch.from_numpy(starts).to(DEVICE)

    # Accumulate over the whole validation split. Terminations are rare enough
    # that a single batch usually contains none at all, so any count taken from
    # one batch would be meaningless.
    squared_error = torch.zeros(len(VISIBLE_DIMS), device=DEVICE)

    # Total scored steps, needed to turn the running sum into a mean.
    n_steps = 0

    # Confusion-matrix counts for the termination head.
    true_positives = predicted_positives = actual_positives = 0

    for i in range(0, len(starts_tensor), BATCH_SIZE):
        # Walk the validation set in order, one batch at a time.
        chunk = starts_tensor[i:i + BATCH_SIZE]

        # Same batch assembly as training, so this measures the same quantity.
        obs, actions, next_obs, terminated = make_batch(tensors, chunk, model)

        # Teacher-forced predictions; this is one-step accuracy, not rollout
        # accuracy. Rollout accuracy is what imagine.py measures.
        predicted, logits, _ = model(obs, actions)

        # Drop the burn-in steps exactly as the training loss does.
        predicted = predicted[:, BURN_IN:]
        logits = logits[:, BURN_IN:]
        next_obs = next_obs[:, BURN_IN:]
        terminated = terminated[:, BURN_IN:]

        # Undo the normalization so the numbers are back in physical units.
        error = model.denormalize(predicted) - model.denormalize(next_obs)

        # Sum rather than mean, because batches at the end may be short.
        squared_error += error.pow(2).sum(dim=(0, 1))

        # Count how many individual step-predictions went into that sum.
        n_steps += error.shape[0] * error.shape[1]

        # A logit above 0 is a predicted probability above 0.5.
        flagged = (logits > 0).float()

        # Multiplying the two masks counts only the steps where the model said
        # "failure" and a failure really happened.
        true_positives += float((flagged * terminated).sum())

        # Every alarm raised, correct or not.
        predicted_positives += float(flagged.sum())

        # Every failure that actually occurred.
        actual_positives += float(terminated.sum())

    # Root of the mean squared error, per dimension, in real units.
    rmse = (squared_error / n_steps).sqrt()

    print("\none-step error on held-out data:")
    print(f"  cart position  {rmse[0]:.5f} m")
    print(f"  pole angle     {rmse[1]:.5f} rad")

    # Recall is what matters for planning: a missed failure means the planner
    # confidently walks into a future where the pole has already fallen.
    # Precision matters less — a few false alarms only make it cautious.
    # Of the failures that happened, how many did the model see coming. The
    # max() guards against dividing by zero if a split contained no failures.
    recall = true_positives / max(actual_positives, 1)

    # Of the alarms it raised, how many were real.
    precision = true_positives / max(predicted_positives, 1)
    print(f"  termination recall    {recall:.1%} "
          f"({true_positives:.0f} of {actual_positives:.0f})")
    print(f"  termination precision {precision:.1%}")

    # Write the numbers down. The article quotes every one of these, and a
    # printed number that nobody saved is a number the next run can change
    # without anyone noticing.
    metrics = {
        "one_step_rmse_position_m": float(rmse[0]),
        "one_step_rmse_angle_rad": float(rmse[1]),
        "termination_recall": float(recall),
        "termination_precision": float(precision),
        "true_positives": int(true_positives),
        "actual_positives": int(actual_positives),
        "false_alarms": int(predicted_positives - true_positives),
        # How many spurious alarms the model raises per real crash it catches.
        # This is the number the article quotes, because "252 false alarms"
        # sounds alarming until you know it caught all 42 crashes.
        "false_alarms_per_crash": float(
            (predicted_positives - true_positives) / max(actual_positives, 1)),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "seq_len": SEQ_LEN,
        "burn_in": BURN_IN,
    }

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"  saved {METRICS_PATH}")

    # Leave the model as we found it.
    model.train()


def save(model, history):
    """Write the checkpoint and the training curve."""
    # Create checkpoints/ on first run; do nothing if it already exists.
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    # state_dict carries the normalization buffers alongside the weights, so
    # loading this file is enough to reproduce the model exactly.
    torch.save(
        {"model": model.state_dict(), "seq_len": SEQ_LEN, "burn_in": BURN_IN},
        CHECKPOINT_PATH,
    )
    print(f"\nsaved {CHECKPOINT_PATH}")

    # Same for figures/, which is where every plot in the article lands.
    os.makedirs(os.path.dirname(CURVE_PATH), exist_ok=True)

    # A single modest-sized panel; this figure has only one thing to say.
    plt.figure(figsize=(6, 4))

    # Training error, one point per epoch.
    plt.plot(history["train"], label="train")

    # Validation error on the same axes. The gap between the two lines is the
    # amount of overfitting, and it should stay roughly constant.
    plt.plot(history["val"], label="validation")

    # Log scale because the loss drops by orders of magnitude early on, which
    # would otherwise squash the entire interesting part of the curve flat.
    plt.yscale("log")

    # Axis labels spelled out, since this figure goes straight into the article.
    plt.xlabel("epoch")
    plt.ylabel("one-step prediction error (normalized MSE)")
    plt.title("World model training")

    # Identify which line is which.
    plt.legend()

    # Stop the labels being clipped at the edges of the image.
    plt.tight_layout()

    # 150 dpi is sharp enough to read on a phone without bloating the file.
    plt.savefig(CURVE_PATH, dpi=150)
    print(f"saved {CURVE_PATH}")


if __name__ == "__main__":
    main()
