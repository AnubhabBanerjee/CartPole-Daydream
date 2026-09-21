"""The world model: a single GRUCell with two prediction heads.

Input at each step is the 2D partial observation (cart position, pole angle)
concatenated with a 2D one-hot action.

Heads read off the hidden state and both answer questions about step t+1:
  - next-observation head predicts the delta o_{t+1} - o_t in normalized space
  - termination head predicts `terminated_{t+1}`, physical failure only

Kept under 100 lines on purpose.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# Which of CartPole's 4 observation dimensions the model is allowed to see:
# index 0 is cart position and index 2 is pole angle. The two velocities are
# deliberately withheld. If we handed over all 4, each next state could be
# computed from the current one alone, the GRU's memory would have nothing to
# do, and a plain feedforward network would work just as well. Hiding the
# velocities forces the hidden state to infer motion from history, which is
# both the honest setting (a camera shows position, not speed) and the reason
# this architecture has a memory at all.
VISIBLE_DIMS = (0, 2)

# Size of the GRU's memory. 64 is plenty for physics this simple.
HIDDEN_SIZE = 64

# CartPole accepts two actions: 0 pushes the cart left, 1 pushes it right.
N_ACTIONS = 2


class WorldModel(nn.Module):
    """Learns to predict what CartPole does next, given what it did last."""

    def __init__(self, obs_dim=len(VISIBLE_DIMS), n_actions=N_ACTIONS,
                 hidden_size=HIDDEN_SIZE):
        # Register this as a PyTorch module before adding any layers to it.
        super().__init__()

        # How many observation numbers go in and come out: 2 here.
        self.obs_dim = obs_dim

        # Needed later to one-hot encode the action.
        self.n_actions = n_actions

        # Kept so callers can build a blank memory of the right width.
        self.hidden_size = hidden_size

        # The memory. A GRUCell processes exactly one timestep per call, which
        # is what we want: the training loop drives it step by step so the
        # unrolling is visible in the code rather than hidden inside nn.GRU.
        # Its input is the observation and the action stuck together.
        self.gru = nn.GRUCell(obs_dim + n_actions, hidden_size)

        # Head 1: the physics. Predicts how much each observed quantity
        # *changes*, not what it becomes. Predicting the absolute next position
        # would let the network score well by copying its input and ignoring
        # the action entirely; asking for the change removes that shortcut.
        self.delta_head = nn.Linear(hidden_size, obs_dim)

        # Head 2: the failure detector. One number, the unsquashed probability
        # that the next state is a physical failure (pole past 12 degrees or
        # cart past 2.4). It predicts `terminated` only, never `truncated`:
        # truncation is the 500-step clock, which the model cannot see and so
        # could only ever guess at.
        self.termination_head = nn.Linear(hidden_size, 1)

        # Normalization statistics, stored on the model rather than beside it.
        # Registering them as buffers means they are saved and loaded with the
        # weights automatically, so imagine.py and plan.py cannot accidentally
        # run the model on differently-scaled inputs than it trained on.
        # Placeholder values until train.py fills them in from the dataset.
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

    # -- normalization ------------------------------------------------------

    def set_normalization(self, mean, std):
        """Record the dataset's mean and spread. Called once, by train.py."""
        # copy_ writes into the existing buffers rather than replacing them,
        # which keeps them registered and therefore saved with the checkpoint.
        self.obs_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.obs_std.copy_(torch.as_tensor(std, dtype=torch.float32))

    def normalize(self, obs):
        """Put raw observations onto a common scale before the network sees them."""
        # Cart position ranges over units while pole angle stays within about
        # 0.2 radians. Left raw, the network would treat a tiny angle error as
        # unimportant next to a position error, when in fact the angle is what
        # kills you. Dividing by each dimension's own spread fixes that.
        return (obs - self.obs_mean) / self.obs_std

    def denormalize(self, obs_norm):
        """Convert the network's output back into real units for plotting."""
        return obs_norm * self.obs_std + self.obs_mean

    # -- one step -----------------------------------------------------------

    def initial_hidden(self, batch_size, device=None):
        """A blank memory: the model starts out knowing nothing about motion."""
        device = device or self.obs_mean.device
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def step(self, obs_norm, action, hidden):
        """Advance the model one timestep.

        obs_norm: (B, obs_dim) normalized observation at time t
        action:   (B,) integer action taken at time t
        hidden:   (B, hidden_size) memory carried in from the past

        Returns the predicted normalized observation at t+1, the termination
        logit for t+1, and the updated memory.
        """
        # Turn each action index into a 2-element vector: 0 becomes [1, 0] and
        # 1 becomes [0, 1]. Feeding the raw integer instead would imply that
        # "right" is numerically bigger than "left", which is meaningless here.
        action_onehot = F.one_hot(action, self.n_actions).float()

        # Glue observation and action into one input vector for this timestep.
        gru_input = torch.cat([obs_norm, action_onehot], dim=-1)

        # Update the memory. Everything the model knows about velocity lives in
        # this vector, inferred from the sequence of positions it has seen.
        hidden = self.gru(gru_input, hidden)

        # Predict the change, then add it on. The addition is what makes the
        # network's job "work out the small correction" rather than "reproduce
        # the whole state", and it is why imagined rollouts track real physics
        # instead of collapsing to a flat line.
        delta = self.delta_head(hidden)
        next_obs_norm = obs_norm + delta

        # One logit per batch element; squeeze off the trailing size-1 axis so
        # it lines up with the (B,) shaped labels in the loss.
        termination_logit = self.termination_head(hidden).squeeze(-1)

        return next_obs_norm, termination_logit, hidden

    # -- many steps ---------------------------------------------------------

    def forward(self, obs_norm_seq, action_seq, hidden=None):
        """Teacher-forced rollout, used for training.

        obs_norm_seq: (B, T, obs_dim) the *real* observations
        action_seq:   (B, T) the real actions

        At every step the model is handed the ground-truth observation, even
        if its own previous prediction was wrong. Errors therefore never
        accumulate here — which is exactly the mismatch that makes `imagine`
        below drift, and the thing the article measures.
        """
        # How many sequences in parallel, and how many steps in each.
        batch_size, horizon, _ = obs_norm_seq.shape

        # Start from a blank memory unless the caller is continuing a sequence.
        if hidden is None:
            hidden = self.initial_hidden(batch_size, obs_norm_seq.device)

        # Collected one timestep at a time, stacked into tensors at the end.
        predicted, logits = [], []

        for t in range(horizon):
            # Note the input: obs_norm_seq[:, t] is the *truth* at time t.
            next_obs_norm, termination_logit, hidden = self.step(
                obs_norm_seq[:, t], action_seq[:, t], hidden
            )

            # Keep this step's prediction; the next iteration ignores it and
            # reads the ground truth again. That is what teacher forcing means.
            predicted.append(next_obs_norm)
            logits.append(termination_logit)

        # Stack the per-step lists back into (B, T, obs_dim) and (B, T).
        # Building the sequence with a Python loop like this is what creates
        # the chain of operations that backpropagation through time runs down.
        return torch.stack(predicted, dim=1), torch.stack(logits, dim=1), hidden

    @torch.no_grad()
    def imagine(self, obs_norm, action_seq, hidden):
        """Autoregressive rollout: the model dreams, with no simulator involved.

        obs_norm:   (B, obs_dim) a single real starting observation
        action_seq: (B, T) the actions to imagine taking
        hidden:     (B, hidden_size) a memory warmed up on real steps, so the
                    model already has a sense of which way things are moving

        The one line that differs from `forward` is which observation gets fed
        back in: its own prediction, not the truth. That single change is what
        turns a predictor into a world model, and what lets small errors
        compound into large ones.
        """
        # How many steps to dream, taken from the action sequence supplied.
        horizon = action_seq.shape[1]

        # Same per-step collection as forward().
        predicted, logits = [], []

        for t in range(horizon):
            # Reassigning obs_norm is the whole trick: the variable holding the
            # input is overwritten by the output.
            obs_norm, termination_logit, hidden = self.step(
                obs_norm, action_seq[:, t], hidden
            )
            # obs_norm has just been overwritten by the model's own output, and
            # that is what the next iteration will consume.
            predicted.append(obs_norm)
            logits.append(termination_logit)

        # Same shapes as forward() returns, so callers can treat them alike.
        return torch.stack(predicted, dim=1), torch.stack(logits, dim=1), hidden
