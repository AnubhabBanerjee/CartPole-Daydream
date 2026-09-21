"""Scoring functions: what counts as a good imagined future.

This file is the article's central argument in one place. Each goal is a small
function with the same signature, scoring a batch of dreamed trajectories. To
give the planner a new objective you swap the function — no retraining, no new
environment steps, no weight updates.

Every score must be computable from what the model actually predicts: cart
position, pole angle, and termination. Nothing else is available.

  balance  - stay alive as long as possible (the default)
  park     - goal A: stay alive and hold the cart at x = +1.0
  corridor - goal B: stay alive and never let |x| exceed 0.5
  waypoint - goal C: reach x = +1.0, hold 100 steps, then return to x = -1.0

Scores are computed over the whole dream, not just its first step, which is what
lets the planner accept a worse next step for a better future.
"""

import os

import matplotlib
import torch

# Draw to a file rather than a window, so this runs the same over SSH.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# Where goal A asks the cart to park, in metres along the rail.
PARK_TARGET = 1.0

# How far from the centre goal B allows the cart to stray. The environment
# itself only fails at 2.4, so this is a self-imposed rule roughly five times
# stricter than anything the model saw while training.
CORRIDOR_HALF_WIDTH = 0.5

# Goal C's schedule: go to +1.0, hold, then go to -1.0.
WAYPOINT_FIRST = 1.0
WAYPOINT_SECOND = -1.0

# How many real steps to spend at the first waypoint before switching.
WAYPOINT_HOLD_STEPS = 100

# How much the planner cares about position relative to staying alive. Survival
# is scored in units of "steps", so a value of 3.0 means being one metre off
# target costs about as much as dying three steps sooner.
POSITION_WEIGHT = 3.0

# How many steps at the end of a dream the "go somewhere" goals are judged on.
#
# This number matters far more than it looks, and getting it wrong was the
# single biggest bug in this project. Judging a dream by its *average* distance
# from the target sounds obviously right and fails completely, because of a
# quirk of the cart: to move right, it must first briefly push left, tipping the
# pole rightwards so it has something to chase. An average-distance score sees
# that opening move as pure loss and refuses to make it, so the planner never
# commits to going anywhere and eventually falls over.
#
# Judging the dream by where it *ends up* instead lets the planner spend a few
# steps going the wrong way in order to arrive. Same model, same dreams, same
# everything else — only the question we ask about them changed.
TERMINAL_WINDOW = 5

# The penalty applied to a dream that leaves goal B's corridor. Large enough to
# dominate any survival gain, which is what makes it a constraint rather than a
# preference the planner can quietly trade away.
CORRIDOR_PENALTY = 10.0

# Column indices into the model's 2-dimensional prediction.
CART_POSITION = 0
POLE_ANGLE = 1


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------

def survival_mask(termination_logits):
    """Work out which imagined steps happen before the pole falls.

    termination_logits: (candidates, horizon) the model's crash predictions.

    Returns a (candidates, horizon) mask that is 1.0 for steps the episode is
    still running and 0.0 for everything at or after the predicted crash.
    """
    # A logit above zero means the model puts the chance of a crash above 50%.
    predicted_crash = (termination_logits > 0).float()

    # Running total of crashes so far along each dream. Once a dream has
    # crashed, every later step has a cumulative count of at least one.
    crashes_so_far = torch.cumsum(predicted_crash, dim=1)

    # Steps before the first predicted crash are alive. Because the cumulative
    # total already includes the current step's own flag, the crash step itself
    # comes out as dead, which is what we want.
    return (crashes_so_far < 1).float()


def survival_score(alive):
    """How many steps the dream is expected to survive.

    This is the backbone of every goal: staying alive is always worth something,
    and the goal-specific terms are adjustments on top of it.
    """
    # Summing the mask counts the steps before the predicted crash. A dream
    # that never crashes scores the full horizon.
    return alive.sum(dim=1)


def terminal_mean(values):
    """Average `values` over the last few steps of the dream.

    Used by the goals that ask the cart to *get somewhere*, so that a dream is
    judged on where it arrives rather than on the scenic route it took. See the
    comment on TERMINAL_WINDOW for why this distinction decides whether the
    planner works at all.
    """
    # No survival masking here: a dream that crashes has already been punished
    # by the survival term, and masking the tail away would leave these goals
    # with nothing to score.
    return values[:, -TERMINAL_WINDOW:].mean(dim=1)


def masked_mean(values, alive):
    """Average `values` over the steps where the dream is still running.

    Averaging over dead steps would let a dream score well by crashing early and
    then sitting motionless in a pretend afterlife at exactly the right spot.
    """
    # Zero out everything after the predicted crash before summing.
    total = (values * alive).sum(dim=1)

    # Divide by the number of live steps, guarding against a dream that was
    # predicted to crash immediately and therefore has none.
    return total / alive.sum(dim=1).clamp(min=1.0)


# ---------------------------------------------------------------------------
# The goals
# ---------------------------------------------------------------------------

def balance(observations, termination_logits, step=0):
    """The default: survive as long as possible.

    observations:       (candidates, horizon, 2) imagined cart position and
                        pole angle, in real units
    termination_logits: (candidates, horizon) imagined crash predictions
    step:               how many real steps have already been taken, which only
                        the time-varying goals care about

    Returns one score per candidate; higher is better.
    """
    # Which steps of each dream happen before the pole falls.
    alive = survival_mask(termination_logits)

    # How upright the pole stayed, averaged over the live part of the dream.
    # Survival alone is a coarse signal — it only changes when a crash appears
    # or disappears — so this term gives the planner a smooth gradient to
    # follow between dreams that all survive the full horizon.
    uprightness = masked_mean(observations[:, :, POLE_ANGLE].abs(), alive)

    # Staying alive is the reward; leaning is a small penalty on top.
    return survival_score(alive) - uprightness


def park(observations, termination_logits, step=0):
    """Goal A: stay alive, and hold the cart at PARK_TARGET.

    The model was never trained on this, and nothing about it appears in the
    dataset. It is expressed entirely as a preference over imagined futures.
    """
    # Same survival backbone as every other goal.
    alive = survival_mask(termination_logits)

    # How far the cart sits from where we want it, at each imagined step.
    distance = (observations[:, :, CART_POSITION] - PARK_TARGET).abs()

    # Judged on where the dream ends up, not where it spent its time, so the
    # planner is allowed the brief wrong-way move that getting there requires.
    off_target = terminal_mean(distance)

    # Keep the pole upright too, or the planner will cheerfully drive the cart
    # to the target while dropping the pole on the way.
    uprightness = masked_mean(observations[:, :, POLE_ANGLE].abs(), alive)

    # Survival, minus the cost of being away from the target.
    return survival_score(alive) - POSITION_WEIGHT * off_target - uprightness


def corridor(observations, termination_logits, step=0):
    """Goal B: stay alive, and never leave a narrow strip of the rail.

    A constraint rather than a target: anywhere inside the corridor is equally
    fine, and leaving it is heavily punished.
    """
    # Same survival backbone as every other goal.
    alive = survival_mask(termination_logits)

    # How far outside the corridor the cart strays, zero when it is inside.
    excess = (observations[:, :, CART_POSITION].abs()
              - CORRIDOR_HALF_WIDTH).clamp(min=0.0)

    # Averaged over every live step, not just the last few. Unlike the "go
    # somewhere" goals, this one is a rule about the whole journey: leaving the
    # corridor and coming back is still a violation.
    violation = masked_mean(excess, alive)

    # Keep the pole up as well.
    uprightness = masked_mean(observations[:, :, POLE_ANGLE].abs(), alive)

    # The large penalty is what turns this from a preference into a rule: no
    # amount of extra survival is worth a sustained trip outside the corridor.
    return (survival_score(alive)
            - CORRIDOR_PENALTY * violation
            - uprightness)


def waypoint(observations, termination_logits, step=0):
    """Goal C: drive to one waypoint, hold, then drive to another.

    The target moves with time, so unlike the other goals this one needs to know
    how far into the run we are.
    """
    # Same survival backbone as every other goal.
    alive = survival_mask(termination_logits)

    # The target moves partway through the episode, and the dream is long
    # enough to straddle that moment: at real step 90, a 29-step dream reaches
    # past the switch at step 100. So the target has to move *along the dream*
    # too. Work out the real time of each imagined step first.
    horizon = observations.shape[1]
    steps_ahead = torch.arange(horizon, device=observations.device) + step

    # Then the target that will actually be in force at each of those moments,
    # rather than freezing whichever one happens to apply right now.
    targets = torch.where(
        steps_ahead < WAYPOINT_HOLD_STEPS,
        torch.full_like(steps_ahead, WAYPOINT_FIRST, dtype=observations.dtype),
        torch.full_like(steps_ahead, WAYPOINT_SECOND, dtype=observations.dtype),
    )

    # Distance from wherever the cart is supposed to be at each imagined step.
    distance = (observations[:, :, CART_POSITION] - targets).abs()

    # Scored at the end of the dream, exactly as park() does and for exactly
    # the same reason.
    off_target = terminal_mean(distance)

    # And keep the pole upright while doing it.
    uprightness = masked_mean(observations[:, :, POLE_ANGLE].abs(), alive)

    # Identical arithmetic to park(); only the target differs, and it moves.
    return survival_score(alive) - POSITION_WEIGHT * off_target - uprightness


# Every goal, keyed by the name plan.py uses on the command line and in its
# figures. Adding a new objective means adding one function and one line here.
GOALS = {
    "balance": balance,
    "park": park,
    "corridor": corridor,
    "waypoint": waypoint,
}


# How far ahead each goal asks the planner to dream.
#
# The horizon measured in imagine.py is an upper bound on how far the dream can
# be *trusted*, not the depth at which planning works best. Those turn out to be
# different numbers, and the sweep in plan.py shows why: the planner picks from
# random action sequences, and the longer those sequences get, the smaller the
# chance that any of the fifty is any good. Depth buys foresight and costs
# search quality.
#
# So the goals that only need the cart to *stay* somewhere do better with short,
# reliable plans, while the ones that need it to *go* somewhere need the full
# depth to see a whole manoeuvre through. Both stay within the measured bound.
SHORT_HORIZON = 15
GOAL_HORIZON = {
    "balance": SHORT_HORIZON,
    "corridor": SHORT_HORIZON,
    "park": None,
    "waypoint": None,
}


# ---------------------------------------------------------------------------
# How each goal is scored after the fact
# ---------------------------------------------------------------------------

def report(goal_name, positions, survived):
    """Turn a real episode into the one number that goal cares about.

    positions: (steps,) the cart positions actually visited
    survived:  how many steps the episode lasted

    This is deliberately separate from the scoring functions above. Those guide
    the planner's imagination; this measures what really happened.
    """
    if goal_name == "park":
        # Mean distance from the target across the whole episode.
        return float((positions - PARK_TARGET).abs().mean())

    if goal_name == "corridor":
        # Percentage of steps spent outside the self-imposed limit. The cast to
        # float is needed because the comparison produces booleans, which have
        # no meaningful average until they are numbers.
        outside = (positions.abs() > CORRIDOR_HALF_WIDTH).float()
        return float(outside.mean() * 100.0)

    if goal_name == "waypoint":
        # Distance from whichever waypoint was active at each step.
        steps = torch.arange(len(positions))

        # Build the schedule the planner was following, step by step.
        targets = torch.where(
            steps < WAYPOINT_HOLD_STEPS,
            torch.full_like(positions, WAYPOINT_FIRST),
            torch.full_like(positions, WAYPOINT_SECOND),
        )
        return float((positions - targets).abs().mean())

    # Plain balancing has no position objective, so survival is the whole story.
    return float(survived)


# Units for each goal's reported number, used to label figures and printouts.
REPORT_UNITS = {
    "balance": "steps survived",
    "park": "mean |x - 1.0| (m)",
    "corridor": "% of steps outside +/-0.5",
    "waypoint": "mean tracking error (m)",
}


# Whether a lower reported number is better. Survival wants more, every
# position-based goal wants less.
LOWER_IS_BETTER = {
    "balance": False,
    "park": True,
    "corridor": True,
    "waypoint": True,
}


# ---------------------------------------------------------------------------
# Self-check: run `python goals.py` to see what each goal actually wants
# ---------------------------------------------------------------------------

# Where the illustration of the scoring functions is written.
PREFERENCE_PLOT = "figures/goal_preferences.png"

# Length of the pretend dreams used below, matching a typical planning horizon.
DEMO_HORIZON = 29


def _constant_dream(position, angle=0.0, crashes_at=None):
    """Build a fake dream that sits at one position for the whole horizon.

    Used only for testing and for the figure: it lets us ask each scoring
    function "how much would you like a future that looks like this?" without
    involving the world model at all.
    """
    # Every step has the same cart position and pole angle.
    observations = torch.tensor(
        [[[position, angle]] * DEMO_HORIZON], dtype=torch.float32
    )

    # Large negative logits mean "no crash predicted" at every step.
    logits = torch.full((1, DEMO_HORIZON), -10.0)

    # Optionally make the dream crash partway through, to test survival.
    if crashes_at is not None:
        logits[0, crashes_at:] = 10.0

    return observations, logits


def _self_check():
    """Verify each goal prefers the future it is supposed to prefer."""
    # A dream parked exactly on goal A's target should beat one parked at zero.
    on_target = park(*_constant_dream(PARK_TARGET))
    off_target = park(*_constant_dream(0.0))
    assert on_target > off_target, "park should prefer the target position"

    # A dream inside goal B's corridor should beat one outside it.
    inside = corridor(*_constant_dream(0.0))
    outside = corridor(*_constant_dream(1.5))
    assert inside > outside, "corridor should prefer staying inside"

    # Surviving the whole horizon should beat crashing halfway through, for
    # every goal, because survival is the backbone of all of them.
    for name, goal in GOALS.items():
        survives = goal(*_constant_dream(0.0))
        crashes = goal(*_constant_dream(0.0, crashes_at=DEMO_HORIZON // 2))
        assert survives > crashes, f"{name} should prefer surviving"

    # Goal C should want +1.0 early on and -1.0 later, from the same dream.
    early = waypoint(*_constant_dream(WAYPOINT_FIRST), step=0)
    late = waypoint(*_constant_dream(WAYPOINT_FIRST),
                    step=WAYPOINT_HOLD_STEPS + 1)
    assert early > late, "waypoint should change its mind after the hold"

    print("all goal self-checks passed")


def _plot_preferences():
    """Draw what each goal thinks of a cart sitting at each position."""
    # Sweep the cart across the rail, well beyond both goals' regions.
    positions = torch.linspace(-2.0, 2.0, 200)

    fig, ax = plt.subplots(figsize=(7, 4))

    # Only the goals with a position preference; plain balancing is a flat line
    # and would say nothing useful here. Goal C is drawn twice, before and
    # after its switch, because a single snapshot of it looks identical to
    # goal A and would hide one line underneath the other.
    curves = [
        ("park", GOALS["park"], 0, "-"),
        ("corridor", GOALS["corridor"], 0, "-"),
        ("waypoint, first leg", GOALS["waypoint"], 0, "--"),
        ("waypoint, after switch", GOALS["waypoint"],
         WAYPOINT_HOLD_STEPS + 1, ":"),
    ]

    for label, goal, step, style in curves:
        # Score a stationary, upright, surviving dream at each position.
        scores = [float(goal(*_constant_dream(float(p)), step=step))
                  for p in positions]
        ax.plot(positions, scores, style, label=label)

    # Mark the two landmarks the goals are defined around.
    ax.axvline(PARK_TARGET, color="grey", ls="--", lw=0.8,
               label=f"park target x={PARK_TARGET}")
    ax.axvspan(-CORRIDOR_HALF_WIDTH, CORRIDOR_HALF_WIDTH, color="grey",
               alpha=0.15, label="corridor")

    ax.set_xlabel("cart position (m)")
    ax.set_ylabel("score of an imagined future sitting there")
    ax.set_title("The same model, three different ideas of a good future")
    ax.legend()
    fig.tight_layout()

    # Create figures/ on first run.
    os.makedirs(os.path.dirname(PREFERENCE_PLOT), exist_ok=True)
    fig.savefig(PREFERENCE_PLOT, dpi=150)
    print(f"saved {PREFERENCE_PLOT}")


if __name__ == "__main__":
    # Confirm the scoring functions rank futures the way they claim to.
    _self_check()

    # Then show, visually, what each of them is asking the planner for.
    _plot_preferences()
