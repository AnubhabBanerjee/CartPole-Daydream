# Project Plan

## Title

**Building a Minimal World Model from Scratch: Predicting CartPole Physics**

## Goal & Audience

A tutorial-style article for Towards Data Science, aimed at beginners. The reader
should finish with a working mental model of what a world model is, plus a repo
they can clone and run end-to-end on a CPU in a few minutes.

Guiding constraint: **the repo and the implementation stay as simple as possible.**
Every design decision below is chosen for teachability first, performance second.

## Repo Layout

```
collect_data.py     # generate the offline dataset
model.py            # the GRU world model (< 100 lines)
train.py            # training loop with BPTT
imagine.py          # autoregressive rollout + drift measurement
plan.py             # act in the real env by dreaming futures (~25 lines)
goals.py            # the scoring functions: balance, park, corridor, waypoints
baseline.py         # minimal DQN, the comparison agent (~80 lines)
requirements.txt    # torch, gymnasium, numpy, matplotlib
README.md
```

Each script runs as `python <script>.py` with no arguments and sane defaults.
No config framework, no experiment tracker, no CLI parsing beyond constants at
the top of each file.

---

## 1. The Core Concept: Learning to Imagine

* **The hook:** A world model is a neural network that learns to *be* the
  environment. Once you have one, an agent can plan and learn inside its own
  imagination without ever touching the real simulator.
* **Why that matters at scale (one-line aside):** imagined rollouts run batched
  on the GPU, while real simulators are stepped one at a time on the CPU. Not a
  real constraint for CartPole, but it is the reason the idea scales.
* **The framing:** this is a *minimal, deterministic* world model. Production
  systems use stochastic latent states to represent uncertainty; a deterministic
  GRU is the foundation you need before that makes any sense. Defer the term
  RSSM to the conclusion.

## 2. The Data Pipeline: Generating Informative Trajectories

* **The environment:** `CartPole-v1`, with its 4D observation vector — cart
  position, cart velocity, pole angle, pole angular velocity.
* **The partial-observability twist:** the model only ever sees **2 of those 4
  dimensions** — cart position and pole angle. Velocities are withheld. This is
  what makes memory genuinely necessary (see section 3). We still record all 4
  dimensions to disk so the article can plot ground truth.
* **The dataset engineering problem:** a purely random agent drops the pole in
  ~15 steps. That data never shows the model what balanced, stable physics looks
  like.
* **The solution — a mixed-policy collector:** blend three sources of actions.
  * A hand-written proportional controller for stable balancing data:
    push right if `angle + 0.5 * angular_velocity > 0`, else push left.
    Three lines, no RL training, no second topic to explain.
  * ε-greedy noise on top of it (ε ≈ 0.2–0.3) for recovery dynamics and edge
    cases near the failure boundary.
  * A small per-episode **drift bias** added to the controller's decision, so
    some episodes balance in place and others balance while sliding steadily
    along the rail. Without it the cart never travels, the dataset contains no
    sustained one-directional pushing, and the planner's questions about going
    somewhere are answered with confident nonsense. See decision D1.
* **`terminated` vs `truncated` — record them separately.** Gymnasium's `step`
  returns two distinct end-of-episode flags and they mean different things:
  * `terminated` is a **physical failure**: `|cart position| > 2.4` or
    `|pole angle| > 12°` (0.2095 rad). It is a deterministic function of the
    observation — and, conveniently, of the two dimensions we keep. Learnable.
  * `truncated` is the `TimeLimit` wrapper firing at **500 steps**. It is a
    function of the step counter, which the model never sees. Not learnable.
  * The collector writes two separate arrays. It never computes
    `done = terminated or truncated`, which is the standard trap: it would label
    the 500th step of a perfectly balanced episode as a failure and teach the
    model to predict a coin flip.
* **Output:** a few hundred episodes saved as a single `.npz` of observations,
  actions, `terminated`, and `truncated`.

## 3. The Architecture: The Deterministic GRU

Core transition module kept under 100 lines.

* **Input:** concatenate the 2D partial observation with the action. CartPole's
  action space is discrete `{0, 1}`; feed it as a **2D one-hot vector** so the
  network treats left and right symmetrically. Input dimension = 4.
* **Preprocessing:** standardize observations using the mean and standard
  deviation computed over the training set. Save these statistics alongside the
  model — `imagine.py` needs them.
* **Memory:** a single `nn.GRUCell`, hidden size 64. Because the velocities are
  hidden, the hidden state must infer them from the history of positions and
  angles. This is the point of the whole architecture.
* **Two output heads** projecting from the hidden state:
  * **Next-observation head** → predicts the **delta** `o_{t+1} - o_t` in
    normalized space, not the absolute next observation. This is the single
    change that decides whether imagined rollouts look like physics or like a
    flat line, and it deserves its own paragraph in the article.
  * **Termination head** → a single logit predicting `terminated_{t+1}`, and
    **only** `terminated`. State the target in one unambiguous sentence in the
    article: *given the hidden state after consuming `(o_t, a_t)`, predict
    whether the resulting next state `o_{t+1}` is a physical failure state.*
    This is the same temporal alignment as the next-observation head, so both
    heads read off the same hidden state and answer questions about step `t+1`.
    Truncation is deliberately excluded — the model has no access to the step
    counter, so asking it to predict the time limit would be asking it to
    hallucinate.
    This head replaces a reward head: CartPole's reward is a constant `1.0` on
    every non-terminal step, so predicting it is trivial and teaches nothing.
    Termination is genuinely informative, and section 5 needs it — an imagined
    rollout has to know when the episode would have been over.

## 4. The Training Loop: Backpropagation Through Time

* **Losses:** MSE on the predicted observation delta, plus BCE-with-logits on
  the termination head. Sum them, with a small weight on the termination term so
  the dynamics loss dominates.
* **How truncation is handled in the loss.** A truncated episode is cut off
  mid-flight, so its last recorded transition is still perfectly valid physics —
  keep it in the dynamics loss. Its termination label is simply `0`, which is
  also correct: nothing failed. So truncation needs no masking anywhere; it just
  must never be folded into the positive class. Say this explicitly rather than
  leaving the reader to wonder whether those steps were dropped.
* **Class imbalance:** there is at most one positive termination label per
  episode, against 50–500 negatives. Use `pos_weight` in `BCEWithLogitsLoss` and
  report termination **recall and precision** separately from the loss, since a
  model that always predicts "not terminated" would otherwise look excellent.
  Accuracy is the wrong summary here for exactly that reason. See decision D4.
* **Temporal unrolling:** sample sequence chunks of 32 steps, carry the hidden
  state across the chunk, and let autograd backpropagate through the whole
  unrolled graph. The first 8 steps are a **burn-in**: they are still unrolled,
  but excluded from the loss, because the memory starts blank and the model
  cannot know a velocity it has not been shown. See decision D3. Explain BPTT concretely here — the graph is the chain of
  GRUCell calls, and the gradient flows backward along it.
* **Teacher forcing — name it explicitly.** During training, every step receives
  the *ground-truth* observation as input. During evaluation the model consumes
  its *own* predictions. That mismatch is the direct cause of the compounding
  error measured in section 5, so this paragraph is what links the two sections
  into a single argument rather than two separate observations.
* Keep it plain: Adam, a fixed number of epochs, gradient clipping, no early
  stopping. The one exception is a cosine-annealing **learning-rate schedule**,
  which eases the rate to zero over training; without it the validation error
  never settles. See decision D2.

## 5. Measuring Imagined Rollouts: How Far Can the Dream Be Trusted?

* **The experiment:** cut the model off from the real simulator. Warm the hidden
  state up on a handful of real steps, then supply only an action sequence and
  let the model autoregressively unroll the future entirely from its own
  predictions.
* **Qualitative result:** plot the real pole angle against the imagined pole
  angle over the rollout horizon. Overlay a few episodes.
* **Quantitative result:** plot prediction RMSE as a function of horizon `h`,
  (root-mean-square, so the numbers stay in metres and radians),
  averaged over many held-out episodes. Alongside it, report the gap between the
  imagined termination step and the real one — the model's answer to "how long
  do I survive?" versus the truth.
* **Termination inside the rollout:** imagine the full horizon for every dream,
  then treat every step at or after the first predicted failure as dead when
  scoring. Equivalent to stopping at the 0.5 crossing, and it keeps all the
  dreams the same length so they can be run as one batch. See decision D5. Since the model only predicts physical
  failure, the rollout has no notion of the 500-step time limit; cap the horizon
  explicitly in `imagine.py` instead. Worth one sentence in the article — it is
  a concrete illustration of why the two flags were kept apart.
* **A "freeze the last state" baseline** is plotted alongside the drift curve:
  a model that predicts nothing ever changes. Without a floor to compare
  against, "the error reaches 0.019 rad by step 29" is a number with no meaning.
  See decision D7.
* **Do not assert a predetermined failure point** — report the measured
  divergence curve and let it make the argument.
* **This section does not end the article.** It ends by turning the error curve
  into a usable number: *the dream is trustworthy for roughly H steps.* That
  number is the **upper bound** on planning depth handed to section 6. It is not
  automatically the best depth to plan at: the sweep in section 6 shows random
  search degrading with depth long before the model does, so the goals that only
  need the cart to stay put use a shorter horizon than the goals that need it to
  travel. See decision D10.

## 6. The Real Payoff: Acting by Dreaming

Sections 1–5 build the **engine** — a network that learned the rules. This
section builds the **player** on top of it, and it is what makes the phrase
"world model" mean something to the reader. Without it the article stops at a
learned simulator.

* **The loop (`plan.py`), run at every real step:**
  1. Take the current real observation and the model's current hidden state.
  2. Invent ~200 random action sequences of length `H`. Fifty was not enough
     once the goals required steering the cart: most random sequences drop the
     pole and are discarded. See decision D8.
  3. Dream all 200 forward **inside the model**, batched. The real simulator is
     never touched during this step.
  4. Score each dream over its **entire** imagined future, not just the next
     step: how many steps before the termination head fires, and how small the
     pole angle stayed. This is the heart of the idea — the planner will accept
     a slightly worse next step to reach a better future fifteen steps out.
  5. Execute only the **first** action of the best-scoring sequence in the real
     environment. Discard the rest, advance the hidden state, repeat.
* **Why this design:** no reinforcement learning, no second training loop, no
  reward model. Roughly 25 lines. It demonstrates exactly the intuition — *it
  considered 200 possible futures and picked the best one* — at a fraction of the
  complexity of training a policy inside the dream. The *file* is much longer
  than 25 lines, because it also carries the evaluation harness, the horizon
  sweep and the figures; the dreaming loop itself is still tiny. See D12.
* **It works at all:** survival time on the real CartPole, planner versus random
  versus the hand-written controller. This is a sanity check, not the headline —
  keep it to one bar chart and move on.
* **The connection back to section 5:** sweep the planning horizon `H` and plot
  survival against it. Too short and the planner is greedy and blind; too long
  and it plans inside the part of the dream that has already drifted into
  fiction. The measured error curve predicted where that sweet spot would be.
  This is what makes section 5 actionable rather than merely interesting.
* **It also retroactively justifies the termination head:** scoring a dream
  means asking "does the pole fall in this imagined future?"

## 7. The Argument: Doing Things It Was Never Trained To Do

The comparison agent is a **small DQN** (`baseline.py`): two hidden layers of
128, trained on CartPole's default reward. Training keeps the best snapshot it
reaches rather than the final weights, because DQN training is unstable enough
that the last episode is not representative. Both choices make the baseline
stronger than the plain version. See decisions D16 and D17.

**What we deliberately do not claim.** We are not comparing raw scores on
standard CartPole, and we are not claiming to need less data. CartPole caps at
500, a DQN solves it outright, and the planner ties at best. Racing on that axis
would be a contest we cannot win and does not matter. Say so in one sentence and
move on — the article is stronger for conceding it early.

**The claim we do make:** the planner can be given a *new goal at run time*, by
editing a scoring function, with **zero new environment steps and zero weight
updates**. The DQN cannot be told anything at run time; it must be retrained.

### The constraint that shapes every goal

The model predicts only **cart position, pole angle, and termination**. Every
scoring function must be expressible from those three. This is a genuine design
discipline, not a limitation to hide — state it plainly in the article.

All three goals live in `goals.py` as small scoring functions with the same
signature, so the article can show that swapping goals means swapping one
function.

### Goal A — Park the cart (the headline)

*"Balance the pole, and hold the cart at x = +1.0."*

* Score: `survival − λ · |x − 1.0| at the end of the dream − upright penalty`.
  Averaging the distance over the *whole* dream fails completely, because
  reaching the target requires an opening move that increases the distance.
  See decision D13 — this is the single most important lesson in the project.
* Metric: mean |x − 1.0| across 100 episodes, plus survival time.
* Figure: two cart-position traces over time — the planner converging onto the
  target line, the DQN wandering wherever momentum takes it.
* Most intuitive of the three and the clearest plot. Lead with this one.

### Goal B — Stay in a safe corridor (the reliable one)

*"Balance, but never let |x| exceed 0.5."*

* The environment's real limit is 2.4, so this constraint is far tighter than
  anything present in training data.
* Score: a large penalty proportional to how far outside the corridor the dream
  strays, averaged over its live steps. Proportional rather than a yes/no
  tripwire, so the planner can tell "just nicked the line" from "toured the
  suburbs" and has a gradient to follow. See decision D14.
* Metric: percentage of steps spent outside the corridor, plus survival time.
* The most robust of the three, because the constraint is local and needs no
  long-horizon planning. It also frames naturally as a safety requirement, which
  is how readers meet this problem in practice. Keep it as the guaranteed win.

### Goal C — Move on command (the stretch)

*"Go to x = +1.0, hold for 100 steps, then return to x = −1.0."*

* A time-varying setpoint rather than a fixed one. The target must move *along
  the dream* as well as along the episode: at real step 90, a 29-step dream
  reaches past the switch at step 100. See decision D15.
* Metric: tracking error against the setpoint schedule over time.
* This is the one that genuinely looks like the model *performing a task* rather
  than surviving. It is also the riskiest: it needs a longer planning horizon,
  which is exactly where drift bites hardest.
* **If it fails, cut it and say why in one sentence.** A measured failure here
  reinforces section 5 rather than undermining the article.

### How the DQN's failure is reported

Precisely, and without overstating it. The DQN does not break — it still scores
~500 on survival. It fails the *new objective* completely, and the reason is the
whole point: there is no mechanism to tell it about a new goal at run time.

Report the contrast as two measurements, not one:

1. The **balance-trained DQN, asked for each new goal with no retraining.** It
   does not break — it plays on happily and ignores the request entirely,
   because there is no way to make the request. This is the measurement that
   demonstrates the claim.
2. The **environment steps needed to retrain** it until it does want the new
   goal, against our zero.

### The cost we disclose rather than hide

Decision-time latency, in milliseconds per action. The DQN is a single forward
pass; we run 200 dreams of `H` steps each and lose. Both sides are timed on the
same clock inside the shared `evaluate()`, so the ratio is measured rather than
asserted. Report it in the same table as the wins. Conceding this openly is what makes the rest
credible.

### Fairness rules

Same evaluation seeds for every agent, DQN results as mean ± std over 5 training
seeds, and a single shared `evaluate()` used by both so that neither agent gets
a bespoke measurement path.

## 8. Closing: Where This Goes Next

Two honest limitations, each pointing at what real systems do:

* **The planner thinks from scratch at every single step**, which is slow. Real
  systems instead let an agent *practice inside the dream* until it has learned
  a policy, then deploy that policy with no planning at play time. Name Dreamer
  here as the thing the reader should look at next.
* **The dream drifts**, because a deterministic model must commit to exactly one
  future. Modelling uncertainty is why the field moved from deterministic GRUs
  to stochastic latent-state models. Introduce the term RSSM here, once.

---

## Decisions That Changed During the Build

This plan was written before the code existed. Building it turned up things the
plan got wrong. The sections above have been amended to describe what was
actually built; this table records what moved and why, so the reasoning is not
lost.

Most of these are not compromises. Three of them (D1, D8, D13) are the reason
the planner works at all, and two are now lessons in the article.

| # | The plan said | We built | Why |
|---|---|---|---|
| D1 | Collector has two action sources | Three: the drift bias was added | The controller corrected every wobble, so the cart never travelled and the data held no sustained pushing. The planner asks about exactly that. **Lesson 2 in the article.** |
| D2 | No learning-rate scheduler | Cosine annealing | At a fixed rate the validation error bounced by a factor of ten between epochs and never settled. |
| D3 | Loss over all 32 unrolled steps | First 8 excluded as burn-in | The memory starts blank and the velocities are hidden, so the first steps ask the model for information it has not been given. Grading them teaches it to hedge. |
| D4 | Report termination accuracy | Report recall and precision | Accuracy is exactly the number a "never crashes" model fakes, which is the failure the plan wanted to catch. Recall is what a planner actually needs. |
| D5 | Stop imagining at p > 0.5 | Imagine the full horizon, mask the dead steps when scoring | Same result, but every dream stays the same length and all 200 run as one batch. |
| D6 | Plot MSE against horizon | Plot RMSE | Keeps the axis in metres and radians, so the reader can compare the error to the 0.2095 rad failure threshold by eye. |
| D7 | No baseline on the drift plot | Added "freeze the last state" | An error curve with no floor to compare against cannot tell you whether the model is good. This is where "beats the lazy baseline for 37 steps" comes from. |
| D8 | ~50 candidate action sequences | 200 | Measured. At 50, the goals requiring the cart to travel failed: most random sequences drop the pole and are thrown away, leaving too few survivors to choose between. |
| D9 | Plan at every real step | First 8 steps use the hand-written controller | The memory is blank for the first few steps, so planning on it is planning blind. Disclosed in the article rather than hidden. |
| D10 | One measured H, handed to the planner | H is an upper bound; balance and corridor use 15 | The horizon sweep showed random search degrading with depth well before the model does. **Lesson 3 in the article.** |
| D11 | 100 evaluation episodes | 30, then corrected back to 100 | 30 was chosen to save compute and never justified. Rerun at 100; the numbers moved by less than 0.005 m, so nothing rested on it. |
| D12 | `plan.py` ~25 lines | 557 lines | The dreaming loop is still tiny. The rest is the evaluation harness, the horizon sweep and the figures, which the plan never budgeted for. |
| D13 | Goal A scores mean distance over the dream | Scores distance at the *end* of the dream, plus an upright penalty | Averaging punishes the opening move, and reaching the target *requires* an opening move in the wrong direction. The plan's formula cannot express the solution. **Lesson 1 in the article.** |
| D14 | Goal B is a hard tripwire | A large proportional penalty | A yes/no penalty gives the planner no gradient: every violating dream looks equally bad, so it cannot steer back. |
| D15 | Goal C target varies with time | Initially frozen per dream; **fixed** to vary along the dream | Caught by the audit. A dream that straddles the switch must see it coming. Fixing it improved tracking error from 1.183 m to 1.006 m. |
| D16 | "Minimal" DQN | Two hidden layers of 128 | A single skinny layer trained less reliably, and an unreliable baseline is an unfair one. |
| D17 | Ordinary DQN training | Keeps the best snapshot, not the final weights | DQN training is unstable enough that the final episode is not representative. This makes the baseline *stronger* than ours. |
| D18 | Show the untrained-for DQN failing the new goal | Initially skipped; **added** | Caught by the audit. This is the measurement the central claim rests on, and only the retraining cost had been measured. |
| D19 | Latency for both agents in ms | DQN was hard-coded to 0.0; **fixed** | Caught by the audit. Measured, the gap is 18x, not the "hundreds of times" the draft had asserted. |

---

## Writing Notes

* Beginner audience throughout. Avoid phrases like "agentic inference stacks,"
  "stochastic latent spaces," or "structural advantage" in the opening sections.
* Every code block in the article should be copy-pasteable and correspond to
  real code in the repo.
* Target roughly 3,500–4,000 words, with 7-8 figures.
