"""SKETCH -- batch rollouts, async, active selection, and resume.

The three things being evaluated have latencies that differ by six orders of
magnitude:

    verifier     ~1ms        pure function
    rollout      ~1s-1min    GPU or API bound
    human        hours-days  out of band

That last one is the constraint that shapes everything else. You cannot await a
person inside a loop, so a study cannot live in a process, so state must be
durable and the optimizer must be resumable. Batch and async fall out as
optimizations; durability is structural.
"""

import hp


# --- 1. the evaluator is batch-first ----------------------------------------
# Batching happens on two axes -- candidates x cases -- and different methods
# want different shapes:
#
#   1 candidate  x N cases     ordinary evaluation
#   N candidates x 1 case      pairwise comparison
#   N candidates x N cases     evolution strategies, population methods

async def evaluate(candidate, cases: list) -> list[hp.Outcome]:
  """One GPU batch, not a loop of single renders."""
  images = await render_batch(candidate, [c.prompt for c in cases])   # batched
  scores = await asyncio.gather(*(judge(i, c) for i, c in zip(images, cases)))
  return [hp.Outcome(score=s, trace=t) for s, t in scores]


# A single-case evaluator is auto-batched, so trivial cases stay trivial:
def evaluate_one(candidate, case) -> float: ...


# --- 2. async with bounded concurrency --------------------------------------
# Verifiers are IO-bound, rollouts are resource-bound. One cap per class.

study = hp.optimize(
  Sampler,
  cases=prompts,
  evaluate=evaluate,
  method=hp.optim.tpe(trials=400),
  concurrency={'rollout': 8, 'verifier': 64},   # GPU slots vs API calls
  batch_size=32,
)


# --- 3. ask / tell, so the loop is yours ------------------------------------
# Every method reduces to this, and it is what makes distributed and
# out-of-band evaluation possible at all.

opt = hp.optim.cmaes(Sampler, store='runs/sweep')
while not opt.done:
  batch = opt.ask(16)                       # a whole population at once
  outcomes = await evaluate(batch, cases)
  opt.tell(list(zip(batch, outcomes)))


# --- 4. active selection ----------------------------------------------------
# Human budget is the scarce resource; spend it where it moves the answer.

pairs = hp.select(
  finalists,
  budget=300,
  by=[
    hp.uncertainty(),        # Elo variance -- who is genuinely unranked
    hp.disagreement(),       # where cheap verifiers contradict each other
    hp.coverage('prompt'),   # do not ask 50 questions about one prompt
  ],
)

# `disagreement` is the cheap one worth having: it needs no model, only the
# verifiers already being run, and it points straight at where the automatic
# signal is unreliable.


# --- 5. out-of-band feedback ------------------------------------------------
# A person is not awaitable. Submit, persist, come back.

study = hp.optimize(Sampler, cases=prompts, store='runs/sweep')
study.submit(pairs, to='cleanser')          # queued for review; process exits

# ... hours or days later, a different process ...
study = hp.resume('runs/sweep')
study.collect()                             # folds in whatever came back
study.step()                                # proposes the next round


# --- 6. resume is a property of the design, not a feature -------------------
# If the optimizer is a pure function of completed trials, resuming is just
# replaying them. Optuna works this way; it is why storage is the source of
# truth rather than the sampler object.

study = hp.resume('runs/sweep')     # rebuilds TPE from trial history

# Methods with genuine internal state -- CMA-ES covariance, an ES population --
# persist a blob alongside, because history alone does not determine them.


# --- 7. the cache makes resume nearly free ----------------------------------

key = (
  hp.stable_hash(candidate),
  case.id,
  evaluator_version,      # checker source + harness commit + lock hash
  environment_hash,
  seed,
)

# Resuming a half-finished study re-runs nothing that already succeeded, which
# is also what makes an interrupted 400-trial sweep survivable.


# NOTES ----------------------------------------------------------------------
#
# Batch-first evaluator
#   + a GPU renders 32 images for barely more than 1; a per-case loop wastes
#     most of the hardware
#   + it is the shape GEPA's adapter and TRL's reward functions already use,
#     so existing evaluators port
#   - failures become partial: one bad case must not fail the batch, so an
#     outcome list can contain a failed status per element
#   - batch size interacts with pruning; a pruned trial has already paid for
#     the whole batch
#
# Async with per-class concurrency caps
#   + an LLM judge and a GPU render want completely different limits, and one
#     global cap starves whichever is cheaper
#   + TRL already runs reward functions concurrently for this reason
#   - it colours the whole evaluation surface async, and a sync escape hatch
#     has to exist for people who do not want it
#
# Ask/tell as the substrate
#   + the loop belongs to the caller, so distributed workers, notebooks and
#     out-of-band humans all work without special support
#   + Optuna's tell() taking a trial *number* means the trial object need not
#     survive the round trip
#   - the optimizer no longer controls pacing, so it cannot do anything clever
#     about when to stop unless asked
#
# Storage as the source of truth
#   + resume becomes replay; no sampler internals to serialize for most methods
#   + two workers can share a study by sharing storage
#   - replaying a long history costs something at startup
#   - stateful methods (CMA-ES, ES) still need a blob, so the property is not
#     universal and must not be assumed by the interface
#
# Active selection
#   + uncertainty and disagreement are both computable from what is already
#     being collected, so neither needs a new model
#   + coverage stops the budget being spent on one easy cluster
#   - selecting adaptively biases the sample, so the held-out set must stay
#     randomly drawn or the final number is not trustworthy
#
# Out-of-band feedback
#   + makes human review a first-class participant rather than a blocking call
#   + the same mechanism covers a long training job or an overnight benchmark
#   - a study now has a lifecycle -- pending, collected, stale -- and answers
#     may arrive for a candidate the search has already moved past
