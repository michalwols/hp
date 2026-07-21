"""SKETCH -- mining cases, validators and priors out of things you already have.

A production log, an existing test suite, or the current implementation each
encode what "typical" and "correct" mean. Four different things can be pulled
out of them, and conflating them is how this gets confusing:

  cases       the input distribution        -- what to evaluate on
  oracle      reference outputs             -- what correct looks like
  validators  assertions already written    -- how correctness is checked
  priors      configs that were tried       -- where to start searching
"""

import hp


# --- 1. cases from a production log -----------------------------------------
# The point of using real traffic is the input *distribution*, which no
# hand-written benchmark reproduces.

cases = hp.cases.from_records(
  read_jsonl('logs/queries-7d.jsonl'),
  input='sql',
  meta=('table', 'rows_scanned', 'duration_ms', 'errored'),
  # a raw log is dominated by a handful of hot queries; sampling it naively
  # tunes the head and silently breaks the tail
  stratify=('table', 'errored'),
  oversample={'errored': 3.0, 'duration_ms > 1000': 2.0},
  n=200,
)


# --- 2. the current system as the oracle ------------------------------------
# Differential testing: whatever the candidate does, it must agree with what
# already ships.

def rows_equivalent(candidate, reference) -> bool:
  """Equivalence, not equality -- row order is not semantic here."""
  return sorted(candidate.rows) == sorted(reference.rows)


oracle = hp.reference(current_planner, equivalent=rows_equivalent)

# For a CUDA kernel the same shape holds, with a numeric comparator:
oracle = hp.reference(torch_matmul, equivalent=lambda a, b: allclose(a, b, rtol=1e-5))


# --- 3. an existing test suite as the validator -----------------------------
# The assertions are already written; reuse them rather than restating them.

validators = hp.validators.from_tests('tests/', select='tests/sql/**')


# --- 4. priors from previous studies ----------------------------------------
# Configs that did well last time are the cheapest possible warm start.

prior = hp.prior.from_history('runs/*/trial.json', top=10)


# --- putting it together ----------------------------------------------------

study = hp.optimize(
  Planner,
  cases=cases,
  verifiers=[oracle, validators, latency],
  weights=[0.0, 0.0, 1.0],          # correctness is a gate, speed is the goal
  constraints={'mismatch_rate': 0.0, 'regression_rate': 0.01},
  prior=prior,
  method=hp.optim.tpe(trials=200, pruner='asha'),
)


# NOTES ----------------------------------------------------------------------
#
# The reference is a constraint, not an objective
#   + this is the whole trick for kernel and query optimization: pin
#     correctness to parity, then optimize speed or cost underneath it
#   + it turns an unmeasurable goal ("is this right?") into a measurable one
#     ("does it agree with what already ships?")
#   - it caps quality at parity by construction. You inherit the reference's
#     bugs and can never score better than it, so a reference oracle can only
#     ever be a safety rail -- never the thing you are trying to improve
#
# Equivalence is pluggable, and is really just a verifier
#   + row-order-insensitive for SQL, tolerance-based for float kernels,
#     semantic for text: one comparator argument covers all three
#   + it composes with the verifier list, so the oracle is not a special case
#   - the comparator is where correctness bugs hide; too loose and the
#     optimizer exploits it, too strict and every candidate fails
#
# Sampling a log
#   + real traffic is the only honest input distribution
#   + stratify by shape and over-sample failures and slow cases, or you tune
#     the hot path and regress the tail where the bugs actually are
#   - production logs carry user data; a sampling policy needs redaction and
#     an approved retention path before any of this runs
#   - the log records what the *current* system was asked, which is itself
#     shaped by what it does well -- optimizing on it reinforces that bias
#
# Cheap oracle vs expensive oracle
#   + the reference is often much cheaper than ground truth (running the old
#     kernel beats a human labelling), which makes it a natural early rung:
#     screen every candidate against the cheap oracle, promote survivors to
#     the expensive check
#   + that maps straight onto ASHA-style multi-fidelity
#   - agreement with a cheap oracle correlates imperfectly with being right,
#     so the final rung still has to be the real thing
#
# Priors and warm starts
#   + previous studies are free information, and stable_hash already gives the
#     join key between a config and its recorded outcome
#   + enqueueing known-good configs costs nothing and often beats the sampler
#     for the first dozen trials
#   - a warm-started trial has consumed more budget than a cold one; comparing
#     them as equals silently rewards the warm start
#   - a prior from a different code version can be worse than nothing, so the
#     evaluator version has to be part of the join key
#
# Leakage
#   - if the optimizer can read the expected outputs, it will eventually
#     memorize them rather than generalize. Cases need a held-out split, and
#     for an agentic optimizer the reference outputs belong behind the same
#     readonly fence as the evaluator itself.
