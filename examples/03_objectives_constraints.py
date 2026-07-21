"""SKETCH -- three ways to attach objectives and constraints to params.

None of `hp.Objective`, `hp.Constraint` or the expression form exist yet. This
file exists to compare them on one problem: tune a retrieval service for
quality, subject to latency and cost limits.
"""

import hp


# --- A. field roles ---------------------------------------------------------
# Objectives and constraints are field types, like Choice and Range.

class ServiceA(hp.Params):
  # inputs
  top_k: int = hp.IntRange(1, 100, default=10)
  rerank: bool = False
  model: str = hp.Choice(('small', 'base', 'large'), default='base')

  # outputs, filled in per trial
  recall = hp.Objective(maximize=True)
  latency_ms = hp.Objective(minimize=True)
  cost_usd = hp.Metric()                       # tracked, not optimized

  # constraints, checked per trial
  latency_budget = hp.Constraint('latency_ms <= 500')
  cost_budget = hp.Constraint('cost_usd <= 0.01')


# --- B. expression graph ----------------------------------------------------
# Params are objects that compose, so constraints are real expressions.

class ServiceB(hp.Params):
  top_k = hp.IntRange(1, 100, default=10)
  rerank = hp.Param(False)
  model = hp.Choice(('small', 'base', 'large'), default='base')

  recall = hp.Objective(maximize=True)
  latency_ms = hp.Objective(minimize=True)
  cost_usd = hp.Metric()

  limits = [
    latency_ms <= 500,
    cost_usd <= 0.01,
    top_k * (2 if rerank else 1) <= 150,      # relates two inputs
  ]


# --- C. functional ----------------------------------------------------------
# The params stay a plain config; the objective returns everything.

class ServiceC(hp.Params):
  top_k: int = hp.IntRange(1, 100, default=10)
  rerank: bool = False
  model: str = hp.Choice(('small', 'base', 'large'), default='base')


def evaluate(config: ServiceC) -> 'hp.Outcome':
  result = run_service(config)
  return hp.Outcome(
    score=(result.recall, -result.latency_ms),
    metrics={'cost_usd': result.cost},
    constraints={
      'latency': result.latency_ms - 500,      # <= 0 is feasible
      'cost': result.cost - 0.01,
    },
  )


# NOTES ----------------------------------------------------------------------
#
# A. field roles
#   + no new machinery: Objective/Constraint/Metric are Field subclasses, so
#     they inherit serialization, help text, provenance and --help for free
#   + the problem is fully described by the class, which is exactly what an
#     agentic optimizer needs to read
#   + a trial is just an instance with the output fields filled in, so
#     stable_hash, diff and sources all keep working unchanged
#   - constraints are strings or callables, so a solver cannot inspect them;
#     you can check feasibility but not optimize against them symbolically
#   - relating two inputs ("top_k doubles when rerank is on") has nowhere
#     natural to live
#
# B. expression graph
#   + constraints are inspectable structure, so they translate to CVXPY or
#     OR-Tools instead of being black boxes -- the one thing A cannot do
#   + conditional fields become graph edges rather than opaque `when=`
#     callables, so a sampler can order draws topologically
#   + derived values and constraints use one mechanism
#   - operator overloading on every param, plus an expression tree, plus a
#     translator per solver backend: by far the most machinery here
#   - `top_k * (2 if rerank else 1)` does not actually work, because a Python
#     conditional evaluates eagerly; it needs hp.where(...), and that is the
#     moment the API stops looking like Python
#   - harder to hand to an LLM than a flat description
#
# C. functional
#   + zero new vocabulary, and the objective can compute anything
#   + matches how Optuna, Ray and TRL already work, so it ports directly
#   - the problem is invisible until you run it: nothing can enumerate the
#     objectives or constraints ahead of time, which blocks both a --help-style
#     summary and an agent reasoning about what to explore
#   - encourages restating the same limits in several places
#
# Suggested mix: A as the declaration, C as the escape hatch, B only where a
# solver genuinely needs symbolic constraints. A and C are complementary --
# Outcome is how a trial reports, field roles are how the problem is declared.
