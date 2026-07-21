"""SKETCH -- verifiers, weights, and what a trial reports back.

Follows the shape the ecosystem converged on: a list of scoring functions,
weighted, where a weight of zero means "track but do not optimize".
"""

import hp


class SQLAgent(hp.Params):
  model: str = 'qwen3-4b'
  temperature: float = hp.Range(0.0, 1.5, default=0.7)
  max_retries: int = hp.IntRange(0, 5, default=2)


# --- verifiers are ordinary functions ---------------------------------------
# Each declares only the arguments it wants; hp injects by name, the same way
# hp.wrap already does.

def correctness(result, expected) -> hp.Outcome:
  if result.rows == expected.rows:
    return hp.Outcome(score=1.0)
  return hp.Outcome(
    score=0.0,
    feedback=f'{len(result.rows ^ expected.rows)} rows differed, '
             f'first at key={next(iter(result.rows ^ expected.rows))}',
    details={'sql': result.sql, 'plan': result.plan},
  )


def latency(result) -> float:
  return -result.elapsed_ms / 1000


def tokens(result) -> float:
  return float(result.tokens)          # weight 0 below: a metric, not a goal


async def judge(result, task) -> hp.Outcome:
  """LLM-as-judge: a verifier that happens to be slow and returns prose."""
  verdict = await ask_judge(task.question, result.sql)
  return hp.Outcome(score=verdict.score, feedback=verdict.reasoning)


study = hp.optimize(
  SQLAgent,
  cases=train_tasks,
  verifiers=[correctness, latency, tokens, judge],
  weights=[1.0, 0.2, 0.0, 0.5],
  constraints={'incorrect_rate': 0.01},        # <= is feasible
  method=hp.optim.tpe(trials=64, pruner='asha'),
)


# NOTES ----------------------------------------------------------------------
#
# Weighted list of verifiers
#   + matches TRL's reward_funcs and the verifiers package's Rubric, so
#     existing scoring functions port with no rewrite
#   + weight=0 unifies "metric" and "objective" into one list, which is
#     strictly fewer concepts than a separate metrics dict
#   + returning None from a verifier can mean "not applicable here", which is
#     how task-specific scorers compose in one sweep
#   - a single weighted sum hides trade-offs; the weights become their own
#     tuning problem, which is the argument for optimizing the Pareto front
#     and picking an operating point afterwards
#
# Verifier returns float | Outcome
#   + trivial scorers stay one-liners
#   + rich ones attach feedback and details without a second mechanism
#   + follows dspy's precedent, where Evaluate reads only `score` and GEPA
#     reads both `score` and `feedback` -- one return type, two consumers
#   - two return types to handle everywhere; needs a normalizer at the boundary
#
# Async verifiers
#   + an LLM judge is IO-bound, and TRL already runs reward functions
#     concurrently for exactly this reason
#   - forces the evaluation loop to be async, or to bridge, which colours a
#     lot of the surface
#
# Feedback as a first-class field
#   + it is the gradient for reflective optimizers; GEPA calls this Actionable
#     Side Information and treats it as the text analogue of a gradient
#   + a human debugging a bad trial reads it too, instead of seeing 0.37
#   - useless to numeric optimizers, so it must be cheap to ignore
