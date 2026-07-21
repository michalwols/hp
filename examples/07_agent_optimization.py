"""SKETCH -- optimizing an agent, including the harness it runs in.

This is the end state: text components, execution traces, verifiers, and an
optimizer allowed to edit some of the code. It is also where the guardrails
matter most, because an optimizer that can edit the harness will hack the
evaluator if you let it.
"""

import hp


class Agent(hp.Params):
  # scalar knobs, searched numerically
  temperature: float = hp.Range(0.0, 1.5, default=0.7)
  max_steps: int = hp.IntRange(1, 20, default=8)
  model: str = 'qwen3-4b'

  # text components, rewritten by a reflective optimizer rather than sampled
  system_prompt: str = hp.Evolve(
    'You are a data agent. Solve the task with the available tools.',
    description='Overall behaviour and planning policy',
  )
  recovery_policy: str = hp.Evolve(
    'When a query fails, inspect the error and revise it.',
    description='Error recovery behaviour',
    group='tools',
  )

  # measured
  correct = hp.Objective(maximize=True)
  latency_ms = hp.Objective(minimize=True)


def evaluate(agent: Agent, task) -> hp.Outcome:
  trace = run_agent(agent, task)                 # full message graph
  check = verify(task, trace)
  return hp.Outcome(
    score=float(check.correct),
    metrics={'latency_ms': trace.elapsed_ms, 'tool_calls': len(trace.tool_calls)},
    constraints={'incorrect': 0.0 if check.correct else 1.0},
    feedback=check.explanation,                  # what a reflective optimizer reads
    trace=trace,                                 # what GEPA-style methods mutate on
    steps=[{'step': i, 'reward': r} for i, r in enumerate(check.step_rewards)],
  )


# Stage 1: numeric search over the scalar knobs
agent = hp.optimize(Agent, cases=tasks, evaluate=evaluate,
                    method=hp.optim.tpe(trials=40)).best

# Stage 2: reflective search over the text components only
agent = hp.optimize(agent, cases=tasks, evaluate=evaluate,
                    method=hp.optim.gepa(reflection_model='gpt-5', budget=150)).best

# Stage 3: let an optimizer edit the harness itself, inside a fence
agent = hp.optimize(
  hp.Repo('agents/sql', editable=['skill.md', 'tools.py'],
          readonly=['evals/**', 'tests/reference/**']),
  cases=tasks,
  evaluate=evaluate,
  method=hp.optim.agent_loop(model='gpt-5', rounds=10, gate='strict'),
).best


# NOTES ----------------------------------------------------------------------
#
# Staged rather than joint
#   + each stage is debuggable and its contribution attributable
#   + the methods genuinely differ: TPE samples, GEPA reflects on text, an
#     agent loop patches code. One optimizer over all three at once cannot
#     credit any of them
#   - the stages interact; a prompt tuned at temperature 0.7 may not be best
#     at 0.3, so it wants a second pass rather than a single sweep
#
# Trace as a first-class outcome field
#   + reflective optimizers need to see *why*, not just the score, and
#     verifiers-the-package makes the whole message graph the argument to
#     every reward function for this reason
#   + a human reads it when debugging
#   - traces are large; they need to live outside the trial record, with the
#     outcome holding a reference
#
# Editable/readonly fence
#   + the single most important guardrail once an optimizer can write code:
#     the evaluator, reference outputs and held-out cases must be off limits,
#     and the diff should be checked before the candidate is ever run
#   + doubles as an audit log of what changed and why
#   - a fence is not a sandbox; a determined edit inside the allowed set can
#     still cheat (caching answers, weakening a check the harness calls), so
#     held-out evaluation stays the real defence
#
# What the agentic optimizer actually needs from hp
#   1. a readable problem      -- Params already serializes with names, ranges,
#                                 units and help text
#   2. history                 -- trials, outcomes, feedback, lineage
#   3. an action space         -- propose config, propose text, propose patch
#   4. guardrails              -- the fence above
#   Items 1 and 2 are incremental on what exists. 3 and 4 are the new work.
