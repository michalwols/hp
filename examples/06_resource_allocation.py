"""SKETCH -- a problem where the constraints must be symbolic.

Allocating a fixed GPU budget across training jobs is a mixed-integer program.
This is the case that decides whether the expression graph is worth building:
a solver cannot see inside a Python callable.
"""

import hp


class Allocation(hp.Params):
  # how many GPUs each job gets
  pretrain = hp.IntRange(0, 64, default=8)
  finetune = hp.IntRange(0, 64, default=8)
  evals = hp.IntRange(0, 16, default=2)

  # derived
  total = hp.Derived(lambda p: p.pretrain + p.finetune + p.evals)
  hourly_cost = hp.Derived(lambda p: p.total * 2.50)

  throughput = hp.Objective(maximize=True)

  limits = [
    total <= 64,                    # cluster size
    hourly_cost <= 100.0,           # budget
    evals >= 1,                     # always keep eval capacity
    finetune >= pretrain / 4,       # ratio between jobs
  ]


# With an expression graph, this compiles:
plan = hp.optimize(Allocation, method=hp.optim.milp())

# Without one, the same problem is a black box and you are back to sampling:
plan = hp.optimize(Allocation, feasible=lambda p: p.total <= 64 and ...,
                   method=hp.optim.tpe(trials=500))


# NOTES ----------------------------------------------------------------------
#
# Symbolic constraints
#   + a MILP solver returns the optimum with a proof, in milliseconds; TPE
#     samples 500 configs and returns something probably decent
#   + infeasible regions are excluded by construction rather than sampled and
#     rejected, which is most of the wasted budget in constrained search
#   + the constraints are readable, so an agent can be told *why* a config was
#     rejected instead of just that it scored badly
#   - needs affine expression support, comparison operators, and a translator
#     per backend (CVXPY, OR-Tools, Pyomo)
#   - only pays off when the objective is also expressible; if throughput can
#     only be measured by running the job, the solver cannot help and you are
#     sampling anyway
#
# The honest test for whether to build the graph: are there problems where the
# objective is cheap or known in closed form? Scheduling, allocation, batch
# sizing and data mixture ratios qualify. Model quality does not -- it has to
# be measured, so black-box search is the only option and the graph buys only
# feasibility filtering.
#
# That suggests building the graph for *constraints* first, where it pays off
# in every problem by pruning infeasible candidates before they are evaluated,
# and leaving symbolic objectives for later.
