# Examples

Two kinds of file live here.

**Runnable** examples work against hp as it is today. Run any of them directly:

```bash
uv run --with pytest python examples/01_ml_training.py
```

**Sketches** are marked `SKETCH` in their docstring. They explore APIs that do
not exist yet, so they are design documents that happen to be written in Python
rather than working code. Each one ends with a `NOTES` block weighing the
approach against the alternatives.

| file | status | what it explores |
|---|---|---|
| `01_ml_training.py` | runnable | layered config, provenance, reproducible run ids |
| `02_derived_values.py` | runnable | `Derived` vs `Computed` vs `@property`, and why derived values must not be searched |
| `03_objectives_constraints.py` | sketch | three ways to attach objectives and constraints to params |
| `04_verifiers_feedback.py` | sketch | verifier lists, weights, and the shape of an `Outcome` |
| `05_paired_preference.py` | sketch | Elo / Bradley-Terry feedback, where there is no absolute score |
| `06_resource_allocation.py` | sketch | symbolic constraints an OR solver can actually read |
| `07_agent_optimization.py` | sketch | text components, traces, and an optimizer that edits a harness |
| `08_priors_and_references.py` | sketch | mining cases, oracles, validators and priors out of logs, test suites and the current system |

## The question these are exploring

Everything hangs off `hp.Params`. The open question is *how much structure*
lives in the class body versus in the calling code:

- **Field roles** — `hp.Objective()`, `hp.Constraint()` as field types. Declarative, serializable, no new machinery.
- **Expression graph** — `latency <= 500` in the class body, building a tree. More power, more machinery.
- **Functional** — the objective returns an `Outcome`. Least magic, least introspectable.

The comparisons in each file argue for a specific mix rather than a winner.

## Where this points

Reading the trade-offs across these files, the same split keeps appearing:

- **Declaration belongs on `hp.Params`.** Field roles (`Objective`, `Constraint`,
  `Metric`, `Derived`, `Evolve`) inherit serialization, help text and provenance,
  and give an agentic optimizer something it can read before running anything.
- **Reporting belongs in an `Outcome`.** Score, metrics, constraints, feedback,
  trace and steps, where each optimizer reads only the fields it understands.
  This is what lets one objective serve TPE, NSGA-II, GEPA and ES at once.
- **The expression graph earns its place on constraints first.** Symbolic
  constraints prune infeasible candidates before they are evaluated, which pays
  off in every problem. Symbolic *objectives* only pay off when the objective is
  cheap or closed-form, which excludes most ML.
- **A reference implementation is a constraint, not an objective.** Pinning
  correctness to parity with what already ships turns an unmeasurable goal into
  a measurable one — but it caps quality at parity, so it is a safety rail
  rather than the thing being improved.
- **Paired feedback needs its own channel**, because reducing it to a win rate
  against a baseline throws away most of the signal — and because it breaks the
  assumption that a finished trial has a final score.

Suggested build order: `Outcome` first (everything depends on its shape), then
field roles, then constraint expressions, then trials and history, then the
agentic layer.
