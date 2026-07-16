from __future__ import annotations

from typing import Any, Callable

from ..fields import Choice, Range


def optimize(
  hp,
  objective: Callable[[Any, Any], float],
  *,
  trials: int = 50,
  direction: str = 'maximize',
  study_name: str | None = None,
  storage: str | None = None,
):
  try:
    import optuna
  except ImportError as error:
    raise ImportError('Install hp[optuna] to use Optuna optimization') from error

  def run(trial):
    candidate = hp.fork()
    for path, field in hp.space().items():
      if isinstance(field, Choice):
        value = trial.suggest_categorical(path, list(field.choices))
      elif isinstance(field, Range):
        if field.integer:
          value = trial.suggest_int(path, int(field.low), int(field.high), step=int(field.step or 1), log=field.log)
        else:
          value = trial.suggest_float(path, float(field.low), float(field.high), step=field.step, log=field.log)
      else:
        continue
      candidate[path] = value
    return objective(candidate, trial)

  study = optuna.create_study(
    direction=direction,
    study_name=study_name,
    storage=storage,
    load_if_exists=bool(study_name and storage),
  )
  study.optimize(run, n_trials=trials)
  return study
