from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Trial:
  number: int
  hp: Any
  value: float | tuple[float, ...] | None = None
  error: Exception | None = None


@dataclass
class Study:
  trials: list[Trial]
  direction: str = 'maximize'

  @property
  def best(self) -> Trial:
    valid = [trial for trial in self.trials if trial.value is not None and not isinstance(trial.value, tuple)]
    if not valid:
      raise RuntimeError('study has no completed scalar trials')
    return (max if self.direction == 'maximize' else min)(valid, key=lambda trial: trial.value)


def random_search(
  hp,
  objective: Callable[[Any], float],
  *,
  trials: int,
  seed: int | None = None,
  direction: str = 'maximize',
) -> Study:
  results: list[Trial] = []
  for number, candidate in enumerate(hp.samples(trials, seed=seed)):
    trial = Trial(number, candidate)
    try:
      trial.value = objective(candidate)
    except Exception as error:
      trial.error = error
    results.append(trial)
  return Study(results, direction)
