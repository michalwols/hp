from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

MISSING = object()


class ValidationError(ValueError):
  pass


@dataclass
class Field:
  default: Any = MISSING
  type: Any = None
  help: str | None = None
  alias: str | None = None
  env: str | None = None
  required: bool = False
  secret: bool = False
  factory: Callable[[], Any] | None = None
  # Gates whether this field participates in a search space. Receives the
  # root params, so conditions can reference values elsewhere in the tree:
  #   group_size = Choice((4, 8), when=lambda root: root.rl.method == 'grpo')
  when: Callable[[Any], bool] | None = None

  name: str | None = None

  @property
  def searchable(self) -> bool:
    return False

  def is_active(self, root: Any = None) -> bool:
    if self.when is None:
      return True
    return bool(self.when(root))

  def clone(self) -> 'Field':
    return copy.deepcopy(self)

  def make_default(self) -> Any:
    if self.factory is not None:
      return self.factory()
    if self.default is MISSING:
      if self.required:
        return MISSING
      return None
    return copy.deepcopy(self.default)

  def validate(self, value: Any) -> None:
    if value is MISSING:
      raise ValidationError(f'{self.name} is required')

  def sample(self, rng: random.Random) -> Any:
    return self.make_default()

  def grid(self) -> Iterable[Any]:
    yield self.make_default()


@dataclass
class Choice(Field):
  choices: Sequence[Any] = ()
  ordered: bool = False

  def __init__(
    self,
    choices: Sequence[Any],
    *,
    default: Any = MISSING,
    ordered: bool = False,
    **kwargs: Any,
  ):
    choices = tuple(choices)
    if not choices:
      raise ValueError('Choice requires at least one value')
    super().__init__(default=choices[0] if default is MISSING else default, **kwargs)
    self.choices = choices
    self.ordered = ordered

  @property
  def searchable(self) -> bool:
    return True

  def validate(self, value: Any) -> None:
    super().validate(value)
    if value not in self.choices:
      raise ValidationError(f'{self.name} must be one of {self.choices!r}, got {value!r}')

  def sample(self, rng: random.Random) -> Any:
    return copy.deepcopy(rng.choice(self.choices))

  def grid(self) -> Iterable[Any]:
    yield from copy.deepcopy(self.choices)


@dataclass
class Range(Field):
  low: int | float = 0
  high: int | float = 1
  step: int | float | None = None
  log: bool = False
  integer: bool = False

  def __init__(
    self,
    low: int | float,
    high: int | float,
    *,
    default: int | float | object = MISSING,
    step: int | float | None = None,
    log: bool = False,
    integer: bool | None = None,
    **kwargs: Any,
  ):
    low, high = sorted((low, high))
    integer = isinstance(low, int) and isinstance(high, int) if integer is None else integer
    if log and low <= 0:
      raise ValueError('log ranges require low > 0')
    super().__init__(default=low if default is MISSING else default, **kwargs)
    self.low = low
    self.high = high
    self.step = step
    self.log = log
    self.integer = integer

  @property
  def searchable(self) -> bool:
    return True

  def validate(self, value: Any) -> None:
    super().validate(value)
    if not self.low <= value <= self.high:
      raise ValidationError(f'{self.name} must be in [{self.low}, {self.high}], got {value}')
    if self.integer and not isinstance(value, int):
      raise ValidationError(f'{self.name} must be an int')

  def sample(self, rng: random.Random) -> int | float:
    if self.log:
      value = math.exp(rng.uniform(math.log(self.low), math.log(self.high)))
    else:
      value = rng.uniform(self.low, self.high)
    if self.integer:
      value = round(value)
    if self.step:
      value = self.low + round((value - self.low) / self.step) * self.step
    return int(value) if self.integer else float(value)

  def grid(self) -> Iterable[int | float]:
    if self.step is None:
      yield self.make_default()
      return
    value = self.low
    epsilon = abs(float(self.step)) * 1e-9
    while value <= self.high + epsilon:
      yield int(value) if self.integer else float(value)
      value += self.step


def LogRange(low: int | float, high: int | float, **kwargs: Any) -> Range:
  return Range(low, high, log=True, **kwargs)


def IntRange(low: int, high: int, **kwargs: Any) -> Range:
  return Range(low, high, integer=True, **kwargs)


def LogIntRange(low: int, high: int, **kwargs: Any) -> Range:
  return Range(low, high, integer=True, log=True, **kwargs)


@dataclass
class Evolve(Field):
  """A text field an outer-loop optimizer is allowed to rewrite.

  Ordinary string fields stay ordinary; only fields marked this way are
  exposed as candidate components to text optimizers like GEPA.

      system_prompt: str = Evolve(
        'You are a data agent...',
        description='Main behavioral instructions',
      )
  """

  group: str | None = None

  def __init__(
    self,
    default: str = '',
    *,
    description: str | None = None,
    group: str | None = None,
    **kwargs: Any,
  ):
    if description is not None:
      kwargs.setdefault('help', description)
    super().__init__(default=default, type=str, **kwargs)
    self.group = group

  @property
  def description(self) -> str | None:
    return self.help
