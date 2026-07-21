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


@dataclass
class Derived(Field):
  """A value computed from the other params, recomputed on every read.

      class Train(hp.Params):
        micro: int = 2
        accum: int = 8
        global_batch = hp.Derived(lambda p: p.micro * p.accum)

  Derived values serialize like any other field but cannot be assigned, and
  never appear in a search space -- searching them alongside their inputs is
  the classic way to waste a sweep on redundant dimensions.
  """

  fn: Callable[[Any], Any] | None = None
  cached: bool = False

  def __init__(self, fn: Callable[[Any], Any], **kwargs: Any):
    super().__init__(**kwargs)
    self.fn = fn
    self.cached = False
    if self.help is None:
      self.help = (fn.__doc__ or '').strip().split('\n')[0] or None

  def compute(self, owner: Any) -> Any:
    return self.fn(owner)

  # data descriptor: __set__ is defined, so it always wins over the instance
  # dict and the value cannot go stale
  def __get__(self, obj: Any, owner: type | None = None) -> Any:
    if obj is None:
      return self
    return self.compute(obj)

  def __set__(self, obj: Any, value: Any) -> None:
    raise AttributeError(f'{self.name!r} is derived and cannot be set')

  def dependencies(self, owner: Any) -> set[str]:
    """Which params this value read, discovered by running it once."""
    seen: set[str] = set()
    self.fn(_Recorder(owner, seen))
    return seen


@dataclass
class Computed(Derived):
  """A value computed once per params object and then fixed.

      created_at = hp.Computed(lambda p: datetime.now(timezone.utc))

  Use this for run ids, timestamps and hostnames -- things that should be
  stable for the life of the object rather than re-evaluated on every read.
  """

  def __init__(self, fn: Callable[[Any], Any], **kwargs: Any):
    super().__init__(fn, **kwargs)
    self.cached = True

  # non-data descriptor: the first read caches into the instance dict, which
  # then shadows it
  def __set__(self, obj: Any, value: Any) -> None:
    raise AttributeError(f'{self.name!r} is computed and cannot be set')

  def __get__(self, obj: Any, owner: type | None = None) -> Any:
    if obj is None:
      return self
    cache = obj.__dict__.get('_computed')
    if cache is None:
      cache = {}
      object.__setattr__(obj, '_computed', cache)
    if self.name not in cache:
      cache[self.name] = self.compute(obj)
    return cache[self.name]


class _Recorder:
  """Stands in for params while a derived function runs, noting what it read."""

  def __init__(self, target: Any, seen: set[str]):
    object.__setattr__(self, '_target', target)
    object.__setattr__(self, '_seen', seen)

  def __getattr__(self, name: str) -> Any:
    if not name.startswith('_'):
      self._seen.add(name)
    return getattr(self._target, name)

  def __getitem__(self, key: str) -> Any:
    self._seen.add(key)
    return self._target[key]


def derived(fn: Callable[[Any], Any]) -> Derived:
  """Decorator form: ``@hp.derived`` on a method of a Params subclass."""
  return Derived(fn)


def computed(fn: Callable[[Any], Any]) -> Computed:
  """Decorator form: ``@hp.computed`` on a method of a Params subclass."""
  return Computed(fn)
