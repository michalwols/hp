"""A process-wide registry of parametrized and tracked targets.

``parametrize`` turns a callable's signature into params and feeds them back in
at call time; ``track`` only records what a callable was called with. Both
register under a name, so the whole program's configuration surface can be
inspected from one place:

    @hp.parametrize
    def train(epochs: int = 10, lr: float = 2e-4):
      ...

    hp.params(train).lr = 1e-4
    train()

    hp.registry()          # {'train': Entry(...)}
"""

from __future__ import annotations

import functools
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from .callable import _signature
from .core import Params, schema


@dataclass
class Entry:
  """What the registry knows about one target."""

  name: str
  target: Any
  params: Params
  calls: list[dict[str, Any]] = field(default_factory=list)
  mode: str = 'parametrize'

  def __repr__(self) -> str:
    return f'Entry(name={self.name!r}, mode={self.mode!r}, calls={len(self.calls)})'


_REGISTRY: OrderedDict[str, Entry] = OrderedDict()


def registry() -> OrderedDict[str, Entry]:
  """Every registered target, keyed by name."""
  return _REGISTRY


def clear() -> None:
  """Forget every registration. Mostly useful in tests."""
  _REGISTRY.clear()


def _name_of(target: Any, name: str | None) -> str:
  if name:
    return name
  return getattr(target, '__qualname__', None) or getattr(target, '__name__', repr(target))


def entry(target: Any) -> Entry:
  """The registry entry for a target, by reference or by name."""
  if isinstance(target, Entry):
    return target
  if isinstance(target, str):
    if target not in _REGISTRY:
      raise KeyError(f'nothing registered under {target!r}')
    return _REGISTRY[target]
  found = getattr(target, 'hp', None)
  if isinstance(found, Entry):
    return found
  for candidate in _REGISTRY.values():
    if candidate.target is target or getattr(target, '__wrapped__', None) is candidate.target:
      return candidate
  raise KeyError(f'{target!r} is not registered; decorate it with hp.parametrize or hp.track')


def params(target: Any) -> Params:
  """The params for a registered target."""
  return entry(target).params


def calls(target: Any) -> list[dict[str, Any]]:
  """The recorded call arguments for a tracked target."""
  return entry(target).calls


def _register(target: Any, params_obj: Params, name: str | None, mode: str) -> Entry:
  record = Entry(name=_name_of(target, name), target=target, params=params_obj, mode=mode)
  _REGISTRY[record.name] = record
  return record


def parametrize(
  target: Any = None,
  *,
  name: str | None = None,
  params: Params | None = None,
  record: bool = False,
):
  """Build params from a callable's signature and supply them at call time.

  Arguments left out of a call are filled from the params; arguments passed
  explicitly win and are written back, so the params always reflect the last
  call. This replaces the old separate ``bind`` and ``wrap`` decorators.
  """

  def apply(target: Any):
    schema_params = params if params is not None else schema(target)()
    signature = _signature(target)
    record_entry = _register(target, schema_params, name, 'parametrize')

    @functools.wraps(target)
    def wrapped(*args: Any, **kwargs: Any):
      bound = signature.bind_partial(*args, **kwargs)
      for argument in schema_params._field_map:
        if argument in bound.arguments:
          schema_params[argument] = bound.arguments[argument]
        elif argument not in kwargs:
          kwargs[argument] = schema_params[argument]
      if record:
        record_entry.calls.append(dict(kwargs))
      return target(*args, **kwargs)

    wrapped.hp = record_entry
    wrapped.__signature__ = signature
    return wrapped

  return apply(target) if target is not None else apply


def track(target: Any = None, *, name: str | None = None):
  """Record what a target is called with, without changing what it does.

  The registry keeps the target's name, a reference to it, and the arguments
  of every call. Nothing is injected and the params are never mutated, so this
  is safe to wrap around anything.
  """

  def apply(target: Any):
    schema_params = schema(target)()
    signature = _signature(target)
    record_entry = _register(target, schema_params, name, 'track')

    @functools.wraps(target)
    def wrapped(*args: Any, **kwargs: Any):
      bound = signature.bind_partial(*args, **kwargs)
      record_entry.calls.append(dict(bound.arguments))
      return target(*args, **kwargs)

    wrapped.hp = record_entry
    wrapped.__signature__ = signature
    return wrapped

  return apply(target) if target is not None else apply


def surface() -> dict[str, Any]:
  """Everything hp knows about this process, in one place.

  Registered targets with their params and call counts, plus the current
  environment and command line.
  """
  from .core import to_dict
  from .views import cli, env

  return {
    'targets': {
      name: {
        'mode': record.mode,
        'params': to_dict(record.params),
        'calls': len(record.calls),
      }
      for name, record in _REGISTRY.items()
    },
    'env': env.to_dict(),
    'cli': cli.to_dict(),
  }
