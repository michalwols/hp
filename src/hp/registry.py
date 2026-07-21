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
import types
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
      from .context import resolve

      schema_params = resolve(record_entry.name, record_entry.params)
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


class ParamsAPI:
  """The registry, decorators and instrumentation, behind explicit names.

  Calling it always means the same thing -- *give me the params for this* --
  and everything else is a named method:

  ============================  ================================================
  ``hp.params(target)``         the params of a registered target or an object
  ``hp.params()``               the registry, name -> Entry
  ``hp.params['name']``         one Entry
  ``hp.params.wrap(fn)``        decorate: supply params, record explicit args
  ``hp.params.track(fn)``       decorate: record only, change nothing
  ``hp.params.instrument(mod)`` wrap a whole module
  ``hp.params.history()``       what everything was called with
  ``hp.params.collect()``       one tree of the whole config surface
  ``hp.params.inject(cfg)``     write config out, to the environment or a scope
  ============================  ================================================
  """

  def __call__(self, target: Any = None, /) -> Any:
    if target is None:
      return _REGISTRY
    if isinstance(target, Params):
      return target
    if isinstance(target, str):
      return entry(target).params
    try:
      return entry(target).params
    except KeyError:
      pass
    from .adapt import from_object

    return from_object(target)

  # -- registry ------------------------------------------------------------

  def __getitem__(self, name: str) -> Entry:
    return entry(name)

  def __contains__(self, target: Any) -> bool:
    try:
      entry(target)
    except KeyError:
      return False
    return True

  def __iter__(self):
    return iter(_REGISTRY)

  def __len__(self) -> int:
    return len(_REGISTRY)

  @property
  def registry(self) -> OrderedDict[str, Entry]:
    return _REGISTRY

  def entry(self, target: Any) -> Entry:
    return entry(target)

  def clear(self) -> None:
    _REGISTRY.clear()

  # -- decorators ----------------------------------------------------------

  def wrap(self, target: Any = None, **kwargs: Any) -> Any:
    """Supply params to a callable and record what it was called with."""
    return parametrize(target, **kwargs)

  def track(self, target: Any = None, **kwargs: Any) -> Any:
    """Record what a callable was called with, changing nothing."""
    return track(target, **kwargs)

  # -- instrumentation -----------------------------------------------------

  def instrument(self, module: Any, **kwargs: Any):
    from .instrument import instrument

    return instrument(module, **kwargs)

  def instrumented(self, module: Any, **kwargs: Any):
    from .instrument import instrumented

    return instrumented(module, **kwargs)

  def restore(self, module: Any) -> None:
    from .instrument import restore

    restore(module)

  # -- inspection ----------------------------------------------------------

  def history(self, target: Any = None) -> Any:
    """Recorded calls: for one target, or every target keyed by name."""
    if target is not None:
      return entry(target).calls
    return {name: record.calls for name, record in _REGISTRY.items() if record.calls}

  def collect(self, *sources: Any, env: bool = False, cli: bool = False) -> Params:
    """The whole config surface as one tree.

    Every registered target contributes a node under its own name, so
    ``hp.params.collect().train.lr`` reaches a parametrized ``train``. Extra
    sources are layered on top, and ``env``/``cli`` fold those in too.
    """
    from .core import Dynamic, _nest, flatten

    values: dict[str, Any] = {}
    for name, record in _REGISTRY.items():
      for path, value in flatten(record.params).items():
        values[f'{name}.{path}'] = value

    config = Dynamic()
    config._update(_nest(values), source='registry')

    layers = list(sources)
    if env:
      from .views import env as env_view

      layers.append(env_view)
    if cli:
      from .views import cli as cli_view

      layers.append(cli_view)
    if layers:
      config._apply_layers(*layers)
    return config

  def inject(
    self,
    config: Params,
    into: Any = None,
    *,
    prefix: str = '',
    uppercase: bool | None = None,
  ) -> Params:
    """Write config values outward.

    With no target the values go into the environment, which is how you hand
    a resolved config to a subprocess. Pass a mapping (``globals()``, a dict)
    to write there instead.
    """
    import os

    from .core import flatten
    from .views import Env

    values = flatten(config)
    if into is None or isinstance(into, Env):
      upper = True if uppercase is None else uppercase
      for path, value in values.items():
        key = f'{prefix}{path.replace(".", "__")}'
        os.environ[key.upper() if upper else key] = str(value)
    else:
      upper = False if uppercase is None else uppercase
      for path, value in values.items():
        key = f'{prefix}{path.replace(".", "_")}'
        into[key.upper() if upper else key] = value
    return config

  def __repr__(self) -> str:
    return f'<hp.params: {len(_REGISTRY)} registered>'


params = ParamsAPI()
