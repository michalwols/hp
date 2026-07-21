"""Record (and optionally override) every call into a module.

    hp.instrument(torch.optim)                       # record constructor calls
    hp.instrument(mylib, select=['Adam'], override=True)

    with hp.instrumented(torch.optim):
      ...                                            # restored on exit

Functions are replaced on the module; classes keep their identity and have
``__init__`` wrapped instead, so ``isinstance`` and subclassing still work.
"""

from __future__ import annotations

import functools
import inspect
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Sequence

from .registry import _register, registry


class Instrumented:
  """Handle for one instrumented module, holding what to put back."""

  def __init__(self, module: Any):
    self.module = module
    self.functions: dict[str, Any] = {}
    self.initializers: dict[type, Any] = {}
    self.names: list[str] = []

  def __repr__(self) -> str:
    return f'Instrumented({self.module.__name__!r}, {len(self.names)} targets)'


_ACTIVE: dict[str, Instrumented] = {}


def _wrap(target: Any, entry: Any, override: bool) -> Callable:
  try:
    signature = inspect.signature(target)
  except (TypeError, ValueError):
    signature = None

  @functools.wraps(target)
  def wrapped(*args: Any, **kwargs: Any):
    if signature is not None:
      try:
        bound = signature.bind_partial(*args, **kwargs)
        entry.calls.append(
          {k: v for k, v in bound.arguments.items() if k not in {'self', 'cls'}},
        )
        if override:
          for name in entry.params._field_map:
            if name not in bound.arguments and entry.params._sources.get(name) != 'default':
              kwargs[name] = entry.params[name]
      except TypeError:
        entry.calls.append({'args': args, 'kwargs': dict(kwargs)})
    else:
      entry.calls.append({'args': args, 'kwargs': dict(kwargs)})
    return target(*args, **kwargs)

  return wrapped


def _should_include(
  name: str,
  value: Any,
  module: Any,
  select: Sequence[str] | None,
  exclude: Sequence[str] | None,
) -> bool:
  if name.startswith('_'):
    return False
  if select is not None:
    return name in select
  if exclude and name in exclude:
    return False
  if not (inspect.isfunction(value) or inspect.isclass(value)):
    return False
  # skip re-exports so a module is not credited with another's API
  origin = getattr(value, '__module__', None)
  return origin is None or origin == module.__name__ or origin.startswith(f'{module.__name__}.')


def instrument(
  module: Any,
  *,
  select: Sequence[str] | None = None,
  exclude: Sequence[str] | None = None,
  override: bool = False,
  prefix: str | None = None,
) -> Instrumented:
  """Wrap a module's callables so their arguments are recorded.

  Args:
    module: the module to patch
    select: only these names (otherwise every public function and class
      defined in the module)
    exclude: names to skip
    override: also let registered params supply arguments the caller omitted,
      for values that were explicitly set
    prefix: registry name prefix, defaulting to the module's name
  """
  if module.__name__ in _ACTIVE:
    raise RuntimeError(f'{module.__name__} is already instrumented; restore it first')

  handle = Instrumented(module)
  label = module.__name__ if prefix is None else prefix

  for name in dir(module):
    value = getattr(module, name, None)
    if not _should_include(name, value, module, select, exclude):
      continue

    from .core import schema

    try:
      params = schema(value)()
    except Exception:
      continue

    entry = _register(value, params, f'{label}.{name}', 'instrument')
    handle.names.append(entry.name)

    if inspect.isclass(value):
      original = value.__init__
      handle.initializers[value] = original
      value.__init__ = _wrap(original, entry, override)
    else:
      handle.functions[name] = value
      setattr(module, name, _wrap(value, entry, override))

  _ACTIVE[module.__name__] = handle
  return handle


def restore(module: Any) -> None:
  """Undo :func:`instrument`, putting the original callables back."""
  handle = _ACTIVE.pop(getattr(module, '__name__', module), None)
  if handle is None:
    return
  for name, original in handle.functions.items():
    setattr(handle.module, name, original)
  for cls, original in handle.initializers.items():
    cls.__init__ = original
  for name in handle.names:
    registry().pop(name, None)


@contextmanager
def instrumented(module: Any, **kwargs: Any) -> Iterator[Instrumented]:
  """Instrument a module for the duration of a block."""
  handle = instrument(module, **kwargs)
  try:
    yield handle
  finally:
    restore(module)
