"""Scoped configuration, isolated per thread and per async task.

Overrides are stored in a :class:`~contextvars.ContextVar` and applied to a
*fork* of the target's params, so nothing mutates shared state. Parallel
trials, threads and asyncio tasks each see their own values:

    with hp.override(train, lr=1e-4):
      train()          # sees lr=1e-4
    train()            # back to the registered value
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from .core import Params, fork

# active params for user code, e.g. inside a training function
_ACTIVE: ContextVar[Params | None] = ContextVar('hp_active', default=None)

# per-target overrides, keyed by registry name
_OVERRIDES: ContextVar[dict[str, Params]] = ContextVar('hp_overrides', default={})


def active() -> Params | None:
  """The params scoped by the innermost :func:`scope`, if any."""
  return _ACTIVE.get()


def overrides() -> dict[str, Params]:
  """Per-target overrides currently in effect."""
  return _OVERRIDES.get()


def resolve(name: str, default: Params) -> Params:
  """The params a registered target should use right now."""
  return _OVERRIDES.get().get(name, default)


@contextmanager
def scope(params: Params) -> Iterator[Params]:
  """Make ``params`` the active configuration for the duration of a block."""
  token = _ACTIVE.set(params)
  try:
    yield params
  finally:
    _ACTIVE.reset(token)


@contextmanager
def override(target: Any = None, **values: Any) -> Iterator[Params]:
  """Temporarily change values, without touching the original params.

  ``target`` may be a registered target (by reference or name), a params
  object, or omitted to override the active scope. The block sees a fork, so
  concurrent tasks never observe each other's overrides.
  """
  from .registry import Entry, _REGISTRY, entry

  name: str | None = None
  base: Params | None = None

  if target is None:
    base = _ACTIVE.get()
    if base is None:
      raise RuntimeError('hp.override() with no target requires an active hp.scope()')
  elif isinstance(target, Params):
    base = target
  else:
    record: Entry = entry(target)
    name, base = record.name, resolve(record.name, record.params)

  forked = fork(base, **values)

  if name is None:
    with scope(forked):
      yield forked
    return

  current = dict(_OVERRIDES.get())
  current[name] = forked
  token = _OVERRIDES.set(current)
  try:
    with scope(forked):
      yield forked
  finally:
    _OVERRIDES.reset(token)
