from __future__ import annotations

from typing import Any

from .callable import fields_from_callable
from .core import Dynamic, Params, UnknownParam, schema
from .fields import (
  Choice,
  Field,
  IntRange,
  LogIntRange,
  LogRange,
  Range,
  ValidationError,
)

# legacy aliases; Params is the canonical name
HP = Params
HyperParams = Params


def wrap(target: Any = None, **kwargs: Any):
  def apply(target):
    params = schema(target)()
    wrapped = params.wrap(target, **kwargs)
    wrapped.hp = params
    return wrapped
  return apply(target) if target is not None else apply


__all__ = [
  'Params',
  'Dynamic',
  'UnknownParam',
  'Field',
  'Choice',
  'Range',
  'LogRange',
  'IntRange',
  'LogIntRange',
  'ValidationError',
  'wrap',
  'schema',
  'fields_from_callable',
  # legacy aliases
  'HP',
  'HyperParams',
]
