from __future__ import annotations

from typing import Any

from .callable import fields_from_callable
from .core import HP
from .fields import (
  Choice,
  Field,
  IntRange,
  LogIntRange,
  LogRange,
  Range,
  ValidationError,
)

HyperParams = HP


def wrap(target: Any = None, **kwargs: Any):
  def apply(target):
    schema = HP.schema(target)
    params = schema()
    wrapped = params.wrap(target, **kwargs)
    wrapped.hp = params
    return wrapped
  return apply(target) if target is not None else apply


def schema(target: Any, **kwargs: Any):
  return HP.schema(target, **kwargs)


__all__ = [
  'HP',
  'HyperParams',
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
]
