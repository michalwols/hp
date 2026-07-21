"""Bridges to other config shapes, matched structurally rather than by import.

Dataclasses, attrs, pydantic, NamedTuples, argparse namespaces and plain
mappings are all recognized by the attributes they expose, so none of those
libraries need to be installed for the rest to work.
"""

from __future__ import annotations

import inspect
from typing import Any, get_type_hints

from .core import Params, Dynamic, schema, to_dict
from .fields import Choice, Field, MISSING, Range


def _asdict(obj: Any) -> dict[str, Any] | None:
  """Values from a config-shaped object, whatever library produced it."""
  if obj is None:
    return None
  if isinstance(obj, Params):
    return to_dict(obj)
  if isinstance(obj, dict):
    return dict(obj)

  # dataclass instance
  if hasattr(type(obj), '__dataclass_fields__') and not isinstance(obj, type):
    import dataclasses

    return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}

  # pydantic v2 then v1
  for method in ('model_dump', 'dict'):
    fn = getattr(obj, method, None)
    if callable(fn) and hasattr(type(obj), 'model_fields' if method == 'model_dump' else '__fields__'):
      try:
        return dict(fn())
      except TypeError:
        pass

  # attrs
  if hasattr(type(obj), '__attrs_attrs__'):
    return {a.name: getattr(obj, a.name) for a in type(obj).__attrs_attrs__}

  # NamedTuple
  if hasattr(obj, '_asdict'):
    return dict(obj._asdict())

  # argparse.Namespace and anything else with a plain __dict__
  if hasattr(obj, '__dict__') and not inspect.isclass(obj):
    return {k: v for k, v in vars(obj).items() if not k.startswith('_')}

  return None


def from_object(obj: Any, *, name: str | None = None) -> Params:
  """Params holding the values of any config-shaped object.

  Works on dataclass, attrs and pydantic instances, NamedTuples,
  ``argparse.Namespace``, mappings, and existing params.
  """
  values = _asdict(obj)
  if values is None:
    raise TypeError(f'cannot read config values from {type(obj).__name__}')

  target = type(obj) if not isinstance(obj, dict) else None
  if target is not None and (
    hasattr(target, '__dataclass_fields__')
    or hasattr(target, '__attrs_attrs__')
    or hasattr(target, 'model_fields')
  ):
    try:
      params = schema(target, name=name)()
      params._update(values)
      return params
    except Exception:
      pass

  node = Dynamic(**values)
  return node


def construct(params: Params, target: Any, **overrides: Any) -> Any:
  """Build ``target`` from params, passing only what its signature accepts."""
  values = {**to_dict(params), **overrides}
  try:
    signature = inspect.signature(target)
    accepted = {
      name
      for name, parameter in signature.parameters.items()
      if name not in {'self', 'cls'}
      and parameter.kind
      not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    }
    takes_kwargs = any(
      p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )
    if not takes_kwargs:
      values = {k: v for k, v in values.items() if k in accepted}
  except (TypeError, ValueError):
    pass
  return target(**values)


def _named_tuple_fields(target: Any) -> dict[str, Field] | None:
  names = getattr(target, '_fields', None)
  if names is None or not isinstance(target, type):
    return None
  defaults = getattr(target, '_field_defaults', {})
  try:
    hints = get_type_hints(target)
  except Exception:
    hints = {}
  fields: dict[str, Field] = {}
  for name in names:
    field = Field(
      default=defaults.get(name, MISSING),
      type=hints.get(name),
      required=name not in defaults,
    )
    field.name = name
    fields[name] = field
  return fields


def from_argparse(parser: Any, *, name: str | None = None) -> Params:
  """Params mirroring an ``ArgumentParser``'s options."""
  namespace = parser.parse_args([]) if hasattr(parser, 'parse_args') else parser
  node = Dynamic()
  for key, value in vars(namespace).items():
    if not key.startswith('_'):
      node[key] = value
  return node


def to_argparse(params: Params, parser: Any = None, **kwargs: Any):
  """An ``ArgumentParser`` populated from the params tree.

  Useful for slotting hp into a codebase that already parses with argparse.
  """
  import argparse

  from .core import field_paths

  parser = parser or argparse.ArgumentParser(**kwargs)
  for path, field in field_paths(params).items():
    flag = '--' + path.replace('_', '-')
    names = [flag] if not field.alias else [flag, f'--{field.alias}']
    current = params[path]

    if isinstance(current, bool) or field.type is bool:
      group = parser.add_mutually_exclusive_group()
      group.add_argument(*names, dest=path, action='store_true', default=current,
                         help=field.help)
      group.add_argument(f'--no-{path.replace("_", "-")}', dest=path,
                         action='store_false', help=argparse.SUPPRESS)
      continue

    options: dict[str, Any] = {'dest': path, 'default': current, 'help': field.help}
    if isinstance(field, Choice):
      options['choices'] = list(field.choices)
    if field.type in (int, float, str):
      options['type'] = field.type
    elif isinstance(field, Range):
      options['type'] = int if field.integer else float
    parser.add_argument(*names, **options)
  return parser
