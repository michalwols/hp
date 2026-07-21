from __future__ import annotations

import functools
import inspect
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, Annotated, get_args, get_origin, get_type_hints

from .fields import Field, MISSING


def _signature(target: Any) -> inspect.Signature:
  if inspect.isclass(target):
    return inspect.signature(target.__init__).replace(
      parameters=[
        p for name, p in inspect.signature(target.__init__).parameters.items()
        if name != 'self'
      ]
    )
  return inspect.signature(target)


def fields_from_callable(target: Any) -> OrderedDict[str, Field]:
  signature = _signature(target)
  try:
    hints = get_type_hints(target, include_extras=True)
  except Exception:
    hints = {}
  fields: OrderedDict[str, Field] = OrderedDict()
  for name, parameter in signature.parameters.items():
    if name in {'self', 'cls'} or parameter.kind in {
      inspect.Parameter.VAR_POSITIONAL,
      inspect.Parameter.VAR_KEYWORD,
    }:
      continue
    annotation = hints.get(name, parameter.annotation)
    metadata: Field | None = None
    if get_origin(annotation) is Annotated:
      base, *extras = get_args(annotation)
      annotation = base
      metadata = next((x for x in extras if isinstance(x, Field)), None)
    default = parameter.default if parameter.default is not inspect.Parameter.empty else MISSING
    field = metadata.clone() if metadata else Field()
    field.name = name
    field.type = None if annotation is inspect.Parameter.empty else annotation
    if field.default is MISSING:
      field.default = default
    field.required = default is MISSING
    fields[name] = field
  return fields


def decorate(
  hp: Any,
  target: Any = None,
  *,
  mode: str,
  mapping: Mapping[str, str] | None = None,
  prefix: str | None = None,
  define: bool | None = None,
):
  def apply(target: Any):
    signature = _signature(target)
    explicit = dict(mapping or {})
    parameter_names = {
      name for name, parameter in signature.parameters.items()
      if name not in {'self', 'cls'}
      and parameter.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    }
    paths = {
      name: explicit.get(name, f'{prefix}.{name}' if prefix else name)
      for name in parameter_names
      if name in explicit or hp.has(f'{prefix}.{name}' if prefix else name)
    }
    should_define = mode in {'watch', 'wrap'} if define is None else define
    if should_define:
      hp.define(target, mapping={name: f'{prefix}.{path}' if prefix and '.' not in path else path for name, path in explicit.items()} if explicit else None)
      paths = {
        name: explicit.get(name, f'{prefix}.{name}' if prefix else name)
        for name in parameter_names
        if hp.has(explicit.get(name, f'{prefix}.{name}' if prefix else name))
      }

    @functools.wraps(target)
    def wrapped(*args: Any, **kwargs: Any):
      bound = signature.bind_partial(*args, **kwargs)
      for arg_name, path in paths.items():
        if arg_name in bound.arguments:
          if mode in {'watch', 'wrap'}:
            hp[path] = bound.arguments[arg_name]
        elif mode in {'bind', 'wrap'}:
          kwargs[arg_name] = hp[path]
      return target(*args, **kwargs)

    wrapped.hp = hp
    wrapped.__signature__ = signature
    return wrapped

  return apply(target) if target is not None else apply
