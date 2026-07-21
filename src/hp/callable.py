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
