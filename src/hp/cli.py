from __future__ import annotations

import json
import sys
from typing import Any

from .core import coerce


def _value(text: str) -> Any:
  try:
    return json.loads(text)
  except json.JSONDecodeError:
    return text


def parse(hp, args: list[str] | None = None):
  args = list(sys.argv[1:] if args is None else args)
  index = 0
  while index < len(args):
    token = args[index]
    if not token.startswith('--'):
      raise ValueError(f'unexpected argument: {token}')
    token = token[2:]
    if '=' in token:
      path, raw = token.split('=', 1)
    elif token.startswith('no-') and hp.has(token[3:]):
      path, raw = token[3:], 'false'
    elif index + 1 < len(args) and not args[index + 1].startswith('--'):
      path, raw = token, args[index + 1]
      index += 1
    else:
      path, raw = token, 'true'
    value = _value(raw)
    if hp.has(path):
      node, name = hp._path(path)
      field = node.fields[name]
      value = coerce(value, field.type, field)
    hp[path] = value
    index += 1
  return hp
