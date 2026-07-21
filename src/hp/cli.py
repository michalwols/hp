from __future__ import annotations

import json
import sys
from typing import Any, get_args, get_origin

from .core import _nest, _unknown_param_warning
from .fields import Choice, Range


def _value(text: str) -> Any:
  try:
    return json.loads(text)
  except json.JSONDecodeError:
    return text


def _norm(path: str) -> str:
  # flags conventionally use dashes, field paths use underscores
  return path.replace('-', '_')


def _type_label(field) -> str:
  if isinstance(field, Choice):
    return '{' + ','.join(str(choice) for choice in field.choices) + '}'
  if isinstance(field, Range):
    return f'[{field.low}..{field.high}]'
  annotation = field.type
  if get_origin(annotation) is not None:
    args = [a for a in get_args(annotation) if a is not type(None)]
    if len(args) == 1:
      annotation = args[0]
    else:
      return '|'.join(getattr(a, '__name__', str(a)) for a in args)
  name = getattr(annotation, '__name__', None)
  return {'int': 'INT', 'float': 'FLOAT', 'str': 'TEXT', 'bool': ''}.get(name, name or 'VALUE')


def help_text(hp, program: str | None = None) -> str:
  """Render the params tree as command line help."""
  program = program or sys.argv[0].rsplit('/', 1)[-1]
  if not program or program == '-':
    program = 'program'
  lines = [f'Usage: {program} [OPTIONS]', '', 'Options:']

  rows: list[tuple[str, str]] = []
  for path, field in hp._field_paths().items():
    flag = '--' + path.replace('_', '-')
    if field.type is bool or isinstance(getattr(field, 'default', None), bool):
      flag = f'{flag} / --no-{path.replace("_", "-")}'
    else:
      label = _type_label(field)
      if label:
        flag = f'{flag} {label}'
    if field.alias:
      flag = f'{flag}, --{field.alias}'

    notes = []
    if field.help:
      notes.append(field.help)
    if field.required:
      notes.append('(required)')
    elif field.secret:
      notes.append('(secret)')
    else:
      try:
        notes.append(f'(default: {hp[path]!r})')
      except (KeyError, AttributeError):
        pass
    rows.append((flag, ' '.join(notes)))

  rows.append(('--help, -h', 'Show this message and exit.'))
  width = min(max((len(flag) for flag, _ in rows), default=0), 34)
  for flag, notes in rows:
    lines.append(f'  {flag.ljust(width)}  {notes}'.rstrip())
  return '\n'.join(lines)


def parse(hp, args: list[str] | None = None, positionals: list[str] | None = None):
  """Apply ``--flag value`` arguments to a params tree.

  Positional arguments are not configuration, so they are collected into
  ``positionals`` when given and otherwise ignored.
  """
  args = list(sys.argv[1:] if args is None else args)

  aliases = {
    field.alias: path
    for path, field in hp._field_paths().items()
    if field.alias
  }

  updates: dict[str, Any] = {}
  index = 0
  while index < len(args):
    token = args[index]
    if token in ('--help', '-h'):
      print(help_text(hp))
      raise SystemExit(0)
    if not token.startswith('--'):
      if positionals is not None:
        positionals.append(token)
      index += 1
      continue
    token = token[2:]
    if '=' in token:
      path, raw = token.split('=', 1)
    elif token.startswith('no-') and (hp._has(_norm(token[3:])) or _norm(token[3:]) in aliases):
      path, raw = token[3:], 'false'
    elif index + 1 < len(args) and not args[index + 1].startswith('--'):
      path, raw = token, args[index + 1]
      index += 1
    else:
      path, raw = token, 'true'
    path = aliases.get(_norm(path), _norm(path))
    updates[path] = _value(raw)
    index += 1

  # applied as one nested update so tagged unions resolve once, and the
  # result does not depend on the order flags happened to appear in
  hp._update(_nest(updates), source='cli')

  # warn only after applying: a flag can be valid for the variant it selects
  for path in updates:
    if hp._has(path):
      continue
    owner = hp
    if '.' in path:
      try:
        owner = hp[path.rsplit('.', 1)[0]]
      except (KeyError, AttributeError):
        owner = hp
    _unknown_param_warning(owner, path.rsplit('.', 1)[-1], stacklevel=4)
  return hp
