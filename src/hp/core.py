from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import types
from abc import ABCMeta
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping, MutableMapping
from itertools import product
from pathlib import Path
from typing import Any, ClassVar, Literal, Union, get_args, get_origin, get_type_hints

from .fields import Choice, Field, MISSING, Range, ValidationError


def _is_params_type(value: Any) -> bool:
  return isinstance(value, type) and issubclass(value, Params)


def _literal_value(annotation: Any) -> Any:
  if get_origin(annotation) is Literal:
    args = get_args(annotation)
    if len(args) == 1:
      return args[0]
  return MISSING


def _params_union_variants(annotation: Any) -> tuple[str, dict[Any, type['Params']]] | None:
  origin = get_origin(annotation)
  if origin not in (Union, types.UnionType):
    return None
  members = [x for x in get_args(annotation) if x is not type(None)]
  if not members or not all(_is_params_type(x) for x in members):
    return None
  common = set(members[0].__fields__)
  for member in members[1:]:
    common &= set(member.__fields__)
  for name in common:
    variants: dict[Any, type[Params]] = {}
    valid = True
    for member in members:
      tag = _literal_value(member.__fields__[name].type)
      if tag is MISSING or tag in variants:
        valid = False
        break
      variants[tag] = member
    if valid:
      return name, variants
  return None


def _matches_type(value: Any, annotation: Any) -> bool:
  if annotation in (None, Any, inspect.Signature.empty):
    return True
  if value is None:
    origin = get_origin(annotation)
    return origin in (Union, types.UnionType) and type(None) in get_args(annotation)
  origin = get_origin(annotation)
  args = get_args(annotation)
  if origin is Literal:
    return value in args
  if origin in (Union, types.UnionType):
    return any(_matches_type(value, arg) for arg in args)
  if origin is list:
    return isinstance(value, list) and (not args or all(_matches_type(v, args[0]) for v in value))
  if origin is tuple:
    return isinstance(value, tuple)
  if origin is dict:
    return isinstance(value, dict)
  try:
    return isinstance(value, annotation)
  except TypeError:
    return True


def _coerce_scalar(value: Any, annotation: Any) -> Any:
  if annotation in (None, Any, inspect.Signature.empty):
    return value
  if value is None:
    return None
  origin = get_origin(annotation)
  args = get_args(annotation)
  if origin is Literal:
    for arg in args:
      if str(value) == str(arg):
        return arg
    return value
  if origin in (Union, types.UnionType):
    for arg in args:
      if arg is type(None):
        continue
      try:
        return _coerce_scalar(value, arg)
      except (TypeError, ValueError):
        pass
    return value
  if annotation is bool and isinstance(value, str):
    lowered = value.lower()
    if lowered in {'1', 'true', 'yes', 'on'}:
      return True
    if lowered in {'0', 'false', 'no', 'off'}:
      return False
  if annotation in (str, int, float, bool, Path) and not isinstance(value, annotation):
    return annotation(value)
  return value


def coerce(value: Any, annotation: Any, field: Field | None = None) -> Any:
  if value is MISSING:
    return value
  if _is_params_type(annotation):
    if isinstance(value, annotation):
      return value
    if isinstance(value, Mapping):
      return annotation(**value)
  variants = _params_union_variants(annotation)
  if variants and isinstance(value, Mapping):
    discriminator, mapping = variants
    tag = value.get(discriminator)
    if tag not in mapping:
      raise ValidationError(
        f'unknown {discriminator} {tag!r}; expected one of {tuple(mapping)!r}'
      )
    return mapping[tag](**value)
  origin = get_origin(annotation)
  args = get_args(annotation)
  if origin is list and isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
    item_type = args[0] if args else Any
    return [coerce(item, item_type) for item in value]
  if origin is tuple and isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
    values = list(value)
    if len(args) == 2 and args[1] is Ellipsis:
      return tuple(coerce(item, args[0]) for item in values)
    return tuple(coerce(item, arg) for item, arg in zip(values, args))
  if isinstance(value, Mapping) and annotation in (None, Any, dict):
    node = Dynamic()
    node.update(value, strict=False)
    return node
  return _coerce_scalar(value, annotation)


class ParamsMeta(ABCMeta):
  def __new__(mcls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs: Any):
    dynamic = kwargs.pop('dynamic', namespace.get('__dynamic__', False))
    fields: OrderedDict[str, Field] = OrderedDict()
    for base in bases:
      fields.update((k, v.clone()) for k, v in getattr(base, '__fields__', {}).items())

    annotations = namespace.get('__annotations__', {})
    reserved = set()
    for base in bases:
      reserved.update(dir(base))

    candidates = {
      key: value
      for key, value in namespace.items()
      if not key.startswith('_')
      and key not in reserved
      and not callable(value)
      and not isinstance(value, (classmethod, staticmethod, property))
    }

    for field_name, annotation in annotations.items():
      if field_name.startswith('_'):
        continue
      if get_origin(annotation) is ClassVar:
        continue
      raw = namespace.get(field_name, MISSING)
      if isinstance(raw, Field):
        field = raw.clone()
        if field.type is None:
          field.type = annotation
      else:
        required = raw is MISSING and not _is_params_type(annotation)
        default = annotation() if raw is MISSING and _is_params_type(annotation) else raw
        field = Field(default=default, type=annotation, required=required)
      field.name = field_name
      fields[field_name] = field
      namespace.pop(field_name, None)

    for field_name, raw in candidates.items():
      if field_name in annotations:
        continue
      if isinstance(raw, Field):
        field = raw.clone()
      else:
        field = Field(default=raw, type=type(raw))
      field.name = field_name
      fields[field_name] = field
      namespace.pop(field_name, None)

    namespace['__fields__'] = fields
    namespace['__dynamic__'] = dynamic
    return super().__new__(mcls, name, bases, namespace)


class Params(MutableMapping[str, Any], metaclass=ParamsMeta):
  __fields__: ClassVar[OrderedDict[str, Field]]
  __dynamic__: ClassVar[bool] = False

  def __init__(self, *args: Mapping[str, Any], **values: Any):
    object.__setattr__(self, '_fields', OrderedDict())
    object.__setattr__(self, '_change_callbacks', [])
    object.__setattr__(self, '_frozen', False)
    for name, field in self.__class__.__fields__.items():
      value = field.make_default()
      if value is not MISSING:
        value = coerce(value, field.type, field)
      object.__setattr__(self, name, value)
    for mapping in args:
      self.update(mapping, strict=True)
    self.update(values, strict=True)

  @property
  def fields(self) -> OrderedDict[str, Field]:
    fields = OrderedDict((k, v) for k, v in self.__class__.__fields__.items())
    fields.update(self._fields)
    return fields

  def __getattr__(self, name: str) -> Any:
    if name.startswith('_') or not self.__dynamic__:
      raise AttributeError(name)
    child = Dynamic()
    field = Field(default=child, type=Params, name=name)
    self._fields[name] = field
    object.__setattr__(self, name, child)
    return child

  def __setattr__(self, name: str, value: Any) -> None:
    if name.startswith('_'):
      object.__setattr__(self, name, value)
      return
    if self._frozen:
      raise TypeError(f'{type(self).__name__} is frozen')
    field = self.fields.get(name)
    if field is None:
      if not self.__dynamic__:
        object.__setattr__(self, name, value)
        return
      if isinstance(value, Mapping) and not isinstance(value, Params):
        node = Dynamic()
        node.update(value, strict=False)
        value = node
      field = Field(default=copy.deepcopy(value), type=type(value), name=name)
      self._fields[name] = field
    else:
      value = coerce(value, field.type, field)
    old = getattr(self, name, MISSING)
    for callback in self._change_callbacks:
      callback(self, name, old, value)
    object.__setattr__(self, name, value)

  def __getitem__(self, key: str) -> Any:
    node, name = self._path(key)
    try:
      return getattr(node, name)
    except AttributeError:
      raise KeyError(key) from None

  def __setitem__(self, key: str, value: Any) -> None:
    node, name = self._path(key, create=True)
    setattr(node, name, value)

  def __delitem__(self, key: str) -> None:
    node, name = self._path(key)
    if name in node.__class__.__fields__:
      raise TypeError(f'cannot delete declared field {key!r}')
    node._fields.pop(name)
    object.__delattr__(node, name)

  def __iter__(self) -> Iterator[str]:
    return iter(self.fields)

  def __len__(self) -> int:
    return len(self.fields)

  def _path(self, path: str, create: bool = False) -> tuple['Params', str]:
    parts = path.split('.')
    node: Params = self
    for part in parts[:-1]:
      try:
        child = getattr(node, part)
      except AttributeError:
        if not create:
          raise KeyError(path) from None
        child = Dynamic()
        if not node.__dynamic__:
          node._fields[part] = Field(default=child, type=Params, name=part)
          object.__setattr__(node, part, child)
        else:
          setattr(node, part, child)
      if not isinstance(child, Params):
        raise KeyError(f'{part!r} in {path!r} is not a Params node')
      node = child
    return node, parts[-1]

  def get_path(self, path: str, default: Any = None) -> Any:
    try:
      return self[path]
    except (KeyError, AttributeError):
      return default

  def has(self, path: str) -> bool:
    sentinel = object()
    return self.get_path(path, sentinel) is not sentinel

  def update(
    self,
    other: Mapping[str, Any] | Iterable[tuple[str, Any]] | None = None,
    *,
    strict: bool = False,
    **values: Any,
  ) -> 'Params':
    incoming = dict(other or {})
    incoming.update(values)
    for key, value in incoming.items():
      if '.' in key:
        self[key] = value
        continue
      if key not in self.fields:
        if strict and not self.__dynamic__:
          raise KeyError(f'unknown parameter {key!r}; expected one of {tuple(self.fields)!r}')
        setattr(self, key, value)
        continue
      current = getattr(self, key, MISSING)
      field = self.fields[key]
      if (
        isinstance(current, Params)
        and isinstance(value, Mapping)
        and _params_union_variants(field.type) is None
      ):
        current.update(value, strict=strict)
      else:
        setattr(self, key, value)
    return self

  def validate(self, recursive: bool = True) -> 'Params':
    for name, field in self.fields.items():
      value = getattr(self, name, MISSING)
      field.validate(value)
      if value is not MISSING and field.type is not None and not _matches_type(value, field.type):
        raise ValidationError(f'{name} expected {field.type!r}, got {type(value)!r}')
      if recursive and isinstance(value, Params):
        value.validate()
    return self

  def to_dict(self, *, secrets: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, field in self.fields.items():
      if field.secret and not secrets:
        continue
      value = getattr(self, name, MISSING)
      if value is MISSING:
        continue
      if isinstance(value, Params):
        value = value.to_dict(secrets=secrets)
      elif isinstance(value, list):
        value = [v.to_dict(secrets=secrets) if isinstance(v, Params) else v for v in value]
      result[name] = copy.deepcopy(value)
    return result

  def flatten(self, prefix: str = '') -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, value in self.items():
      path = f'{prefix}.{name}' if prefix else name
      if isinstance(value, Params):
        output.update(value.flatten(path))
      else:
        output[path] = value
    return output

  def fork(self, **updates: Any) -> 'Params':
    clone = copy.deepcopy(self)
    object.__setattr__(clone, '_frozen', False)
    clone.update(updates, strict=True)
    return clone

  def freeze(self, recursive: bool = True) -> 'Params':
    if recursive:
      for value in self.values():
        if isinstance(value, Params):
          value.freeze()
    object.__setattr__(self, '_frozen', True)
    return self

  def unfreeze(self, recursive: bool = True) -> 'Params':
    object.__setattr__(self, '_frozen', False)
    if recursive:
      for value in self.values():
        if isinstance(value, Params):
          value.unfreeze()
    return self

  def stable_hash(self, length: int = 12) -> str:
    payload = json.dumps(self.to_dict(secrets=True), sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:length]

  def diff(self, other: 'Params | Mapping[str, Any]') -> dict[str, tuple[Any, Any]]:
    left = self.flatten()
    right = other.flatten() if isinstance(other, Params) else Dynamic(**other).flatten()
    return {
      key: (left.get(key, MISSING), right.get(key, MISSING))
      for key in left.keys() | right.keys()
      if left.get(key, MISSING) != right.get(key, MISSING)
    }

  def on_change(self, callback: Any) -> Any:
    self._change_callbacks.append(callback)
    return callback

  def space(self, prefix: str = '') -> dict[str, Field]:
    output: dict[str, Field] = {}
    for name, field in self.fields.items():
      path = f'{prefix}.{name}' if prefix else name
      value = getattr(self, name, MISSING)
      if isinstance(value, Params):
        output.update(value.space(path))
      elif field.searchable:
        output[path] = field
    return output

  def sample(self, seed: int | None = None) -> 'Params':
    import random
    rng = random.Random(seed)
    clone = self.fork()
    for path, field in self.space().items():
      clone[path] = field.sample(rng)
    return clone

  def samples(self, count: int, seed: int | None = None) -> Iterator['Params']:
    import random
    rng = random.Random(seed)
    for _ in range(count):
      clone = self.fork()
      for path, field in self.space().items():
        clone[path] = field.sample(rng)
      yield clone

  def grid(self) -> Iterator['Params']:
    space = self.space()
    paths = tuple(space)
    for values in product(*(tuple(space[path].grid()) for path in paths)):
      clone = self.fork()
      for path, value in zip(paths, values):
        clone[path] = value
      yield clone

  @classmethod
  def schema(cls, target: Any, *, name: str | None = None) -> type['Params']:
    from .callable import fields_from_callable
    fields = fields_from_callable(target)
    namespace: dict[str, Any] = {'__annotations__': {}}
    for field_name, field in fields.items():
      namespace['__annotations__'][field_name] = field.type or Any
      namespace[field_name] = field
    return ParamsMeta(name or f'{getattr(target, "__name__", "Callable").title()}Params', (cls,), namespace)

  def define(self, target: Any, *, mapping: Mapping[str, str] | None = None) -> 'Params':
    from .callable import fields_from_callable
    fields = fields_from_callable(target)
    mapping = mapping or {}
    for arg, field in fields.items():
      path = mapping.get(arg, arg)
      node, name = self._path(path, create=True)
      if name in node.fields:
        continue
      node._fields[name] = field
      value = field.make_default()
      if value is not MISSING:
        object.__setattr__(node, name, coerce(value, field.type, field))
    return self

  def bind(self, target: Any = None, **kwargs: Any) -> Any:
    from .callable import decorate
    return decorate(self, target, mode='bind', **kwargs)

  def watch(self, target: Any = None, **kwargs: Any) -> Any:
    from .callable import decorate
    return decorate(self, target, mode='watch', **kwargs)

  def wrap(self, target: Any = None, **kwargs: Any) -> Any:
    from .callable import decorate
    return decorate(self, target, mode='wrap', **kwargs)

  @classmethod
  def from_env(cls, prefix: str = '', *, separator: str = '__') -> 'Params':
    hp = cls()
    for key, value in os.environ.items():
      if not key.startswith(prefix):
        continue
      path = key[len(prefix):].lower().replace(separator.lower(), '.')
      if hp.has(path):
        field = hp._path(path)[0].fields[hp._path(path)[1]]
        value = coerce(value, field.type, field)
      hp[path] = value
    return hp

  @classmethod
  def from_command(cls, args: str | list[str] | None = None) -> 'Params':
    from .cli import parse
    if isinstance(args, str):
      import shlex
      args = shlex.split(args)
    hp = cls()
    return parse(hp, args)

  from_cli = from_command

  @classmethod
  def load(cls, path: str | Path) -> 'Params':
    from .io import load
    return cls(**load(path))

  def save(self, path: str | Path) -> None:
    from .io import save
    save(self.to_dict(secrets=True), path)

  def __repr__(self) -> str:
    body = ', '.join(f'{k}={v!r}' for k, v in self.items())
    return f'{type(self).__name__}({body})'


class Dynamic(Params, dynamic=True):
  """Schemaless params tree: unknown names become fields on assignment.

  Nested nodes auto-vivify, so intermediate levels need no declaration::

      config = Dynamic()
      config.rollout.temperature = 0.8

  Declared classes can opt into the same behavior with
  ``class Config(Params, dynamic=True)``.
  """
