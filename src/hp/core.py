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


class UnknownParam(UserWarning):
  """Warned when a value is set for a name with no declared field.

  The value is still accepted and becomes a real field; the warning exists
  so typos in config files and command lines are visible instead of silent.
  """


def _unknown_param_warning(owner: 'Params', name: str, stacklevel: int = 3) -> None:
  if owner.__dynamic__:
    return
  import difflib
  import warnings

  close = difflib.get_close_matches(name, tuple(owner.fields), n=3)
  hint = f'; did you mean {" or ".join(repr(c) for c in close)}?' if close else ''
  warnings.warn(
    f'{type(owner).__name__} has no declared field {name!r}{hint}',
    UnknownParam,
    stacklevel=stacklevel,
  )


def _nest(flat: Mapping[str, Any]) -> dict[str, Any]:
  """Expand dotted keys into nested dicts so one update() sees whole nodes.

  Applying ``optimizer.name`` and ``optimizer.momentum`` together lets a
  tagged union resolve to the right variant once, rather than mutating
  whichever variant happened to be there first.
  """
  output: dict[str, Any] = {}
  for path, value in flat.items():
    parts = path.split('.')
    node = output
    for part in parts[:-1]:
      child = node.get(part)
      if not isinstance(child, dict):
        child = {}
        node[part] = child
      node = child
    node[parts[-1]] = value
  return output


def _restamp(node: 'Params', source: str) -> None:
  for name, origin in node._sources.items():
    if origin == 'code':
      node._sources[name] = source
  for value in node.values():
    if isinstance(value, Params):
      _restamp(value, source)


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
    if isinstance(value, Params):
      tag = getattr(value, discriminator, None)
      # already the right variant; rebuilding would discard its provenance
      if tag in mapping and isinstance(value, mapping[tag]):
        return value
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
    object.__setattr__(self, '_sources', {})
    object.__setattr__(self, '_source', 'default')
    for name, field in self.__class__.__fields__.items():
      value = field.make_default()
      if value is not MISSING:
        value = coerce(value, field.type, field)
      object.__setattr__(self, name, value)
      self._sources[name] = 'default'
    object.__setattr__(self, '_source', 'code')
    for mapping in args:
      self.update(mapping)
    self.update(values)

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
      # unknown names become real fields, so they serialize like declared ones
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
    self._sources[name] = self._source
    if isinstance(value, Params) and self._source != 'code':
      # a node built while applying a layer belongs to that layer, not to
      # the constructor call that materialized it
      _restamp(value, self._source)

  def __getitem__(self, key: str) -> Any:
    node, name = self._path(key)
    try:
      return getattr(node, name)
    except AttributeError:
      raise KeyError(key) from None

  def __setitem__(self, key: str, value: Any) -> None:
    node, name = self._path(key, create=True)
    setattr(node, name, value)
    self._resolve_variant(key)

  def _resolve_variant(self, path: str) -> None:
    """Rebuild a tagged-union node when its discriminator was just set.

    Without this, ``params['optimizer.name'] = 'sgd'`` would leave an AdamW
    instance claiming to be an SGD one, which then deserializes as a
    different type than the process was running with.
    """
    parts = path.split('.')
    if len(parts) < 2:
      return
    try:
      owner, child = self._path('.'.join(parts[:-1]))
    except KeyError:
      return
    field = owner.fields.get(child)
    if field is None:
      return
    variants = _params_union_variants(field.type)
    if variants is None:
      return
    discriminator, mapping = variants
    if parts[-1] != discriminator:
      return
    current = getattr(owner, child, None)
    tag = getattr(current, discriminator, None)
    if tag not in mapping or isinstance(current, mapping[tag]):
      return
    target = mapping[tag]
    # carry across only what was explicitly set; the outgoing variant's own
    # defaults should not override the incoming one's
    carried = {
      key: value
      for key, value in current.to_dict(secrets=True).items()
      if key in target.__fields__
      and key != discriminator
      and current._sources.get(key, 'default') != 'default'
    }
    setattr(owner, child, target(**carried))

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

  def _resolve_alias(self, key: str) -> str:
    for name, field in self.fields.items():
      if field.alias == key:
        return name
    return key

  def update(
    self,
    other: Mapping[str, Any] | Iterable[tuple[str, Any]] | None = None,
    *,
    strict: bool = False,
    source: str | None = None,
    **values: Any,
  ) -> 'Params':
    incoming = dict(other or {})
    incoming.update(values)
    previous = self._source
    if source is not None:
      object.__setattr__(self, '_source', source)
    try:
      for key, value in incoming.items():
        if '.' in key:
          self[key] = value
          continue
        key = self._resolve_alias(key)
        if key not in self.fields:
          if strict and not self.__dynamic__:
            raise KeyError(f'unknown parameter {key!r}; expected one of {tuple(self.fields)!r}')
          _unknown_param_warning(self, key)
          setattr(self, key, value)
          continue
        current = getattr(self, key, MISSING)
        field = self.fields[key]
        if isinstance(current, Params) and isinstance(value, Mapping):
          variants = _params_union_variants(field.type)
          # replace only when a different variant was actually requested;
          # a partial update just merges into the variant already there
          if variants is None or variants[0] not in value:
            current.update(value, strict=strict, source=source)
            continue
        setattr(self, key, value)
    finally:
      object.__setattr__(self, '_source', previous)
    return self

  def source(self, path: str) -> str:
    """Which layer last set this value: default, code, cli, env, or a filename."""
    node, name = self._path(path)
    return node._sources.get(name, 'default')

  def sources(self, prefix: str = '') -> dict[str, str]:
    """Provenance for every value in the tree, keyed by dotted path."""
    output: dict[str, str] = {}
    for name in self.fields:
      path = f'{prefix}.{name}' if prefix else name
      value = getattr(self, name, MISSING)
      if isinstance(value, Params):
        output.update(value.sources(path))
      else:
        output[path] = self._sources.get(name, 'default')
    return output

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

  def field_paths(self, prefix: str = '') -> OrderedDict[str, Field]:
    """Every field in the tree, keyed by dotted path."""
    output: OrderedDict[str, Field] = OrderedDict()
    for name, field in self.fields.items():
      path = f'{prefix}.{name}' if prefix else name
      value = getattr(self, name, MISSING)
      if isinstance(value, Params):
        output.update(value.field_paths(path))
      else:
        output[path] = field
    return output

  def space(
    self,
    prefix: str = '',
    root: 'Params | None' = None,
    *,
    active_only: bool = True,
  ) -> dict[str, Field]:
    """Searchable fields keyed by dotted path.

    Fields carrying a ``when`` condition are excluded when that condition is
    false; pass ``active_only=False`` to get the full space regardless.
    """
    root = self if root is None else root
    output: dict[str, Field] = {}
    for name, field in self.fields.items():
      path = f'{prefix}.{name}' if prefix else name
      value = getattr(self, name, MISSING)
      if isinstance(value, Params):
        output.update(value.space(path, root, active_only=active_only))
      elif field.searchable and (not active_only or field.is_active(root)):
        output[path] = field
    return output

  def _sampled(self, rng: Any) -> 'Params':
    candidates = self.space(active_only=False)
    clone = self.fork()
    for path, field in candidates.items():
      clone[path] = field.sample(rng)
    # a condition may only become false once the fields it depends on are
    # drawn, so settle inactive fields back to their defaults afterwards
    for path, field in candidates.items():
      if not field.is_active(clone):
        clone[path] = field.make_default()
    return clone

  def sample(self, seed: int | None = None) -> 'Params':
    import random
    return self._sampled(random.Random(seed))

  def samples(self, count: int, seed: int | None = None) -> Iterator['Params']:
    import random
    rng = random.Random(seed)
    for _ in range(count):
      yield self._sampled(rng)

  def grid(self) -> Iterator['Params']:
    space = self.space(active_only=False)
    paths = tuple(space)
    seen: set[str] = set()
    for values in product(*(tuple(space[path].grid()) for path in paths)):
      clone = self.fork()
      for path, value in zip(paths, values):
        clone[path] = value
      for path in paths:
        if not space[path].is_active(clone):
          clone[path] = space[path].make_default()
      # conditions collapse distinct draws onto the same config
      digest = clone.stable_hash()
      if digest in seen:
        continue
      seen.add(digest)
      yield clone

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
    """Populate from environment variables.

    ``APP__OPTIM__LR`` maps to ``optim.lr`` under prefix ``APP``. Only
    declared fields are populated — the environment is ambient, so unknown
    names are ignored rather than becoming config — unless the class is
    dynamic. A field's explicit ``env`` name takes precedence.
    """
    hp = cls()
    hp.update(_nest(hp._env_updates(prefix, separator)), source='env')
    return hp

  def _env_updates(self, prefix: str = '', separator: str = '__') -> dict[str, Any]:
    updates: dict[str, Any] = {}

    explicit = {
      field.env: path
      for path, field in self.field_paths().items()
      if field.env
    }
    for name, path in explicit.items():
      if name in os.environ:
        updates[path] = os.environ[name]

    for key, value in os.environ.items():
      if key in explicit or not key.startswith(prefix):
        continue
      remainder = key[len(prefix):]
      if separator and remainder.startswith(separator):
        remainder = remainder[len(separator):]
      if not remainder:
        continue
      path = remainder.lower().replace(separator.lower(), '.')
      if self.has(path) or self.__dynamic__:
        updates.setdefault(path, value)
    return updates

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
    hp = cls()
    hp.update(load(path), source=str(path))
    return hp

  @classmethod
  def layered(cls, *sources: Any, separator: str = '__') -> 'Params':
    """Compose ordered sources, later ones winning.

    Each source is a mapping, a config file path, or the string ``'env'`` or
    ``'cli'``. Use :meth:`sources` afterwards to see which layer won a value.

        params = TrainParams.layered('base.yaml', 'experiment.yaml', 'env', 'cli')
    """
    hp = cls()
    for entry in sources:
      prefix = ''
      if isinstance(entry, tuple) and entry and entry[0] == 'env':
        entry, prefix = 'env', entry[1] if len(entry) > 1 else ''
      if entry == 'env':
        hp.update(_nest(hp._env_updates(prefix, separator)), source='env')
      elif entry == 'cli':
        from .cli import parse
        parse(hp, None)
      elif isinstance(entry, Mapping):
        hp.update(entry, source='mapping')
      else:
        from .io import load
        hp.update(load(entry), source=str(entry))
    return hp

  def save(self, path: str | Path) -> None:
    from .io import save
    save(self.to_dict(secrets=True), path)

  def __hash__(self) -> int:
    # by current content, like any other value object; do not mutate a
    # params object while it is in use as a dict key
    return hash(self.stable_hash())

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


def _class_name(target: Any) -> str:
  raw = getattr(target, '__name__', 'callable')
  parts = [part for part in raw.strip('<>').split('_') if part]
  if not parts:
    return 'CallableParams'
  return ''.join(part.title() for part in parts) + 'Params'


def schema(
  target: Any,
  *,
  name: str | None = None,
  base: type[Params] | None = None,
) -> type[Params]:
  """Build a Params subclass from a callable's signature.

  ``base`` selects the class to derive from, so schemas can inherit shared
  fields or opt into dynamic behavior.
  """
  from .callable import fields_from_callable

  fields = fields_from_callable(target)
  namespace: dict[str, Any] = {'__annotations__': {}}
  for field_name, field in fields.items():
    namespace['__annotations__'][field_name] = field.type or Any
    namespace[field_name] = field
  return ParamsMeta(name or _class_name(target), (base or Params,), namespace)


def evolvable(params: Params, *, group: str | None = None) -> dict[str, str]:
  """Dotted paths and current values for every Evolve field.

  This is the seed candidate for a text optimizer; pass ``group`` to take
  only one labelled subset.
  """
  from .fields import Evolve

  return {
    path: params[path]
    for path, field in params.field_paths().items()
    if isinstance(field, Evolve) and (group is None or field.group == group)
  }


def apply_candidate(params: Params, candidate: Mapping[str, str]) -> Params:
  """Fork ``params`` and apply a candidate's text components by dotted path."""
  clone = params.fork()
  clone.update(_nest(dict(candidate)), source='candidate')
  return clone
