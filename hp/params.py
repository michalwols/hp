from abc import ABCMeta
from collections import OrderedDict
from copy import deepcopy
from .parser import get_arg_parser


class Field:
  def __init__(
    self, help=None, type=None, required=False, default=None
  ):
    self.help = help
    self.type = type
    self.required = required
    self.default = default

  def validate(self, val):
    if self.type:
      assert isinstance(val, self.type)

  def __repr__(self):
    return f"{self.__class__.__name__}(type={self.type}, default={self.default})"

  def __str__(self):
    return f"{self.__class__.__name__}(type={self.type}, default={self.default})"


class Choice(Field):
  def __init__(self, choices=None, **kwargs):
    super().__init__(**kwargs)
    self.choices = choices

  def validate(self, val):
    super().validate(val)
    assert val in self.choices


class Range(Field):
  def __init__(self, start, end):
    super(Range, self).__init__()
    self.min = min(start, end)
    self.max = max(start, end)

  def validate(self, val):
    super().validate(val)
    assert self.min <= val <= self.max


class HyperParamsBase:
  def __init__(self, **args):
    self._change_callbacks = []

    for k, v in args.items():
      if k in self.__fields__:
        setattr(self, k, v)
      else:
        raise ValueError(
          f'Unknown parameter: {k}, should be one of {", ".join(self.__fields__)}'
        )

  def validate(self):
    for k, f in self.__fields__.items():
      try:
        f.validate(getattr(self, k))
      except Exception as e:
        raise Exception(f"{k} failed validation. {e}")

  @classmethod
  def arg_parser(cls, **kwargs):
    return get_arg_parser(
      cls.__fields__, description=cls.__doc__, **kwargs
    )

  @classmethod
  def from_command(cls, cmd=None, validate=False, **kwargs):
    parser = cls.arg_parser(**kwargs)
    parsed = parser.parse_args(
      cmd.split() if isinstance(cmd, str) else cmd
    )
    params = cls(**vars(parsed))

    if validate:
      params.validate()

    return params

  @classmethod
  def from_env(cls, prefix=''):
    raise NotImplementedError()

  @classmethod
  def load(cls, path):

    if path.endswith(('.yaml', '.yml')):
      import yaml
      with open(path, 'r') as f:
        data = yaml.safe_load(f)
        return cls(**data)
    elif path.endswith('.json'):
      import json
      with open(path, 'r') as f:
        data = json.load(f)
        return cls(**data)

  def save(self, path):
    import pathlib
    if isinstance(path, (str, pathlib.Path)):
      if path.endswith(('.yaml', 'yml')):
        import yaml
        with open(path, 'w') as f:
          yaml.dump(dict(self), f)
      if path.endswith('.json'):
        import json
        with open(path, 'w') as f:
          json.dump(dict(self), f)

  def on_change(self, callback):
    self._change_callbacks.append(callback)


  def bind(self, x):

    def decorated(*args, **kwargs):



      return decorated

  def __setattr__(self, k, v):
    if k in self.__fields__:
      for c in self._change_callbacks:
        c(k, v)
    super.__setattr__(self, k, v)

  def __getitem__(self, item):
    if isinstance(item, (tuple, list)):
      return tuple(getattr(self, k) for k in item)
    return getattr(self, item)

  def __setitem__(self, k, v):
    setattr(self, k, v)

  def fork(self, **args):
    return HyperParams(**{**self.items(), **args})

  def __repr__(self):
    return (
      f'{self.__class__.__name__}('
      f"{', '.join(f'{k}={v}' for k, v in self.items())}"
      ')'
    )

  def __str__(self):
    return (
      f'{self.__class__.__name__}(\n' +
      ',\n'.join('  {}={}'.format(k, v) for k, v in self.items()) +
      '\n)'
    )

  def __len__(self):
    return len(self.keys())

  def __hash__(self):
    return hash(tuple(sorted(self.items())))

  def keys(self):
    return self.__fields__.keys()

  def values(self):
    return {getattr(self, k) for k in self.keys()}

  def items(self):
    return ((k, getattr(self, k)) for k in self.keys())

  def inject(self, scope=None, uppercase=True):
    scope = globals() if scope is None else scope
    for k, v in self.items():
      scope[k.upper() if uppercase else k] = v

  @classmethod
  def collect(
    cls,
    scope=None,
    types=(int, str, float, bool),
    upper_only=True,
    lowercase=True
  ):
    scope = globals() if scope is None else scope

    d = {}
    for k, v in scope.items():
      if types and not isinstance(v, types):
        continue
      if upper_only and not k.isupper():
        continue

      d[k.lower() if lowercase else k] = v

    return cls(**d)

  def __eq__(self, other):
    return hash(self) == hash(other)


class MetaHyperParams(ABCMeta):
  def __new__(metaclass, class_name, bases, namespace):
    fields = OrderedDict()

    for base in reversed(bases):
      if issubclass(
        base, HyperParamsBase
      ) and base != HyperParamsBase:
        fields.update(deepcopy(base.__fields__))

    existing_attributes = set(dir(HyperParamsBase)) | set(fields)

    new_attributes = {
      k: v
      for (k, v) in namespace.items()
      if k not in existing_attributes and not k.startswith('_')
    }

    for name, annotation in namespace.get('__annotations__',
                                          {}).items():
      if name in existing_attributes:
        continue
      if isinstance(annotation, Field):
        fields[name] = annotation
      else:
        fields[name] = Field(type=annotation)

    for name, value in new_attributes.items():
      if name in fields:
        fields[name].default = value
      else:
        fields[name] = Field(default=value, type=type(value))

    return super().__new__(
      metaclass, class_name, bases, {
        '__fields__': fields,
        **namespace,
      }
    )


class HyperParams(HyperParamsBase, metaclass=MetaHyperParams):
  pass
