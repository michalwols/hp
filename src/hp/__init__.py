"""Configuration, parameter trees, and hyperparameter search.

``Params`` deliberately has an empty public namespace so that every attribute
name is available for user fields. Operations are module-level functions taking
the params as their first argument:

    import hp

    class TrainParams(hp.Params):
      sample: int = 4
      freeze: bool = True
      update: str = 'ema'

    params = hp.from_command(TrainParams)
    hp.to_dict(params)
"""

from __future__ import annotations

from .callable import fields_from_callable
from .core import (
  Dynamic,
  Params,
  UnknownParam,
  apply_candidate,
  define,
  diff,
  evolvable,
  field_paths,
  fields,
  flatten,
  fork,
  freeze,
  get,
  grid,
  has,
  items,
  keys,
  load,
  on_change,
  sample,
  samples,
  save,
  schema,
  serializable,
  source,
  sources,
  space,
  stable_hash,
  to_dict,
  unfreeze,
  update,
  validate,
  values,
)
from .cli import help_text
from .fields import (
  Choice,
  Computed,
  Derived,
  Evolve,
  Field,
  IntRange,
  LogIntRange,
  LogRange,
  Range,
  ValidationError,
  computed,
  derived,
)
from .adapt import construct, to_argparse
from .context import active, override, scope
from .registry import Entry, params, parametrize as wrap, track
from .views import cli, env

__all__ = [
  # types
  'Params',
  'Dynamic',
  'Field',
  'Choice',
  'Range',
  'LogRange',
  'IntRange',
  'LogIntRange',
  'Evolve',
  'Derived',
  'Computed',
  'derived',
  'computed',
  'ValidationError',
  'UnknownParam',
  'Entry',
  # construction
  'load',
  'schema',
  'define',
  'fields_from_callable',
  # access
  'fields',
  'field_paths',
  'keys',
  'values',
  'items',
  'get',
  'has',
  'to_dict',
  'serializable',
  'flatten',
  'source',
  'sources',
  # mutation
  'update',
  'fork',
  'freeze',
  'unfreeze',
  'on_change',
  'validate',
  'save',
  # comparison
  'diff',
  'stable_hash',
  # search
  'space',
  'sample',
  'samples',
  'grid',
  # text optimization
  'evolvable',
  'apply_candidate',
  # registry, decorator and instrumenter, in one object
  'params',
  'wrap',
  'track',
  # live views
  'env',
  'cli',
  # scoping
  'scope',
  'override',
  'active',
  # interop
  'construct',
  'to_argparse',
  # cli
  'help_text',
]
