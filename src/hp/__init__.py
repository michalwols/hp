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
  from_command,
  from_env,
  get,
  grid,
  has,
  items,
  keys,
  layered,
  load,
  on_change,
  sample,
  samples,
  save,
  schema,
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
  Evolve,
  Field,
  IntRange,
  LogIntRange,
  LogRange,
  Range,
  ValidationError,
)
from .registry import (
  Entry,
  calls,
  clear,
  entry,
  parametrize,
  params,
  registry,
  track,
)

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
  'ValidationError',
  'UnknownParam',
  'Entry',
  # construction
  'from_command',
  'from_env',
  'load',
  'layered',
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
  # registry
  'parametrize',
  'track',
  'params',
  'calls',
  'entry',
  'registry',
  'clear',
  # cli
  'help_text',
]
