from typing import Literal

import hp


class OptimParams(hp.Params):
  lr: float = 2e-4
  weight_decay: float = 0.01


class TrainParams(hp.Params):
  seed: int = 42
  optim: OptimParams = OptimParams()


def test_declared_nested_values_are_independent():
  a = TrainParams()
  b = TrainParams()
  a.optim.lr = 1e-4
  assert b.optim.lr == 2e-4


def test_dotted_update_and_flatten():
  p = TrainParams()
  p['optim.lr'] = 1e-4
  assert p.optim.lr == 1e-4
  assert hp.flatten(p)['optim.lr'] == 1e-4


def test_dynamic_autovivification():
  p = hp.Dynamic()
  p.opt.foo = 5
  p.model.encoder.layers = 12
  assert hp.to_dict(p) == {
    'opt': {'foo': 5},
    'model': {'encoder': {'layers': 12}},
  }


def test_dynamic_accepts_initial_values():
  p = hp.Dynamic(seed=1)
  p.extra = 'x'
  assert p.seed == 1
  assert hp.to_dict(p) == {'seed': 1, 'extra': 'x'}


def test_declared_class_can_opt_into_dynamic():
  class OpenParams(hp.Params, dynamic=True):
    seed: int = 42

  p = OpenParams()
  p.rollout.temperature = 0.8
  assert p.seed == 42
  assert hp.to_dict(p) == {'seed': 42, 'rollout': {'temperature': 0.8}}


def test_unknown_write_becomes_a_real_field():
  p = TrainParams()
  p.extra = 5
  assert p.extra == 5
  assert hp.to_dict(p)['extra'] == 5
  assert 'extra' in hp.fields(p)


def test_unknown_read_still_raises():
  p = TrainParams()
  try:
    p.sed  # typo for seed
  except AttributeError:
    pass
  else:
    raise AssertionError('undeclared reads should raise on non-dynamic params')


def test_unknown_key_warns_with_suggestion():
  import pytest

  with pytest.warns(hp.UnknownParam, match='did you mean'):
    p = TrainParams(sed=1)
  assert p.sed == 1


def test_dynamic_does_not_warn():
  import warnings

  with warnings.catch_warnings():
    warnings.simplefilter('error')
    p = hp.Dynamic(anything=1)
    p.other = 2
  assert hp.to_dict(p) == {'anything': 1, 'other': 2}


def test_update_can_still_be_strict():
  p = TrainParams()
  try:
    hp.update(p, {'nope': 1}, strict=True)
  except KeyError:
    pass
  else:
    raise AssertionError('strict=True should still reject unknown keys')


def test_parametrize_fills_and_records_arguments():
  hp.params.clear()

  @hp.params
  def train(epochs: int = 10, lr: float = 2e-4):
    return epochs, lr

  assert train() == (10, 2e-4)          # filled from params
  assert train(lr=1e-3) == (10, 1e-3)   # explicit wins
  assert hp.params(train).lr == 1e-3    # and is written back
  assert train() == (10, 1e-3)          # so it sticks


def test_track_records_without_changing_behavior():
  hp.params.clear()

  @hp.track
  def train(epochs: int = 10, lr: float = 2e-4):
    return epochs, lr

  assert train(lr=1e-3) == (10, 1e-3)
  assert train(5) == (5, 2e-4)          # defaults untouched by params

  assert hp.params.calls(train) == [{'lr': 1e-3}, {'epochs': 5}]
  assert hp.params(train).lr == 2e-4    # never mutated


def test_registry_collects_targets():
  hp.params.clear()

  @hp.params(name='alpha')
  def alpha(a: int = 1):
    return a

  @hp.track(name='custom')
  def beta(b: int = 2):
    return b

  assert list(hp.params.registry) == ['alpha', 'custom']
  assert hp.params.entry('custom').target is beta.__wrapped__
  assert hp.params.registry['alpha'].mode == 'parametrize'


def test_schema_from_callable():
  def train(epochs: int = 10, lr: float = 2e-4):
    pass

  Schema = hp.schema(train)
  p = Schema(lr=1e-4)
  assert p.epochs == 10
  assert p.lr == 1e-4


def test_parametrize_exposes_params_on_the_target():
  hp.params.clear()

  @hp.params
  def train(epochs: int = 10):
    return epochs

  hp.params(train).epochs = 5
  assert train() == 5


class AdamWParams(hp.Params):
  method: Literal['adamw'] = 'adamw'
  lr: float = 2e-4


class SGDParams(hp.Params):
  method: Literal['sgd'] = 'sgd'
  lr: float = 1e-2
  momentum: float = 0.9


class VariantParams(hp.Params):
  optimizer: AdamWParams | SGDParams = AdamWParams()


def test_tagged_union_coercion():
  p = VariantParams(optimizer={'method': 'sgd', 'momentum': 0.95})
  assert isinstance(p.optimizer, SGDParams)
  assert p.optimizer.momentum == 0.95


def test_search_space_sample_and_grid():
  class SearchParams(hp.Params):
    lr: float = hp.LogRange(1e-5, 1e-3, default=1e-4)
    batch: int = hp.Choice((2, 4), default=2)

  p = SearchParams()
  sample = hp.sample(p, seed=1)
  assert 1e-5 <= sample.lr <= 1e-3
  assert sample.batch in {2, 4}
  grid = list(hp.grid(p))
  assert len(grid) == 2


def test_freeze_and_hash():
  p = hp.freeze(TrainParams())
  before = hp.stable_hash(p)
  try:
    p.seed = 1
  except TypeError:
    pass
  else:
    raise AssertionError('frozen params should reject updates')
  assert hp.stable_hash(p) == before


def test_mapping_contains():
  p = TrainParams()
  assert 'seed' in p
  assert 'missing_key' not in p
  try:
    p['missing_key']
  except KeyError:
    pass
  else:
    raise AssertionError('unknown key should raise KeyError')


def test_params_namespace_is_empty():
  # every attribute name stays available for user fields
  assert [n for n in dir(hp.Params) if not n.startswith('_')] == []

  class P(hp.Params):
    sample: int = 4
    freeze: bool = True
    update: str = 'ema'
    values: list = []

  p = P()
  assert hp.to_dict(p) == {'sample': 4, 'freeze': True, 'update': 'ema', 'values': []}


def test_schema_is_module_level_only():
  def train_model(epochs: int = 10):
    pass

  Schema = hp.schema(train_model)
  assert Schema.__name__ == 'TrainModelParams'
  assert not hasattr(hp.Params, 'schema')


def test_schema_accepts_a_base_class():
  class Shared(hp.Params):
    seed: int = 42

  def train(epochs: int = 10):
    pass

  Schema = hp.schema(train, base=Shared, name='TrainSchema')
  p = Schema()
  assert Schema.__name__ == 'TrainSchema'
  assert p.seed == 42
  assert p.epochs == 10
