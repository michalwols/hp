from typing import Literal

import hp
from hp import Choice, HP, LogRange


class OptimHP(HP):
  lr: float = 2e-4
  weight_decay: float = 0.01


class Params(HP):
  seed: int = 42
  optim: OptimHP = OptimHP()


def test_declared_nested_values_are_independent():
  a = Params()
  b = Params()
  a.optim.lr = 1e-4
  assert b.optim.lr == 2e-4


def test_dotted_update_and_flatten():
  p = Params()
  p['optim.lr'] = 1e-4
  assert p.optim.lr == 1e-4
  assert p.flatten()['optim.lr'] == 1e-4


def test_dynamic_autovivification():
  p = HP.dynamic()
  p.opt.foo = 5
  p.model.encoder.layers = 12
  assert p.to_dict() == {
    'opt': {'foo': 5},
    'model': {'encoder': {'layers': 12}},
  }


def test_bind_watch_wrap():
  p = Params()

  @p.bind(mapping={'lr': 'optim.lr'})
  def bound(lr=1.0):
    return lr

  assert bound() == 2e-4
  assert bound(0.5) == 0.5
  assert p.optim.lr == 2e-4

  @p.watch(mapping={'lr': 'optim.lr'})
  def watched(lr=1.0):
    return lr

  watched(0.3)
  assert p.optim.lr == 0.3

  @p.wrap(mapping={'lr': 'optim.lr'})
  def wrapped(lr=1.0):
    return lr

  assert wrapped() == 0.3
  wrapped(0.2)
  assert p.optim.lr == 0.2


def test_schema_from_callable():
  def train(epochs: int = 10, lr: float = 2e-4):
    pass

  TrainHP = HP.schema(train)
  p = TrainHP(lr=1e-4)
  assert p.epochs == 10
  assert p.lr == 1e-4


def test_module_wrap():
  @hp.wrap
  def train(epochs: int = 10):
    return epochs

  train.hp.epochs = 5
  assert train() == 5


class AdamWHP(HP):
  method: Literal['adamw'] = 'adamw'
  lr: float = 2e-4


class SGDHP(HP):
  method: Literal['sgd'] = 'sgd'
  lr: float = 1e-2
  momentum: float = 0.9


class VariantParams(HP):
  optimizer: AdamWHP | SGDHP = AdamWHP()


def test_tagged_union_coercion():
  p = VariantParams(optimizer={'method': 'sgd', 'momentum': 0.95})
  assert isinstance(p.optimizer, SGDHP)
  assert p.optimizer.momentum == 0.95


def test_search_space_sample_and_grid():
  class SearchHP(HP):
    lr: float = LogRange(1e-5, 1e-3, default=1e-4)
    batch: int = Choice((2, 4), default=2)

  p = SearchHP()
  sample = p.sample(seed=1)
  assert 1e-5 <= sample.lr <= 1e-3
  assert sample.batch in {2, 4}
  grid = list(p.grid())
  assert len(grid) == 2


def test_freeze_and_hash():
  p = Params().freeze()
  before = p.stable_hash()
  try:
    p.seed = 1
  except TypeError:
    pass
  else:
    raise AssertionError('frozen HP should reject updates')
  assert p.stable_hash() == before


def test_mapping_contains():
  p = Params()
  assert 'seed' in p
  assert 'missing_key' not in p
  try:
    p['missing_key']
  except KeyError:
    pass
  else:
    raise AssertionError('unknown key should raise KeyError')
