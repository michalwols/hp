"""Live env/cli views, third-party config bridges, and module instrumentation."""

import argparse
import dataclasses
import os
import types
import typing

import pytest

import hp


@pytest.fixture(autouse=True)
def clean():
  hp.params.clear()
  hp.cli.reset()
  yield
  hp.params.clear()
  hp.cli.reset()


# --- live views -----------------------------------------------------------

def test_env_reads_and_writes_os_environ(monkeypatch):
  monkeypatch.delenv('HP_TEST_VAR', raising=False)
  hp.env.HP_TEST_VAR = 3
  assert os.environ['HP_TEST_VAR'] == '3'
  assert hp.env.HP_TEST_VAR == '3'
  assert hp.env.hp_test_var == '3'  # lower case spelling resolves
  assert 'HP_TEST_VAR' in hp.env
  del hp.env.HP_TEST_VAR
  assert 'HP_TEST_VAR' not in os.environ


def test_env_is_a_layer(monkeypatch):
  class Config(hp.Params):
    seed: int = 1

  monkeypatch.setenv('APP__SEED', '7')
  params = hp.load(Config, hp.env(prefix='APP'))
  assert params.seed == 7
  assert hp.source(params, 'seed') == 'env'


def test_cli_is_a_layer_and_a_view(monkeypatch):
  class Config(hp.Params):
    seed: int = 1

  params = hp.load(Config, hp.cli(['--seed', '9']))
  assert params.seed == 9

  monkeypatch.setattr('sys.argv', ['prog', '--seed', '5'])
  hp.cli.reset()
  assert hp.cli.seed == 5


# --- interop --------------------------------------------------------------

@dataclasses.dataclass
class DataCfg:
  lr: float = 1e-3
  name: str = 'x'


def test_dataclass_round_trip():
  params = hp.load(DataCfg(lr=0.5))
  assert hp.to_dict(params) == {'lr': 0.5, 'name': 'x'}
  assert hp.construct(params, DataCfg) == DataCfg(lr=0.5, name='x')


def test_namedtuple_schema():
  class NT(typing.NamedTuple):
    a: int = 1
    b: str = 'x'

  assert hp.to_dict(hp.schema(NT)()) == {'a': 1, 'b': 'x'}


def test_namespace_and_mapping():
  assert hp.to_dict(hp.load(argparse.Namespace(seed=3))) == {'seed': 3}
  assert hp.to_dict(hp.load({'a': 1})) == {'a': 1}


def test_from_object_rejects_unreadable():
  with pytest.raises(TypeError):
    hp.load(42)


def test_to_argparse_round_trips():
  class Config(hp.Params):
    lr: float = 1e-3
    debug: bool = False
    method: str = hp.Choice(('sft', 'grpo'), default='sft')

  parser = hp.to_argparse(Config(), prog='train')
  parsed = vars(parser.parse_args(['--lr', '0.9', '--debug']))
  assert parsed['lr'] == 0.9
  assert parsed['debug'] is True
  assert parsed['method'] == 'sft'

  with pytest.raises(SystemExit):
    parser.parse_args(['--method', 'nope'])  # choices enforced


def test_construct_filters_to_the_signature():
  params = hp.Dynamic(lr=0.5, name='x', extra='ignored')
  assert hp.construct(params, DataCfg) == DataCfg(lr=0.5, name='x')


# --- instrumentation ------------------------------------------------------

def fake_module():
  module = types.ModuleType('fakelib')

  def build(width: int = 8, depth: int = 2):
    return width * depth

  class Model:
    def __init__(self, hidden: int = 16):
      self.hidden = hidden

  build.__module__ = Model.__module__ = 'fakelib'
  module.build, module.Model = build, Model
  return module, build, Model


def test_instrument_records_calls_and_restores():
  module, build, Model = fake_module()

  with hp.params.instrumented(module) as handle:
    module.build(width=4)
    instance = module.Model(hidden=32)

    assert sorted(handle.names) == ['fakelib.Model', 'fakelib.build']
    assert hp.params.calls('fakelib.build') == [{'width': 4}]
    assert hp.params.calls('fakelib.Model') == [{'hidden': 32}]  # no self
    assert isinstance(instance, Model)

  assert module.build is build
  assert list(hp.params.registry) == []


def test_instrument_select_and_override():
  module, build, _ = fake_module()

  hp.params(module, select=['build'], override=True)
  assert 'fakelib.Model' not in hp.params.registry

  hp.params('fakelib.build').width = 100
  assert module.build() == 200  # width overridden, depth default
  assert module.build(width=2) == 4  # explicit still wins

  hp.params.restore(module)
  assert module.build() == 16


def test_instrumenting_twice_raises():
  module, _, _ = fake_module()
  hp.params(module)
  try:
    with pytest.raises(RuntimeError, match='already instrumented'):
      hp.params(module)
  finally:
    hp.params.restore(module)


def test_surface_reports_everything():
  @hp.params(name='train')
  def train(epochs: int = 3):
    return epochs

  train()
  report = hp.params.surface()
  assert report['targets']['train']['params'] == {'epochs': 3}
  assert 'env' in report and 'cli' in report


def test_cli_collects_positionals_instead_of_failing(monkeypatch):
  monkeypatch.setattr('sys.argv', ['prog', 'data.csv', '--seed', '5', 'more'])
  hp.cli.reset()
  assert hp.cli.seed == 5
  assert hp.cli.args == ['data.csv', 'more']


def test_from_command_ignores_positionals():
  class Config(hp.Params):
    seed: int = 1

  assert hp.load(Config, hp.cli(['input.txt', '--seed', '4'])).seed == 4
