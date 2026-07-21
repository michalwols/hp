"""Union switching, provenance, layering, conditionals, and CLI help."""

import warnings
from typing import Literal

import pytest

import hp


class AdamW(hp.Params):
  name: Literal['adamw'] = 'adamw'
  lr: float = 2e-4


class SGD(hp.Params):
  name: Literal['sgd'] = 'sgd'
  lr: float = 1e-2
  momentum: float = 0.9


class Cfg(hp.Params):
  optimizer: AdamW | SGD = AdamW()
  seed: int = 1


def test_cli_switches_union_variant():
  cfg = hp.load(Cfg, hp.cli(['--optimizer.name', 'sgd', '--optimizer.momentum', '0.8']))
  assert isinstance(cfg.optimizer, SGD)
  assert cfg.optimizer.momentum == 0.8
  assert cfg.optimizer.lr == 1e-2  # SGD's default, not AdamW's


def test_union_switch_is_order_independent():
  first = hp.load(Cfg, hp.cli(['--optimizer.name', 'sgd', '--optimizer.momentum', '0.8']))
  second = hp.load(Cfg, hp.cli(['--optimizer.momentum', '0.8', '--optimizer.name', 'sgd']))
  assert hp.to_dict(first) == hp.to_dict(second)


def test_union_survives_a_round_trip(tmp_path):
  cfg = hp.load(Cfg, hp.cli(['--optimizer.name', 'sgd']))
  path = tmp_path / 'c.json'
  hp.save(cfg, path)
  reloaded = hp.load(Cfg, path)
  assert isinstance(reloaded.optimizer, SGD)
  assert hp.to_dict(reloaded) == hp.to_dict(cfg)


def test_explicit_values_carry_across_a_variant_switch():
  cfg = hp.load(Cfg, hp.cli(['--optimizer.lr', '5e-4', '--optimizer.name', 'sgd']))
  assert isinstance(cfg.optimizer, SGD)
  assert cfg.optimizer.lr == 5e-4


def test_dotted_assignment_switches_variant():
  cfg = Cfg()
  cfg['optimizer.name'] = 'sgd'
  assert isinstance(cfg.optimizer, SGD)
  assert cfg.optimizer.lr == 1e-2


def test_partial_update_merges_into_current_variant():
  cfg = Cfg()
  hp.update(cfg, {'optimizer': {'lr': 0.5}})
  assert isinstance(cfg.optimizer, AdamW)
  assert cfg.optimizer.lr == 0.5


def test_env_reads_nested_paths_and_ignores_ambient(monkeypatch):
  monkeypatch.setenv('APP__OPTIMIZER__LR', '0.5')
  monkeypatch.setenv('SOMETHING_UNRELATED', 'x')
  cfg = hp.load(Cfg, hp.env('APP'))
  assert cfg.optimizer.lr == 0.5
  assert 'something_unrelated' not in hp.to_dict(cfg)


def test_env_honors_explicit_field_env(monkeypatch):
  class Svc(hp.Params):
    token: str = hp.Field(default='', env='SERVICE_TOKEN')

  monkeypatch.setenv('SERVICE_TOKEN', 'abc')
  assert hp.load(Svc, hp.env).token == 'abc'


def test_provenance_records_the_winning_layer():
  cfg = hp.load(Cfg, hp.cli(['--optimizer.name', 'sgd']))
  assert hp.source(cfg, 'optimizer.name') == 'cli'
  assert hp.source(cfg, 'seed') == 'default'
  assert hp.sources(cfg)['optimizer.lr'] == 'default'


def test_layered_composition_later_wins(tmp_path, monkeypatch):
  base = tmp_path / 'base.json'
  base.write_text('{"seed": 7}')
  monkeypatch.setenv('L__OPTIMIZER__LR', '0.3')

  cfg = hp.load(Cfg, base, ('env', 'L'))
  assert cfg.seed == 7
  assert cfg.optimizer.lr == 0.3
  assert hp.source(cfg, 'seed') == str(base)
  assert hp.source(cfg, 'optimizer.lr') == 'env'


def test_alias_resolves_on_cli_and_update():
  class P(hp.Params):
    weight_decay: float = hp.Field(default=0.01, alias='wd')

  assert hp.load(P, hp.cli(['--wd', '0.5'])).weight_decay == 0.5
  assert hp.update(P(), {'wd': 0.2}).weight_decay == 0.2


def test_help_lists_fields_and_exits(capsys):
  class P(hp.Params):
    seed: int = hp.Field(default=42, help='random seed')
    token: str = hp.Field(default='shh', secret=True)

  with pytest.raises(SystemExit) as excinfo:
    hp.load(P, hp.cli(['--help']))
  assert excinfo.value.code == 0

  out = capsys.readouterr().out
  assert '--seed INT' in out
  assert 'random seed' in out
  assert 'default: 42' in out
  assert 'shh' not in out  # secrets are not printed


def test_conditional_fields_gate_the_search_space():
  class RL(hp.Params):
    method: str = hp.Choice(('sft', 'grpo'), default='grpo')
    group_size: int = hp.Choice((4, 8), default=8, when=lambda root: root.rl.method == 'grpo')

  class Root(hp.Params):
    rl: RL = RL()

  cfg = Root()
  assert 'rl.group_size' in hp.space(cfg)
  cfg.rl.method = 'sft'
  assert 'rl.group_size' not in hp.space(cfg)
  assert 'rl.group_size' in hp.space(cfg, active_only=False)


def test_sampling_settles_inactive_conditionals():
  class RL(hp.Params):
    method: str = hp.Choice(('sft', 'grpo'), default='grpo')
    group_size: int = hp.Choice((4, 16), default=8, when=lambda root: root.rl.method == 'grpo')

  class Root(hp.Params):
    rl: RL = RL()

  for candidate in hp.samples(Root(), 30, seed=0):
    if candidate.rl.method != 'grpo':
      assert candidate.rl.group_size == 8  # reverted to the default


def test_grid_dedupes_collapsed_conditionals():
  class G(hp.Params):
    mode: str = hp.Choice(('on', 'off'), default='on')
    detail: int = hp.Choice((1, 2, 3), default=1, when=lambda root: root.mode == 'on')

  assert len(list(hp.grid(G()))) == 4  # on x 3 + off x 1


def test_params_are_hashable():
  assert len({Cfg(), Cfg(), Cfg(seed=2)}) == 2


def test_unknown_flag_still_warns_after_union_resolution():
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    hp.load(Cfg, hp.cli(['--optimizer.name', 'sgd', '--optimizer.nope', '1']))
  assert any(issubclass(w.category, hp.UnknownParam) for w in caught)


class Agent(hp.Params):
  system_prompt: str = hp.Evolve('You are a data agent.', description='Main behavior')
  sql_tool: str = hp.Evolve('Run read-only SQL.', group='tools')
  model: str = 'Qwen3-4B'


class AgentCfg(hp.Params):
  agent: Agent = Agent()


def test_evolvable_exposes_only_marked_text_fields():
  candidate = hp.evolvable(AgentCfg())
  assert set(candidate) == {'agent.system_prompt', 'agent.sql_tool'}
  assert candidate['agent.system_prompt'] == 'You are a data agent.'


def test_evolvable_filters_by_group():
  assert list(hp.evolvable(AgentCfg(), group='tools')) == ['agent.sql_tool']


def test_evolve_fields_are_not_searchable():
  assert hp.space(AgentCfg()) == {}


def test_apply_candidate_forks_and_records_source():
  original = AgentCfg()
  best = hp.apply_candidate(original, {'agent.system_prompt': 'IMPROVED'})
  assert best.agent.system_prompt == 'IMPROVED'
  assert original.agent.system_prompt == 'You are a data agent.'
  assert hp.source(best, 'agent.system_prompt') == 'candidate'


class Secretive(hp.Params):
  lr: float = 0.1
  api_key: str = hp.Field(default='sk-live-abc', secret=True)
  handle: object = object()


def test_flatten_omits_secrets_by_default():
  # a logger flattening for columnar storage must not write the key out
  assert 'api_key' not in hp.flatten(Secretive())
  assert 'api_key' in hp.flatten(Secretive(), secrets=True)


def test_serializable_stringifies_unloggable_values():
  import json

  values = hp.serializable(Secretive())
  assert values['lr'] == 0.1
  assert values['handle'] == 'object'      # type name, not the object
  assert 'api_key' not in values
  json.dumps(values)                        # writable to json/parquet


def test_serializable_recurses():
  class Outer(hp.Params):
    inner: Secretive = Secretive()
    items: list = [1, object()]

  values = hp.serializable(Outer())
  assert values['inner']['lr'] == 0.1
  assert 'api_key' not in values['inner']
  assert values['items'] == [1, 'object']


def test_asdict_duck_type_hook():
  # tools that already read NamedTuples pick params up for free
  assert Secretive()._asdict()['lr'] == 0.1
