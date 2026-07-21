"""Union switching, provenance, layering, conditionals, and CLI help."""

import os
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
  cfg = Cfg.from_command(['--optimizer.name', 'sgd', '--optimizer.momentum', '0.8'])
  assert isinstance(cfg.optimizer, SGD)
  assert cfg.optimizer.momentum == 0.8
  assert cfg.optimizer.lr == 1e-2  # SGD's default, not AdamW's


def test_union_switch_is_order_independent():
  first = Cfg.from_command(['--optimizer.name', 'sgd', '--optimizer.momentum', '0.8'])
  second = Cfg.from_command(['--optimizer.momentum', '0.8', '--optimizer.name', 'sgd'])
  assert first.to_dict() == second.to_dict()


def test_union_survives_a_round_trip(tmp_path):
  cfg = Cfg.from_command(['--optimizer.name', 'sgd'])
  path = tmp_path / 'c.json'
  cfg.save(path)
  reloaded = Cfg.load(path)
  assert isinstance(reloaded.optimizer, SGD)
  assert reloaded.to_dict() == cfg.to_dict()


def test_explicit_values_carry_across_a_variant_switch():
  cfg = Cfg.from_command(['--optimizer.lr', '5e-4', '--optimizer.name', 'sgd'])
  assert isinstance(cfg.optimizer, SGD)
  assert cfg.optimizer.lr == 5e-4


def test_dotted_assignment_switches_variant():
  cfg = Cfg()
  cfg['optimizer.name'] = 'sgd'
  assert isinstance(cfg.optimizer, SGD)
  assert cfg.optimizer.lr == 1e-2


def test_partial_update_merges_into_current_variant():
  cfg = Cfg()
  cfg.update({'optimizer': {'lr': 0.5}})
  assert isinstance(cfg.optimizer, AdamW)
  assert cfg.optimizer.lr == 0.5


def test_env_reads_nested_paths_and_ignores_ambient(monkeypatch):
  monkeypatch.setenv('APP__OPTIMIZER__LR', '0.5')
  monkeypatch.setenv('SOMETHING_UNRELATED', 'x')
  cfg = Cfg.from_env('APP')
  assert cfg.optimizer.lr == 0.5
  assert 'something_unrelated' not in cfg.to_dict()


def test_env_honors_explicit_field_env(monkeypatch):
  class Svc(hp.Params):
    token: str = hp.Field(default='', env='SERVICE_TOKEN')

  monkeypatch.setenv('SERVICE_TOKEN', 'abc')
  assert Svc.from_env().token == 'abc'


def test_provenance_records_the_winning_layer():
  cfg = Cfg.from_command(['--optimizer.name', 'sgd'])
  assert cfg.source('optimizer.name') == 'cli'
  assert cfg.source('seed') == 'default'
  assert cfg.sources()['optimizer.lr'] == 'default'


def test_layered_composition_later_wins(tmp_path, monkeypatch):
  base = tmp_path / 'base.json'
  base.write_text('{"seed": 7}')
  monkeypatch.setenv('L__OPTIMIZER__LR', '0.3')

  cfg = Cfg.layered(base, ('env', 'L'))
  assert cfg.seed == 7
  assert cfg.optimizer.lr == 0.3
  assert cfg.source('seed') == str(base)
  assert cfg.source('optimizer.lr') == 'env'


def test_alias_resolves_on_cli_and_update():
  class P(hp.Params):
    weight_decay: float = hp.Field(default=0.01, alias='wd')

  assert P.from_command(['--wd', '0.5']).weight_decay == 0.5
  assert P().update({'wd': 0.2}).weight_decay == 0.2


def test_help_lists_fields_and_exits(capsys):
  class P(hp.Params):
    seed: int = hp.Field(default=42, help='random seed')
    token: str = hp.Field(default='shh', secret=True)

  with pytest.raises(SystemExit) as excinfo:
    P.from_command(['--help'])
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
  assert 'rl.group_size' in cfg.space()
  cfg.rl.method = 'sft'
  assert 'rl.group_size' not in cfg.space()
  assert 'rl.group_size' in cfg.space(active_only=False)


def test_sampling_settles_inactive_conditionals():
  class RL(hp.Params):
    method: str = hp.Choice(('sft', 'grpo'), default='grpo')
    group_size: int = hp.Choice((4, 16), default=8, when=lambda root: root.rl.method == 'grpo')

  class Root(hp.Params):
    rl: RL = RL()

  for candidate in Root().samples(30, seed=0):
    if candidate.rl.method != 'grpo':
      assert candidate.rl.group_size == 8  # reverted to the default


def test_grid_dedupes_collapsed_conditionals():
  class G(hp.Params):
    mode: str = hp.Choice(('on', 'off'), default='on')
    detail: int = hp.Choice((1, 2, 3), default=1, when=lambda root: root.mode == 'on')

  assert len(list(G().grid())) == 4  # on x 3 + off x 1


def test_params_are_hashable():
  assert len({Cfg(), Cfg(), Cfg(seed=2)}) == 2


def test_unknown_flag_still_warns_after_union_resolution():
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    Cfg.from_command(['--optimizer.name', 'sgd', '--optimizer.nope', '1'])
  assert any(issubclass(w.category, hp.UnknownParam) for w in caught)
