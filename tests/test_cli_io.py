import hp


class OptimParams(hp.Params):
  lr: float = 2e-4
  weight_decay: float = 0.01


class TrainParams(hp.Params):
  debug: bool = False
  optim: OptimParams = OptimParams()


def test_cli():
  p = hp.load(TrainParams, hp.cli(['--optim.lr', '0.001', '--debug']))
  assert p.optim.lr == 0.001
  assert p.debug is True


def test_json_round_trip(tmp_path):
  path = tmp_path / 'config.json'
  hp.save(TrainParams(), path)
  p = hp.load(TrainParams, path)
  assert p.optim.lr == 2e-4


def test_from_command_string_and_dashes():
  p = hp.load(TrainParams, hp.cli('--optim.lr 0.001 --debug'))
  assert p.optim.lr == 0.001
  assert p.debug is True

  p = hp.load(TrainParams, hp.cli(['--optim.weight-decay=0.1']))
  assert p.optim.weight_decay == 0.1


def test_unknown_flag_warns_but_is_kept():
  import pytest

  with pytest.warns(hp.UnknownParam):
    p = hp.load(TrainParams, hp.cli(['--lr', '0.5']))

  assert p.lr == 0.5
  assert p.optim.lr == 2e-4  # the declared field is untouched
  assert hp.to_dict(p)['lr'] == 0.5


def test_known_flags_do_not_warn():
  import warnings

  with warnings.catch_warnings():
    warnings.simplefilter('error')
    p = hp.load(TrainParams, hp.cli(['--optim.lr', '0.001', '--debug']))
  assert p.optim.lr == 0.001
