import hp


class OptimParams(hp.Params):
  lr: float = 2e-4
  weight_decay: float = 0.01


class TrainParams(hp.Params):
  debug: bool = False
  optim: OptimParams = OptimParams()


def test_cli():
  p = TrainParams.from_command(['--optim.lr', '0.001', '--debug'])
  assert p.optim.lr == 0.001
  assert p.debug is True


def test_json_round_trip(tmp_path):
  path = tmp_path / 'config.json'
  TrainParams().save(path)
  p = TrainParams.load(path)
  assert p.optim.lr == 2e-4


def test_from_command_string_and_dashes():
  p = TrainParams.from_command('--optim.lr 0.001 --debug')
  assert p.optim.lr == 0.001
  assert p.debug is True

  p = TrainParams.from_command(['--optim.weight-decay=0.1'])
  assert p.optim.weight_decay == 0.1
