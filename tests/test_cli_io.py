from hp import HP


class OptimHP(HP):
  lr: float = 2e-4


class Params(HP):
  debug: bool = False
  optim: OptimHP = OptimHP()


def test_cli():
  p = Params.from_cli(['--optim.lr', '0.001', '--debug'])
  assert p.optim.lr == 0.001
  assert p.debug is True


def test_json_round_trip(tmp_path):
  path = tmp_path / 'config.json'
  Params().save(path)
  p = Params.load(path)
  assert p.optim.lr == 2e-4
