import hp
import pytest


def test_constructor():
  class Params(hp.HyperParams):
    """
    Description
    """
    name: str
    batch_size = 32

  assert 'name' in Params.__fields__
  assert 'batch_size' in Params.__fields__


  params = Params()
  assert params.batch_size == 32
  assert not hasattr(params, 'name')

  params = Params(name='foo', batch_size=8)
  assert params.batch_size == 8
  assert params.name == 'foo'


def test_command_line_parsing():

  class Params(hp.HyperParams):
    """Description"""
    name: str
    batch_size = 32



  parser = Params.arg_parser()

  assert parser.description == 'Description'

  params = Params.from_command('-bs 10 -n foo')

  assert params.name == 'foo'
  assert params.batch_size == 10


  with pytest.raises(SystemExit):
    params = Params.from_command('-bs 10 -n foo -f invalid_arg')


def test_json(tmpdir):
  path = tmpdir / 'params.json'

  class Params(hp.HyperParams):
    """Description"""
    name: str
    batch_size = 32

  params = Params(name='foo', batch_size=8)

  params.save(str(path))
  assert path.exists()

  p = Params.load(str(path))

  assert p.name == 'foo'
  assert p.batch_size == 8