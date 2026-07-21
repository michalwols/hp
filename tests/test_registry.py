"""parametrize, track, and the process-wide registry."""

import pytest

import hp


@pytest.fixture(autouse=True)
def clean_registry():
  hp.params.clear()
  yield
  hp.params.clear()


def test_parametrize_reads_and_writes_params():
  @hp.params
  def train(epochs: int = 10, lr: float = 2e-4):
    return epochs, lr

  assert train() == (10, 2e-4)
  assert train(lr=1e-3) == (10, 1e-3)
  assert hp.params(train).lr == 1e-3
  assert train() == (10, 1e-3)


def test_parametrize_accepts_existing_params():
  class Shared(hp.Params):
    epochs: int = 3

  shared = Shared()

  @hp.params(params=shared)
  def train(epochs: int = 10):
    return epochs

  assert train() == 3
  assert hp.params(train) is shared


def test_parametrize_can_record_calls():
  @hp.params(record=True)
  def train(epochs: int = 10):
    return epochs

  train()
  train(epochs=2)
  assert len(hp.params.calls(train)) == 2


def test_track_records_without_mutating():
  @hp.track
  def train(epochs: int = 10, lr: float = 2e-4):
    return epochs, lr

  assert train(lr=1e-3) == (10, 1e-3)
  assert train(5) == (5, 2e-4)

  assert hp.params.calls(train) == [{'lr': 1e-3}, {'epochs': 5}]
  assert hp.params(train).lr == 2e-4


def test_track_keeps_a_reference_to_the_target():
  def train(epochs: int = 10):
    return epochs

  tracked = hp.track(train)
  tracked()

  record = hp.params.entry(tracked)
  assert record.target is train
  assert record.mode == 'track'


def test_registry_is_keyed_by_name():
  @hp.params(name='alpha')
  def one(a: int = 1):
    return a

  @hp.track(name='beta')
  def two(b: int = 2):
    return b

  assert list(hp.params.registry) == ['alpha', 'beta']
  assert hp.params('alpha').a == 1
  assert hp.params.entry('beta').mode == 'track'


def test_unregistered_callable_is_decorated():
  # hp.params dispatches: an unregistered callable gets parametrized
  def plain(a: int = 1):
    return a

  wrapped = hp.params(plain)
  assert wrapped() == 1
  assert hp.params(wrapped).a == 1


def test_unknown_name_raises():
  with pytest.raises(KeyError, match='nothing registered'):
    hp.params('no-such-target')


def test_parametrize_works_on_classes():
  @hp.params
  class Model:
    def __init__(self, width: int = 8):
      self.width = width

  assert Model().width == 8
  hp.params(Model).width = 16
  assert Model().width == 16
