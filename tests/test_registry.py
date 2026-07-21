"""parametrize, track, and the process-wide registry."""

import pytest

import hp


@pytest.fixture(autouse=True)
def clean_registry():
  hp.params.clear()
  yield
  hp.params.clear()


def test_parametrize_reads_and_writes_params():
  @hp.wrap
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

  @hp.wrap(params=shared)
  def train(epochs: int = 10):
    return epochs

  assert train() == 3
  assert hp.params(train) is shared


def test_parametrize_can_record_calls():
  @hp.wrap(record=True)
  def train(epochs: int = 10):
    return epochs

  train()
  train(epochs=2)
  assert len(hp.params.history(train)) == 2


def test_track_records_without_mutating():
  @hp.track
  def train(epochs: int = 10, lr: float = 2e-4):
    return epochs, lr

  assert train(lr=1e-3) == (10, 1e-3)
  assert train(5) == (5, 2e-4)

  assert hp.params.history(train) == [{'lr': 1e-3}, {'epochs': 5}]
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
  @hp.wrap(name='alpha')
  def one(a: int = 1):
    return a

  @hp.track(name='beta')
  def two(b: int = 2):
    return b

  assert list(hp.params.registry) == ['alpha', 'beta']
  assert hp.params('alpha').a == 1
  assert hp.params.entry('beta').mode == 'track'


def test_calling_params_never_decorates():
  # hp.params(x) always means "the params of x", never "wrap x"
  def plain(a: int = 1):
    return a

  wrapped = hp.wrap(plain)
  assert hp.params(wrapped).a == 1
  assert hp.params(wrapped) is hp.params.entry(wrapped).params


def test_unknown_name_raises():
  with pytest.raises(KeyError, match='nothing registered'):
    hp.params('no-such-target')


def test_parametrize_works_on_classes():
  @hp.wrap
  class Model:
    def __init__(self, width: int = 8):
      self.width = width

  assert Model().width == 8
  hp.params(Model).width = 16
  assert Model().width == 16
