"""Context-scoped configuration."""

import asyncio
import threading

import pytest

import hp


@pytest.fixture(autouse=True)
def clean():
  hp.params.clear()
  yield
  hp.params.clear()


def test_override_is_temporary():
  @hp.wrap
  def train(lr: float = 1e-3):
    return lr

  assert train() == 1e-3
  with hp.override(train, lr=1e-4):
    assert train() == 1e-4
  assert train() == 1e-3


def test_override_does_not_mutate_the_original():
  @hp.wrap
  def train(lr: float = 1e-3):
    return lr

  original = hp.params(train)
  with hp.override(train, lr=1e-4):
    assert hp.params(train).lr == 1e-3  # the registered object is untouched
  assert original.lr == 1e-3


def test_override_nests():
  @hp.wrap
  def train(lr: float = 1e-3):
    return lr

  with hp.override(train, lr=1e-4):
    with hp.override(train, lr=1e-5):
      assert train() == 1e-5
    assert train() == 1e-4


def test_scope_sets_active_params():
  class Config(hp.Params):
    seed: int = 1

  params = Config(seed=9)
  assert hp.active() is None
  with hp.scope(params) as scoped:
    assert hp.active() is scoped
    assert scoped.seed == 9
  assert hp.active() is None


def test_override_without_target_needs_a_scope():
  with pytest.raises(RuntimeError, match='active hp.scope'):
    with hp.override(lr=1e-4):
      pass


def test_override_of_the_active_scope():
  class Config(hp.Params):
    lr: float = 1e-3

  with hp.scope(Config()):
    with hp.override(lr=1e-4) as scoped:
      assert scoped.lr == 1e-4
      assert hp.active().lr == 1e-4
    assert hp.active().lr == 1e-3


def test_threads_do_not_see_each_others_overrides():
  @hp.wrap
  def train(lr: float = 1e-3):
    return lr

  seen = {}

  def worker(name, value):
    with hp.override(train, lr=value):
      seen[name] = train()

  threads = [
    threading.Thread(target=worker, args=(f't{i}', 10.0 ** -i)) for i in range(1, 4)
  ]
  for t in threads:
    t.start()
  for t in threads:
    t.join()

  assert seen == {'t1': 0.1, 't2': 0.01, 't3': 0.001}
  assert train() == 1e-3


def test_async_tasks_are_isolated():
  @hp.wrap
  def train(lr: float = 1e-3):
    return lr

  async def run(value):
    with hp.override(train, lr=value):
      await asyncio.sleep(0)
      return train()

  async def main():
    return await asyncio.gather(run(0.1), run(0.2), run(0.3))

  assert asyncio.run(main()) == [0.1, 0.2, 0.3]
