"""Derived and computed values."""

import pytest

import hp


class Batch(hp.Params):
  micro: int = 2
  accum: int = 8
  world_size: int = 1

  global_batch = hp.Derived(lambda p: p.micro * p.accum * p.world_size)

  @hp.derived
  def steps_per_sample(self):
    """Optimizer steps per sample seen."""
    return 1 / (self.micro * self.accum)


def test_derived_recomputes():
  batch = Batch()
  assert batch.global_batch == 16
  batch.accum = 16
  assert batch.global_batch == 32


def test_derived_decorator_and_docstring_help():
  batch = Batch()
  assert batch.steps_per_sample == 1 / 16
  assert hp.fields(batch)['steps_per_sample'].help == 'Optimizer steps per sample seen.'


def test_derived_cannot_be_set():
  with pytest.raises(AttributeError, match='derived'):
    Batch().global_batch = 5


def test_derived_serializes_but_is_not_searchable():
  class Search(hp.Params):
    micro: int = hp.Choice((1, 2, 4), default=2)
    accum: int = 8
    global_batch = hp.Derived(lambda p: p.micro * p.accum)

  assert 'global_batch' in hp.to_dict(Search())
  assert 'global_batch' not in hp.space(Search())  # never a search dimension


def test_derived_reports_its_dependencies():
  assert hp.fields(Batch())['global_batch'].dependencies(Batch()) == {
    'micro',
    'accum',
    'world_size',
  }


def test_computed_is_evaluated_once():
  counter = {'n': 0}

  def stamp(params):
    counter['n'] += 1
    return counter['n']

  class Run(hp.Params):
    created = hp.Computed(stamp)

  run = Run()
  assert run.created == run.created == 1
  assert counter['n'] == 1

  assert Run().created == 2  # per instance, not per class


def test_computed_survives_a_fork():
  class Run(hp.Params):
    seed: int = 1

    @hp.computed
    def run_id(self):
      return f'run-{id(self)}'

  run = Run()
  first = run.run_id
  assert run.run_id == first
