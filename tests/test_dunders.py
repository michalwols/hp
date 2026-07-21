"""Params behaves like a value object and a read-only mapping."""

import copy
import pickle

import hp


class Inner(hp.Params):
  depth: int = 2


class Cfg(hp.Params):
  seed: int = 42
  lr: float = 3e-4
  token: str = hp.Field(default='sk-secret', secret=True)
  inner: Inner = Inner()


def test_truthiness_does_not_follow_length():
  class Empty(hp.Params):
    pass

  # without __bool__, __len__ makes an empty config falsy and `if config:`
  # silently takes the wrong branch
  assert bool(Empty()) is True
  assert bool(Cfg()) is True


def test_mapping_reads():
  config = Cfg()
  assert config['seed'] == 42
  assert config['inner.depth'] == 2
  assert 'seed' in config
  assert 'nope' not in config
  assert len(config) == 4
  assert sorted(config) == ['inner', 'lr', 'seed', 'token']


def test_merge_operators():
  config = Cfg()

  merged = config | {'seed': 7}
  assert merged.seed == 7
  assert config.seed == 42          # not mutated

  merged = config | Cfg(seed=8)
  assert merged.seed == 8

  config |= {'seed': 9}
  assert config.seed == 9           # in place

  assert config.__or__(3) is NotImplemented


def test_equality_and_hashing():
  assert Cfg() == Cfg()
  assert Cfg() != Cfg(seed=1)
  assert Cfg().__eq__(3) is NotImplemented
  assert len({Cfg(), Cfg(), Cfg(seed=1)}) == 2


def test_str_is_readable_and_annotated():
  config = hp.load(Cfg, hp.cli(['--lr', '0.01']))
  text = str(config)

  assert text.splitlines()[0] == 'Cfg('
  assert 'inner.depth' in text          # flattened paths
  assert '# cli' in text                # provenance annotated
  assert 'sk-secret' not in text        # secrets never printed


def test_repr_is_one_line():
  assert repr(Cfg()).startswith('Cfg(seed=42,')
  assert '\n' not in repr(Cfg())


def test_format_spec_selects_the_long_form():
  config = Cfg()
  assert f'{config}' == repr(config)
  assert f'{config:v}' == str(config)


def test_html_repr_for_notebooks():
  html = Cfg()._repr_html_()
  assert '<table>' in html
  assert 'inner.depth' in html
  assert 'sk-secret' not in html        # secrets never rendered


def test_rich_repr():
  assert dict(Cfg().__rich_repr__())['seed'] == 42


def test_dir_lists_fields():
  assert {'seed', 'lr', 'inner'} <= set(dir(Cfg()))


def test_copy_and_pickle_round_trip():
  config = Cfg(seed=5)
  assert copy.copy(config).seed == 5
  assert copy.deepcopy(config).inner.depth == 2
  assert pickle.loads(pickle.dumps(config)) == config
