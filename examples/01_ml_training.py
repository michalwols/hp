"""A training config you can reproduce, audit and sweep. RUNNABLE.

Shows the parts of hp that already exist: layered loading, provenance,
content-addressed run ids, and a search space.
"""

import hp


class Optim(hp.Params):
  name: str = hp.Choice(('adamw', 'sgd'))
  lr: float = hp.LogRange(1e-6, 1e-2, default=3e-4, help='peak learning rate')
  weight_decay: float = 0.01


class Train(hp.Params):
  seed: int = 42
  epochs: int = 10
  micro_batch: int = hp.Choice((8, 16, 32), default=16)
  accum: int = hp.IntRange(1, 32, default=4)
  optim: Optim = Optim()

  # derived, so the sweep never searches batch size twice
  global_batch = hp.Derived(lambda p: p.micro_batch * p.accum)


def main() -> None:
  # later sources win; every value remembers which one set it
  config = hp.load(Train, {'seed': 7}, hp.cli(['--optim.lr', '1e-4']))

  print('config     ', hp.to_dict(config))
  print('provenance ', hp.sources(config))
  print('run id     ', hp.stable_hash(config))
  print('global batch', config.global_batch, '(derived)')

  # the same class is the search space
  print('search dims', sorted(hp.space(config)))
  for candidate in hp.samples(config, 3, seed=0):
    print('  trial', hp.stable_hash(candidate), hp.to_dict(candidate)['optim']['lr'])

  # freezing makes accidental mutation mid-run an error
  hp.freeze(config)
  try:
    config.seed = 1
  except TypeError as error:
    print('frozen     ', error)


if __name__ == '__main__':
  main()
