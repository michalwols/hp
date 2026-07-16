# hp

A small Python-native package for configuration, callable binding, nested parameter trees, and hyperparameter search.

```python
from typing import Literal
from hp import HP, Choice, LogRange

class AdamWHP(HP):
  method: Literal['adamw'] = 'adamw'
  lr: float = LogRange(1e-6, 1e-3, default=2e-4)
  weight_decay: float = 0.01

class TrainHP(HP):
  seed: int = 42
  optimizer: AdamWHP = AdamWHP()

p = TrainHP.from_cli()
p.freeze()
```

## Nested and dynamic parameters

Declared trees are strict:

```python
p.optimizer.lr = 1e-4
p['optimizer.lr'] = 5e-5
```

Dynamic trees auto-vivify:

```python
p = HP.dynamic()
p.opt.foo = 5
p.rollout.temperature = 0.8
```

## Bind, watch, and wrap

```python
class Params(HP):
  batch_size: int = 32

p = Params()

@p.bind
def train(batch_size=8):
  return batch_size

train()  # 32; explicit calls do not mutate p
```

```python
@p.watch
def train(batch_size=8):
  return batch_size

train(batch_size=64)
assert p.batch_size == 64
```

```python
@p.wrap
def train(batch_size=8):
  return batch_size

train()               # reads 64 from p
train(batch_size=128) # updates p and calls with 128
```

Function-first usage:

```python
import hp

@hp.wrap
def train(epochs: int = 10, lr: float = 2e-4):
  ...

train.hp.lr = 1e-4
train()
```

## Callable schemas

```python
def train(epochs: int = 10, lr: float = 2e-4): ...

TrainHP = HP.schema(train)
p = TrainHP(lr=1e-4)
```

## Search spaces

```python
class Params(HP):
  lr: float = LogRange(1e-6, 1e-3, default=2e-4)
  batch_size: int = Choice((2, 4, 8), default=4)

p = Params()
for candidate in p.samples(20, seed=1):
  train(candidate)
```

Optional Optuna integration is available in `hp.optimize.optuna`.
