# hp

A small Python-native package for configuration, callable binding, nested parameter trees, and hyperparameter search.

```python
from typing import Literal
import hp

class AdamWParams(hp.Params):
  method: Literal['adamw'] = 'adamw'
  lr: float = hp.LogRange(1e-6, 1e-3, default=2e-4)
  weight_decay: float = 0.01

class TrainParams(hp.Params):
  seed: int = 42
  optimizer: AdamWParams = AdamWParams()

params = TrainParams.from_command()
params.freeze()
```

`import hp` is all you need — everything hangs off the module.

## Nested parameters

```python
params.optimizer.lr = 1e-4
params['optimizer.lr'] = 5e-5   # dotted paths create intermediate nodes
```

## Writes are open, reads are strict

Setting a name that was never declared is allowed, and it becomes a real field —
so it round-trips through `to_dict()` and `save()` like any other:

```python
params.notes = 'sweep A'
params.save('config.json')   # notes is in there
```

Reading a name that was never set still raises, which is what catches typos:

```python
params.optimzer   # AttributeError, rather than a silent empty value
```

That asymmetry is deliberate. A typo'd read is the dangerous case — without it
`if params.use_amp:` would quietly evaluate to false forever. A typo'd write only
adds an unused key, so it warns instead:

```python
TrainParams(sed=1)
# UnknownParam: TrainParams has no declared field 'sed'; did you mean 'seed'?
```

Command line flags behave the same way — an unrecognized flag is kept and warned
about, never silently dropped. Use `params.update(data, strict=True)` to turn
unknown keys into a `KeyError` instead, and filter or raise on `hp.UnknownParam`
to tune how loud it is.

## Dynamic trees

`hp.Dynamic` additionally auto-vivifies on read, so intermediate nodes need no
declaration and nothing warns:

```python
config = hp.Dynamic()
config.opt.foo = 5
config.rollout.temperature = 0.8
```

A declared class can opt into the same behavior:

```python
class OpenParams(hp.Params, dynamic=True):
  seed: int = 42
```

## Command line

```python
params = TrainParams.from_command()                      # sys.argv
params = TrainParams.from_command('--optimizer.lr 1e-4') # or a string / argv list
```

Dashed flags map onto underscore field paths, so `--weight-decay=0.1` and
`--weight_decay=0.1` are equivalent. Booleans accept `--debug` and `--no-debug`.

## Bind, watch, and wrap

```python
params = TrainParams()

@params.bind
def train(seed=8):
  return seed

train()  # 42; explicit calls do not mutate params
```

```python
@params.watch
def train(seed=8):
  return seed

train(seed=64)
assert params.seed == 64
```

```python
@params.wrap
def train(seed=8):
  return seed

train()          # reads 64 from params
train(seed=128)  # updates params and calls with 128
```

Function-first usage, where the schema comes from the signature:

```python
@hp.wrap
def train(epochs: int = 10, lr: float = 2e-4):
  ...

train.hp.lr = 1e-4
train()
```

## Callable schemas

```python
def train(epochs: int = 10, lr: float = 2e-4): ...

TrainParams = hp.schema(train)
params = TrainParams(lr=1e-4)
```

## Search spaces

```python
class SearchParams(hp.Params):
  lr: float = hp.LogRange(1e-6, 1e-3, default=2e-4)
  batch_size: int = hp.Choice((2, 4, 8), default=4)

params = SearchParams()
for candidate in params.samples(20, seed=1):
  train(candidate)
```

Optional Optuna integration is available in `hp.optimize.optuna`.

## Run tooling

```python
params.stable_hash()        # content-addressed id for a config
params.diff(other)          # {path: (mine, theirs)} for changed values
params.fork(seed=7)         # copy with updates
params.on_change(callback)  # (params, name, old, new) on every set
params.save('config.yaml')  # json / yaml
```

`hp.Params` is the canonical base class; `hp.HP` and `hp.HyperParams` remain as aliases.
