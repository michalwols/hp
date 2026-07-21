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

## Typed unions

A field typed as a union of `Params` classes resolves to the variant matching its
discriminator — any field the variants share as a distinct `Literal`:

```python
class AdamW(hp.Params):
  name: Literal['adamw'] = 'adamw'
  lr: float = 2e-4

class SGD(hp.Params):
  name: Literal['sgd'] = 'sgd'
  lr: float = 1e-2
  momentum: float = 0.9

class TrainParams(hp.Params):
  optimizer: AdamW | SGD = AdamW()
```

Selecting a variant works the same from a dict, a config file, or the command line:

```
--optimizer.name sgd --optimizer.momentum 0.8
```

Flags are applied together rather than one at a time, so the result does not depend
on their order. Values you set explicitly carry across a switch; the outgoing
variant's own defaults do not.

## Command line

```python
params = TrainParams.from_command()                      # sys.argv
params = TrainParams.from_command('--optimizer.lr 1e-4') # or a string / argv list
```

Dashed flags map onto underscore field paths, so `--weight-decay=0.1` and
`--weight_decay=0.1` are equivalent. Booleans accept `--debug` and `--no-debug`.
`--help` renders from the field tree:

```
Usage: train.py [OPTIONS]

Options:
  --seed INT                        random seed (default: 42)
  --debug / --no-debug              (default: False)
  --method {sft,grpo}               training method (default: 'sft')
  --optim.lr [1e-06..0.001]         peak learning rate (default: 0.0002)
  --optim.weight-decay FLOAT, --wd  AdamW weight decay (default: 0.01)
  --help, -h                        Show this message and exit.
```

`Field(help=...)` supplies the description, `Field(alias='wd')` adds a second spelling
(also accepted by `update()`, which makes it a migration path for renamed fields), and
`secret=True` fields never print their value.

## Layering and provenance

Sources compose in order, later ones winning:

```python
params = TrainParams.layered('base.yaml', 'experiment.yaml', ('env', 'APP'), 'cli')
```

Every value remembers which layer set it:

```python
params.source('optim.lr')   # 'experiment.yaml'
params.sources()            # {'seed': 'default', 'optim.lr': 'experiment.yaml', ...}
```

Environment variables map `APP__OPTIM__LR` onto `optim.lr`. Only declared fields are
read — the environment is ambient, so unrecognized names are ignored rather than
becoming config — and `Field(env='SERVICE_TOKEN')` binds an explicit name.

## Conditional fields

`when` gates whether a field participates in a search space, which keeps
mutually-irrelevant options out of a sweep:

```python
class RLParams(hp.Params):
  method: str = hp.Choice(('sft', 'grpo'), default='grpo')
  group_size: int = hp.Choice((4, 8, 16), default=8,
                              when=lambda root: root.rl.method == 'grpo')
```

The callable receives the root params, so conditions can reference anything in the
tree. `space()` hides inactive fields, `sample()` settles them back to their defaults
once the fields they depend on are drawn, and `grid()` drops the duplicate
configurations that conditions collapse together.

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

TrainParams = hp.schema(train)   # -> class TrainParams
params = TrainParams(lr=1e-4)
```

Pass `base=` to inherit shared fields, and `name=` to control the class name:

```python
class Shared(hp.Params):
  seed: int = 42

TrainParams = hp.schema(train, base=Shared)
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
