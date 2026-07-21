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

params = hp.from_command(TrainParams)
hp.freeze(params)
```

`import hp` is all you need — everything hangs off the module.

## Params has no methods

Operations are module-level functions taking the params first, so that **every
attribute name stays available for your fields**:

```python
class TrainParams(hp.Params):
  sample: int = 4      # would collide with a .sample() method
  freeze: bool = True
  update: str = 'ema'
  values: list = []

hp.to_dict(params)
hp.fork(params, seed=7)
hp.space(params)
hp.from_command(TrainParams)
```

`hp.Params` intentionally has an empty public namespace. Attribute access
(`params.lr`), item access (`params['optim.lr']`), `in`, `len()` and iteration
all still work — they are dunders, so they cost you no names.

## Nested parameters

```python
params.optimizer.lr = 1e-4
params['optimizer.lr'] = 5e-5   # dotted paths create intermediate nodes
```

## Writes are open, reads are strict

Setting a name that was never declared is allowed, and it becomes a real field —
so it round-trips through `hp.to_dict()` and `hp.save()` like any other:

```python
params.notes = 'sweep A'
hp.save(params, 'config.json')   # notes is in there
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
about, never silently dropped. Use `hp.update(params, data, strict=True)` to turn
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
params = hp.from_command(TrainParams)                      # sys.argv
params = hp.from_command(TrainParams, '--optimizer.lr 1e-4') # or a string / argv list
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
(also accepted by `hp.update()`, which makes it a migration path for renamed fields), and
`secret=True` fields never print their value.

## Layering and provenance

Sources compose in order, later ones winning:

```python
params = hp.layered(TrainParams, 'base.yaml', 'experiment.yaml', ('env', 'APP'), 'cli')
```

Every value remembers which layer set it:

```python
hp.source(params, 'optim.lr')   # 'experiment.yaml'
hp.sources(params)          # {'seed': 'default', 'optim.lr': 'experiment.yaml', ...}
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
tree. `hp.space()` hides inactive fields, `hp.sample()` settles them back to their defaults
once the fields they depend on are drawn, and `hp.grid()` drops the duplicate
configurations that conditions collapse together.

## Parametrized callables

`parametrize` turns a signature into params and supplies them at call time.
Arguments you pass explicitly win, and are written back:

```python
@hp.parametrize
def train(epochs: int = 10, lr: float = 2e-4):
  ...

hp.params(train).lr = 1e-4
train()            # uses lr=1e-4
train(lr=1e-3)     # explicit wins, and params.lr becomes 1e-3
```

`track` only observes — it records the target's name, a reference to it, and the
arguments of every call, without injecting anything or mutating params:

```python
@hp.track
def evaluate(threshold: float = 0.5):
  ...

evaluate(threshold=0.7)
hp.calls(evaluate)   # [{'threshold': 0.7}]
```

Both register under a name, so the whole program's configuration surface is
reachable from one place:

```python
hp.registry()        # {'train': Entry(...), 'evaluate': Entry(...)}
hp.params('train')   # by name as well as by reference
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
for candidate in hp.samples(params, 20, seed=1):
  train(candidate)
```

Optional Optuna integration is available in `hp.optimize.optuna`.

## Run tooling

```python
hp.stable_hash(params)        # content-addressed id for a config
hp.diff(params, other)        # {path: (mine, theirs)} for changed values
hp.fork(params, seed=7)       # copy with updates
hp.on_change(params, fn)      # (params, name, old, new) on every set
hp.save(params, 'config.yaml')  # json / yaml
```
