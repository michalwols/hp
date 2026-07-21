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

params = hp.load(TrainParams, hp.cli)
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
```

`hp.Params` intentionally has an empty public namespace. Everything a config
object should do is a dunder, so it costs you no names:

```python
params.lr                      # attribute access
params['optim.lr']             # dotted item access
'lr' in params, len(params)    # membership and size
for name in params: ...        # iteration
params | {'lr': 1e-4}          # merge, like a dict
params == other, hash(params)  # value semantics
print(params)                  # aligned, one value per line, with provenance
```

In a notebook `params` renders as a table. `repr()` stays a single line;
`f'{params:v}'` gives the long form.

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

## Loading

One verb builds params from anything, and extra arguments are layers applied in
order:

```python
hp.load(Config)                              # defaults
hp.load(Config, 'base.yaml', 'exp.yaml')     # layered files
hp.load(Config, hp.env(prefix='APP'), hp.cli)  # env then command line
hp.load('config.yaml')                       # a bare file
hp.load(Cfg(lr=0.5))                         # a dataclass / attrs / pydantic object
hp.load(arg_parser)                          # an argparse parser
hp.load({'lr': 0.5})                         # a mapping
```

## Command line

```python
params = hp.load(Config, hp.cli)                    # sys.argv
params = hp.load(Config, hp.cli('--optim.lr 1e-4')) # or a string / argv list
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

## Derived and computed values

`Derived` recomputes from the other params on every read, so it can never
disagree with its inputs — and it is excluded from search spaces, because
searching a value alongside the inputs it is computed from wastes the sweep on
a redundant dimension:

```python
class Batch(hp.Params):
  micro: int = hp.Choice((1, 2, 4), default=2)
  accum: int = hp.IntRange(1, 32, default=8)

  global_batch = hp.Derived(lambda p: p.micro * p.accum)

  @hp.derived
  def tokens_per_step(self):
    """Sequence tokens per optimizer step."""
    return self.global_batch * 2048
```

`Computed` is evaluated once per params object and then fixed, which is what a
timestamp, run id or hostname wants:

```python
started_at = hp.Computed(lambda p: datetime.now(timezone.utc).isoformat())
```

Both serialize like ordinary fields, so a saved config records what was actually
used, and neither can be assigned. A derived field can report what it reads:

```python
hp.fields(batch)['global_batch'].dependencies(batch)   # {'micro', 'accum'}
```

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

`hp.params` is one object holding the registry, the decorators and the
instrumentation. Calling it always means the same thing — *the params of this* —
and everything else has a name:

```python
hp.params(train)          # the params of a registered target
hp.params('train')        # the same, by name
hp.params()               # the registry
hp.params['train']        # one entry, with its mode and call log

hp.params.wrap(fn)        # decorate: supply params, record explicit args
hp.params.track(fn)       # decorate: record only
hp.params.instrument(mod) # wrap a whole module

hp.params.history()       # what everything was called with
hp.params.collect()       # the whole config surface as one tree
hp.params.inject(config)  # write config back out
```

`@hp.wrap` and `@hp.track` are the same decorators at the top level.

## Collecting and injecting

`collect` gathers every registered target into one tree, optionally folding in
the environment and command line — a single object holding the program's whole
configuration:

```python
config = hp.params.collect(env=True, cli=True)
config.train.lr           # from a parametrized train()
config.torch.optim.Adam   # from an instrumented module
```

`inject` writes values back out. With no target they go to the environment,
which is how you hand a resolved config to a subprocess:

```python
hp.params.inject(config)                 # -> os.environ, TRAIN__LR=0.0001
subprocess.run(['python', 'train.py'])   # child reads it via hp.load(Cfg, hp.env)

hp.params.inject(config, globals())      # or into a namespace
```


## Examples

### A training run you can reproduce

```python
import hp

class Optim(hp.Params):
  name: str = hp.Choice(('adamw', 'sgd'))
  lr: float = hp.LogRange(1e-6, 1e-2, default=3e-4, help='peak learning rate')
  weight_decay: float = 0.01

class Train(hp.Params):
  seed: int = 42
  epochs: int = 10
  batch_size: int = hp.Choice((16, 32, 64), default=32)
  data_dir: str = hp.Field(default='./data', env='DATA_DIR')
  optim: Optim = Optim()

config = hp.load(Train, 'configs/base.yaml', hp.env(prefix='TRAIN'), hp.cli)
hp.validate(config)

run_id = hp.stable_hash(config)          # identical config -> identical id
hp.save(config, f'runs/{run_id}/config.yaml')
hp.freeze(config)                        # nothing mutates it mid-run

hp.sources(config)
# {'seed': 'configs/base.yaml', 'optim.lr': 'cli', 'data_dir': 'env', ...}
```

When a run misbehaves, `hp.sources` answers *where did this value come from* —
usually faster than reading four config layers by hand.

### Swapping optimizers without branching

```python
from typing import Literal

class AdamW(hp.Params):
  name: Literal['adamw'] = 'adamw'
  lr: float = 3e-4
  weight_decay: float = 0.01

class SGD(hp.Params):
  name: Literal['sgd'] = 'sgd'
  lr: float = 0.1
  momentum: float = 0.9
  nesterov: bool = True

class Train(hp.Params):
  optim: AdamW | SGD = AdamW()

config = hp.load(Train, hp.cli)   # --optim.name sgd --optim.momentum 0.95
```

`config.optim` is now an `SGD`, with `momentum` available and `weight_decay`
gone. Each optimizer declares only the arguments it actually takes, and
`hp.construct` passes only what the constructor accepts:

```python
IMPL = {'adamw': torch.optim.AdamW, 'sgd': torch.optim.SGD}
optimizer = hp.construct(config.optim, IMPL[config.optim.name], params=model.parameters())
```

### A sweep where some options only apply sometimes

```python
class RL(hp.Params):
  method: str = hp.Choice(('sft', 'dpo', 'grpo'))
  kl_coef: float = hp.LogRange(1e-4, 0.2, default=0.02,
                               when=lambda root: root.rl.method in {'dpo', 'grpo'})
  group_size: int = hp.Choice((4, 8, 16), default=8,
                              when=lambda root: root.rl.method == 'grpo')

class Sweep(hp.Params):
  rl: RL = RL()
  lr: float = hp.LogRange(1e-6, 1e-3, default=2e-4)

for trial in hp.samples(Sweep(), 50, seed=0):
  score = train(trial)
  results.append((hp.stable_hash(trial), score, hp.to_dict(trial)))
```

`group_size` is only drawn for GRPO trials, so the sweep does not waste runs
distinguishing values that cannot matter.

### What did that library actually use?

```python
import torch.optim

hp.params.instrument(torch.optim)
trainer.train()                      # builds an optimizer somewhere inside

hp.params.history('torch.optim.AdamW')
# [{'lr': 3e-05, 'weight_decay': 0.01, 'eps': 1e-08}]
```

Turn it around with `override=True` to set values in code you do not control:

```python
hp.params.instrument(torch.optim, select=['AdamW'], override=True)
hp.params('torch.optim.AdamW').lr = 1e-5   # applies wherever it gets constructed
```

### Parallel trials that do not interfere

```python
@hp.wrap
def train(lr: float = 1e-3, seed: int = 0):
  ...

async def trial(lr):
  with hp.override(train, lr=lr):
    return await asyncio.to_thread(train)

scores = await asyncio.gather(*(trial(x) for x in (1e-3, 3e-4, 1e-4)))
```

Each override applies to a fork held in a `ContextVar`, so the three trials
never see each other's values — and the registered params are unchanged
afterwards.

### Service configuration and feature flags

```python
class Service(hp.Params):
  rollout: str = hp.Choice(('off', 'shadow', 'on'), default='off')
  timeout_s: float = hp.Range(0.1, 30.0, default=5.0)
  api_key: str = hp.Field(default='', secret=True, env='SERVICE_API_KEY')

config = hp.load(Service, '/etc/svc/config.yaml', hp.env(prefix='SVC'), hp.cli)

with hp.override(config, rollout='on'):
  handle(request)            # hp.active().rollout == 'on', only in this block
```

`secret=True` keeps the key out of `hp.to_dict()`, `hp.flatten()`, `--help` and
saved configs, while passing `secrets=True` to either still gets it when you
genuinely need to serialize everything.

For logging, `hp.serializable(config)` is the safe shape: secrets dropped and
anything a columnar store cannot hold — a model, a DataLoader, an open file —
replaced by its type name rather than crashing the writer.

### Handing config to a subprocess

```python
config = hp.load(Train, hp.cli)
hp.params.inject(config, prefix='TRAIN__')     # -> TRAIN__OPTIM__LR=0.0003
subprocess.run(['python', 'worker.py'])
```

```python
# worker.py
config = hp.load(Train, hp.env(prefix='TRAIN'))
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
