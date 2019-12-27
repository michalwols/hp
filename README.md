# hp: Hyperparameter Management Library

Defining, tuning and tracking hyperparameters in machine learning experiments can get messy. hp uses python classes to declaratively define your hyperparameters. 

## Benefits

1. Type annotated container for your parameters
2. Automatically generated command line interface 
3. Handles saving and loading of parameters


## Install

Install with pip

```
pip install hp
```

## Tour

### Define Your Parameters

```python
import hp


class Params(hp.HyperParams):
  learning_rate: hp.Range(0.001, 0.1) = 0.03
  optimizer: hp.Choice(('SGD', 'Adam')) = 'SGD'

  batch_size = 32
  seed = 1
```

### Command Line Parser

```python
# parse from command line arguments
params = Params.from_command()
```

### Environment Variables

```python

params = Params.from_env(prefix='HP_')
```


### Global Constants

```python
params = Params.from_constants()  # load all CAP_CASE variables
```


### YAML / JSON 

```python
hp.save(params, 'params.yaml')

params = Params.load('params.yaml')
```


### Binding

```python

params = hp.HyperParams()

@params.bind
def train(epochs=10):
  pass


train()  # use current param value (default to function default)

train(epochs=4)  # override params
```


```python
trainer.batch_size = params.bind('batch_size')
```

```python
params.bind(Optimizer, fields={'lr': 'learning_rate'})
```

### Grid / Random Search

```python

# grid search
for params in Params.grid():
  pass
 
# random samples without replacement
for params in Params.samples():
  pass
```


### Change Tracking

```python

@params.on_change
def log_changes(params, key, value):
  print(f"changing {key} from {params[key]} to {value}")
  
params.learning_rate = 0.001
# >> changing learning_rate from 0.03 to 0.001
```
