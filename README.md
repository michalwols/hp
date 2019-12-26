# hp - Hyperparameter Management Library


## Getting Started

Install with pip

```
pip install hp
```

Define your parameters

```python
import hp


class Params(hp.HyperParams):
  learning_rate: hp.Range(0.001, 0.1) = 0.03
  optimizer: hp.Choice(('SGD', 'Adam')) = 'SGD'

  batch_size = 32
  seed = 1

# parse from command line arguments
params = Params.from_command()
```
