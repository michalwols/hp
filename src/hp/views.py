"""Live views over the process environment and the command line.

Both read through to the real thing rather than snapshotting it, and both
double as layers for :func:`hp.layered`:

    hp.env.CUDA_VISIBLE_DEVICES = '0'   # writes os.environ
    hp.cli.optim.lr                     # parsed --optim.lr

    params = hp.layered(Config, 'base.yaml', hp.env(prefix='APP'), hp.cli)
"""

from __future__ import annotations

import os
import sys
from typing import Any, Iterator


class EnvSource:
  """An environment layer, optionally restricted to a prefix."""

  def __init__(self, prefix: str = '', separator: str = '__'):
    self.prefix = prefix
    self.separator = separator

  def __repr__(self) -> str:
    return f'EnvSource(prefix={self.prefix!r})'


class CliSource:
  """A command line layer over an explicit argv."""

  def __init__(self, argv: str | list[str] | None = None):
    if isinstance(argv, str):
      import shlex

      argv = shlex.split(argv)
    self.argv = argv

  def __repr__(self) -> str:
    return f'CliSource(argv={self.argv!r})'


def _env_name(name: str) -> str:
  """Env vars are conventionally upper case; accept either spelling."""
  if name in os.environ:
    return name
  upper = name.upper()
  return upper if upper in os.environ else name


class Env:
  """Read and write ``os.environ`` as attributes or items.

  Assignment writes through, so subprocesses and libraries that read the
  environment later see the change.
  """

  def __call__(self, prefix: str = '', separator: str = '__') -> EnvSource:
    return EnvSource(prefix, separator)

  def __getattr__(self, name: str) -> str | None:
    if name.startswith('_'):
      raise AttributeError(name)
    return os.environ.get(_env_name(name))

  def __setattr__(self, name: str, value: Any) -> None:
    if name.startswith('_'):
      object.__setattr__(self, name, value)
      return
    os.environ[_env_name(name)] = str(value)

  def __delattr__(self, name: str) -> None:
    os.environ.pop(_env_name(name), None)

  def __getitem__(self, key: str) -> str:
    return os.environ[_env_name(key)]

  def __setitem__(self, key: str, value: Any) -> None:
    os.environ[_env_name(key)] = str(value)

  def __delitem__(self, key: str) -> None:
    del os.environ[_env_name(key)]

  def __contains__(self, key: str) -> bool:
    return _env_name(key) in os.environ

  def __iter__(self) -> Iterator[str]:
    return iter(os.environ)

  def __len__(self) -> int:
    return len(os.environ)

  def get(self, key: str, default: Any = None) -> Any:
    return os.environ.get(_env_name(key), default)

  def to_dict(self, prefix: str = '') -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k.startswith(prefix)}

  def __repr__(self) -> str:
    return f'<hp.env: {len(os.environ)} variables>'


class Cli:
  """The parsed command line, as a dynamic params tree.

  Parsing is lazy and cached; call :meth:`reset` after mutating ``sys.argv``.
  """

  def __init__(self):
    object.__setattr__(self, '_parsed', None)
    object.__setattr__(self, '_args', [])

  def __call__(self, argv: str | list[str] | None = None) -> CliSource:
    return CliSource(argv)

  @property
  def parsed(self):
    if self._parsed is None:
      from .cli import parse
      from .core import Dynamic

      positionals: list[str] = []
      parsed = parse(Dynamic(), list(sys.argv[1:]), positionals)
      object.__setattr__(self, '_args', positionals)
      object.__setattr__(self, '_parsed', parsed)
    return self._parsed

  @property
  def args(self) -> list[str]:
    """Positional arguments, which are not configuration."""
    self.parsed
    return self._args

  def reset(self) -> None:
    """Drop the cached parse, so the next access re-reads sys.argv."""
    object.__setattr__(self, '_parsed', None)
    object.__setattr__(self, '_args', [])

  def __getattr__(self, name: str) -> Any:
    if name.startswith('_'):
      raise AttributeError(name)
    return getattr(self.parsed, name)

  def __getitem__(self, key: str) -> Any:
    return self.parsed[key]

  def __contains__(self, key: str) -> bool:
    return key in self.parsed

  def __iter__(self) -> Iterator[str]:
    return iter(self.parsed)

  def to_dict(self) -> dict[str, Any]:
    from .core import to_dict

    return to_dict(self.parsed)

  def __repr__(self) -> str:
    return f'<hp.cli: {" ".join(sys.argv[1:]) or "no arguments"}>'


env = Env()
cli = Cli()
