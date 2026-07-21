from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load(path: str | Path) -> dict[str, Any]:
  path = Path(path)
  if path.suffix == '.json':
    return json.loads(path.read_text())
  if path.suffix in {'.yaml', '.yml'}:
    try:
      import yaml
    except ImportError as error:
      raise ImportError('Install hp[yaml] to load YAML') from error
    return yaml.safe_load(path.read_text()) or {}
  if path.suffix == '.toml':
    import tomllib
    return tomllib.loads(path.read_text())
  raise ValueError(f'unsupported config format: {path.suffix}')


def save(value: dict[str, Any], path: str | Path) -> None:
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  if path.suffix == '.json':
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')
    return
  if path.suffix in {'.yaml', '.yml'}:
    try:
      import yaml
    except ImportError as error:
      raise ImportError('Install hp[yaml] to save YAML') from error
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    return
  raise ValueError(f'unsupported config format: {path.suffix}')
