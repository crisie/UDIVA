"""Parser config loading."""

from __future__ import annotations

import json
try:
    import tomllib
except ModuleNotFoundError:
    import pip._vendor.tomli as tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ParserConfig:
    path: Path
    data: dict[str, Any]

    @property
    def base_dir(self) -> Path:
        return self.path.parent

    @property
    def input(self) -> dict[str, Any]:
        value = self.data.get("input", {})
        return value if isinstance(value, dict) else {}

    @property
    def defaults(self) -> dict[str, Any]:
        value = self.data.get("defaults", {})
        return value if isinstance(value, dict) else {}

    @property
    def outputs(self) -> dict[str, dict[str, Any]]:
        value = self.data.get("outputs", {})
        if not isinstance(value, dict) or not value:
            raise ValueError("Config must define at least one [outputs.<name>] block")
        return {
            str(name): output
            for name, output in value.items()
            if isinstance(output, dict)
        }


def load_config(path: str | Path) -> ParserConfig:
    config_path = Path(path).resolve()
    suffix = config_path.suffix.lower()

    if suffix == ".json":
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    elif suffix in {".toml", ".tml"}:
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
    elif suffix in {".yaml", ".yml"}:
        data = _load_yaml(config_path)
    else:
        raise ValueError(f"Unsupported config type: {config_path.suffix}")

    if not isinstance(data, dict):
        raise ValueError("Config must contain an object/table at the top level")
    return ParserConfig(path=config_path, data=data)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "YAML configs require PyYAML. Use TOML or JSON for a stdlib-only config."
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("YAML config must contain a mapping at the top level")
    return data
