"""Load and merge configuration.

Precedence, lowest to highest: schema defaults, the YAML file, environment
variables. The loader is deliberately dumb -- all validation lives in the
schema, so there is one place to look when a value is rejected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ppe.config.schema import AppConfig
from ppe.core.errors import ConfigError

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def load_config(path: str | Path | None = None) -> AppConfig:
    """Build an :class:`AppConfig` from a YAML file plus the environment.

    Args:
        path: YAML file to read. Defaults to ``configs/default.yaml``. A
            missing default file is not an error -- schema defaults are used --
            but a missing *explicitly requested* file is.

    Raises:
        ConfigError: The file is unreadable, is not a mapping, or fails
            validation against the schema.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    explicit = path is not None

    if not config_path.exists():
        if explicit:
            msg = f"config file not found: {config_path}"
            raise ConfigError(msg)
        raw: dict[str, Any] = {}
    else:
        raw = _read_yaml(config_path)

    try:
        return AppConfig(**raw)
    except Exception as exc:  # re-raised as our own error type
        msg = f"invalid configuration in {config_path}: {exc}"
        raise ConfigError(msg) from exc


def _read_yaml(config_path: Path) -> dict[str, Any]:
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"could not read config file {config_path}: {exc}"
        raise ConfigError(msg) from exc

    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        msg = f"could not parse YAML in {config_path}: {exc}"
        raise ConfigError(msg) from exc

    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        msg = f"config root must be a mapping, got {type(parsed).__name__} in {config_path}"
        raise ConfigError(msg)
    return parsed
