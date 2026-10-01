"""Loading and fingerprinting for versioned Driftroom run configuration."""

import hashlib
import json
import tomllib
from pathlib import Path

from .domain import (
    AgentConfig,
    AgentTraits,
    RunConfig,
    RuntimeConfig,
    SamplingConfig,
    SchedulerConfig,
)


def load_run_config(path: Path) -> RunConfig:
    with path.open("rb") as config_file:
        data = tomllib.load(config_file)
    return RunConfig.model_validate(data)


def canonical_config_json(config: RunConfig) -> str:
    return json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def run_config_hash(config: RunConfig) -> str:
    return hashlib.sha256(canonical_config_json(config).encode("utf-8")).hexdigest()
