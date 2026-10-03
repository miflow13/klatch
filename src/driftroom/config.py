"""Loading and fingerprinting for versioned Driftroom run configuration."""

import hashlib
import json
import tomllib
from collections.abc import Mapping
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


ENGINE_VERSION = "driftroom-engine-0.1.0"


def run_fingerprint(
    config: RunConfig,
    *,
    prompt_hash: str,
    model_digests: Mapping[str, str | None],
    engine_version: str = ENGINE_VERSION,
) -> str:
    """Hash the whole experimental regime: config, prompts, model digests, engine."""
    regime = {
        "config": config.model_dump(mode="json"),
        "prompt_hash": prompt_hash,
        "model_digests": dict(model_digests),
        "engine_version": engine_version,
        "memory": "disabled",
    }
    canonical = json.dumps(
        regime, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
