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

ENGINE_VERSION = "driftroom-engine-0.1.4"
# Placeholder regime for memory. The memory task must replace this with real memory
# parameters so runs with and without memory cannot hash the same.
MEMORY_REGIME = "disabled"


def _canonical_json(obj: object) -> str:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def load_run_config(path: Path) -> RunConfig:
    with path.open("rb") as config_file:
        data = tomllib.load(config_file)
    return RunConfig.model_validate(data)


def canonical_config_json(config: RunConfig) -> str:
    return _canonical_json(config.model_dump(mode="json"))


def run_config_hash(config: RunConfig) -> str:
    return hashlib.sha256(canonical_config_json(config).encode("utf-8")).hexdigest()


def run_fingerprint(
    config: RunConfig,
    *,
    prompt_hash: str,
    model_digests: Mapping[str, str | None],
    engine_version: str = ENGINE_VERSION,
) -> str:
    """Hash the whole experimental regime: config, prompts, model digests, engine."""
    missing = sorted({agent.model for agent in config.agents} - model_digests.keys())
    if missing:
        raise ValueError(f"model_digests is missing configured model(s): {', '.join(missing)}")
    regime = {
        "config": config.model_dump(mode="json"),
        "prompt_hash": prompt_hash,
        "model_digests": dict(model_digests),
        "engine_version": engine_version,
        "memory": MEMORY_REGIME,
    }
    return hashlib.sha256(_canonical_json(regime).encode("utf-8")).hexdigest()
