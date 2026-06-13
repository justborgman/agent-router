"""Configuration loader for Agent Router."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _resolve_env_vars(value: Any) -> Any:
    """Recursively resolve ${VAR} references in config values."""
    if isinstance(value, str):
        pattern = re.compile(r"\$\{(\w+)\}")
        def replace(match: re.Match) -> str:
            var_name = match.group(1)
            env_val = os.environ.get(var_name)
            if env_val is None:
                raise ValueError(f"Environment variable {var_name} is not set")
            return env_val
        return pattern.sub(replace, value)
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


@dataclass
class ModelConfig:
    id: str
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0
    max_tokens: int = 4096


@dataclass
class ProviderConfig:
    name: str
    type: str
    api_key: str
    base_url: str
    priority: int = 10
    weight: int = 1
    rate_limit: int = 100
    timeout: float = 30.0
    models: list[ModelConfig] = field(default_factory=list)


@dataclass
class BalancerConfig:
    strategy: str = "round_robin"


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1


@dataclass
class SettingsConfig:
    default_timeout: float = 30.0
    max_retries: int = 3
    retry_backoff_base: float = 1.0
    retry_backoff_max: float = 10.0
    cache_enabled: bool = True
    cache_ttl: int = 300
    log_level: str = "INFO"


@dataclass
class RouterConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    settings: SettingsConfig = field(default_factory=SettingsConfig)
    balancer: BalancerConfig = field(default_factory=BalancerConfig)
    providers: list[ProviderConfig] = field(default_factory=list)


def load_config(path: str | Path) -> RouterConfig:
    """Load router configuration from a YAML file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    raw = _resolve_env_vars(raw)

    server_data = raw.get("server", {})
    server = ServerConfig(
        host=server_data.get("host", "0.0.0.0"),
        port=server_data.get("port", 8000),
        workers=server_data.get("workers", 1),
    )

    settings_data = raw.get("settings", {})
    settings = SettingsConfig(
        default_timeout=settings_data.get("default_timeout", 30.0),
        max_retries=settings_data.get("max_retries", 3),
        retry_backoff_base=settings_data.get("retry_backoff_base", 1.0),
        retry_backoff_max=settings_data.get("retry_backoff_max", 10.0),
        cache_enabled=settings_data.get("cache_enabled", True),
        cache_ttl=settings_data.get("cache_ttl", 300),
        log_level=settings_data.get("log_level", "INFO"),
    )

    balancer_data = raw.get("balancer", {})
    balancer = BalancerConfig(strategy=balancer_data.get("strategy", "round_robin"))

    providers = []
    for p in raw.get("providers", []):
        models = [
            ModelConfig(
                id=m["id"],
                input_cost_per_1k=m.get("input_cost_per_1k", 0.0),
                output_cost_per_1k=m.get("output_cost_per_1k", 0.0),
                max_tokens=m.get("max_tokens", 4096),
            )
            for m in p.get("models", [])
        ]
        providers.append(
            ProviderConfig(
                name=p["name"],
                type=p["type"],
                api_key=p.get("api_key", ""),
                base_url=p.get("base_url", ""),
                priority=p.get("priority", 10),
                weight=p.get("weight", 1),
                rate_limit=p.get("rate_limit", 100),
                timeout=p.get("timeout", 30.0),
                models=models,
            )
        )

    return RouterConfig(
        server=server,
        settings=settings,
        balancer=balancer,
        providers=providers,
    )
