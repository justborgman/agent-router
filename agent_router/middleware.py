"""Middleware pipeline for Agent Router: logging, token counting, cost tracking, caching."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


# Token counting

def count_tokens(text: str) -> int:
    """Count tokens using tiktoken, falling back to word estimate."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        # Rough fallback: ~0.75 tokens per word
        return max(1, int(len(text.split()) * 1.3))


def estimate_request_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate total tokens in a messages array."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content) + 4  # message overhead
        total += count_tokens(msg.get("role", ""))
    return total


# Cost tracking

@dataclass
class CostRecord:
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    timestamp: float = field(default_factory=time.time)


class CostTracker:
    """Tracks cumulative costs across providers."""

    def __init__(self) -> None:
        self.records: list[CostRecord] = []
        self._total_cost: float = 0.0
        self._cost_by_provider: dict[str, float] = {}
        self._cost_by_model: dict[str, float] = {}

    def record(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
    ) -> None:
        rec = CostRecord(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
        )
        self.records.append(rec)
        self._total_cost += cost
        self._cost_by_provider[provider] = (
            self._cost_by_provider.get(provider, 0.0) + cost
        )
        self._cost_by_model[model] = self._cost_by_model.get(model, 0.0) + cost

    @property
    def total_cost(self) -> float:
        return self._total_cost

    def get_summary(self) -> dict[str, Any]:
        return {
            "total_cost": round(self._total_cost, 6),
            "total_requests": len(self.records),
            "by_provider": {k: round(v, 6) for k, v in self._cost_by_provider.items()},
            "by_model": {k: round(v, 6) for k, v in self._cost_by_model.items()},
        }


# Caching

class ResponseCache:
    """Simple LRU response cache with TTL."""

    def __init__(self, max_size: int = 256, ttl: int = 300):
        self._cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl

    @staticmethod
    def _make_key(payload: dict[str, Any]) -> str:
        serializable = {
            k: v for k, v in payload.items()
            if k in ("model", "messages", "temperature", "max_tokens")
        }
        raw = json.dumps(serializable, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        key = self._make_key(payload)
        if key in self._cache:
            ts, value = self._cache[key]
            if time.monotonic() - ts < self._ttl:
                self._cache.move_to_end(key)
                return value
            else:
                del self._cache[key]
        return None

    def set(self, payload: dict[str, Any], response: dict[str, Any]) -> None:
        key = self._make_key(payload)
        self._cache[key] = (time.monotonic(), response)
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()


# Request logging

@dataclass
class RequestLog:
    timestamp: float
    provider: str
    model: str
    latency: float
    input_tokens: int
    output_tokens: int
    cost: float
    cached: bool
    success: bool
    error: str | None = None


class RequestLogger:
    """Collects request logs for analysis."""

    def __init__(self) -> None:
        self.logs: list[RequestLog] = []

    def log(self, entry: RequestLog) -> None:
        self.logs.append(entry)
        logger.info(
            "Request: provider=%s model=%s latency=%.2fs tokens=%d/%d cost=$%.6f cached=%s",
            entry.provider,
            entry.model,
            entry.latency,
            entry.input_tokens,
            entry.output_tokens,
            entry.cost,
            entry.cached,
        )

    def get_summary(self) -> dict[str, Any]:
        total = len(self.logs)
        successes = sum(1 for l in self.logs if l.success)
        return {
            "total_requests": total,
            "successes": successes,
            "failures": total - successes,
            "avg_latency": (
                sum(l.latency for l in self.logs) / total if total else 0
            ),
            "cached_hits": sum(1 for l in self.logs if l.cached),
        }
