"""Provider implementations for Agent Router."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


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


class Provider:
    """Represents a single LLM provider with health tracking and rate limiting."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.name = config.name
        self.type = config.type
        self.priority = config.priority
        self.weight = config.weight
        self.models = {m.id: m for m in config.models}

        # Health tracking
        self.healthy = True
        self.last_health_check = 0.0
        self.consecutive_failures = 0
        self.circuit_breaker_until = 0.0

        # Latency tracking (exponential moving average)
        self._latency_ema: float = 0.0
        self._latency_alpha: float = 0.3

        # Rate limiting
        self._request_timestamps: list[float] = []
        self._rate_lock = asyncio.Lock()

    @property
    def latency(self) -> float:
        return self._latency_ema if self._latency_ema > 0 else 1.0

    def get_model(self, model_id: str) -> ModelConfig | None:
        return self.models.get(model_id)

    def get_model_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a request."""
        model = self.models.get(model_id)
        if not model:
            return 0.0
        input_cost = (input_tokens / 1000) * model.input_cost_per_1k
        output_cost = (output_tokens / 1000) * model.output_cost_per_1k
        return input_cost + output_cost

    async def check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        async with self._rate_lock:
            now = time.monotonic()
            window = 60.0
            self._request_timestamps = [
                t for t in self._request_timestamps if now - t < window
            ]
            if len(self._request_timestamps) >= self.config.rate_limit:
                return False
            self._request_timestamps.append(now)
            return True

    def is_available(self) -> bool:
        """Check if provider is available (healthy and not circuit-broken)."""
        if not self.healthy:
            return False
        if time.monotonic() < self.circuit_breaker_until:
            return False
        return True

    def record_success(self, latency: float) -> None:
        """Record a successful request."""
        self.consecutive_failures = 0
        self.healthy = True
        self.circuit_breaker_until = 0.0
        if self._latency_ema == 0:
            self._latency_ema = latency
        else:
            self._latency_ema = (
                self._latency_alpha * latency
                + (1 - self._latency_alpha) * self._latency_ema
            )

    def record_failure(self) -> None:
        """Record a failed request. Open circuit breaker after 3 consecutive failures."""
        self.consecutive_failures += 1
        if self.consecutive_failures >= 3:
            self.healthy = False
            self.circuit_breaker_until = time.monotonic() + 30.0
            logger.warning(
                "Circuit breaker opened for provider %s (failures=%d)",
                self.name,
                self.consecutive_failures,
            )

    def _build_headers(self) -> dict[str, str]:
        """Build provider-specific headers."""
        if self.type == "anthropic":
            return {
                "x-api-key": self.config.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-type": "application/json",
        }

    def _build_request_body(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Build provider-specific request body."""
        if self.type == "anthropic":
            messages = payload.get("messages", [])
            system_msg = ""
            filtered = []
            for m in messages:
                if m.get("role") == "system":
                    system_msg = m.get("content", "")
                else:
                    filtered.append(m)
            body: dict[str, Any] = {
                "model": payload.get("model", ""),
                "messages": filtered,
                "max_tokens": payload.get("max_tokens", 4096),
            }
            if system_msg:
                body["system"] = system_msg
            if "temperature" in payload:
                body["temperature"] = payload["temperature"]
            return body
        # OpenAI-compatible (openai, openrouter, vllm, etc.)
        return payload

    def _get_endpoint(self) -> str:
        """Get the chat completions endpoint URL."""
        if self.type == "anthropic":
            return f"{self.config.base_url.rstrip('/')}/v1/messages"
        return f"{self.config.base_url.rstrip('/')}/chat/completions"

    def _normalize_response(
        self, raw: dict[str, Any], model: str
    ) -> dict[str, Any]:
        """Normalize provider response to OpenAI format."""
        if self.type == "anthropic":
            content = ""
            if raw.get("content"):
                content = raw["content"][0].get("text", "")
            usage = raw.get("usage", {})
            return {
                "id": raw.get("id", ""),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content,
                        },
                        "finish_reason": raw.get("stop_reason", "stop"),
                    }
                ],
                "usage": {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("input_tokens", 0)
                    + usage.get("output_tokens", 0),
                },
            }
        return raw

    async def complete(
        self,
        payload: dict[str, Any],
        client: httpx.AsyncClient,
    ) -> dict[str, Any]:
        """Send a completion request to this provider."""
        url = self._get_endpoint()
        headers = self._build_headers()
        body = self._build_request_body(payload)

        start = time.monotonic()
        response = await client.post(
            url,
            json=body,
            headers=headers,
            timeout=self.config.timeout,
        )
        latency = time.monotonic() - start

        response.raise_for_status()
        raw = response.json()

        self.record_success(latency)
        return self._normalize_response(raw, payload.get("model", ""))

    async def health_check(self, client: httpx.AsyncClient) -> bool:
        """Perform a lightweight health check."""
        try:
            url = self._get_endpoint()
            headers = self._build_headers()
            resp = await client.get(
                url.rsplit("/", 1)[0],
                headers=headers,
                timeout=5.0,
            )
            # Any non-5xx means provider is reachable
            is_ok = resp.status_code < 500
            self.healthy = is_ok
            self.last_health_check = time.monotonic()
            return is_ok
        except Exception:
            self.healthy = False
            self.last_health_check = time.monotonic()
            return False


# Pre-configured provider factories


def create_openai_provider(api_key: str, base_url: str = "https://api.openai.com/v1", **kwargs: Any) -> Provider:
    config = ProviderConfig(
        name=kwargs.get("name", "openai"),
        type="openai",
        api_key=api_key,
        base_url=base_url,
        models=[
            ModelConfig("gpt-4o", 0.0025, 0.01, 128000),
            ModelConfig("gpt-4o-mini", 0.00015, 0.0006, 128000),
        ],
        **{k: v for k, v in kwargs.items() if k != "name"},
    )
    return Provider(config)


def create_anthropic_provider(api_key: str, **kwargs: Any) -> Provider:
    config = ProviderConfig(
        name=kwargs.get("name", "anthropic"),
        type="anthropic",
        api_key=api_key,
        base_url="https://api.anthropic.com",
        models=[
            ModelConfig("claude-sonnet-4-20250514", 0.003, 0.015, 200000),
            ModelConfig("claude-3-5-haiku-20241022", 0.001, 0.005, 200000),
        ],
        **{k: v for k, v in kwargs.items() if k != "name"},
    )
    return Provider(config)
