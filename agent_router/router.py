"""Core Router with multi-provider dispatch, fallback, and retry."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from agent_router.balancer import Balancer
from agent_router.middleware import (
    CostTracker,
    RequestLog,
    RequestLogger,
    ResponseCache,
    estimate_request_tokens,
)
from agent_router.providers import Provider

logger = logging.getLogger(__name__)


class Router:
    """Multi-provider LLM router with intelligent fallback and load balancing."""

    def __init__(
        self,
        providers: list[Provider],
        balancer: Balancer | None = None,
        max_retries: int = 3,
        retry_backoff_base: float = 1.0,
        retry_backoff_max: float = 10.0,
        cache_enabled: bool = True,
        cache_ttl: int = 300,
    ):
        self.providers = providers
        self.balancer = balancer or Balancer("round_robin")
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base
        self.retry_backoff_max = retry_backoff_max
        self.cost_tracker = CostTracker()
        self.request_logger = RequestLogger()
        self.cache = ResponseCache(ttl=cache_ttl) if cache_enabled else None
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient()
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Route a chat completion request through providers with fallback and retry."""
        model_id = payload.get("model", "")

        # Check cache
        if self.cache:
            cached = self.cache.get(payload)
            if cached:
                self.request_logger.log(
                    RequestLog(
                        timestamp=time.time(),
                        provider="cache",
                        model=model_id,
                        latency=0.0,
                        input_tokens=0,
                        output_tokens=0,
                        cost=0.0,
                        cached=True,
                        success=True,
                    )
                )
                return cached

        # Sort providers by priority for fallback order
        sorted_providers = sorted(self.providers, key=lambda p: p.priority)

        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            # Select provider via balancer from available providers
            selected = self.balancer.select(sorted_providers, model_id)
            if selected is None:
                # All providers are down - try any that exist
                available_any = [p for p in sorted_providers]
                if not available_any:
                    raise RuntimeError("No providers configured")
                selected = available_any[0]

            # Try each provider in priority order for this attempt
            providers_to_try = sorted(
                [p for p in sorted_providers if p.is_available()],
                key=lambda p: p.priority,
            )
            if not providers_to_try:
                providers_to_try = sorted_providers

            for provider in providers_to_try:
                try:
                    # Check rate limit
                    if not await provider.check_rate_limit():
                        logger.warning("Rate limited: %s", provider.name)
                        continue

                    client = await self._get_client()
                    start = time.monotonic()
                    response = await provider.complete(payload, client)
                    latency = time.monotonic() - start

                    # Track costs
                    usage = response.get("usage", {})
                    input_tokens = usage.get("prompt_tokens", 0)
                    output_tokens = usage.get("completion_tokens", 0)
                    cost = provider.get_model_cost(model_id, input_tokens, output_tokens)
                    self.cost_tracker.record(
                        provider.name, model_id, input_tokens, output_tokens, cost
                    )

                    self.request_logger.log(
                        RequestLog(
                            timestamp=time.time(),
                            provider=provider.name,
                            model=model_id,
                            latency=latency,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            cost=cost,
                            cached=False,
                            success=True,
                        )
                    )

                    # Cache response
                    if self.cache:
                        self.cache.set(payload, response)

                    return response

                except httpx.HTTPStatusError as e:
                    last_error = e
                    provider.record_failure()
                    logger.warning(
                        "Provider %s failed (HTTP %d): %s",
                        provider.name,
                        e.response.status_code,
                        str(e),
                    )
                    continue

                except (httpx.RequestError, httpx.TimeoutException) as e:
                    last_error = e
                    provider.record_failure()
                    logger.warning("Provider %s failed: %s", provider.name, str(e))
                    continue

                except Exception as e:
                    last_error = e
                    provider.record_failure()
                    logger.error("Provider %s unexpected error: %s", provider.name, str(e))
                    continue

            # Backoff before retry
            if attempt < self.max_retries - 1:
                backoff = min(
                    self.retry_backoff_base * (2**attempt),
                    self.retry_backoff_max,
                )
                logger.info("Retrying in %.1fs (attempt %d/%d)", backoff, attempt + 1, self.max_retries)
                await asyncio.sleep(backoff)

        # All attempts exhausted
        self.request_logger.log(
            RequestLog(
                timestamp=time.time(),
                provider="none",
                model=model_id,
                latency=0.0,
                input_tokens=0,
                output_tokens=0,
                cost=0.0,
                cached=False,
                success=False,
                error=str(last_error),
            )
        )
        raise RuntimeError(
            f"All providers failed after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )
