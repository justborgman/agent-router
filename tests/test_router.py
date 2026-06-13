"""Tests for router fallback logic and balancer strategies."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_router.balancer import Balancer, BalancerStrategy
from agent_router.providers import Provider, ProviderConfig, ModelConfig
from agent_router.router import Router


def _make_provider(name: str, priority: int = 1, weight: int = 1, cost: float = 0.001) -> Provider:
    return Provider(
        ProviderConfig(
            name=name,
            type="openai",
            api_key="test-key",
            base_url="https://api.example.com/v1",
            priority=priority,
            weight=weight,
            models=[ModelConfig("test-model", cost, cost * 2, 4096)],
        )
    )


class TestBalancer:
    def test_round_robin(self):
        providers = [_make_provider("a"), _make_provider("b"), _make_provider("c")]
        balancer = Balancer("round_robin")

        results = [balancer.select(providers).name for _ in range(6)]
        assert results == ["a", "b", "c", "a", "b", "c"]

    def test_round_robin_skips_unhealthy(self):
        p1 = _make_provider("a")
        p2 = _make_provider("b")
        p1.healthy = False
        balancer = Balancer("round_robin")

        results = [balancer.select([p1, p2]).name for _ in range(3)]
        assert all(r == "b" for r in results)

    def test_least_cost(self):
        cheap = _make_provider("cheap", cost=0.0001)
        expensive = _make_provider("expensive", cost=0.1)
        balancer = Balancer("least_cost")

        selected = balancer.select([cheap, expensive], model_id="test-model")
        assert selected.name == "cheap"

    def test_least_latency(self):
        fast = _make_provider("fast")
        fast._latency_ema = 0.1
        slow = _make_provider("slow")
        slow._latency_ema = 2.0
        balancer = Balancer("least_latency")

        selected = balancer.select([fast, slow])
        assert selected.name == "fast"

    def test_weighted_random(self):
        heavy = _make_provider("heavy", weight=90)
        light = _make_provider("light", weight=10)
        balancer = Balancer("weighted_random")

        counts: dict[str, int] = {}
        for _ in range(1000):
            p = balancer.select([heavy, light])
            counts[p.name] = counts.get(p.name, 0) + 1

        assert counts["heavy"] > counts["light"]
        assert counts["heavy"] > 700  # Should be ~900

    def test_select_with_model_filter(self):
        p1 = _make_provider("a")
        p2 = _make_provider("b")
        p2.models = {"other-model": ModelConfig("other-model", 0.01, 0.02, 4096)}
        balancer = Balancer("round_robin")

        selected = balancer.select([p1, p2], model_id="test-model")
        assert selected.name == "a"

    def test_select_returns_none_when_all_down(self):
        p1 = _make_provider("a")
        p1.healthy = False
        balancer = Balancer("round_robin")

        selected = balancer.select([p1])
        assert selected is None


class TestRouter:
    @pytest.mark.asyncio
    async def test_router_success_on_first_provider(self):
        p1 = _make_provider("primary", priority=1)
        p2 = _make_provider("fallback", priority=2)

        mock_response = {
            "id": "test-123",
            "object": "chat.completion",
            "created": 1000,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        with patch.object(p1, "complete", new_callable=AsyncMock, return_value=mock_response) as m1:
            router = Router([p1, p2], max_retries=2, cache_enabled=False)
            result = await router.chat_completion(
                {"model": "test-model", "messages": [{"role": "user", "content": "Hi"}]}
            )
            assert result["choices"][0]["message"]["content"] == "Hello!"
            m1.assert_called_once()
            await router.close()

    @pytest.mark.asyncio
    async def test_router_fallback_on_failure(self):
        import httpx

        p1 = _make_provider("primary", priority=1)
        p2 = _make_provider("fallback", priority=2)

        mock_response = {
            "id": "test-456",
            "object": "chat.completion",
            "created": 1000,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Fallback response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        with patch.object(
            p1, "complete", new_callable=AsyncMock, side_effect=httpx.ConnectError("Connection refused")
        ), patch.object(
            p2, "complete", new_callable=AsyncMock, return_value=mock_response
        ):
            router = Router([p1, p2], max_retries=1, retry_backoff_base=0.01, cache_enabled=False)
            result = await router.chat_completion(
                {"model": "test-model", "messages": [{"role": "user", "content": "Hi"}]}
            )
            assert result["choices"][0]["message"]["content"] == "Fallback response"
            await router.close()

    @pytest.mark.asyncio
    async def test_router_raises_when_all_fail(self):
        import httpx

        p1 = _make_provider("a", priority=1)
        p2 = _make_provider("b", priority=2)

        with patch.object(
            p1, "complete", new_callable=AsyncMock, side_effect=httpx.ConnectError("fail")
        ), patch.object(
            p2, "complete", new_callable=AsyncMock, side_effect=httpx.ConnectError("fail")
        ):
            router = Router([p1, p2], max_retries=1, retry_backoff_base=0.01, cache_enabled=False)
            with pytest.raises(RuntimeError, match="All providers failed"):
                await router.chat_completion(
                    {"model": "test-model", "messages": [{"role": "user", "content": "Hi"}]}
                )
            await router.close()

    @pytest.mark.asyncio
    async def test_router_caching(self):
        p1 = _make_provider("a")

        mock_response = {
            "id": "cached",
            "object": "chat.completion",
            "created": 1000,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Cached!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        with patch.object(p1, "complete", new_callable=AsyncMock, return_value=mock_response) as m:
            router = Router([p1], max_retries=1, cache_enabled=True, cache_ttl=60)
            payload = {"model": "test-model", "messages": [{"role": "user", "content": "Hi"}]}

            r1 = await router.chat_completion(payload)
            r2 = await router.chat_completion(payload)

            assert r1["id"] == r2["id"]
            m.assert_called_once()  # Second call hit cache
            await router.close()
