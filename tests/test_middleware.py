"""Tests for middleware: token counting and cost calculation."""

from agent_router.middleware import (
    CostTracker,
    ResponseCache,
    RequestLogger,
    RequestLog,
    count_tokens,
    estimate_request_tokens,
)


class TestTokenCounting:
    def test_count_tokens_basic(self):
        tokens = count_tokens("Hello, world!")
        assert tokens > 0
        assert tokens < 10

    def test_count_tokens_empty(self):
        tokens = count_tokens("")
        # tiktoken may return 0 or 1 for empty string depending on encoding
        assert tokens <= 1

    def test_count_tokens_longer_text(self):
        text = "The quick brown fox jumps over the lazy dog. " * 10
        tokens = count_tokens(text)
        assert tokens > 20

    def test_estimate_request_tokens(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        tokens = estimate_request_tokens(messages)
        assert tokens > 0
        assert tokens < 50


class TestCostTracker:
    def test_record_and_total(self):
        tracker = CostTracker()
        tracker.record("openai", "gpt-4o", 1000, 500, 0.01)
        tracker.record("openai", "gpt-4o", 2000, 1000, 0.02)
        assert tracker.total_cost == 0.03

    def test_summary(self):
        tracker = CostTracker()
        tracker.record("openai", "gpt-4o", 1000, 500, 0.01)
        tracker.record("anthropic", "claude-sonnet-4-20250514", 1000, 500, 0.02)

        summary = tracker.get_summary()
        assert summary["total_cost"] == 0.03
        assert summary["total_requests"] == 2
        assert summary["by_provider"]["openai"] == 0.01
        assert summary["by_provider"]["anthropic"] == 0.02

    def test_empty_tracker(self):
        tracker = CostTracker()
        assert tracker.total_cost == 0.0
        summary = tracker.get_summary()
        assert summary["total_requests"] == 0


class TestResponseCache:
    def test_cache_hit(self):
        cache = ResponseCache(max_size=10, ttl=60)
        payload = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]}
        response = {"choices": [{"message": {"content": "Hello"}}]}

        cache.set(payload, response)
        result = cache.get(payload)
        assert result == response

    def test_cache_miss(self):
        cache = ResponseCache(max_size=10, ttl=60)
        payload = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]}
        assert cache.get(payload) is None

    def test_cache_eviction(self):
        cache = ResponseCache(max_size=2, ttl=60)
        for i in range(5):
            payload = {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": f"Message {i}"}],
            }
            cache.set(payload, {"id": i})

        assert len(cache._cache) == 2

    def test_cache_clear(self):
        cache = ResponseCache(max_size=10, ttl=60)
        payload = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]}
        cache.set(payload, {"choices": []})
        cache.clear()
        assert cache.get(payload) is None


class TestRequestLogger:
    def test_log_and_summary(self):
        rl = RequestLogger()
        rl.log(RequestLog(
            timestamp=0, provider="openai", model="gpt-4o",
            latency=0.5, input_tokens=100, output_tokens=50,
            cost=0.001, cached=False, success=True,
        ))
        rl.log(RequestLog(
            timestamp=0, provider="openai", model="gpt-4o",
            latency=1.0, input_tokens=100, output_tokens=50,
            cost=0.001, cached=False, success=False, error="timeout",
        ))

        summary = rl.get_summary()
        assert summary["total_requests"] == 2
        assert summary["successes"] == 1
        assert summary["failures"] == 1
