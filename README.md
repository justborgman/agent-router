# Agent Router

**Multi-provider LLM router with intelligent fallback, load balancing, and cost tracking.**

[![Tests](https://github.com/justborgman/agent-router/actions/workflows/test.yml/badge.svg)](https://github.com/justborgman/agent-router/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Agent Router provides a unified, OpenAI-compatible API that distributes requests across multiple LLM providers with automatic failover, intelligent load balancing, and real-time cost tracking. Designed for production workloads where reliability and cost optimization are critical.

## Architecture

```
                        ┌─────────────────────────────────────┐
                        │          Client Application         │
                        │   (OpenAI SDK, curl, any HTTP)      │
                        └──────────────┬──────────────────────┘
                                       │
                              POST /v1/chat/completions
                                       │
                        ┌──────────────▼──────────────────────┐
                        │        FastAPI Server                │
                        │    (OpenAI-compatible endpoint)      │
                        └──────────────┬──────────────────────┘
                                       │
                        ┌──────────────▼──────────────────────┐
                        │         Agent Router Core            │
                        │  ┌─────────────────────────────┐    │
                        │  │   Middleware Pipeline         │    │
                        │  │  • Request Logging            │    │
                        │  │  • Token Counting (tiktoken)  │    │
                        │  │  • Cost Tracking              │    │
                        │  │  • Response Caching (LRU)     │    │
                        │  └─────────────────────────────┘    │
                        │                                       │
                        │  ┌─────────────────────────────┐    │
                        │  │   Load Balancer               │    │
                        │  │  • Round Robin                │    │
                        │  │  • Least Cost                 │    │
                        │  │  • Least Latency              │    │
                        │  │  • Weighted Random            │    │
                        │  └──────────────┬──────────────┘    │
                        │                 │                     │
                        │  ┌──────────────▼──────────────┐    │
                        │  │   Fallback & Retry Engine    │    │
                        │  │  • Priority-based failover   │    │
                        │  │  • Exponential backoff       │    │
                        │  │  • Circuit breaker           │    │
                        │  └─────────────────────────────┘    │
                        └──────────────┬──────────────────────┘
                                       │
              ┌────────────┬───────────┼───────────┬────────────┐
              │            │           │           │            │
        ┌─────▼─────┐ ┌───▼───┐ ┌────▼────┐ ┌───▼───┐ ┌─────▼─────┐
        │  OpenAI    │ │Anthropic│ │OpenRouter│ │vLLM/  │ │  Custom   │
        │  GPT-4o    │ │Claude  │ │Llama,etc │ │Local  │ │ Endpoint  │
        └───────────┘ └───────┘ └─────────┘ └───────┘ └───────────┘
```

## Features

- **Multi-Provider Support**: OpenAI, Anthropic, OpenRouter, local vLLM, any OpenAI-compatible endpoint
- **Intelligent Fallback**: Priority-based failover with circuit breaker pattern
- **Load Balancing**: Round-robin, least-cost, least-latency, weighted random strategies
- **Cost Tracking**: Real-time per-request cost calculation and aggregation by provider/model
- **Response Caching**: LRU cache with configurable TTL to reduce costs and latency
- **Rate Limiting**: Per-provider request rate limiting with sliding window
- **OpenAI-Compatible API**: Drop-in replacement for any OpenAI SDK client
- **Production Ready**: Docker support, structured logging, health checks, CI/CD

## Quick Start

### Installation

```bash
git clone https://github.com/justborgman/agent-router.git
cd agent-router
pip install -r requirements.txt
```

### Configuration

Copy the example config and set your API keys:

```bash
cp config.example.yaml config.yaml

# Set environment variables
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENROUTER_API_KEY="sk-or-..."
```

### Run the Server

```bash
# Direct
python -m agent_router.server

# With custom config
AGENT_ROUTER_CONFIG=my-config.yaml python -m agent_router.server

# Docker
docker build -t agent-router .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY="sk-..." \
  -e ANTHROPIC_API_KEY="sk-ant-..." \
  agent-router
```

### Use with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # Keys are configured server-side
)

response = client.chat.completions.create(
    model="gpt-4o",  # Or any configured model
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Use with curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Python Client

```python
import asyncio
from agent_router import Router
from agent_router.providers import create_openai_provider, create_anthropic_provider
from agent_router.balancer import Balancer

async def main():
    router = Router(
        providers=[
            create_openai_provider(api_key="sk-...", priority=1),
            create_anthropic_provider(api_key="sk-ant-...", priority=2),
        ],
        balancer=Balancer("least_cost"),
    )

    response = await router.chat_completion({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello!"}],
    })
    print(response["choices"][0]["message"]["content"])

    # View cost summary
    print(router.cost_tracker.get_summary())

    await router.close()

asyncio.run(main())
```

## Configuration Reference

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4

settings:
  default_timeout: 30.0
  max_retries: 3
  retry_backoff_base: 1.0
  retry_backoff_max: 10.0
  cache_enabled: true
  cache_ttl: 300

balancer:
  strategy: "least_cost"  # round_robin | least_cost | least_latency | weighted_random

providers:
  - name: "openai"
    type: "openai"        # openai | anthropic | openrouter | openai_compatible
    api_key: "${OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"
    priority: 1           # Lower = preferred
    weight: 40            # For weighted_random strategy
    rate_limit: 100       # Requests per minute
    timeout: 30.0
    models:
      - id: "gpt-4o"
        input_cost_per_1k: 0.0025
        output_cost_per_1k: 0.01
        max_tokens: 128000
```

Environment variables in `${VAR}` syntax are resolved at load time.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (OpenAI-compatible) |
| `/v1/models` | GET | List available models |
| `/v1/stats` | GET | Cost and request statistics |
| `/health` | GET | Health check |

## Development

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run with auto-reload
uvicorn agent_router.server:create_app --reload
```

## License

MIT License - see [LICENSE](LICENSE) for details.
