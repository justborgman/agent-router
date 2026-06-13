"""FastAPI server exposing an OpenAI-compatible /v1/chat/completions endpoint."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent_router.balancer import Balancer
from agent_router.config import load_config
from agent_router.providers import Provider, ProviderConfig, ModelConfig
from agent_router.router import Router

logger = logging.getLogger(__name__)

# Global router instance
_router: Router | None = None


def get_router() -> Router:
    global _router
    if _router is None:
        raise RuntimeError("Router not initialized")
    return _router


class ChatMessage(BaseModel):
    role: str
    content: str | list[Any] = ""


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stream: bool = False
    stop: str | list[str] | None = None


def build_router_from_config(config_path: str) -> Router:
    """Build a Router instance from config file."""
    cfg = load_config(config_path)

    providers = []
    for pc in cfg.providers:
        models = [
            ModelConfig(
                id=m.id,
                input_cost_per_1k=m.input_cost_per_1k,
                output_cost_per_1k=m.output_cost_per_1k,
                max_tokens=m.max_tokens,
            )
            for m in pc.models
        ]
        providers.append(
            Provider(
                ProviderConfig(
                    name=pc.name,
                    type=pc.type,
                    api_key=pc.api_key,
                    base_url=pc.base_url,
                    priority=pc.priority,
                    weight=pc.weight,
                    rate_limit=pc.rate_limit,
                    timeout=pc.timeout,
                    models=models,
                )
            )
        )

    balancer = Balancer(cfg.balancer.strategy)
    return Router(
        providers=providers,
        balancer=balancer,
        max_retries=cfg.settings.max_retries,
        retry_backoff_base=cfg.settings.retry_backoff_base,
        retry_backoff_max=cfg.settings.retry_backoff_max,
        cache_enabled=cfg.settings.cache_enabled,
        cache_ttl=cfg.settings.cache_ttl,
    )


def create_app(config_path: str | None = None) -> FastAPI:
    """Create the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _router
        path = config_path or os.environ.get("AGENT_ROUTER_CONFIG", "config.yaml")
        _router = build_router_from_config(path)
        logger.info("Router initialized with %d providers", len(_router.providers))
        yield
        await _router.close()
        _router = None

    app = FastAPI(
        title="Agent Router",
        description="Multi-provider LLM router with OpenAI-compatible API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        router = get_router()
        models = []
        for provider in router.providers:
            for model_id, model in provider.models.items():
                models.append({
                    "id": model_id,
                    "object": "model",
                    "owned_by": provider.name,
                })
        return {"object": "list", "data": models}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        if request.stream:
            raise HTTPException(
                status_code=501,
                detail="Streaming is not yet supported",
            )

        router = get_router()
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop is not None:
            payload["stop"] = request.stop

        try:
            result = await router.chat_completion(payload)
            return result
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail=str(e))
        except Exception as e:
            logger.exception("Unexpected error")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/stats")
    async def stats():
        router = get_router()
        return {
            "cost": router.cost_tracker.get_summary(),
            "requests": router.request_logger.get_summary(),
        }

    return app


def main():
    """Run the server standalone."""
    import uvicorn

    config_path = os.environ.get("AGENT_ROUTER_CONFIG", "config.yaml")
    cfg = load_config(config_path)

    app = create_app(config_path)
    uvicorn.run(
        app,
        host=cfg.server.host,
        port=cfg.server.port,
        workers=cfg.server.workers,
        log_level=cfg.settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
