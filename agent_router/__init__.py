"""Agent Router - Multi-provider LLM router with intelligent fallback and load balancing."""

__version__ = "0.1.0"

from agent_router.router import Router
from agent_router.config import load_config, RouterConfig
from agent_router.providers import Provider, ProviderConfig, ModelConfig
from agent_router.balancer import Balancer, BalancerStrategy

__all__ = [
    "Router",
    "load_config",
    "RouterConfig",
    "Provider",
    "ProviderConfig",
    "ModelConfig",
    "Balancer",
    "BalancerStrategy",
]
