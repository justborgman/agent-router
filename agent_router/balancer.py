"""Load balancing strategies for Agent Router."""

from __future__ import annotations

import random
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_router.providers import Provider


class BalancerStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_COST = "least_cost"
    LEAST_LATENCY = "least_latency"
    WEIGHTED_RANDOM = "weighted_random"


class Balancer:
    """Selects providers using configurable load balancing strategies."""

    def __init__(self, strategy: str = "round_robin"):
        self.strategy = BalancerStrategy(strategy)
        self._rr_index = 0

    def select(
        self,
        providers: list[Provider],
        model_id: str | None = None,
    ) -> Provider | None:
        """Select a provider from the available list."""
        available = [p for p in providers if p.is_available()]
        if not available:
            return None

        # If model_id specified, filter to providers that have it
        if model_id:
            with_model = [p for p in available if p.get_model(model_id)]
            if with_model:
                available = with_model

        if not available:
            return None

        if self.strategy == BalancerStrategy.ROUND_ROBIN:
            return self._round_robin(available)
        elif self.strategy == BalancerStrategy.LEAST_COST:
            return self._least_cost(available, model_id)
        elif self.strategy == BalancerStrategy.LEAST_LATENCY:
            return self._least_latency(available)
        elif self.strategy == BalancerStrategy.WEIGHTED_RANDOM:
            return self._weighted_random(available)
        else:
            return self._round_robin(available)

    def _round_robin(self, providers: list[Provider]) -> Provider:
        idx = self._rr_index % len(providers)
        self._rr_index += 1
        return providers[idx]

    def _least_cost(self, providers: list[Provider], model_id: str | None) -> Provider:
        def cost_key(p: Provider) -> float:
            if model_id:
                model = p.get_model(model_id)
                if model:
                    return model.input_cost_per_1k + model.output_cost_per_1k
            return float("inf")
        return min(providers, key=cost_key)

    def _least_latency(self, providers: list[Provider]) -> Provider:
        return min(providers, key=lambda p: p.latency)

    def _weighted_random(self, providers: list[Provider]) -> Provider:
        weights = [p.weight for p in providers]
        total = sum(weights)
        if total == 0:
            return random.choice(providers)
        r = random.uniform(0, total)
        cumulative = 0.0
        for p, w in zip(providers, weights):
            cumulative += w
            if r <= cumulative:
                return p
        return providers[-1]
