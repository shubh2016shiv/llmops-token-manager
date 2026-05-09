"""
Token queue package - Layer 4 queue absorption and DLQ handling.

Architecture:
-------------
    ┌────────────────────────────┐     ┌────────────────────────────┐
    │ API / services hot path    │────▶│ publisher.py               │
    │ TokenAllocationPublisher   │     │ typed RabbitMQ publishing  │
    └────────────────────────────┘     └─────────────┬──────────────┘
                                                     │
                                                     ▼
                    ┌─────────────────────────────────────────────────────┐
                    │ topology.py                                         │
                    │ work queue + retry stages + DLQ declarations        │
                    └─────────────┬──────────────────────────────┬────────┘
                                  │                              │
                                  ▼                              ▼
                    ┌────────────────────────────┐   ┌────────────────────┐
                    │ consumer.py                │   │ handlers.py        │
                    │ raw Kombu consumer loop    │──▶│ persistence + DLQ  │
                    │ work + DLQ processing      │   │ alert side effects │
                    └────────────────────────────┘   └────────────────────┘

Dependencies:
    - app/core/config.py - broker, heartbeat, and retry settings
    - app/models/resilience_models.py - typed queue payload contracts
    - app/resilience/circuit_breaker - RabbitMQ circuit breaker registry

Author: Engineering Team
Last Updated: 2026-05-09
"""

from app.resilience.circuit_breaker import get_rmq_circuit_breaker
from app.resilience.token_queue.consumer import (
    TokenQueueConsumerService,
    run_token_queue_consumer,
)
from app.resilience.token_queue.publisher import TokenAllocationPublisher
from app.resilience.token_queue.topology import (
    TOKEN_ALLOCATION_DLQ,
    TOKEN_ALLOCATION_QUEUE,
    declare_token_queues,
)

__all__ = [
    "TOKEN_ALLOCATION_DLQ",
    "TOKEN_ALLOCATION_QUEUE",
    "TokenAllocationPublisher",
    "TokenQueueConsumerService",
    "declare_token_queues",
    "get_rmq_circuit_breaker",
    "run_token_queue_consumer",
]
