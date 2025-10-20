"""
Celery Application
-----------------
Celery configuration and initialization for distributed task processing.
Handles LLM request queueing and worker management.
"""

from celery import Celery
from loguru import logger

from app.core.config_manager import settings


# Initialize Celery app
celery_app = Celery(
    "llm_token_manager",
    broker=settings.broker_url,
    backend=settings.celery_result_backend,
    include=["app.llm_client_provisioning.llm_tasks"],
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Task execution settings
    task_soft_time_limit=settings.celery_task_soft_time_limit,
    task_time_limit=settings.celery_task_time_limit,
    task_acks_late=True,  # Acknowledge tasks after completion
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,  # Fetch one task at a time for better load distribution
    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_backend_transport_options={
        "master_name": "mymaster",
        "visibility_timeout": 3600,
    },
    # Retry settings
    task_default_retry_delay=30,  # 30 seconds
    task_max_retries=3,
    # Worker settings
    worker_max_tasks_per_child=1000,  # Restart worker after 1000 tasks (prevent memory leaks)
    worker_disable_rate_limits=True,
    # Logging
    worker_log_format="%(asctime)s - %(levelname)s - %(message)s",
    worker_task_log_format="%(asctime)s - %(task_name)s[%(task_id)s] - %(levelname)s - %(message)s",
)

# Task routing configuration
celery_app.conf.task_routes = {
    "app.llm_client_provisioning.llm_tasks.process_llm_request": {
        "queue": "llm_requests"
    },
    "app.llm_client_provisioning.llm_tasks.process_priority_llm_request": {
        "queue": "priority_requests"
    },
}

# Queue priorities
celery_app.conf.broker_transport_options = {
    "priority_steps": [0, 3, 6, 9],  # 4 priority levels
    "queue_order_strategy": "priority",
}

logger.info("Celery application configured successfully")
