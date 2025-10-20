"""Workers module - Celery tasks and worker configuration."""

# Import celery_app for testing purposes
# This allows test patches to access app.llm_client_provisioning.celery_app.celery_app
try:
    from .celery_app import celery_app
except ImportError:
    # If celery_app can't be imported (e.g., missing dependencies),
    # create a placeholder for testing
    celery_app = None
