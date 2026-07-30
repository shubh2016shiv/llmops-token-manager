"""
API router aggregation module.

Why this file exists (enterprise pattern):
1) Single source of truth for route registration:
   - All endpoint routers are imported and included here in one place.
   - This prevents route registration from being scattered across bootstrap code.

2) Cleaner application bootstrap:
   - `app/app.py` can include one router (`api_router`) instead of N routers.
   - Startup wiring becomes easier to read, review, and maintain.

3) Controlled route order:
   - FastAPI matches routes in definition/include order.
   - Keeping include order centralized lowers the risk of accidental precedence bugs.

4) Better modularity for teams:
   - Feature teams can add/modify endpoint modules without touching app bootstrap.
   - Import churn and merge conflicts in `app/app.py` are reduced.

Terminology:
- "Aggregated router": an APIRouter composed of child routers from multiple modules.
- "Re-export": exposing imported router symbols from this package so callers can
  import from `app.api` instead of deep module paths.
"""

from fastapi import APIRouter

from app.api.auth_endpoints import router as auth_router
from app.api.health_endpoints import router as health_router

# llm_service concerns, not token management — commented out, module renamed *_OBSELETE.
# from app.api.llm_configuration_endpoints import router as llm_configuration_router
# from app.api.llm_inference_endpoints import router as llm_inference_router
from app.api.token_manager_endpoints import router as token_manager_router

# from app.api.user_endpoints import router as user_router
# from app.api.user_entitlement_endpoints import router as user_entitlement_router
# from app.llm_gateway.result_store import result_store_router

api_router = APIRouter()

# Enterprise registration note:
# Keep this order intentional and stable. If new routers are added, append with
# care and evaluate path precedence interactions.
api_router.include_router(health_router)
api_router.include_router(auth_router)
# api_router.include_router(llm_configuration_router)
# api_router.include_router(user_router)
# api_router.include_router(user_entitlement_router)
api_router.include_router(token_manager_router)
# api_router.include_router(llm_inference_router)   # POST /api/v1/llm/jobs  — submit job
# api_router.include_router(result_store_router)     # GET  /api/v1/llm/jobs/{id} — poll result

__all__ = [
    "api_router",
    "auth_router",
    "health_router",
    # "llm_configuration_router",
    # "llm_inference_router",
    # "user_router",
    # "user_entitlement_router",
    "token_manager_router",
    # "result_store_router",
]
