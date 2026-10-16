"""Signal Workbench — FastAPI application for 1-bit LLM inference and calibration.

Endpoints:
- /api/health: Health check
- /api/models: Model registry and in-memory session management
- /api/inference: Token inference against warm model sessions
- /api/benchmarks: Perplexity calibration and activation comparison
- /api/sizing: Hardware memory planning and throughput estimation
- /api/compiler: Compiler and CPU feature diagnostics
- /api/advisory: Opt-in LLM advisory (grounded, labeled, provider-agnostic)
- /api/audit: Immutable audit log retrieval
"""
from __future__ import annotations

import datetime as dt
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.core.config import settings
from app.core.database import init_db, log_audit_event
from app.core.security import UserContext, get_current_user, require_role
from app.models.schemas import (
    AuditLogEntry,
    BenchmarkRunRequest,
    BenchmarkRunResponse,
    CompilerDiagnosticResponse,
    InferenceRequest,
    InferenceResponse,
    LLMAdvisoryRequest,
    LLMAdvisoryResponse,
    ModelInfo,
    ModelLoadRequest,
    SizingRequest,
    SizingResponse,
)
from app.services.benchmark_service import list_benchmark_history, run_benchmark
from app.services.llm_client import call_llm_advisory
from app.services.model_service import list_models, register_and_load_model, run_inference
from app.engine.compiler_check import run_compiler_diagnostics
from app.engine.memory_planner import calculate_memory_and_throughput


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = settings.VERSION
    environment: str = settings.ENVIRONMENT
    demo_mode: bool = settings.DEMO_MODE


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize database on startup."""
    init_db()
    yield


app = FastAPI(
    title="Signal Workbench",
    version=settings.VERSION,
    description="Inference framework and calibration workbench for 1-bit BitNet b1.58 LLMs",
    lifespan=lifespan,
    docs_url="/api/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/api/redoc" if settings.ENVIRONMENT != "production" else None,
)

# CORS for local SPA development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3000", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check (unauthenticated)
@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
        status="ok",
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        demo_mode=settings.DEMO_MODE,
    )


# Model Registry & In-Memory Sessions
@app.post(
    "/api/models",
    response_model=ModelInfo,
    status_code=status.HTTP_201_CREATED,
    tags=["Models"],
)
async def load_model(
    req: ModelLoadRequest,
    user: UserContext = Depends(require_role("operator")),
):
    """Load a 1-bit model into system RAM for persistent inference (Issue #206)."""
    try:
        model_info = register_and_load_model(req, user.user_id)
        return model_info
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/models", response_model=list[ModelInfo], tags=["Models"])
async def get_models(user: UserContext = Depends(get_current_user)):
    """List all registered models with in-memory residency status."""
    return list_models()


@app.get("/api/models/{model_id}", response_model=ModelInfo, tags=["Models"])
async def get_model(
    model_id: str,
    user: UserContext = Depends(get_current_user),
):
    """Get details for a specific model."""
    models = list_models()
    for m in models:
        if m.id == model_id:
            return m
    raise HTTPException(status_code=404, detail="Model not found")


@app.delete(
    "/api/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Models"]
)
async def unload_model(
    model_id: str,
    user: UserContext = Depends(require_role("operator")),
):
    """Evict a model from in-memory session (persistent record remains)."""
    from app.services.model_service import _ACTIVE_SESSIONS
    from app.core.database import get_connection

    if model_id not in _ACTIVE_SESSIONS:
        raise HTTPException(status_code=404, detail="Model not loaded in memory")

    session = _ACTIVE_SESSIONS.pop(model_id)

    # Update database status to "stored_only"
    now_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            "UPDATE models SET status = ?, updated_at = ? WHERE id = ?",
            ("stored_only", now_utc, model_id),
        )
        conn.commit()

    log_audit_event(
        action="model_unload",
        resource_type="model",
        resource_id=model_id,
        user_id=user.user_id,
        status="success",
        details={
            "was_resident": True,
            "memory_freed_bytes": session.resident_memory_bytes,
        },
    )


# Inference against warm sessions
@app.post("/api/inference", response_model=InferenceResponse, tags=["Inference"])
async def create_inference(
    req: InferenceRequest,
    user: UserContext = Depends(require_role("operator")),
):
    """Execute token inference against a warm in-memory model session."""
    try:
        return run_inference(req, user.user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


# Benchmarking & Perplexity Calibration
@app.post(
    "/api/benchmarks",
    response_model=BenchmarkRunResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Benchmarks"],
)
async def create_benchmark(
    req: BenchmarkRunRequest,
    user: UserContext = Depends(require_role("operator")),
):
    """Run perplexity benchmark comparing activation functions (Issue #588)."""
    try:
        return run_benchmark(req, user.user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get(
    "/api/benchmarks", response_model=list[BenchmarkRunResponse], tags=["Benchmarks"]
)
async def get_benchmarks(
    limit: int = 20,
    user: UserContext = Depends(get_current_user),
):
    """Retrieve historical benchmark runs."""
    return list_benchmark_history(limit)


# Hardware Sizing & Memory Planning
@app.post("/api/sizing", response_model=SizingResponse, tags=["Sizing"])
async def calculate_sizing(
    req: SizingRequest,
    user: UserContext = Depends(get_current_user),
):
    """Estimate memory footprint and throughput for BitNet model scales."""
    return calculate_memory_and_throughput(
        model_scale=req.model_scale,
        context_length=req.context_length,
        device_memory_gb=req.device_memory_gb,
        memory_bandwidth_gbps=req.memory_bandwidth_gbps,
    )


# Compiler & Hardware Diagnostics
@app.get("/api/compiler", response_model=CompilerDiagnosticResponse, tags=["Diagnostics"])
async def get_compiler_diagnostics(user: UserContext = Depends(get_current_user)):
    """Run CPU feature detection and compiler compatibility checks (Issues 180, 334, 185, 480)."""
    return run_compiler_diagnostics()


# LLM Advisory (opt-in, server-side only, grounded)
@app.post("/api/advisory", response_model=LLMAdvisoryResponse, tags=["Advisory"])
async def create_advisory(
    req: LLMAdvisoryRequest,
    user: UserContext = Depends(require_role("operator")),
):
    """Generate grounded advisory response for benchmark findings or configuration questions.

    Advisory is opt-in, labeled as non-authoritative, and uses only server-side LLM credentials.
    When an LLM endpoint is configured and reachable, the response is enhanced with an LLM-generated
    summary.  When the LLM call fails or is not configured, a grounded fallback response is returned.
    """
    if not settings.LLM_API_KEY:
        log_audit_event(
            action="advisory_request",
            resource_type="advisory",
            resource_id=req.topic,
            user_id=user.user_id,
            status="rejected",
            details={"reason": "LLM_API_KEY not configured"},
        )
        raise HTTPException(
            status_code=503,
            detail="Advisory service unavailable: LLM_API_KEY not configured on server",
        )

    disclaimer = (
        "ADVISORY ONLY: This response may be generated by an LLM for informational purposes. "
        "It is not authoritative. Verify all technical claims against official documentation "
        "and empirical benchmarks."
    )

    grounded_context = {
        "activation_bug": "Issue #588 confirms ReLU² (not SiLU) is correct for BitNet b1.58 FFN.",
        "build_issues": "Issues #180, #334, #185, #480 document compiler/flag fixes for clang/CMake.",
        "server_request": "Issue #206 requests persistent in-memory model server.",
        "larger_models": "Issue #8: 70B needs ~17.5 GB, 405B exceeds consumer RAM even at 1.58-bit.",
    }
    references = [
        "https://github.com/microsoft/BitNet/issues/588",
        "https://github.com/microsoft/BitNet/issues/206",
        "https://github.com/microsoft/BitNet/issues/480",
        "https://github.com/microsoft/BitNet/issues/8",
    ]
    summary = f"Grounded advisory for: {req.topic}"

    # Attempt opt-in LLM call through the configured provider.
    ll_content, ll_error = await call_llm_advisory(req.topic, req.context or {})
    if ll_content is not None:
        summary = ll_content
        grounded_context["llm_enhanced"] = True
        grounded_context["llm_provider"] = settings.LLM_PROVIDER
        grounded_context["llm_model"] = settings.LLM_MODEL
    else:
        grounded_context["llm_enhanced"] = False
        grounded_context["llm_error"] = ll_error

    log_audit_event(
        action="advisory_request",
        resource_type="advisory",
        resource_id=req.topic,
        user_id=user.user_id,
        status="success",
        details={"llm_enhanced": ll_content is not None, "provider": settings.LLM_PROVIDER},
    )

    return LLMAdvisoryResponse(
        topic=req.topic,
        summary=summary,
        details=grounded_context,
        references=references,
        disclaimer=disclaimer,
    )


# Audit Log
@app.get("/api/audit", response_model=list[AuditLogEntry], tags=["Audit"])
async def get_audit_logs(
    limit: int = 100,
    offset: int = 0,
    user: UserContext = Depends(require_role("admin")),
):
    """Retrieve paginated immutable audit log (admin only)."""
    from app.core.database import query_audit_logs

    return query_audit_logs(limit=limit, offset=offset)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development",
    )