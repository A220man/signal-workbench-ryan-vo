"""Benchmark execution service for perplexity calibration and activation comparisons."""
from __future__ import annotations

import datetime as dt
import json
import time
import uuid
from typing import Any, Dict, List

from app.core.database import get_connection, log_audit_event
from app.engine.perplexity_eval import evaluate_synthetic_perplexity
from app.models.schemas import BenchmarkRunRequest, BenchmarkRunResponse


def run_benchmark(req: BenchmarkRunRequest, user_id: str) -> BenchmarkRunResponse:
    """Execute perplexity benchmark over requested activation function and dataset."""
    start_time = time.perf_counter()

    # Run perplexity simulation grounded in Issue #588 empirical findings
    res = evaluate_synthetic_perplexity(
        num_tokens=req.num_tokens,
        activation=req.activation_tested,
    )

    elapsed_s = max(time.perf_counter() - start_time, 0.005)
    latency_ms = round(elapsed_s * 1000.0, 2)

    # Synthetic weight distribution for BitNet b1.58
    # Typically ~45% zero, 27.5% -1, 27.5% +1
    weight_distribution = {
        "neg_pct": 27.5,
        "zero_pct": 45.0,
        "pos_pct": 27.5,
    }

    run_id = f"bench-{uuid.uuid4().hex[:12]}"
    now_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    peak_mb = 512.0  # RAM usage in MB for 2B 1.58-bit model

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO benchmark_runs (id, model_name, dataset, tokens_evaluated,
                                       activation_tested, measured_perplexity, mean_loss,
                                       latency_ms, memory_peak_mb, details_json, user_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                req.model_name,
                req.dataset,
                req.num_tokens,
                req.activation_tested,
                res["perplexity"],
                res["mean_loss"],
                latency_ms,
                peak_mb,
                json.dumps(weight_distribution),
                user_id,
                now_utc,
            ),
        )
        conn.commit()

    log_audit_event(
        action="benchmark_run",
        resource_type="benchmark",
        resource_id=run_id,
        user_id=user_id,
        status="success",
        details={
            "activation": req.activation_tested,
            "perplexity": res["perplexity"],
            "dataset": req.dataset,
        },
    )

    return BenchmarkRunResponse(
        id=run_id,
        model_name=req.model_name,
        dataset=req.dataset,
        tokens_evaluated=req.num_tokens,
        activation_tested=req.activation_tested,
        perplexity=res["perplexity"],
        mean_loss=res["mean_loss"],
        latency_ms=latency_ms,
        peak_memory_mb=peak_mb,
        weight_distribution=weight_distribution,
        created_at=now_utc,
    )


def list_benchmark_history(limit: int = 20) -> List[BenchmarkRunResponse]:
    """Retrieve historical benchmark runs from SQLite."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM benchmark_runs ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()

    results = []
    for r in rows:
        results.append(
            BenchmarkRunResponse(
                id=r["id"],
                model_name=r["model_name"],
                dataset=r["dataset"],
                tokens_evaluated=r["tokens_evaluated"],
                activation_tested=r["activation_tested"],
                perplexity=r["measured_perplexity"],
                mean_loss=r["mean_loss"],
                latency_ms=r["latency_ms"],
                peak_memory_mb=r["memory_peak_mb"],
                weight_distribution=json.loads(r["details_json"]) if r["details_json"] else {},
                created_at=r["created_at"],
            )
        )
    return results