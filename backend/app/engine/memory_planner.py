"""Hardware memory planner and throughput estimation for BitNet b1.58 scales."""
from __future__ import annotations

from typing import Dict, List, Optional
from app.models.schemas import SizingResponse

# Known BitNet architectures with configuration specs
BITNET_SPECS = {
    "0.6B": {
        "params": 600_000_000,
        "layers": 24,
        "hidden_dim": 1024,
        "heads": 16,
        "kv_heads": 16,
        "intermediate_dim": 2816,
    },
    "2B": {
        "params": 2_000_000_000,
        "layers": 30,
        "hidden_dim": 2048,
        "heads": 32,
        "kv_heads": 32,
        "intermediate_dim": 5632,
    },
    "7B": {
        "params": 7_000_000_000,
        "layers": 32,
        "hidden_dim": 4096,
        "heads": 32,
        "kv_heads": 32,
        "intermediate_dim": 11008,
    },
    "70B": {
        "params": 70_000_000_000,
        "layers": 80,
        "hidden_dim": 8192,
        "heads": 64,
        "kv_heads": 8,
        "intermediate_dim": 28672,
    },
    "405B": {
        "params": 405_000_000_000,
        "layers": 126,
        "hidden_dim": 16384,
        "heads": 128,
        "kv_heads": 16,
        "intermediate_dim": 53248,
    },
}


def calculate_memory_and_throughput(
    model_scale: str,
    context_length: int,
    device_memory_gb: float,
    memory_bandwidth_gbps: float,
) -> SizingResponse:
    """Compute detailed memory requirements and token throughput for BitNet models.

    Addresses Issue #8: Can models like 70B or 405B run on consumer devices?
    With 1.58-bit ternary quantization:
    - 70B model requires only ~17.5 GB for weights (fits on 32GB Mac/PC).
    - FP16 baseline requires 140 GB (requiring multiple server GPUs).
    """
    spec = BITNET_SPECS.get(model_scale)
    if not spec:
        spec = BITNET_SPECS["2B"]
        model_scale = "2B"

    params = spec["params"]
    layers = spec["layers"]
    kv_heads = spec["kv_heads"]
    head_dim = spec["hidden_dim"] // spec["heads"]

    # Weight memory
    fp16_weight_bytes = params * 2.0  # 16 bits = 2 bytes
    # BitNet packed: 2 bits per parameter in i2_s format = 0.25 bytes
    # Plus scaling factors and embeddings (~0.05 bytes overhead)
    ternary_weight_bytes = params * 0.25

    # KV Cache memory: 2 * layers * kv_heads * head_dim * context_length * bytes_per_elem (FP16 or INT8)
    # Using 16-bit (2 bytes) for KV cache states
    kv_cache_bytes = 2.0 * layers * kv_heads * head_dim * context_length * 2.0

    # Total memory required
    fp16_weight_gb = round(fp16_weight_bytes / (1024**3), 2)
    ternary_weight_gb = round(ternary_weight_bytes / (1024**3), 2)
    kv_cache_gb = round(kv_cache_bytes / (1024**3), 3)

    # Allow 1.0 GB overhead for activations and runtime buffers
    runtime_overhead_gb = 1.0
    total_memory_required_gb = round(ternary_weight_gb + kv_cache_gb + runtime_overhead_gb, 2)

    reduction_pct = round(((fp16_weight_gb - ternary_weight_gb) / fp16_weight_gb) * 100.0, 1)
    fits = total_memory_required_gb <= device_memory_gb

    # Estimated throughput in memory-bandwidth-bound single-batch generation:
    # Throughput (tokens/s) = Bandwidth (GB/s) / Model Size (GB)
    estimated_tps = (
        round(memory_bandwidth_gbps / max(0.1, ternary_weight_gb), 1) if ternary_weight_gb > 0 else 0.0
    )

    notes: List[str] = [
        f"BitNet {model_scale} weights compress from {fp16_weight_gb} GB (FP16) to {ternary_weight_gb} GB in 2-bit i2_s format.",
        f"Memory savings: {reduction_pct}% reduction in parameter footprint.",
        f"Context length of {context_length} tokens adds {kv_cache_gb} GB KV-cache.",
    ]

    if fits:
        notes.append(
            f"FITS cleanly in {device_memory_gb} GB RAM with {round(device_memory_gb - total_memory_required_gb, 1)} GB headroom."
        )
    else:
        notes.append(
            f"EXCEEDS device memory by {round(total_memory_required_gb - device_memory_gb, 1)} GB; requires offloading or higher RAM capacity."
        )

    return SizingResponse(
        model_scale=model_scale,
        fp16_weight_gb=fp16_weight_gb,
        ternary_weight_gb=ternary_weight_gb,
        kv_cache_gb=kv_cache_gb,
        total_memory_required_gb=total_memory_required_gb,
        reduction_pct=reduction_pct,
        fits=fits,
        estimated_tps=estimated_tps,
        notes=notes,
    )