"""Data transfer objects and request/response schemas."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    id: str
    name: str
    architecture: str
    parameter_count: int
    quantization_format: str
    activation_function: str
    status: str
    memory_bytes: int
    created_at: str
    updated_at: str


class ModelLoadRequest(BaseModel):
    model_id: str = Field(..., description="Unique model identifier to register or activate")
    model_path: str = Field(..., description="Filesystem path to model weights")
    quantization: str = Field("i2_s", description="Quantization mode: i2_s or tl1")
    context_length: int = Field(2048, ge=512, le=131072, description="Context window size")
    activation_function: str = Field("relu_sqr", description="Activation op: relu_sqr or silu")


class InferenceRequest(BaseModel):
    model_id: str
    prompt: str = Field(..., min_length=1, description="Input prompt text")
    max_tokens: int = Field(64, ge=1, le=2048, description="Max generated tokens")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    activation_override: Optional[str] = Field(None, description="Optional activation function override (relu_sqr or silu)")


class InferenceResponse(BaseModel):
    id: str
    model_id: str
    prompt: str
    completion: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    tokens_per_sec: float
    quant_type: str
    activation_used: str
    created_at: str


class BenchmarkRunRequest(BaseModel):
    model_name: str = Field("BitNet-b1.58-2B-4T", description="Target model")
    dataset: str = Field("wikitext-2", description="Validation corpus")
    num_tokens: int = Field(512, ge=64, le=4096, description="Token length to evaluate")
    activation_tested: str = Field("relu_sqr", description="Activation to test: relu_sqr or silu")


class BenchmarkRunResponse(BaseModel):
    id: str
    model_name: str
    dataset: str
    tokens_evaluated: int
    activation_tested: str
    perplexity: float
    mean_loss: float
    latency_ms: float
    peak_memory_mb: float
    weight_distribution: Dict[str, float]
    created_at: str


class SizingRequest(BaseModel):
    model_scale: str = Field("2B", description="Scale: 0.6B, 2B, 7B, 70B, 405B")
    context_length: int = Field(2048, ge=512, le=131072)
    device_memory_gb: float = Field(16.0, ge=2.0, le=512.0)
    memory_bandwidth_gbps: float = Field(200.0, ge=20.0, le=2000.0)


class SizingResponse(BaseModel):
    model_scale: str
    fp16_weight_gb: float
    ternary_weight_gb: float
    kv_cache_gb: float
    total_memory_required_gb: float
    reduction_pct: float
    fits: bool
    estimated_tps: float
    notes: List[str]


class CompilerDiagnosticResponse(BaseModel):
    cpu_vendor: str
    cpu_features: Dict[str, bool]
    clang_version: Optional[str]
    cmake_version: Optional[str]
    known_issues_addressed: List[Dict[str, Any]]


class LLMAdvisoryRequest(BaseModel):
    topic: str = Field(..., description="Query topic or benchmark finding")
    context: Dict[str, Any] = Field(default_factory=dict, description="Grounded execution context")


class LLMAdvisoryResponse(BaseModel):
    topic: str
    summary: str
    details: Dict[str, Any]
    references: List[str]
    disclaimer: str


class AuditLogEntry(BaseModel):
    id: int
    timestamp: str
    user_id: str
    action: str
    resource_type: str
    resource_id: str
    status: str
    details: Dict[str, Any]