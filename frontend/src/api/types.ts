/** Typed API contracts derived from backend response envelopes. */

export interface HealthResponse {
  status: string;
  version: string;
  environment: string;
  demo_mode: boolean;
}

export interface ModelInfo {
  id: string;
  name: string;
  architecture: string;
  parameter_count: number;
  quantization_format: string;
  activation_function: string;
  status: string;
  memory_bytes: number;
  created_at: string;
  updated_at: string;
}

export interface ModelLoadRequest {
  model_id: string;
  model_path: string;
  quantization?: string;
  context_length?: number;
  activation_function?: string;
}

export interface InferenceRequest {
  model_id: string;
  prompt: string;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  activation_override?: string | null;
}

export interface InferenceResponse {
  id: string;
  model_id: string;
  prompt: string;
  completion: string;
  prompt_tokens: number;
  completion_tokens: number;
  latency_ms: number;
  tokens_per_sec: number;
  quant_type: string;
  activation_used: string;
  created_at: string;
}

export interface BenchmarkRunRequest {
  model_name?: string;
  dataset?: string;
  num_tokens?: number;
  activation_tested?: string;
}

export interface WeightDistribution {
  neg_pct: number;
  zero_pct: number;
  pos_pct: number;
}

export interface BenchmarkRunResponse {
  id: string;
  model_name: string;
  dataset: string;
  tokens_evaluated: number;
  activation_tested: string;
  perplexity: number;
  mean_loss: number;
  latency_ms: number;
  peak_memory_mb: number;
  weight_distribution: WeightDistribution;
  created_at: string;
}

export interface SizingRequest {
  model_scale: string;
  context_length: number;
  device_memory_gb: number;
  memory_bandwidth_gbps: number;
}

export interface SizingResponse {
  model_scale: string;
  fp16_weight_gb: number;
  ternary_weight_gb: number;
  kv_cache_gb: number;
  total_memory_required_gb: number;
  reduction_pct: number;
  fits: boolean;
  estimated_tps: number;
  notes: string[];
}

export interface CompilerDiagnosticResponse {
  cpu_vendor: string;
  cpu_features: Record<string, boolean>;
  clang_version: string | null;
  cmake_version: string | null;
  known_issues_addressed: Array<Record<string, unknown>>;
}

export interface LLMAdvisoryRequest {
  topic: string;
  context?: Record<string, unknown>;
}

export interface LLMAdvisoryResponse {
  topic: string;
  summary: string;
  details: Record<string, unknown>;
  references: string[];
  disclaimer: string;
}

export interface AuditLogEntry {
  id: number;
  timestamp: string;
  user_id: string;
  action: string;
  resource_type: string;
  resource_id: string;
  status: string;
  details: Record<string, unknown>;
}