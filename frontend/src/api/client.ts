/** HTTP client for the Signal Workbench API, derived from backend routes. */
import type {
  AuditLogEntry,
  BenchmarkRunRequest,
  BenchmarkRunResponse,
  CompilerDiagnosticResponse,
  HealthResponse,
  InferenceRequest,
  InferenceResponse,
  LLMAdvisoryRequest,
  LLMAdvisoryResponse,
  ModelInfo,
  ModelLoadRequest,
  SizingRequest,
  SizingResponse,
} from "./types";

const API_BASE = import.meta.env?.VITE_API_BASE_URL ?? "";

export class ApiError extends Error {
  constructor(message: string, public status: number) {
    super(message);
    this.name = "ApiError";
  }
}

export class ApiClient {
  private token: string | null = null;

  setToken(token: string | null): void {
    this.token = token;
  }

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...(init.headers as Record<string, string> | undefined),
    };
    if (this.token) {
      headers.Authorization = `Bearer ${this.token}`;
    }

    const response = await fetch(`${API_BASE}${path}`, { ...init, headers });

    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const data = await response.json();
        if (typeof data?.detail === "string") detail = data.detail;
      } catch {
        // non-JSON body
      }
      throw new ApiError(detail, response.status);
    }

    if (response.status === 204) {
      return undefined as T;
    }

    return response.json() as Promise<T>;
  }

  /** GET /api/health — public health check */
  getHealth = (): Promise<HealthResponse> => this.request<HealthResponse>("/api/health");

  /** POST /api/models — load a model into RAM */
  loadModel = (req: ModelLoadRequest): Promise<ModelInfo> =>
    this.request<ModelInfo>("/api/models", {
      method: "POST",
      body: JSON.stringify(req),
    });

  /** GET /api/models — list all registered models */
  listModels = (): Promise<ModelInfo[]> => this.request<ModelInfo[]>("/api/models");

  /** GET /api/models/{id} — get a single model */
  getModel = (id: string): Promise<ModelInfo> => this.request<ModelInfo>(`/api/models/${id}`);

  /** DELETE /api/models/{id} — evict a model from RAM */
  unloadModel = (id: string): Promise<void> =>
    this.request<void>(`/api/models/${id}`, { method: "DELETE" });

  /** POST /api/inference — run inference against a warm session */
  runInference = (req: InferenceRequest): Promise<InferenceResponse> =>
    this.request<InferenceResponse>("/api/inference", {
      method: "POST",
      body: JSON.stringify(req),
    });

  /** POST /api/benchmarks — run a perplexity benchmark */
  createBenchmark = (req: BenchmarkRunRequest): Promise<BenchmarkRunResponse> =>
    this.request<BenchmarkRunResponse>("/api/benchmarks", {
      method: "POST",
      body: JSON.stringify(req),
    });

  /** GET /api/benchmarks — list recent benchmark runs */
  listBenchmarks = (limit = 20): Promise<BenchmarkRunResponse[]> =>
    this.request<BenchmarkRunResponse[]>(`/api/benchmarks?limit=${limit}`);

  /** POST /api/sizing — estimate memory and throughput */
  calculateSizing = (req: SizingRequest): Promise<SizingResponse> =>
    this.request<SizingResponse>("/api/sizing", {
      method: "POST",
      body: JSON.stringify(req),
    });

  /** GET /api/compiler — run CPU/toolchain diagnostics */
  getCompilerDiagnostics = (): Promise<CompilerDiagnosticResponse> =>
    this.request<CompilerDiagnosticResponse>("/api/compiler");

  /** GET /api/audit — list audit logs (admin only) */
  getAuditLogs = (limit = 100, offset = 0): Promise<AuditLogEntry[]> =>
    this.request<AuditLogEntry[]>(`/api/audit?limit=${limit}&offset=${offset}`);

  /** POST /api/advisory — opt-in grounded LLM advisory */
  createAdvisory = (req: LLMAdvisoryRequest): Promise<LLMAdvisoryResponse> =>
    this.request<LLMAdvisoryResponse>("/api/advisory", {
      method: "POST",
      body: JSON.stringify(req),
    });
}

export const apiClient = new ApiClient();