import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { DashboardPage } from "../pages/DashboardPage";

const { mockGetHealth, mockListModels, mockListBenchmarks } = vi.hoisted(() => ({
  mockGetHealth: vi.fn(),
  mockListModels: vi.fn(),
  mockListBenchmarks: vi.fn(),
}));

vi.mock("../api/client", () => ({
  apiClient: {
    getHealth: mockGetHealth,
    listModels: mockListModels,
    listBenchmarks: mockListBenchmarks,
  },
  ApiError: class ApiError extends Error {
    status: number;
    constructor(message: string, status: number) {
      super(message);
      this.status = status;
      this.name = "ApiError";
    }
  },
}));

describe("DashboardPage", () => {
  beforeEach(() => {
    mockGetHealth.mockReset();
    mockListModels.mockReset();
    mockListBenchmarks.mockReset();
  });

  it("shows loading state initially", () => {
    mockGetHealth.mockResolvedValue({ status: "ok", version: "1.0.0", environment: "test", demo_mode: true });
    mockListModels.mockResolvedValue([]);
    mockListBenchmarks.mockResolvedValue([]);

    render(<DashboardPage />);
    expect(screen.getByText("Loading dashboard…")).toBeInTheDocument();
  });

  it("renders dashboard summary after data loads", async () => {
    mockGetHealth.mockResolvedValue({
      status: "ok", version: "1.0.0", environment: "test", demo_mode: true,
    });
    mockListModels.mockResolvedValue([
      { id: "m1", name: "m1", architecture: "BitNet-b1.58-2B-4T", parameter_count: 2000000000,
        quantization_format: "i2_s", activation_function: "relu_sqr", status: "loaded_in_ram",
        memory_bytes: 450000000, created_at: "2024-01-01", updated_at: "2024-01-01" },
    ]);
    mockListBenchmarks.mockResolvedValue([
      { id: "bench-1", model_name: "BitNet-b1.58-2B-4T", dataset: "wikitext-2",
        tokens_evaluated: 256, activation_tested: "relu_sqr", perplexity: 17.1,
        mean_loss: 2.84, latency_ms: 50.0, peak_memory_mb: 512.0,
        weight_distribution: { neg_pct: 27.5, zero_pct: 45.0, pos_pct: 27.5 },
        created_at: "2024-01-01" },
    ]);

    render(<DashboardPage />);

    await waitFor(() => {
      expect(screen.getByText("Signal Workbench Dashboard")).toBeInTheDocument();
    });

    expect(screen.getByText("1.0.0")).toBeInTheDocument();
  });

  it("shows error state when API fails", async () => {
    mockGetHealth.mockRejectedValue(new Error("Failed to fetch"));
    mockListModels.mockResolvedValue([]);
    mockListBenchmarks.mockResolvedValue([]);

    render(<DashboardPage />);

    await waitFor(() => {
      expect(screen.getByText("Failed to fetch")).toBeInTheDocument();
    });
  });
});