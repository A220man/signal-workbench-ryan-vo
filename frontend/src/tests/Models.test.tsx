import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { ModelsPage } from "../pages/ModelsPage";

const { mockListModels, mockUnloadModel, mockLoadModel } = vi.hoisted(() => ({
  mockListModels: vi.fn(),
  mockUnloadModel: vi.fn(),
  mockLoadModel: vi.fn(),
}));

vi.mock("../api/client", () => ({
  apiClient: {
    listModels: mockListModels,
    unloadModel: mockUnloadModel,
    loadModel: mockLoadModel,
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

describe("ModelsPage", () => {
  beforeEach(() => {
    mockListModels.mockReset();
    mockUnloadModel.mockReset();
    mockLoadModel.mockReset();
  });

  it("shows loading state then models list", async () => {
    mockListModels.mockResolvedValue([
      { id: "model-1", name: "model-1", architecture: "BitNet-b1.58-2B-4T",
        parameter_count: 2000000000, quantization_format: "i2_s",
        activation_function: "relu_sqr", status: "loaded_in_ram",
        memory_bytes: 450000000, created_at: "2024-01-01", updated_at: "2024-01-01" },
    ]);

    render(<ModelsPage />);

    expect(screen.getByText("Loading models…")).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText("model-1")).toBeInTheDocument();
    });
    expect(screen.getByText("BitNet-b1.58-2B-4T")).toBeInTheDocument();
    expect(screen.getByText("loaded_in_ram")).toBeInTheDocument();
  });

  it("shows empty state when no models registered", async () => {
    mockListModels.mockResolvedValue([]);

    render(<ModelsPage />);

    await waitFor(() => {
      expect(screen.getByText("No models registered yet.")).toBeInTheDocument();
    });
  });

  it("unloads model when unload button clicked", async () => {
    mockListModels.mockResolvedValue([
      { id: "to-unload", name: "to-unload", architecture: "BitNet-b1.58-2B-4T",
        parameter_count: 2000000000, quantization_format: "i2_s",
        activation_function: "relu_sqr", status: "loaded_in_ram",
        memory_bytes: 450000000, created_at: "2024-01-01", updated_at: "2024-01-01" },
    ]);
    mockUnloadModel.mockResolvedValue(undefined);

    render(<ModelsPage />);

    await waitFor(() => {
      expect(screen.getByText("to-unload")).toBeInTheDocument();
    });

    const unloadBtn = screen.getByText("Unload");
    fireEvent.click(unloadBtn);

    await waitFor(() => {
      expect(mockUnloadModel).toHaveBeenCalledWith("to-unload");
    });
  });
});