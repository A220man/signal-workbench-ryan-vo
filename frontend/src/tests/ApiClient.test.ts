import { describe, it, expect, vi, beforeEach, type Mock } from "vitest";
import { ApiClient, ApiError } from "../api/client";

/** Tests for the typed API client: HTTP envelope parsing and error mapping. */
describe("ApiClient", () => {
  let client: ApiClient;
  let mockFetch: Mock;

  beforeEach(() => {
    client = new ApiClient();
    mockFetch = vi.fn();
    globalThis.fetch = mockFetch as unknown as typeof fetch;
  });

  it("parses JSON on success", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({ status: "ok", version: "1.0.0", environment: "test", demo_mode: true }),
    });

    const result = await client.getHealth();
    expect(result.status).toBe("ok");
    expect(result.version).toBe("1.0.0");
    expect(result.demo_mode).toBe(true);
  });

  it("throws ApiError with detail on 404", async () => {
    mockFetch.mockResolvedValue({
      ok: false,
      status: 404,
      json: async () => ({ detail: "Model not found" }),
    });

    try {
      await client.getModel("nonexistent");
      throw new Error("Should have thrown");
    } catch (err) {
      expect(err).toBeInstanceOf(ApiError);
      expect((err as ApiError).status).toBe(404);
      expect((err as Error).message).toBe("Model not found");
    }
  });

  it("returns undefined on 204 No Content", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 204,
      json: async () => {
        throw new Error("Should not parse body");
      },
    });

    const result = await client.unloadModel("test-id");
    expect(result).toBeUndefined();
  });

  it("sends Authorization header when token set", async () => {
    client.setToken("test-token-123");

    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({ status: "ok", version: "1.0.0", environment: "test", demo_mode: false }),
    });

    await client.getHealth();

    const call = mockFetch.mock.calls[0];
    expect(call[1].headers.Authorization).toBe("Bearer test-token-123");
  });

  it("serializes request body as JSON for POST", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 201,
      json: async () => ({ id: "test", name: "test" }),
    });

    await client.loadModel({ model_id: "test", model_path: "/m" });

    const call = mockFetch.mock.calls[0];
    expect(call[1].method).toBe("POST");
    expect(call[1].body).toBe(JSON.stringify({ model_id: "test", model_path: "/m" }));
  });
});