import React, { useEffect, useState } from "react";
import { apiClient, ApiError } from "../api/client";
import type { ModelInfo, InferenceRequest, InferenceResponse } from "../api/types";
import { LoadingState } from "../components/LoadingState";
import { ErrorState } from "../components/ErrorState";
import { EmptyState } from "../components/EmptyState";

/** Run token inference against a warm in-memory model session. */
export function InferencePage(): React.JSX.Element {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [prompt, setPrompt] = useState("");
  const [maxTokens, setMaxTokens] = useState(64);
  const [activationOverride, setActivationOverride] = useState<"" | "relu_sqr" | "silu">("");
  const [result, setResult] = useState<InferenceResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [modelsLoading, setModelsLoading] = useState(true);

  const loadModels = async () => {
    setModelsLoading(true);
    try {
      const data = await apiClient.listModels();
      setModels(data);
      const loaded = data.find((m) => m.status === "loaded_in_ram");
      if (loaded) setSelectedModel(loaded.id);
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : String(err);
      setError(msg);
    } finally {
      setModelsLoading(false);
    }
  };

  useEffect(() => {
    void loadModels();
  }, []);

  async function handleRun(e: React.FormEvent) {
    e.preventDefault();
    if (!selectedModel || !prompt.trim()) return;

    setLoading(true);
    setError(null);
    try {
      const req: InferenceRequest = {
        model_id: selectedModel,
        prompt: prompt.trim(),
        max_tokens: maxTokens,
        temperature: 0.7,
        top_p: 0.9,
        activation_override: activationOverride || null,
      };
      const resp = await apiClient.runInference(req);
      setResult(resp);
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : String(err);
      setError(msg);
    } finally {
      setLoading(false);
    }
  }

  const loadedModels = models.filter((m) => m.status === "loaded_in_ram");

  if (modelsLoading) return <LoadingState message="Loading models…" />;

  return (
    <div>
      <h2>Inference</h2>

      {error !== null && <ErrorState message={error} onRetry={loadModels} />}

      {loadedModels.length === 0 ? (
        <EmptyState
          message="No warm model sessions. Load a model from the Models tab first."
          actionLabel="Reload models"
          onAction={loadModels}
        />
      ) : (
        <div className="card">
          <h3>Run Inference</h3>
          <form onSubmit={handleRun}>
            <div className="form-group">
              <label>Warm Model Session</label>
              <select
                className="form-select"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                required
              >
                <option value="">Select a loaded model…</option>
                {loadedModels.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.id} ({m.quantization_format})
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label>Prompt</label>
              <textarea
                className="form-input"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Enter your prompt…"
                rows={4}
                required
              />
            </div>

            <div className="grid grid-cols-2">
              <div className="form-group">
                <label>Max Tokens</label>
                <input
                  type="number"
                  className="form-input"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(Number(e.target.value))}
                  min={1}
                  max={2048}
                />
              </div>
              <div className="form-group">
                <label>Activation Override</label>
                <select
                  className="form-select"
                  value={activationOverride}
                  onChange={(e) => setActivationOverride(e.target.value as "" | "relu_sqr" | "silu")}
                >
                  <option value="">Use model default</option>
                  <option value="relu_sqr">ReLU²</option>
                  <option value="silu">SiLU</option>
                </select>
              </div>
            </div>

            <button type="submit" className="btn btn-primary" disabled={loading || !selectedModel || !prompt.trim()}>
              {loading ? "Running…" : "Run Inference"}
            </button>
          </form>
        </div>
      )}

      {result !== null && (
        <div className="card">
          <h3>Inference Result</h3>
          <div className="code-block">
            <strong>Completion:</strong> {result.completion}
          </div>
          <div className="grid grid-cols-3" style={{ marginTop: "0.75rem" }}>
            <div className="stat">
              <div className="stat-value">{result.prompt_tokens}</div>
              <div className="stat-label">Prompt Tokens</div>
            </div>
            <div className="stat">
              <div className="stat-value">{result.completion_tokens}</div>
              <div className="stat-label">Completion Tokens</div>
            </div>
            <div className="stat">
              <div className="stat-value">{result.tokens_per_sec.toFixed(1)}</div>
              <div className="stat-label">Tokens/sec</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}