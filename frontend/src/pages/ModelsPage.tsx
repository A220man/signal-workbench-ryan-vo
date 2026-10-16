import React, { useEffect, useState } from "react";
import { apiClient, ApiError } from "../api/client";
import type { ModelInfo, ModelLoadRequest } from "../api/types";
import { LoadingState } from "../components/LoadingState";
import { ErrorState } from "../components/ErrorState";
import { EmptyState } from "../components/EmptyState";

/** Model registry: list, load new models, and evict from RAM. */
export function ModelsPage(): React.JSX.Element {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [formModelId, setFormModelId] = useState("");
  const [formModelPath, setFormModelPath] = useState("");
  const [formQuantization, setFormQuantization] = useState<"i2_s" | "tl1">("i2_s");
  const [formActivation, setFormActivation] = useState<"relu_sqr" | "silu">("relu_sqr");

  const loadModels = async () => {
    setLoading(true);
    try {
      const data = await apiClient.listModels();
      setModels(data);
      setError(null);
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : String(err);
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadModels();
  }, []);

  async function handleLoad(e: React.FormEvent) {
    e.preventDefault();
    try {
      const req: ModelLoadRequest = {
        model_id: formModelId,
        model_path: formModelPath,
        quantization: formQuantization,
        context_length: 2048,
        activation_function: formActivation,
      };
      await apiClient.loadModel(req);
      setShowForm(false);
      setFormModelId("");
      setFormModelPath("");
      await loadModels();
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : String(err);
      setError(msg);
    }
  }

  async function handleUnload(id: string) {
    try {
      await apiClient.unloadModel(id);
      await loadModels();
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : String(err);
      setError(msg);
    }
  }

  if (loading) return <LoadingState message="Loading models…" />;

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
        <h2>Model Registry</h2>
        <button className="btn btn-primary" onClick={() => setShowForm(!showForm)}>
          {showForm ? "Cancel" : "+ Load Model"}
        </button>
      </div>

      {error !== null && <ErrorState message={error} onRetry={loadModels} />}

      {showForm && (
        <div className="card">
          <h3>Load New Model</h3>
          <form onSubmit={handleLoad}>
            <div className="form-group">
              <label>Model ID</label>
              <input
                className="form-input"
                value={formModelId}
                onChange={(e) => setFormModelId(e.target.value)}
                placeholder="e.g. BitNet-b1.58-2B-4T"
                required
              />
            </div>
            <div className="form-group">
              <label>Model Path</label>
              <input
                className="form-input"
                value={formModelPath}
                onChange={(e) => setFormModelPath(e.target.value)}
                placeholder="/models/bitnet-2b"
                required
              />
            </div>
            <div className="form-group">
              <label>Quantization</label>
              <select
                className="form-select"
                value={formQuantization}
                onChange={(e) => setFormQuantization(e.target.value as "i2_s" | "tl1")}
              >
                <option value="i2_s">i2_s (scalar 2-bit)</option>
                <option value="tl1">tl1 (LUT-based)</option>
              </select>
            </div>
            <div className="form-group">
              <label>Activation Function</label>
              <select
                className="form-select"
                value={formActivation}
                onChange={(e) => setFormActivation(e.target.value as "relu_sqr" | "silu")}
              >
                <option value="relu_sqr">ReLU² (correct for b1.58)</option>
                <option value="silu">SiLU (regression — see #588)</option>
              </select>
            </div>
            <button type="submit" className="btn btn-primary">
              Load Model
            </button>
          </form>
        </div>
      )}

      {models.length === 0 ? (
        <EmptyState
          message="No models registered yet."
          actionLabel="Load a model to get started"
          onAction={() => setShowForm(true)}
        />
      ) : (
        <div className="card">
          <h3>Registered Models ({models.length})</h3>
          <table className="table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Architecture</th>
                <th>Params</th>
                <th>Quant</th>
                <th>Activation</th>
                <th>Status</th>
                <th>Memory</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.id}>
                  <td>{m.id}</td>
                  <td>{m.architecture}</td>
                  <td>{(m.parameter_count / 1e9).toFixed(1)}B</td>
                  <td><span className="tag">{m.quantization_format}</span></td>
                  <td><span className="tag">{m.activation_function}</span></td>
                  <td>
                    <span
                      className={`badge ${
                        m.status === "loaded_in_ram" ? "badge-loaded" : "badge-stored"
                      }`}
                    >
                      {m.status}
                    </span>
                  </td>
                  <td>{(m.memory_bytes / (1024 * 1024)).toFixed(0)} MB</td>
                  <td>
                    {m.status === "loaded_in_ram" && (
                      <button className="btn btn-sm btn-danger" onClick={() => handleUnload(m.id)}>
                        Unload
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}