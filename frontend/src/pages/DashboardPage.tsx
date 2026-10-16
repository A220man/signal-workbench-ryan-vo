import React, { useEffect, useState } from "react";
import { apiClient } from "../api/client";
import type { HealthResponse, ModelInfo, BenchmarkRunResponse } from "../api/types";
import { LoadingState } from "../components/LoadingState";
import { ErrorState } from "../components/ErrorState";

/** Application overview: health, model registry snapshot, and recent benchmarks. */
export function DashboardPage(): React.JSX.Element {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [benchmarks, setBenchmarks] = useState<BenchmarkRunResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadAll = async () => {
    setError(null);
    try {
      const [h, m, b] = await Promise.all([
        apiClient.getHealth(),
        apiClient.listModels(),
        apiClient.listBenchmarks(5),
      ]);
      setHealth(h);
      setModels(m);
      setBenchmarks(b);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadAll();
  }, []);

  if (loading) return <LoadingState message="Loading dashboard…" />;

  const loadedCount = models.filter((m) => m.status === "loaded_in_ram").length;
  const siluRuns = benchmarks.filter((b) => b.activation_tested === "silu").length;
  const reluRuns = benchmarks.filter((b) => b.activation_tested === "relu_sqr").length;

  return (
    <div>
      <h2>Signal Workbench Dashboard</h2>
      {error !== null && <ErrorState message={error} onRetry={loadAll} />}
      <div className="grid grid-cols-4">
        <div className="stat">
          <div className="stat-value">{health?.version ?? "—"}</div>
          <div className="stat-label">Version</div>
        </div>
        <div className="stat">
          <div className="stat-value">{loadedCount}</div>
          <div className="stat-label">Models Loaded</div>
        </div>
        <div className="stat">
          <div className="stat-value">{siluRuns}</div>
          <div className="stat-label">SiLU Benchmarks</div>
        </div>
        <div className="stat">
          <div className="stat-value">{reluRuns}</div>
          <div className="stat-label">ReLU² Benchmarks</div>
        </div>
      </div>

      {benchmarks.length > 0 && (
        <div className="card">
          <h2>Recent Benchmarks</h2>
          <table className="table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Activation</th>
                <th>Perplexity</th>
                <th>Loss</th>
                <th>Latency (ms)</th>
                <th>Created</th>
              </tr>
            </thead>
            <tbody>
              {benchmarks.map((b) => (
                <tr key={b.id}>
                  <td>{b.id.slice(0, 10)}…</td>
                  <td>
                    <span className="tag">{b.activation_tested}</span>
                  </td>
                  <td>{b.perplexity.toFixed(2)}</td>
                  <td>{b.mean_loss.toFixed(4)}</td>
                  <td>{b.latency_ms}</td>
                  <td>{new Date(b.created_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {models.length > 0 && (
        <div className="card">
          <h2>Registered Models</h2>
          <table className="table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Architecture</th>
                <th>Quantization</th>
                <th>Activation</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.id}>
                  <td>{m.id}</td>
                  <td>{m.architecture}</td>
                  <td>
                    <span className="tag">{m.quantization_format}</span>
                  </td>
                  <td>
                    <span className="tag">{m.activation_function}</span>
                  </td>
                  <td>
                    <span className={`badge badge-${m.status.includes("loaded") ? "loaded" : "stored"}`}>
                      {m.status}
                    </span>
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