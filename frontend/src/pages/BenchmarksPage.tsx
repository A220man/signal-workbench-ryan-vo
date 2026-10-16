import React, { useEffect, useState } from "react";
import { apiClient, ApiError } from "../api/client";
import type { BenchmarkRunResponse, BenchmarkRunRequest } from "../api/types";
import { LoadingState } from "../components/LoadingState";
import { ErrorState } from "../components/ErrorState";
import { EmptyState } from "../components/EmptyState";

/** Benchmark history viewer and new run launcher. */
export function BenchmarksPage(): React.JSX.Element {
  const [runs, setRuns] = useState<BenchmarkRunResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activation, setActivation] = useState<"relu_sqr" | "silu">("relu_sqr");
  const [numTokens, setNumTokens] = useState(512);

  const loadRuns = async () => {
    setLoading(true);
    try {
      const data = await apiClient.listBenchmarks(20);
      setRuns(data);
      setError(null);
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : String(err);
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadRuns();
  }, []);

  async function handleRun(e: React.FormEvent) {
    e.preventDefault();
    setRunning(true);
    setError(null);
    try {
      const req: BenchmarkRunRequest = {
        model_name: "BitNet-b1.58-2B-4T",
        dataset: "wikitext-2",
        num_tokens: numTokens,
        activation_tested: activation,
      };
      await apiClient.createBenchmark(req);
      await loadRuns();
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : String(err);
      setError(msg);
    } finally {
      setRunning(false);
    }
  }

  if (loading) return <LoadingState message="Loading benchmarks…" />;

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
        <h2>Perplexity Benchmarks</h2>
      </div>

      {error !== null && <ErrorState message={error} onRetry={loadRuns} />}

      <div className="card">
        <h3>Run New Benchmark</h3>
        <form onSubmit={handleRun} style={{ display: "flex", gap: "1rem", alignItems: "flex-end" }}>
          <div className="form-group" style={{ marginBottom: 0, minWidth: "140px" }}>
            <label>Activation</label>
            <select
              className="form-select"
              value={activation}
              onChange={(e) => setActivation(e.target.value as "relu_sqr" | "silu")}
            >
              <option value="relu_sqr">ReLU² (correct)</option>
              <option value="silu">SiLU (regression)</option>
            </select>
          </div>
          <div className="form-group" style={{ marginBottom: 0, minWidth: "120px" }}>
            <label>Tokens</label>
            <input
              type="number"
              className="form-input"
              value={numTokens}
              onChange={(e) => setNumTokens(Number(e.target.value))}
              min={64}
              max={4096}
            />
          </div>
          <button type="submit" className="btn btn-primary" disabled={running}>
            {running ? "Running…" : "Run Benchmark"}
          </button>
        </form>
        <p style={{ fontSize: "0.75rem", color: "var(--color-text-muted)", marginTop: "0.5rem" }}>
          Grounded in upstream issue #588: ReLU² produces PPL ≈ 17.1, SiLU ≈ 99.8.
        </p>
      </div>

      {runs.length === 0 ? (
        <EmptyState message="No benchmarks have been run yet." actionLabel="Run one above" onAction={() => undefined} />
      ) : (
        <div className="card">
          <h3>Benchmark History</h3>
          <table className="table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Activation</th>
                <th>Perplexity</th>
                <th>Mean Loss</th>
                <th>Latency</th>
                <th>Memory</th>
                <th>Created</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => {
                const isRegression = r.activation_tested === "silu";
                return (
                  <tr key={r.id}>
                    <td>{r.id.slice(0, 10)}…</td>
                    <td>
                      <span className={`badge ${isRegression ? "badge-stored" : "badge-loaded"}`}>
                        {r.activation_tested}
                      </span>
                    </td>
                    <td style={{ color: isRegression ? "var(--color-danger)" : "var(--color-success)" }}>
                      {r.perplexity.toFixed(2)}
                    </td>
                    <td>{r.mean_loss.toFixed(4)}</td>
                    <td>{r.latency_ms} ms</td>
                    <td>{r.peak_memory_mb.toFixed(0)} MB</td>
                    <td style={{ fontSize: "0.75rem", color: "var(--color-text-muted)" }}>
                      {new Date(r.created_at).toLocaleString()}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}