import React, { useState } from "react";
import { apiClient, ApiError } from "../api/client";
import type { SizingRequest, SizingResponse } from "../api/types";
import { LoadingState } from "../components/LoadingState";
import { ErrorState } from "../components/ErrorState";

/** Hardware memory planner: estimate weight/KV memory and throughput for BitNet scales. */
export function SizingPage(): React.JSX.Element {
  const [modelScale, setModelScale] = useState<"0.6B" | "2B" | "7B" | "70B" | "405B">("2B");
  const [contextLength, setContextLength] = useState(2048);
  const [deviceMemory, setDeviceMemory] = useState(16.0);
  const [memoryBandwidth, setMemoryBandwidth] = useState(200.0);
  const [result, setResult] = useState<SizingResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const req: SizingRequest = {
        model_scale: modelScale,
        context_length: contextLength,
        device_memory_gb: deviceMemory,
        memory_bandwidth_gbps: memoryBandwidth,
      };
      const resp = await apiClient.calculateSizing(req);
      setResult(resp);
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : String(err);
      setError(msg);
    } finally {
      setLoading(false);
    }
  }

  const scales = ["0.6B", "2B", "7B", "70B", "405B"];

  return (
    <div>
      <h2>Hardware Sizing Planner</h2>
      <p style={{ color: "var(--color-text-muted)", marginBottom: "1rem", fontSize: "0.9rem" }}>
        Estimate memory footprint and throughput for BitNet b1.58 model scales (Issue #8).
      </p>

      {error !== null && <ErrorState message={error} />}

      <div className="card">
        <h3>Configuration</h3>
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Model Scale</label>
            <select
              className="form-select"
              value={modelScale}
              onChange={(e) => setModelScale(e.target.value as "0.6B" | "2B" | "7B" | "70B" | "405B")}
            >
              {scales.map((s) => (
                <option key={s} value={s}>
                  BitNet {s}
                </option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-3">
            <div className="form-group">
              <label>Context Length</label>
              <input
                type="number"
                className="form-input"
                value={contextLength}
                onChange={(e) => setContextLength(Number(e.target.value))}
                min={512}
                max={131072}
              />
            </div>
            <div className="form-group">
              <label>Device Memory (GB)</label>
              <input
                type="number"
                className="form-input"
                value={deviceMemory}
                onChange={(e) => setDeviceMemory(Number(e.target.value))}
                min={2}
                max={512}
                step={0.5}
              />
            </div>
            <div className="form-group">
              <label>Memory Bandwidth (GB/s)</label>
              <input
                type="number"
                className="form-input"
                value={memoryBandwidth}
                onChange={(e) => setMemoryBandwidth(Number(e.target.value))}
                min={20}
                max={2000}
                step={10}
              />
            </div>
          </div>

          <button type="submit" className="btn btn-primary" disabled={loading}>
            {loading ? "Calculating…" : "Calculate"}
          </button>
        </form>
      </div>

      {loading && <LoadingState message="Sizing…" />}

      {result !== null && !loading && (
        <div className="card">
          <h3>Results: BitNet {result.model_scale}</h3>
          <div className="grid grid-cols-3">
            <div className="stat">
              <div className="stat-value">{result.fp16_weight_gb.toFixed(2)}</div>
              <div className="stat-label">FP16 Weight (GB)</div>
            </div>
            <div className="stat">
              <div className="stat-value">{result.ternary_weight_gb.toFixed(2)}</div>
              <div className="stat-label">Ternary Weight (GB)</div>
            </div>
            <div className="stat">
              <div className="stat-value">{result.kv_cache_gb.toFixed(3)}</div>
              <div className="stat-label">KV Cache (GB)</div>
            </div>
          </div>

          <div className="grid grid-cols-3" style={{ marginTop: "0.75rem" }}>
            <div className="stat">
              <div className="stat-value">{result.reduction_pct.toFixed(1)}%</div>
              <div className="stat-label">Weight Reduction</div>
            </div>
            <div className="stat">
              <div className="stat-value">{result.estimated_tps.toFixed(1)}</div>
              <div className="stat-label">Est. Tokens/s</div>
            </div>
            <div className="stat">
              <div className={`stat-value ${result.fits ? "" : "text-warning"}`}>
                {result.fits ? "FITS" : "EXCEEDS"}
              </div>
              <div className="stat-label">Memory</div>
            </div>
          </div>

          <div style={{ marginTop: "1rem" }}>
            {result.notes.map((note, i) => (
              <p key={i} style={{ fontSize: "0.8rem", color: "var(--color-text-muted)", marginBottom: "0.25rem" }}>
                • {note}
              </p>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
