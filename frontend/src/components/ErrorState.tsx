import React from "react";

interface ErrorStateProps {
  message: string;
  onRetry?: () => void;
}

/** Centered error display with optional retry button. */
export function ErrorState({ message, onRetry }: ErrorStateProps): React.JSX.Element {
  return (
    <div className="state-message" style={{ color: "var(--color-danger)" }}>
      <div className="icon">⚠</div>
      <div style={{ marginTop: "0.5rem", marginBottom: "0.75rem" }}>{message}</div>
      {onRetry !== undefined && (
        <button className="btn btn-sm" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}