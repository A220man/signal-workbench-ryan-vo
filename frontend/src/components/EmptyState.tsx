import React from "react";

interface EmptyStateProps {
  message: string;
  actionLabel?: string;
  onAction?: () => void;
}

/** Centered empty state with optional action button. */
export function EmptyState({ message, actionLabel, onAction }: EmptyStateProps): React.JSX.Element {
  return (
    <div className="state-message">
      <div className="icon">📋</div>
      <div style={{ marginTop: "0.5rem", marginBottom: "0.75rem" }}>{message}</div>
      {actionLabel !== undefined && onAction !== undefined && (
        <button className="btn btn-primary btn-sm" onClick={onAction}>
          {actionLabel}
        </button>
      )}
    </div>
  );
}