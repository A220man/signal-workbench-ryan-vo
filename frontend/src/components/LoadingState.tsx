import React from "react";

interface LoadingStateProps {
  message?: string;
}

/** Full-width loading spinner with optional label. */
export function LoadingState({ message = "Loading…" }: LoadingStateProps): React.JSX.Element {
  return (
    <div className="state-message">
      <div className="spinner" />
      <div style={{ marginTop: "0.75rem" }}>{message}</div>
    </div>
  );
}