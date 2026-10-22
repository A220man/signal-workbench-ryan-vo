/** Login page: initiates OIDC PKCE flow. */
import React, { useEffect, useState } from "react";
import { useAuth } from "./AuthContext";
import { LoadingState } from "../components/LoadingState";
import { ErrorState } from "../components/ErrorState";

export function LoginPage(): React.JSX.Element {
  const { login, isLoading } = useAuth();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Auto-initiate login on mount
    void (async () => {
      try {
        await login();
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    })();
  }, [login]);

  if (isLoading) {
    return <LoadingState message="Redirecting to identity provider…" />;
  }

  if (error) {
    return (
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: "60vh" }}>
        <ErrorState message={error} />
      </div>
    );
  }

  return <LoadingState message="Preparing login…" />;
}
