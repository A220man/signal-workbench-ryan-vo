/** Callback page: handles OIDC authorization code redirect. */
import React, { useEffect, useState } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import { handleCallback } from "./oidc";
import { LoadingState } from "../components/LoadingState";
import { ErrorState } from "../components/ErrorState";

/** OIDC callback route (/callback). */
export function CallbackPage(): React.JSX.Element {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  useEffect(() => {
    void (async () => {
      const code = searchParams.get("code");
      const state = searchParams.get("state");
      const oauthError = searchParams.get("error");
      const errorDesc = searchParams.get("error_description");

      if (oauthError) {
        setError(`${oauthError}: ${errorDesc ?? "Unknown error"}`);
        return;
      }

      if (!code || !state) {
        setError("Missing authorization code or state parameter");
        return;
      }

      try {
        await handleCallback(code, state);
        setDone(true);
        // Small delay to show success, then redirect to dashboard
        setTimeout(() => navigate("/", { replace: true }), 500);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    })();
  }, [searchParams, navigate]);

  if (done) {
    return <LoadingState message="Authentication successful, redirecting…" />;
  }

  if (error) {
    return (
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: "60vh" }}>
        <ErrorState message={error} onRetry={() => navigate("/")} />
      </div>
    );
  }

  return <LoadingState message="Completing authentication…" />;
}
