import React, { useEffect, useState } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import type { HealthResponse } from "./api/types";
import { apiClient, ApiError } from "./api/client";
import { Layout } from "./components/Layout";
import { DashboardPage } from "./pages/DashboardPage";
import { ModelsPage } from "./pages/ModelsPage";
import { InferencePage } from "./pages/InferencePage";
import { BenchmarksPage } from "./pages/BenchmarksPage";
import { SizingPage } from "./pages/SizingPage";
import { AuthProvider, useAuth } from "./auth/AuthContext";
import { LoginPage } from "./auth/LoginPage";
import { CallbackPage } from "./auth/CallbackPage";
import { LoadingState } from "./components/LoadingState";

/** Inner app with routing, wrapped by AuthProvider. */
function AppInner(): React.JSX.Element {
  const { isAuthenticated, isLoading: authLoading, refreshAuth } = useAuth();
  const [activeTab, setActiveTab] = useState("dashboard");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [globalError, setGlobalError] = useState<string | null>(null);

  useEffect(() => {
    void (async () => {
      try {
        const h = await apiClient.getHealth();
        setHealth(h);
      } catch (err) {
        const msg = err instanceof ApiError ? err.message : String(err);
        setGlobalError(msg);
      }
    })();
  }, []);

  // Refresh auth when health loads (demo mode may not need OIDC)
  useEffect(() => {
    if (health?.demo_mode) {
      // In demo mode, backend accepts requests without token
      return;
    }
    void refreshAuth();
  }, [health, refreshAuth]);

  const renderPage = (): React.ReactNode => {
    switch (activeTab) {
      case "models":
        return <ModelsPage />;
      case "inference":
        return <InferencePage />;
      case "benchmarks":
        return <BenchmarksPage />;
      case "sizing":
        return <SizingPage />;
      default:
        return <DashboardPage />;
    }
  };

  // Protected routes wrapper
  const ProtectedRoute = ({ children }: { children: React.ReactNode }) => {
    if (authLoading) return <LoadingState message="Verifying authentication…" />;
    if (!health?.demo_mode && !isAuthenticated) {
      return <Navigate to="/login" replace />;
    }
    return <>{children}</>;
  };

  return (
    <Layout
      health={health}
      activeTab={activeTab}
      onTabChange={setActiveTab}
      error={globalError}
    >
      <Routes>
        <Route
          path="/"
          element={
            <ProtectedRoute>
              {renderPage()}
            </ProtectedRoute>
          }
        />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/callback" element={<CallbackPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}

/** Top-level application with routing and auth provider. */
export function App(): React.JSX.Element {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AppInner />
      </AuthProvider>
    </BrowserRouter>
  );
}