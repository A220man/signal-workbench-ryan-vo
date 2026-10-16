import React, { useEffect, useState } from "react";
import type { HealthResponse } from "./api/types";
import { apiClient, ApiError } from "./api/client";
import { Layout } from "./components/Layout";
import { DashboardPage } from "./pages/DashboardPage";
import { ModelsPage } from "./pages/ModelsPage";
import { InferencePage } from "./pages/InferencePage";
import { BenchmarksPage } from "./pages/BenchmarksPage";
import { SizingPage } from "./pages/SizingPage";

/** Top-level application: health monitoring, tab navigation, and page routing. */
export function App(): React.JSX.Element {
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

  return (
    <Layout
      health={health}
      activeTab={activeTab}
      onTabChange={setActiveTab}
      error={globalError}
    >
      {renderPage()}
    </Layout>
  );
}