import React, { ReactNode } from "react";
import type { HealthResponse } from "../api/types";

interface LayoutProps {
  health: HealthResponse | null;
  activeTab: string;
  onTabChange: (tab: string) => void;
  error: string | null;
  children: ReactNode;
}

/** Application shell: header, navigation tabs, main content, footer. */
export function Layout({
  health,
  activeTab,
  onTabChange,
  error,
  children,
}: LayoutProps): React.JSX.Element {
  const tabs = [
    { id: "dashboard", label: "Dashboard" },
    { id: "models", label: "Models" },
    { id: "inference", label: "Inference" },
    { id: "benchmarks", label: "Benchmarks" },
    { id: "sizing", label: "Sizing" },
  ];

  return (
    <div className="app-container">
      <header className="app-header">
        <h1 className="app-title">
          Signal <span className="accent">Workbench</span>
        </h1>
        <div className="header-status">
          {health ? (
            <>
              {health.demo_mode ? "Demo mode" : "Production"} · v{health.version}
            </>
          ) : (
            "Checking…"
          )}
        </div>
      </header>

      <nav className="app-nav">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            className={`nav-tab ${activeTab === tab.id ? "active" : ""}`}
            onClick={() => onTabChange(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      {error !== null && error !== "" && (
        <div
          style={{
            padding: "0.5rem 1.5rem",
            background: "rgba(248, 113, 113, 0.1)",
            color: "var(--color-danger)",
            fontSize: "0.8rem",
            borderBottom: "1px solid var(--color-danger)",
          }}
        >
          {error}
        </div>
      )}

      <main className="app-main">{children}</main>

      <footer className="app-footer">
        Signal Workbench — 1-bit LLM inference &amp; calibration framework
      </footer>
    </div>
  );
}