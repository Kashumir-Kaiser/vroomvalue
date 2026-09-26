"use client";

import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Metrics = {
  prediction_count: number;
  feedback_count: number;
  request_count: number;
  error_rate: number;
  invalid_input_rate: number;
  avg_latency_ms: number;
  current_model_version: string | null;
  readiness: string;
};

function errorMessage(body: unknown): string {
  if (!body || typeof body !== "object") return "Metrics unavailable";
  const detail = (body as { detail?: unknown }).detail;
  return typeof detail === "string" ? detail : "Metrics unavailable";
}

export default function AdminPage() {
  const [data, setData] = useState<Metrics | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [lastRefreshed, setLastRefreshed] = useState<Date | null>(null);

  async function load() {
    setLoading(true);
    setError("");

    try {
      const response = await fetch(
        `${API}/v1/admin/metrics?refresh=${Date.now()}`,
        { cache: "no-store" },
      );
      const body: unknown = await response.json().catch(() => null);

      if (!response.ok) throw new Error(errorMessage(body));
      setData(body as Metrics);
      setLastRefreshed(new Date());
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Metrics unavailable");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
  }, []);

  return (
    <main className="shell narrow">
      <header className="page-head">
        <span className="eyebrow">OPERATIONS STATUS</span>
        <h1>Admin</h1>
        <p>
          Minimal runtime view for the MVP. These counters are operational
          signals, not model-quality metrics.
        </p>
      </header>

      <section className="admin-sheet">
        <div className="sheet-title">
          <span>LIVE SUMMARY</span>
          <button
            type="button"
            className="mini-button"
            onClick={() => void load()}
            disabled={loading}
          >
            {loading ? "Refreshing…" : "Refresh"}
          </button>
        </div>

        <p className="refresh-status" role="status" aria-live="polite">
          {loading
            ? "Fetching current metrics…"
            : lastRefreshed
              ? `Last refreshed: ${lastRefreshed.toLocaleTimeString()}`
              : "Waiting for metrics…"}
        </p>

        {error && <p className="error">{error}</p>}
        {data && (
          <table>
            <tbody>
              <tr><th>Readiness</th><td>{data.readiness}</td></tr>
              <tr><th>Model version</th><td>{data.current_model_version ?? "—"}</td></tr>
              <tr><th>Requests</th><td>{data.request_count}</td></tr>
              <tr><th>Predictions</th><td>{data.prediction_count}</td></tr>
              <tr><th>Feedback records</th><td>{data.feedback_count}</td></tr>
              <tr><th>Error rate</th><td>{(data.error_rate * 100).toFixed(2)}%</td></tr>
              <tr><th>Invalid-input rate</th><td>{(data.invalid_input_rate * 100).toFixed(2)}%</td></tr>
              <tr><th>Average latency</th><td>{data.avg_latency_ms.toFixed(2)} ms</td></tr>
            </tbody>
          </table>
        )}
      </section>
    </main>
  );
}
