"use client";

import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const money = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

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

type FeedbackItem = {
  id: number;
  prediction_id: string;
  actual_sale_price: number;
  sale_date: string;
  submitted_at: string;
  predicted_price: number;
  interval_lower: number;
  interval_upper: number;
  support: string;
  model_version: string;
  review_status: "pending" | "accepted" | "closed";
  reviewed_at: string | null;
};

function errorMessage(body: unknown): string {
  if (!body || typeof body !== "object") return "Admin data unavailable";
  const detail = (body as { detail?: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        if (!item || typeof item !== "object") return "";
        const message = (item as { msg?: unknown }).msg;
        return typeof message === "string" ? message : "";
      })
      .filter(Boolean);
    if (messages.length) return messages.join(" ");
  }
  return "Admin data unavailable";
}

function adminHeaders(token: string, json = false): HeadersInit {
  const headers: Record<string, string> = {};
  if (token) headers["X-Admin-Token"] = token;
  if (json) headers["Content-Type"] = "application/json";
  return headers;
}

export default function AdminPage() {
  const [data, setData] = useState<Metrics | null>(null);
  const [feedback, setFeedback] = useState<FeedbackItem[] | null>(null);
  const [adminToken, setAdminToken] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [loading, setLoading] = useState(false);
  const [reviewing, setReviewing] = useState<number | null>(null);
  const [lastRefreshed, setLastRefreshed] = useState<Date | null>(null);

  async function load(token = adminToken) {
    setLoading(true);
    setError("");

    try {
      const headers = adminHeaders(token);
      const [metricsResponse, feedbackResponse] = await Promise.all([
        fetch(`${API}/v1/admin/metrics`, {
          cache: "no-store",
          headers,
        }),
        fetch(`${API}/v1/admin/feedback`, {
          cache: "no-store",
          headers,
        }),
      ]);
      const [metricsBody, feedbackBody]: [unknown, unknown] = await Promise.all([
        metricsResponse.json().catch(() => null),
        feedbackResponse.json().catch(() => null),
      ]);

      if (!metricsResponse.ok) throw new Error(errorMessage(metricsBody));
      if (!feedbackResponse.ok) throw new Error(errorMessage(feedbackBody));

      setData(metricsBody as Metrics);
      setFeedback(feedbackBody as FeedbackItem[]);
      setLastRefreshed(new Date());
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Admin data unavailable");
    } finally {
      setLoading(false);
    }
  }

  async function review(feedbackId: number, status: "accepted" | "closed") {
    setReviewing(feedbackId);
    setError("");
    setNotice("");

    try {
      const response = await fetch(
        `${API}/v1/admin/feedback/${feedbackId}/review`,
        {
          method: "POST",
          headers: adminHeaders(adminToken, true),
          body: JSON.stringify({ status }),
        },
      );
      const body: unknown = await response.json().catch(() => null);
      if (!response.ok) throw new Error(errorMessage(body));

      setNotice(
        status === "accepted"
          ? `Feedback #${feedbackId} accepted.`
          : `Feedback #${feedbackId} closed.`,
      );
      await load(adminToken);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Review update failed");
    } finally {
      setReviewing(null);
    }
  }

  useEffect(() => {
    let storedToken = "";
    try {
      storedToken = sessionStorage.getItem("vroomvalue:adminToken") ?? "";
    } catch {
      storedToken = "";
    }
    setAdminToken(storedToken);
    void load(storedToken);
  }, []);

  function updateToken(value: string) {
    setAdminToken(value);
    try {
      if (value) {
        sessionStorage.setItem("vroomvalue:adminToken", value);
      } else {
        sessionStorage.removeItem("vroomvalue:adminToken");
      }
    } catch {
      // The token can still be used for this page load if storage is unavailable.
    }
  }

  return (
    <main className="shell narrow admin-page">
      <header className="page-head">
        <span className="eyebrow">OPERATIONS STATUS</span>
        <h1>Admin</h1>
        <p>
          Runtime health and a human review queue for submitted sale outcomes.
          Accept only feedback you consider credible; closed feedback remains
          stored for audit but is not treated as approved.
        </p>
      </header>

      <section className="admin-auth">
        <label>
          Admin token
          <input
            type="password"
            autoComplete="off"
            value={adminToken}
            onChange={(event) => updateToken(event.target.value)}
            placeholder="Only required when ADMIN_TOKEN is configured"
          />
        </label>
        <button type="button" onClick={() => void load()} disabled={loading}>
          Apply and refresh
        </button>
      </section>

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
            ? "Fetching current admin data…"
            : lastRefreshed
              ? `Last refreshed: ${lastRefreshed.toLocaleTimeString()}`
              : "Waiting for metrics…"}
        </p>

        {error && <p className="error">{error}</p>}
        {notice && <p className="form-status">{notice}</p>}
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

      <section className="admin-sheet feedback-queue">
        <div className="sheet-title">
          <span>FEEDBACK REVIEW QUEUE</span>
          <span>{feedback ? `${feedback.length} RECORDS` : "LOADING"}</span>
        </div>

        {feedback === null && (
          <div className="review-skeleton" aria-label="Loading feedback">
            <span />
            <span />
            <span />
          </div>
        )}

        {feedback?.length === 0 && (
          <p className="queue-empty">No feedback has been submitted yet.</p>
        )}

        {feedback?.map((item) => {
          const delta = item.actual_sale_price - item.predicted_price;
          return (
            <article className="feedback-review-card" key={item.id}>
              <div className="feedback-review-head">
                <div>
                  <b>Feedback #{item.id}</b>
                  <span className={`review-status ${item.review_status}`}>
                    {item.review_status}
                  </span>
                </div>
                <time dateTime={item.submitted_at}>
                  {new Date(item.submitted_at).toLocaleString()}
                </time>
              </div>

              <dl className="feedback-review-grid">
                <div><dt>Actual sale</dt><dd>{money.format(item.actual_sale_price)}</dd></div>
                <div><dt>Model estimate</dt><dd>{money.format(item.predicted_price)}</dd></div>
                <div><dt>Difference</dt><dd>{money.format(delta)}</dd></div>
                <div><dt>Sale date</dt><dd>{item.sale_date}</dd></div>
                <div><dt>80% range</dt><dd>{money.format(item.interval_lower)} – {money.format(item.interval_upper)}</dd></div>
                <div><dt>Support</dt><dd>{item.support.replaceAll("_", " ")}</dd></div>
              </dl>

              <p className="prediction-reference">
                Prediction: <code>{item.prediction_id}</code> · Model {item.model_version}
              </p>

              <div className="review-actions">
                <button
                  type="button"
                  className="accept-button"
                  disabled={reviewing === item.id || item.review_status === "accepted"}
                  onClick={() => void review(item.id, "accepted")}
                >
                  {item.review_status === "accepted" ? "Accepted" : "Accept"}
                </button>
                <button
                  type="button"
                  className="close-button"
                  disabled={reviewing === item.id || item.review_status === "closed"}
                  onClick={() => void review(item.id, "closed")}
                >
                  {item.review_status === "closed" ? "Closed" : "Close"}
                </button>
              </div>
            </article>
          );
        })}
      </section>
    </main>
  );
}
