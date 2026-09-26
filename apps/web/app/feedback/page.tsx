"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const EARLIEST_SALE_DATE = "1886-01-29";
const MAX_SALE_PRICE = 10_000_000;

function utcToday(): string {
  return new Date().toISOString().slice(0, 10);
}

function apiErrorMessage(body: unknown): string {
  if (!body || typeof body !== "object") return "Feedback could not be saved.";

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

  return "Feedback could not be saved.";
}

export default function FeedbackPage() {
  const [predictionId, setPredictionId] = useState("");
  const [salePrice, setSalePrice] = useState("");
  const [status, setStatus] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const latestSaleDate = useMemo(utcToday, []);

  useEffect(() => {
    try {
      setPredictionId(localStorage.getItem("vroomvalue:lastPredictionId") ?? "");
    } catch {
      setPredictionId("");
    }
  }, []);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setStatus("");

    const form = new FormData(event.currentTarget);
    const price = Number(salePrice);
    const saleDate = String(form.get("sale_date") ?? "");

    if (!salePrice || !Number.isFinite(price) || price <= 0 || price > MAX_SALE_PRICE) {
      setStatus("Actual sale price must be greater than $0 and no more than $10,000,000.");
      return;
    }

    if (!saleDate || saleDate < EARLIEST_SALE_DATE || saleDate > latestSaleDate) {
      setStatus(
        `Sale date must be between ${EARLIEST_SALE_DATE} and ${latestSaleDate}.`,
      );
      return;
    }

    setSubmitting(true);
    try {
      const response = await fetch(`${API}/v1/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prediction_id: predictionId,
          actual_sale_price: price,
          sale_date: saleDate,
        }),
      });
      const body: unknown = await response.json().catch(() => null);
      setStatus(
        response.ok
          ? "Feedback accepted for review."
          : apiErrorMessage(body),
      );
    } catch {
      setStatus("Feedback service could not be reached.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="shell narrow">
      <header className="page-head">
        <span className="eyebrow">VERIFIED OUTCOME CANDIDATE</span>
        <h1>Sale feedback</h1>
        <p>
          Attach the realized sale price to a prediction. Feedback is stored
          separately and is not added to training automatically.
        </p>
      </header>

      <form className="spec-sheet feedback-form" onSubmit={submit}>
        <div className="sheet-title">
          <span>SALE RECORD</span>
          <span>Form VV-02</span>
        </div>

        <label>
          Prediction ID
          <input
            name="prediction_id"
            value={predictionId}
            onChange={(event) => setPredictionId(event.target.value)}
            required
          />
        </label>

        <label>
          Actual sale price (USD)
          <input
            name="actual_sale_price"
            type="text"
            inputMode="decimal"
            value={salePrice}
            onChange={(event) => {
              const next = event.target.value;
              if (/^\d*(?:\.\d{0,2})?$/.test(next)) setSalePrice(next);
            }}
            placeholder="28000.00"
            aria-describedby="sale-price-help"
            required
          />
          <small id="sale-price-help">
            Positive USD amount only; maximum $10,000,000.
          </small>
        </label>

        <label>
          Sale date
          <input
            name="sale_date"
            type="date"
            min={EARLIEST_SALE_DATE}
            max={latestSaleDate}
            required
          />
          <small>
            Allowed range: {EARLIEST_SALE_DATE} through {latestSaleDate}.
          </small>
        </label>

        <button disabled={submitting}>
          {submitting ? "Submitting…" : "Submit feedback"}
        </button>
        {status && (
          <p className="form-status" role="status" aria-live="polite">
            {status}
          </p>
        )}
      </form>
    </main>
  );
}
