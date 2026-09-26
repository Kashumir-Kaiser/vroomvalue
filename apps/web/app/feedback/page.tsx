"use client";

import { FormEvent, useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function FeedbackPage() {
  const [predictionId, setPredictionId] = useState("");
  const [status, setStatus] = useState("");
  useEffect(() => { setPredictionId(localStorage.getItem("vroomvalue:lastPredictionId") ?? ""); }, []);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setStatus("");
    const fd = new FormData(event.currentTarget);
    const response = await fetch(`${API}/v1/feedback`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prediction_id: String(fd.get("prediction_id")), actual_sale_price: Number(fd.get("actual_sale_price")), sale_date: String(fd.get("sale_date")) }),
    });
    const body = await response.json();
    setStatus(response.ok ? "Feedback accepted for review." : String(body.detail ?? "Feedback could not be saved."));
  }

  return <main className="shell narrow"><header className="page-head"><span className="eyebrow">VERIFIED OUTCOME CANDIDATE</span><h1>Sale feedback</h1><p>Attach the realized sale price to a prediction. Feedback is stored separately and is not added to training automatically.</p></header>
    <form className="spec-sheet feedback-form" onSubmit={submit}>
      <div className="sheet-title"><span>SALE RECORD</span><span>Form VV-02</span></div>
      <label>Prediction ID<input name="prediction_id" value={predictionId} onChange={e=>setPredictionId(e.target.value)} required /></label>
      <label>Actual sale price (USD)<input name="actual_sale_price" type="number" min="1" step="1" required /></label>
      <label>Sale date<input name="sale_date" type="date" required /></label>
      <button>Submit feedback</button>{status && <p className="form-status" role="status">{status}</p>}
    </form>
  </main>;
}