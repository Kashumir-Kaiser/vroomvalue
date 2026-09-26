"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

export type Metadata = {
  categories: Record<string, string[]>;
  make_models: Record<string, string[]>;
  field_rules: { engine_size: { min: number; max: number; step: number } };
};

type Prediction = {
  prediction_id: string;
  estimated_price: { amount: number; currency: string };
  interval_80: { lower: number; upper: number };
  support: "in_distribution" | "low_confidence";
  top_factors: { feature: string; direction: "up" | "down" }[];
  warnings: string[];
  model: { version: string; as_of_date: string };
};

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const money = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

function selectOptions(values: string[], unknown = false) {
  return (
    <>
      {unknown && <option value="">Unknown</option>}
      {values.map((value) => (
        <option key={value} value={value}>{value}</option>
      ))}
    </>
  );
}

function apiErrorMessage(body: unknown, fallback: string): string {
  if (!body || typeof body !== "object") return fallback;
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
  return fallback;
}

export default function PredictionForm({
  initialMetadata,
}: {
  initialMetadata: Metadata | null;
}) {
  const [meta, setMeta] = useState<Metadata | null>(initialMetadata);
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const firstMake = initialMetadata?.categories?.Make?.[0] ?? "";
  const firstModel =
    (firstMake && initialMetadata?.make_models?.[firstMake]?.[0]) ?? "";
  const [make, setMake] = useState(firstMake);
  const [model, setModel] = useState(firstModel);
  const models = useMemo(() => meta?.make_models?.[make] ?? [], [meta, make]);

  useEffect(() => {
    if (meta) return;
    let cancelled = false;

    async function loadMetadata() {
      try {
        const response = await fetch(`${API}/v1/metadata`, { cache: "no-store" });
        const body: unknown = await response.json().catch(() => null);
        if (!response.ok) {
          throw new Error(apiErrorMessage(body, "Service is not ready."));
        }
        if (!cancelled) {
          const loaded = body as Metadata;
          setMeta(loaded);
          const loadedMake = loaded.categories?.Make?.[0] ?? "";
          setMake(loadedMake);
          setModel(
            (loadedMake && loaded.make_models?.[loadedMake]?.[0]) ?? "",
          );
        }
      } catch (caught) {
        if (!cancelled) {
          setError(caught instanceof Error ? caught.message : "Service is not ready.");
        }
      }
    }

    void loadMetadata();
    return () => { cancelled = true; };
  }, [meta]);

  useEffect(() => {
    if (!models.length) return;
    setModel((current) => models.includes(current) ? current : models[0]);
  }, [models]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    const fd = new FormData(event.currentTarget);
    const optionalText = (name: string) => String(fd.get(name) ?? "") || null;
    const optionalNumber = (name: string) =>
      String(fd.get(name) ?? "") === "" ? null : Number(fd.get(name));

    const payload = {
      make: String(fd.get("make")),
      model: String(fd.get("model")),
      year: Number(fd.get("year")),
      fuel_type: String(fd.get("fuel_type")),
      transmission: optionalText("transmission"),
      engine_size: optionalNumber("engine_size"),
      mileage: Number(fd.get("mileage")),
      horsepower: optionalNumber("horsepower"),
      torque: optionalNumber("torque"),
      owners: Number(fd.get("owners")),
      accident_history: optionalNumber("accident_history"),
      service_history: optionalText("service_history"),
      color: optionalText("color"),
      body_type: String(fd.get("body_type")),
      drivetrain: String(fd.get("drivetrain")),
      fuel_efficiency: optionalNumber("fuel_efficiency"),
      location: optionalText("location"),
    };

    try {
      const response = await fetch(`${API}/v1/predictions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body: unknown = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error(apiErrorMessage(body, "Prediction failed."));
      }

      const prediction = body as Prediction;
      setResult(prediction);
      try {
        localStorage.setItem("vroomvalue:lastPredictionId", prediction.prediction_id);
      } catch {
        // Prediction succeeded even if browser storage is unavailable.
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Prediction failed.");
    } finally {
      setLoading(false);
    }
  }

  const lanePosition = result
    ? Math.max(
        0,
        Math.min(
          100,
          ((result.estimated_price.amount - result.interval_80.lower)
            / Math.max(1, result.interval_80.upper - result.interval_80.lower))
            * 100,
        ),
      )
    : 50;

  return (
    <main className="shell">
      <header className="masthead">
        <div>
          <span className="eyebrow">USED VEHICLE PRICING</span>
          <h1>VroomValue</h1>
        </div>
        <div className="stamp">MODEL-ASSISTED<br />MARKET ESTIMATE</div>
      </header>

      <section className="intro">
        <p>
          Enter the vehicle as it sits today. Unknown optional fields stay
          unknown — the estimator never fills in a favorable answer for you.
        </p>
        <div className="unit-strip">
          <b>Display assumptions</b>
          <span>Mileage: miles</span>
          <span>Torque: lb-ft</span>
          <span>Efficiency: MPG / MPGe</span>
          <span>Price: USD</span>
        </div>
      </section>

      <div className="workspace">
        <form className="spec-sheet" onSubmit={submit}>
          <div className="sheet-title">
            <span>VEHICLE SPECIFICATION</span>
            <span>Form VV-01</span>
          </div>

          <div className="grid">
            <label>
              Make
              <select name="make" value={make} onChange={(e) => setMake(e.target.value)} disabled={!meta} required>
                {meta
                  ? selectOptions(meta.categories?.Make ?? [])
                  : <option value="">Loading makes…</option>}
              </select>
            </label>
            <label>
              Model
              <select
                name="model"
                value={model}
                onChange={(e) => setModel(e.target.value)}
                disabled={!meta}
                required
              >
                {meta
                  ? selectOptions(models)
                  : <option value="">Loading models…</option>}
              </select>
            </label>
            <label>Year<input name="year" type="number" min="2005" max="2024" defaultValue="2022" required /></label>
            <label>Fuel type<select name="fuel_type" disabled={!meta}>{meta ? selectOptions(meta.categories?.Fuel_Type ?? []) : <option value="">Loading…</option>}</select></label>
            <label>Transmission<select name="transmission" disabled={!meta}>{meta ? selectOptions(meta.categories?.Transmission ?? [], true) : <option value="">Loading…</option>}</select></label>
            <label>Engine size (L)<input name="engine_size" type="number" min="0" max="5.7" step="0.1" placeholder="Unknown" defaultValue="2.5" /></label>
            <label>Mileage (mi)<input name="mileage" type="number" min="0" defaultValue="42000" required /></label>
            <label>Horsepower<input name="horsepower" type="number" step="0.1" placeholder="Unknown" defaultValue="203" /></label>
            <label>Torque (lb-ft)<input name="torque" type="number" step="0.1" placeholder="Unknown" defaultValue="184" /></label>
            <label>Owners<input name="owners" type="number" min="1" defaultValue="1" required /></label>
            <label>Accident history<select name="accident_history"><option value="">Unknown</option><option value="0">No recorded accident</option><option value="1">Recorded accident</option></select></label>
            <label>Service history<select name="service_history" disabled={!meta}>{meta ? selectOptions(meta.categories?.Service_History ?? [], true) : <option value="">Loading…</option>}</select></label>
            <label>Color<select name="color" disabled={!meta}>{meta ? selectOptions(meta.categories?.Color ?? [], true) : <option value="">Loading…</option>}</select></label>
            <label>Body type<select name="body_type" disabled={!meta}>{meta ? selectOptions(meta.categories?.Body_Type ?? []) : <option value="">Loading…</option>}</select></label>
            <label>Drivetrain<select name="drivetrain" disabled={!meta}>{meta ? selectOptions(meta.categories?.Drivetrain ?? []) : <option value="">Loading…</option>}</select></label>
            <label>Fuel efficiency (MPG/MPGe)<input name="fuel_efficiency" type="number" step="0.1" placeholder="Unknown" defaultValue="32" /></label>
            <label>Location<select name="location" disabled={!meta}>{meta ? selectOptions(meta.categories?.Location ?? [], true) : <option value="">Loading…</option>}</select></label>
          </div>

          <button disabled={loading || !meta}>
            {loading ? "Estimating…" : "Estimate market price"}
          </button>
          {error && <p className="error" role="alert">{error}</p>}
        </form>

        <aside className="result-card" aria-live="polite">
          <div className="sheet-title">
            <span>PRICE WINDOW</span>
            <span>{result ? `Model ${result.model.version}` : "Awaiting vehicle"}</span>
          </div>

          {!result ? (
            <div className="empty">
              <strong>No estimate yet.</strong>
              <p>
                Complete the specification sheet to generate a point estimate,
                calibrated 80% range, and the strongest model factors.
              </p>
            </div>
          ) : (
            <>
              <span className={`support ${result.support}`}>
                {result.support === "in_distribution"
                  ? "IN TRAINING SUPPORT"
                  : "LOW SUPPORT"}
              </span>
              <div className="price copyable">{money.format(result.estimated_price.amount)}</div>
              <p className="prediction-id">
                Prediction ID: <code className="copyable">{result.prediction_id}</code>
              </p>
              <div className="range-label">
                <span>{money.format(result.interval_80.lower)}</span>
                <b>80% RANGE</b>
                <span>{money.format(result.interval_80.upper)}</span>
              </div>
              <div className="price-lane">
                <div className="lane-fill" />
                <div className="lane-marker" style={{ left: `${lanePosition}%` }}>
                  <span>{money.format(result.estimated_price.amount)}</span>
                </div>
              </div>
              <div className="factors">
                <h2>What moved this estimate</h2>
                {result.top_factors.map((factor, index) => (
                  <div className="factor" key={`${factor.feature}-${index}`}>
                    <span>{factor.feature.replaceAll("_", " ")}</span>
                    <b>{factor.direction === "up" ? "↑ higher" : "↓ lower"}</b>
                  </div>
                ))}
              </div>
              {result.warnings.length > 0 && (
                <div className="warnings">
                  <b>Support notes</b>
                  {result.warnings.map((warning, index) => (
                    <p key={`${warning}-${index}`}>{warning}</p>
                  ))}
                </div>
              )}
              <a className="feedback-link" href="/feedback">
                Report the actual sale price →
              </a>
              <p className="disclaimer">
                Decision assistance only — not an appraisal guarantee. Model
                factors describe model behavior, not causal price effects.
              </p>
            </>
          )}
        </aside>
      </div>
    </main>
  );
}
