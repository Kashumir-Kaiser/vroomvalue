"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

type Metadata = {
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
const money = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });

function selectOptions(values: string[], unknown = false) {
  return <>{unknown && <option value="">Unknown</option>}{values.map(v => <option key={v} value={v}>{v}</option>)}</>;
}

export default function PredictionForm() {
  const [meta, setMeta] = useState<Metadata | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [make, setMake] = useState("Toyota");
  const models = useMemo(() => meta?.make_models?.[make] ?? [], [meta, make]);

  useEffect(() => {
    fetch(`${API}/v1/metadata`).then(async r => {
      if (!r.ok) throw new Error((await r.json()).detail ?? "Service is not ready.");
      return r.json();
    }).then(setMeta).catch(e => setError(String(e.message ?? e)));
  }, []);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setLoading(true); setError(""); setResult(null);
    const fd = new FormData(event.currentTarget);
    const optionalText = (name: string) => String(fd.get(name) ?? "") || null;
    const optionalNumber = (name: string) => String(fd.get(name) ?? "") === "" ? null : Number(fd.get(name));
    const payload = {
      make: String(fd.get("make")), model: String(fd.get("model")), year: Number(fd.get("year")),
      fuel_type: String(fd.get("fuel_type")), transmission: optionalText("transmission"),
      engine_size: optionalNumber("engine_size"), mileage: Number(fd.get("mileage")),
      horsepower: optionalNumber("horsepower"), torque: optionalNumber("torque"), owners: Number(fd.get("owners")),
      accident_history: optionalNumber("accident_history"), service_history: optionalText("service_history"),
      color: optionalText("color"), body_type: String(fd.get("body_type")), drivetrain: String(fd.get("drivetrain")),
      fuel_efficiency: optionalNumber("fuel_efficiency"), location: optionalText("location"),
    };
    try {
      const response = await fetch(`${API}/v1/predictions`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail?.[0]?.msg ?? body.detail ?? "Prediction failed.");
      setResult(body);
      localStorage.setItem("vroomvalue:lastPredictionId", body.prediction_id);
    } catch (e) { setError(e instanceof Error ? e.message : "Prediction failed."); }
    finally { setLoading(false); }
  }

  const lanePosition = result ? Math.max(0, Math.min(100, ((result.estimated_price.amount - result.interval_80.lower) / Math.max(1, result.interval_80.upper - result.interval_80.lower)) * 100)) : 50;

  return <main className="shell">
    <header className="masthead"><div><span className="eyebrow">USED VEHICLE PRICING</span><h1>VroomValue</h1></div><div className="stamp">MODEL-ASSISTED<br/>MARKET ESTIMATE</div></header>
    <section className="intro"><p>Enter the vehicle as it sits today. Unknown optional fields stay unknown — the estimator never fills in a favorable answer for you.</p><div className="unit-strip"><b>Display assumptions</b><span>Mileage: miles</span><span>Torque: lb-ft</span><span>Efficiency: MPG / MPGe</span><span>Price: USD</span></div></section>

    <div className="workspace">
      <form className="spec-sheet" onSubmit={submit}>
        <div className="sheet-title"><span>VEHICLE SPECIFICATION</span><span>Form VV-01</span></div>
        <div className="grid">
          <label>Make<select name="make" value={make} onChange={e => setMake(e.target.value)} required>{selectOptions(meta?.categories?.Make ?? ["Toyota"])}</select></label>
          <label>Model<select name="model" required>{selectOptions(models.length ? models : ["Camry"])}</select></label>
          <label>Year<input name="year" type="number" min="2005" max="2024" defaultValue="2022" required /></label>
          <label>Fuel type<select name="fuel_type">{selectOptions(meta?.categories?.Fuel_Type ?? ["Petrol","Diesel","Electric","Hybrid"])}</select></label>
          <label>Transmission<select name="transmission">{selectOptions(meta?.categories?.Transmission ?? ["Automatic","Manual"], true)}</select></label>
          <label>Engine size (L)<input name="engine_size" type="number" min="0" max="5.7" step="0.1" placeholder="Unknown" defaultValue="2.5" /></label>
          <label>Mileage (mi)<input name="mileage" type="number" min="0" defaultValue="42000" required /></label>
          <label>Horsepower<input name="horsepower" type="number" step="0.1" placeholder="Unknown" defaultValue="203" /></label>
          <label>Torque (lb-ft)<input name="torque" type="number" step="0.1" placeholder="Unknown" defaultValue="184" /></label>
          <label>Owners<input name="owners" type="number" min="1" defaultValue="1" required /></label>
          <label>Accident history<select name="accident_history"><option value="">Unknown</option><option value="0">No recorded accident</option><option value="1">Recorded accident</option></select></label>
          <label>Service history<select name="service_history">{selectOptions(meta?.categories?.Service_History ?? ["No Service","Partial Service","Full Service"], true)}</select></label>
          <label>Color<select name="color">{selectOptions(meta?.categories?.Color ?? ["White","Black","Silver"], true)}</select></label>
          <label>Body type<select name="body_type">{selectOptions(meta?.categories?.Body_Type ?? ["Sedan","SUV","Hatchback","Coupe","Truck"])}</select></label>
          <label>Drivetrain<select name="drivetrain">{selectOptions(meta?.categories?.Drivetrain ?? ["FWD","RWD","AWD","4WD"])}</select></label>
          <label>Fuel efficiency (MPG/MPGe)<input name="fuel_efficiency" type="number" step="0.1" placeholder="Unknown" defaultValue="32" /></label>
          <label>Location<select name="location">{selectOptions(meta?.categories?.Location ?? ["TX","CA","NY"], true)}</select></label>
        </div>
        <button disabled={loading || !meta}>{loading ? "Estimating…" : "Estimate market price"}</button>
        {error && <p className="error" role="alert">{error}</p>}
      </form>

      <aside className="result-card" aria-live="polite">
        <div className="sheet-title"><span>PRICE WINDOW</span><span>{result ? `Model ${result.model.version}` : "Awaiting vehicle"}</span></div>
        {!result ? <div className="empty"><strong>No estimate yet.</strong><p>Complete the specification sheet to generate a point estimate, calibrated 80% range, and the strongest model factors.</p></div> : <>
          <span className={`support ${result.support}`}>{result.support === "in_distribution" ? "IN TRAINING SUPPORT" : "LOW SUPPORT"}</span>
          <div className="price">{money.format(result.estimated_price.amount)}</div>
          <div className="range-label"><span>{money.format(result.interval_80.lower)}</span><b>80% RANGE</b><span>{money.format(result.interval_80.upper)}</span></div>
          <div className="price-lane"><div className="lane-fill"/><div className="lane-marker" style={{left:`${lanePosition}%`}}><span>{money.format(result.estimated_price.amount)}</span></div></div>
          <div className="factors"><h2>What moved this estimate</h2>{result.top_factors.map((f,i)=><div className="factor" key={`${f.feature}-${i}`}><span>{f.feature.replaceAll("_"," ")}</span><b>{f.direction === "up" ? "↑ higher" : "↓ lower"}</b></div>)}</div>
          {result.warnings.length > 0 && <div className="warnings"><b>Support notes</b>{result.warnings.map(w=><p key={w}>{w}</p>)}</div>}
          <a className="feedback-link" href="/feedback">Report the actual sale price →</a>
          <p className="disclaimer">Decision assistance only — not an appraisal guarantee. Model factors describe model behavior, not causal price effects.</p>
        </>}
      </aside>
    </div>
  </main>;
}