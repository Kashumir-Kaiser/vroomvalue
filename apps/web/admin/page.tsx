"use client";

import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
type Metrics = {prediction_count:number;feedback_count:number;request_count:number;error_rate:number;invalid_input_rate:number;avg_latency_ms:number;current_model_version:string|null;readiness:string};

export default function AdminPage(){
  const [data,setData]=useState<Metrics|null>(null); const [error,setError]=useState("");
  async function load(){try{const r=await fetch(`${API}/v1/admin/metrics`,{cache:"no-store"});const b=await r.json();if(!r.ok)throw new Error(b.detail??"Metrics unavailable");setData(b);setError("");}catch(e){setError(e instanceof Error?e.message:"Metrics unavailable");}}
  useEffect(()=>{load();},[]);
  return <main className="shell narrow"><header className="page-head"><span className="eyebrow">OPERATIONS STATUS</span><h1>Admin</h1><p>Minimal runtime view for the MVP. These counters are operational signals, not model-quality metrics.</p></header>
    <section className="admin-sheet"><div className="sheet-title"><span>LIVE SUMMARY</span><button className="mini-button" onClick={load}>Refresh</button></div>
      {error && <p className="error">{error}</p>}{data && <table><tbody>
        <tr><th>Readiness</th><td>{data.readiness}</td></tr><tr><th>Model version</th><td>{data.current_model_version??"—"}</td></tr><tr><th>Requests</th><td>{data.request_count}</td></tr><tr><th>Predictions</th><td>{data.prediction_count}</td></tr><tr><th>Feedback records</th><td>{data.feedback_count}</td></tr><tr><th>Error rate</th><td>{(data.error_rate*100).toFixed(2)}%</td></tr><tr><th>Invalid-input rate</th><td>{(data.invalid_input_rate*100).toFixed(2)}%</td></tr><tr><th>Average latency</th><td>{data.avg_latency_ms.toFixed(2)} ms</td></tr>
      </tbody></table>}
    </section>
  </main>;
}