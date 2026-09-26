import PredictionForm, { type Metadata } from "../components/PredictionForm";

const INTERNAL_API =
  process.env.API_INTERNAL_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000";

async function loadMetadata(): Promise<Metadata | null> {
  try {
    const response = await fetch(`${INTERNAL_API}/v1/metadata`, {
      cache: "no-store",
    });
    if (!response.ok) return null;
    return (await response.json()) as Metadata;
  } catch {
    return null;
  }
}

export default async function Home() {
  const metadata = await loadMetadata();
  return <PredictionForm initialMetadata={metadata} />;
}
