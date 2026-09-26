const BODY = `# VroomValue

> VroomValue estimates used-vehicle selling prices in USD with an 80% calibrated prediction interval, support warnings, and model explanation factors.

## Primary pages

- [Estimate](/): Enter vehicle specifications and request a market-price estimate.
- [Feedback](/feedback): Submit the realized sale price for an existing prediction.
- [Admin](/admin): Review runtime metrics and moderate submitted sale feedback.
`;

export const dynamic = "force-static";

export function GET() {
  return new Response(BODY, {
    headers: {
      "content-type": "text/plain; charset=utf-8",
      "cache-control": "public, max-age=3600",
    },
  });
}
