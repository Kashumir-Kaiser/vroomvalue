# VroomValue

VroomValue is a used-vehicle price prediction MVP built around the supplied 5,500-row automobile dataset. It trains a CatBoost regressor, compares it with a hierarchical Make-Model-Year median baseline, calibrates an 80% prediction range, exposes a FastAPI service, stores predictions and feedback in PostgreSQL, and serves a Next.js vehicle-pricing interface.

## Data contract

The source file is `data/automobile_dataset.csv`. It is project data and must not be generated, replaced with synthetic rows, or silently substituted. The API and trainer fail closed when it is missing or empty. The dataset contains 18 columns total: 17 predictors plus `Selling_Price`.

`Engine_Size` is constrained to **0.0–5.7 with at most one decimal place (0.1 increments)** end-to-end. Missing engine size remains allowed. A zero engine size is valid for electric vehicles; on a non-electric vehicle it is accepted but flagged as low support rather than silently changed.

The dataset does not include explicit unit/currency columns. For this MVP the UI and model card therefore state these assumptions visibly: **miles**, **lb-ft**, **MPG / MPGe**, and **USD**.

## Architecture

- `ml/contracts`: executable dataset and validation contract.
- `ml/features`: deterministic features shared by training and serving.
- `ml/training`: grouped evaluation, baseline, CatBoost training, conformal calibration, MLflow logging.
- `apps/api`: FastAPI + Pydantic + SQLAlchemy prediction/feedback API.
- `apps/web`: Next.js TypeScript UI styled as a vehicle specification/window-sticker sheet rather than a generic dashboard.
- `infra/docker`: container definitions.
- `tests`: unit, data-contract, model-gate, and API integration tests.

The model uses a frozen reference year of **2026** for `vehicle_age`, plus `mileage_per_year`, `log1p(Mileage)`, and explicit missing indicators. Raw-price and `log1p(price)` targets are compared using grouped cross-validation. The final uncertainty layer is split conformal with four prediction-price buckets, calibrated on a dedicated set that is not used for point-model selection.

## Local setup

Python 3.12 and Node 22 are the supported local versions.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e '.[dev]'
python -m ml.training.train
```

Training must finish with both release gates passing: CatBoost improves holdout MAE over the naive baseline by at least 10%, and the nominal 80% interval covers 77–83% of the locked interpolation holdout. The command writes `models/auto_price.joblib`, `models/split_manifest.json`, updates `model_card.md`, and records the run in local `mlruns/` through MLflow.

Start the stack only after training:

```bash
docker compose up --build
```

Open `http://localhost:3000`. API docs are at `http://localhost:8000/docs`.

## API

- `GET /v1/metadata` — supported categories, Make→Model mapping, numeric support, units, engine-size step, model/schema versions.
- `POST /v1/predictions` — validated single-vehicle scoring with USD estimate, 80% interval, support label, warnings, and three SHAP factors.
- `POST /v1/feedback` — verified-sale candidate attached to a prediction id.
- `GET /v1/admin/metrics` — minimal prediction/feedback/model status summary.
- `GET /health/live` — process liveness.
- `GET /health/ready` — model loaded and dataset contract passed.

## Tests

After training the artifact:

```bash
pytest -q
```

The data tests explicitly cover the real dataset, missing file, header-only file, missing columns, and the one-decimal `Engine_Size` contract. The model test fails unless the stored training metrics satisfy both release gates. The integration test uses SQLite only as an isolated test persistence backend; the application runtime uses PostgreSQL.

## Clean-system acceptance gate

Before calling the MVP finished:

1. Clone the repository into a clean directory with no previous `.venv`, `node_modules`, containers, volumes, `mlruns`, or model artifacts.
2. Follow only this README to install Python dependencies and run `python -m ml.training.train`.
3. Run `pytest -q` and confirm every automated layer passes.
4. Run `docker compose up --build`, submit a valid vehicle, confirm the estimate/range/factors, submit feedback, and confirm `/v1/admin/metrics` increments.
5. Stop the stack. Temporarily rename `data/automobile_dataset.csv`; confirm training exits non-zero with `Dataset not found at data/automobile_dataset.csv. Add the file and retry.` and `/health/ready` returns 503 with that message. Restore it.
6. Repeat with a header-only file; confirm the exact no-row failure, restore the real CSV, then perform one final browser journey.

## Repository description

Suggested GitHub description:

> Used-car price prediction MVP with CatBoost, FastAPI, Next.js, PostgreSQL, calibrated uncertainty, SHAP explanations, MLflow, and Docker.

## Phase 2 ideas

Batch CSV scoring, saved scenario comparisons, scheduled retraining, richer drift reporting, authenticated role-based administration, external market data, VIN decoding, image-based condition assessment, and production edge/security infrastructure are intentionally deferred.