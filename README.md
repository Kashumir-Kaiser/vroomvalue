# VroomValue

VroomValue is an end-to-end used-vehicle price prediction system built from the supplied 5,500-row automobile dataset. It combines a CatBoost regression pipeline, calibrated prediction intervals, SHAP-based local explanations, a FastAPI inference service, PostgreSQL persistence, and a Next.js web interface.

The project is designed as a reproducible machine-learning application rather than a standalone notebook. Training, validation, serving, persistence, UI behavior, model metadata, and CI checks are all kept in the repository.

## Project objectives

Given a vehicle description, VroomValue returns:

- an estimated selling price in USD;
- an 80% calibrated prediction interval;
- an in-distribution or low-confidence support label;
- warnings when inputs fall outside observed training support;
- up to three local SHAP factors and their direction; if explanation generation fails, pricing still returns with an explicit warning;
- the model version, schema version, and model date attached to the prediction.

The result is intended for decision support. It is not an appraisal guarantee and the SHAP factors describe model behavior rather than causal price effects.

## Technology stack

| Layer | Technology | Purpose |
| --- | --- | --- |
| Machine learning | CatBoost, scikit-learn, NumPy, pandas | Regression, grouped evaluation, feature preparation, calibration |
| Explainability | SHAP | Local feature-attribution factors |
| Experiment tracking | MLflow | Local training-run metrics and artifacts |
| API | FastAPI, Pydantic | Typed validation and inference endpoints |
| Persistence | PostgreSQL, SQLAlchemy | Prediction and feedback records |
| Frontend | Next.js 15, React 19, TypeScript | Vehicle input, results, feedback, and admin views |
| Packaging | Docker, Docker Compose | Reproducible local application stack |
| Testing | pytest, Ruff, GitHub Actions | Unit, data-contract, model, integration, lint, and build checks |

## Repository structure

```text
vroomvalue/
├── apps/
│   ├── api/
│   │   └── app/
│   │       ├── database.py       # SQLAlchemy models and persistence
│   │       ├── main.py           # FastAPI routes, middleware, health checks
│   │       ├── model_runtime.py  # Model loading, integrity checks, inference
│   │       └── schemas.py        # Pydantic request/response validation
│   └── web/
│       ├── app/                  # Next.js pages and global styles
│       ├── components/           # Prediction form and result UI
│       ├── admin/                # Runtime metrics page
│       └── feedback/             # Actual-sale feedback page
├── data/
│   └── automobile_dataset.csv   # Supplied project dataset
├── infra/
│   └── docker/                  # API and web Dockerfiles
├── ml/
│   ├── contracts/               # Executable dataset contract
│   ├── features/                # Shared deterministic feature engineering
│   └── training/                # Model training, calibration, evaluation
├── models/                      # Generated model artifacts
├── tests/
│   ├── data/                    # Dataset-contract tests
│   ├── integration/             # API and persistence tests
│   ├── model/                   # Saved-model release gates
│   └── unit/                    # Schema, features, calibration logic
├── .github/workflows/ci.yml
├── docker-compose.yml
├── model_card.md
├── pyproject.toml
└── README.md
```

## Dataset

The source dataset is `data/automobile_dataset.csv` and contains **5,500 rows and 18 columns**: 17 predictors plus the target `Selling_Price`.

The predictors are:

`Make`, `Model`, `Year`, `Fuel_Type`, `Transmission`, `Engine_Size`, `Mileage`, `Horsepower`, `Torque`, `Owners`, `Accident_History`, `Service_History`, `Color`, `Body_Type`, `Drivetrain`, `Fuel_Efficiency`, and `Location`.

The training and API paths use the same data contract. The application does not create synthetic replacement data if the source file is missing or malformed.

### Engine size contract

`Engine_Size` is constrained to:

- minimum: **0.0**
- maximum: **5.7**
- precision: **at most one decimal place**
- valid increment: **0.1**
- nullable: **yes**

Examples such as `2.5` and `3.0` are valid. A value such as `2.55` is rejected rather than rounded. `0.0` is valid for electric vehicles; a non-electric vehicle with `0.0` is accepted but marked as low-confidence support.

### Missing values

Optional missing values are preserved rather than replaced with favorable defaults. Categorical missing values are represented as an explicit `MISSING` category for model input, while numeric missing values remain missing for CatBoost and receive companion missing-indicator features.

The dataset contract also rejects malformed numeric values, non-finite values, missing required fields, invalid integer fields, invalid accident-history values, out-of-range years, and non-positive target prices.

### Units and currency

The dataset does not contain explicit unit or currency columns. The application therefore makes the following assumptions visible in the UI and model card:

- mileage: miles;
- torque: lb-ft;
- fuel efficiency: MPG, or MPGe for electric vehicles;
- price: USD.

## Feature engineering

Training and inference use the same deterministic feature builder.

Additional derived features include:

- `vehicle_age = reference_year - Year`;
- `mileage_per_year = Mileage / max(vehicle_age, 1)`;
- `log_mileage = log1p(Mileage)`;
- missing-value indicators for partially missing fields.

The model reference year is frozen at **2026** inside the artifact so that training and serving use the same age calculation.

## Model training

The primary model is `CatBoostRegressor`. Two target formulations are evaluated:

1. raw selling price;
2. `log1p(Selling_Price)`.

Grouped cross-validation selects the formulation with the lower development MAE.

A hierarchical median baseline is also fitted using:

1. Make + Model + Year;
2. Make + Model;
3. Make;
4. global median.

The trained CatBoost model must improve holdout MAE over this baseline by at least **10%** before the model artifact is accepted.

### Leakage control

Rows are fingerprinted using vehicle identity and pricing fields before splitting. Duplicate fingerprints are kept in a single split so equivalent rows cannot leak across training, calibration, and holdout data.

The pipeline uses separate:

- training data;
- calibration data;
- locked interpolation holdout data.

The code explicitly checks that duplicate groups do not overlap across these sets.

## Prediction intervals

VroomValue does not return only a point estimate.

An **80% split-conformal interval** is calibrated from held-out residuals. Calibration uses four prediction-price buckets so uncertainty can vary across the price range.

For tied predictions that produce an empty calibration bucket, the implementation falls back to the global finite-sample conformal quantile instead of generating an undefined interval.

The release gate requires overall holdout coverage between **77% and 83%**.

When an input is classified as low support, the serving layer widens the calibrated half-width by 1.35× and returns a warning with the prediction.

## Current model results

The currently documented training result in `model_card.md` is:

| Metric | Result |
| --- | ---: |
| Selected target formulation | `log1p` |
| Grouped-CV MAE — raw target | $1,063.72 |
| Grouped-CV MAE — log1p target | $1,024.37 |
| Baseline holdout MAE | $2,143.13 |
| CatBoost holdout MAE | $1,116.91 |
| MAE improvement over baseline | 47.9% |
| Holdout RMSE | $1,845.98 |
| Holdout R² | 0.9800 |
| Holdout WAPE | 9.07% |
| 80% interval coverage | 81.09% |

A separate Make-Model cold-start stress split is derived **only from the 4,400-row primary training partition**, so calibration and locked holdout rows are not reused in that stress metric. The current stress result is **$4,251.48 MAE** and **29.09% WAPE** across 903 held-out cold-start rows, showing that predictions for unseen vehicle families are materially harder than interpolation within supported vehicle groups.

## Model artifacts

Training writes:

```text
models/auto_price.joblib
models/auto_price.joblib.sha256
models/split_manifest.json
model_card.md
```

The model bundle contains the fitted CatBoost model, target formulation, reference year, **ordered feature schema**, conformal interval data, observed support ranges/categories, model metadata, and evaluation metrics.

The API verifies the SHA-256 sidecar before deserializing the model artifact. A missing checksum, mismatched checksum, missing required artifact keys, or load failure keeps readiness false. At inference, generated features are checked against the saved schema: reordered columns are explicitly reindexed to the training order, while missing or unexpected features fail closed instead of being silently passed to the model.

## API

The FastAPI service exposes:

### `GET /health/live`

Confirms that the API process is running.

### `GET /health/ready`

Returns ready only when the dataset contract passes and the model artifact can be loaded and verified.

### `GET /v1/metadata`

Returns:

- supported categorical values;
- Make → Model mappings;
- observed numeric support ranges;
- units;
- the Engine Size validation rule;
- model version;
- schema version;
- model reference year.

### `POST /v1/predictions`

Validates one vehicle and returns:

- prediction ID;
- request ID;
- estimated price;
- 80% prediction interval;
- support status;
- warnings;
- up to three local SHAP factors;
- model metadata.

### `POST /v1/feedback`

Associates an actual sale price and sale date with an existing prediction. Duplicate feedback for the same prediction is rejected.

### `GET /v1/admin/metrics`

Returns prediction count, feedback count, request count, error rate, invalid-input rate, average request latency, current model version, and readiness state. Service counters are stored in the database and updated atomically, so they aggregate across API workers rather than resetting per process. Metric writes run only after the response body has been sent and are offloaded to Starlette's thread pool, avoiding synchronous database work on the async event loop. Health checks and the metrics endpoint itself are excluded from the service-rate counters. The invalid-input rate counts malformed or oversized request statuses 400, 413, and 422; authentication, missing-resource, conflict, and server failures remain part of the broader error rate but not the invalid-input rate.

If `ADMIN_TOKEN` is configured, this route requires the token through the `X-Admin-Token` header.

## API validation and request handling

Input models reject:

- unknown JSON fields;
- NaN and Infinity;
- blank required strings;
- values outside numeric bounds;
- Engine Size values with more than one decimal place;
- Owners outside 1–5;
- invalid accident-history values;
- malformed prediction IDs;
- feedback sale dates in the future.

Required text fields are trimmed and blank values return the explicit validation message `Field must not be blank.` Optional blank strings are normalized to unknown values. Location codes are normalized to uppercase.

The API also applies a configurable request-body limit to both fixed-length and streamed/chunked requests, so omitting `Content-Length` does not bypass the limit. CORS wraps the observability and body-limit middleware, so browser clients still receive readable CORS headers on direct 400/413 body-limit responses. Request/security headers are injected at the ASGI response-start boundary rather than by mutating a response object returned from `call_next`. Observability emits one `request.completed` record for normal responses and for interrupted responses that had already started streaming. The API sanitizes externally supplied request IDs, returns request IDs in responses, and adds basic defensive response headers.

## Support detection

A prediction is marked `low_confidence` when the request contains an unsupported condition such as:

- a categorical value not observed in training;
- an unseen Make-Model combination;
- a numeric value outside the training support range;
- `Engine_Size = 0.0` for a non-electric vehicle.

These conditions do not produce a server error. The prediction is returned with warnings and a wider interval.

## Persistence

Predictions and user-submitted outcomes are stored separately.

The `predictions` table records:

- prediction ID;
- timestamp;
- estimated price;
- interval bounds;
- support state;
- model version.

The `feedback` table records:

- feedback ID;
- prediction ID;
- actual sale price;
- sale date;
- timestamp.

A unique constraint allows only one feedback record per prediction.

PostgreSQL is used by the application stack. SQLite is used only inside isolated integration tests.

## Frontend

The Next.js frontend provides three primary views:

- the vehicle specification and prediction page;
- an actual-sale feedback page;
- a minimal operations/admin metrics page.

The vehicle form retrieves supported metadata from the API, uses cascading Make → Model selection, keeps optional fields explicitly unknown when not provided, and applies the Engine Size step of `0.1` in the browser.

The result panel displays the price estimate, calibrated range, support status, warnings, model version, and local factors. SHAP is treated as best-effort explanation logic: an explainer initialization or per-request SHAP failure does not suppress a valid price prediction; the API returns an empty factor list plus an explanation-unavailable warning.

## Local development

### Requirements

- `uv` for the Python toolchain and virtual environment;
- Python **3.12**, installed and managed locally by `uv` for this project; a global Python 3.12 installation is not required;
- Node.js 22;
- Docker and Docker Compose for the full stack.

### Python environment

VroomValue standardizes local development on **Python 3.12**, matching CI. Do not create the project environment from whatever `python` happens to be on the global PATH, because that may select an unsupported interpreter such as Python 3.14.

Install `uv` once if it is not already available:

```powershell
# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

From the repository root, install the project-local Python 3.12 toolchain and create the virtual environment explicitly from it:

```bash
uv python install 3.12
uv venv --python 3.12 .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Verify that the project environment is using Python 3.12:

```bash
python --version
```

The output must begin with `Python 3.12`.

Install the project and development dependencies into that environment:

```bash
uv pip install -e ".[dev]"
```

This setup does not require Python 3.12 to be installed globally or added to the system PATH; `uv` downloads and manages the interpreter used by `.venv`.

### Train the model

```bash
python -m ml.training.train
```

Release training logs the run to the local MLflow file store under `mlruns/`.

For test/CI runs where MLflow logging is intentionally skipped:

```bash
python -m ml.training.train --skip-mlflow
```

### Run the application

After training has produced the model artifact and checksum:

```bash
docker compose up --build
```

The local services are:

- web UI: `http://localhost:3000`
- API: `http://localhost:8000`
- OpenAPI docs: `http://localhost:8000/docs`

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql+psycopg://vroomvalue:vroomvalue@db:5432/vroomvalue` | SQLAlchemy database connection |
| `DATASET_PATH` | `data/automobile_dataset.csv` | Dataset used by readiness checks |
| `MODEL_ARTIFACT_PATH` | `models/auto_price.joblib` | Model bundle loaded by the API |
| `RUNTIME_REFRESH_TTL_SECONDS` | `1.0` | Minimum interval between dataset/model filesystem-change checks per API process; read when each `Runtime` is constructed, malformed/non-finite values fall back to 1.0, and negative values clamp to 0 |
| `CORS_ORIGINS` | `http://localhost:3000` | Allowed browser origins |
| `MAX_BODY_BYTES` | `32768` | Maximum accepted request body size |
| `ADMIN_TOKEN` | unset | Optional protection for the admin metrics route |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | API base URL used by the frontend |

## Tests

Run the complete Python test suite after training:

```bash
pytest -q
```

The automated test areas cover:

- schema validation;
- Engine Size precision;
- non-finite inputs;
- deterministic feature engineering;
- malformed and missing dataset cases;
- calibration edge cases;
- inference feature-order/schema enforcement;
- SHAP failure fallback behavior;
- streamed request-body size enforcement and CORS preservation on 413 responses;
- cold-start isolation to the primary training partition;
- explicit admin metrics response schema;
- release model gates;
- API prediction flow;
- feedback persistence.

Static checks can be run with:

```bash
python -m compileall -q apps ml tests
ruff check apps ml tests
```

The frontend production build can be checked with:

```bash
cd apps/web
npm install
npm run build
```

## Continuous integration

GitHub Actions runs three jobs:

### `python-fast`

- installs the Python project;
- byte-compiles application, ML, and test modules;
- runs Ruff;
- runs unit and data-contract tests.

### `model-and-integration`

- installs the Python project;
- trains a fresh model against the supplied dataset;
- enforces model quality and interval-coverage gates;
- runs model and API integration tests.

### `web`

- installs Node dependencies;
- builds the Next.js application in production mode.

A change is considered clean only when all three jobs pass.

## Failure behavior

Training stops with a non-zero exit instead of fabricating data when the source dataset is unavailable or unusable.

Examples include:

```text
Dataset not found at data/automobile_dataset.csv. Add the file and retry.
```

and:

```text
Dataset at data/automobile_dataset.csv contains no rows. Add data and retry.
```

The API exposes the same dataset/model problems through `/health/ready` with HTTP 503 rather than reporting a healthy service without a usable model. Dataset/model change detection is throttled by a 1-second monotonic TTL by default, avoiding two filesystem `stat()` calls on every request while still allowing near-immediate local artifact refreshes. Runtime signatures start from an explicit unset sentinel, so a first refresh correctly reports missing files even if startup loading was skipped. Malformed or non-finite TTL environment values log a warning and fall back to 1.0 seconds. A non-missing model artifact that fails checksum/deserialization is not signature-cached as successful and is retried after the next TTL window. Persistent load failures emit a structured `model.load_failed` error event immediately and then at most once every 60 seconds per process; a successful recovery emits `model.load_recovered`.

## Known limitations

The supplied dataset does not provide listing dates, transaction-source identifiers, VIN data, image-based condition information, or explicit unit/currency columns. As a result:

- the model cannot directly measure time-based market drift;
- source-specific bias cannot be quantified;
- vehicle condition beyond the supplied structured fields is not modeled;
- units and currency remain documented assumptions;
- cold-start accuracy is weaker for unseen Make-Model groups than for interpolation within known groups.

The 1.35× widening applied to low-support predictions is a serving heuristic rather than a separately calibrated coverage guarantee.

## License

This repository is released under the MIT License.
