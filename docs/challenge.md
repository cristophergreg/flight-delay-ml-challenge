# Flight Delay ML Challenge

## Overview

This repository contains a **productionized version** of the notebook: a minimal, testable classifier of flight delays, and a **FastAPI service** that exposes predictions.

---

### Model
- **Algorithm:** Logistic Regression (`class_weight=balanced`)
- **Features:** Engineered categorical features

---

### API
- Endpoints: `/health` and `/predict`
- Domain validation included
- Model trained once at startup

---

### Packaging
- Python 3.10 Docker image
- Runs as **non-root** user
- Correct file permissions
- Served with **Uvicorn**

---

### Tests
- Run with `make model-test` and `make api-test`
- All tests pass locally

---

### Stress Test
- Performed with **Locust**
- 60 seconds, 100 users, 1 user/s spawn rate
- **No failures**

---

### Cloud Deploy
- **Cloud Run–ready**
- Stress test run locally (billing not enabled at review time)
- Instructions included to swap to Cloud Run URL when available

---

## Data & Target

**Source file:** `data/data.csv`

---

**Columns of interest:**

- `OPERA` (airline/operator)  
- `TIPOVUELO` (`N` domestic / `I` international)  
- `MES` (`1..12`)  
- `Fecha-I` (scheduled) and `Fecha-O` (actual)

---

**Target definition (rebuilt if missing):**

```python
delay = 1 if (Fecha-O − Fecha-I) > 15 minutes else 0
```

**Rationale:** This mirrors the notebook’s logic, but implemented in a vectorized way for performance and consistency.

---

## Feature Engineering (serving-time)

To satisfy the tests and keep the feature contract stable, I use **exactly 10 one-hot columns**:

```text
OPERA_Latin American Wings  
OPERA_Grupo LATAM  
OPERA_Sky Airline  
OPERA_Copa Air  
TIPOVUELO_I  
MES_4  
MES_7  
MES_10  
MES_11  
MES_12
```
---
### Implementation details

`pd.get_dummies` on `OPERA`, `TIPOVUELO`, and `MES`, followed by a `.reindex()` to the exact set, filling missing columns with `0`.

This ensures training and serving are robust regardless of input distributions.


> **Note:** Other notebook features (e.g., “Period of Day”, “High Season”, minute diffs) were **not carried to serving** because the unit tests and the provided baseline expect this exact feature set.  
> They can be re-introduced later behind a versioned schema.

---

## Model Choice

I evaluated the DS notebook’s options and chose Logistic Regression with:

- `class_weight="balanced"`  
- `max_iter=1000`, `solver="lbfgs"`, `random_state=42`

---

### Why LR (plain) over XGBoost here?

- **Simplicity & reliability:** fewer moving parts, very stable under class imbalance with `class_weight`.  
- **Interpretability:** coefficients map directly to categorical effects.  
- **Operational fit:** fast, low memory, deterministic, and easy to validate in tests.  
- **Notebook parity:** with this small, categorical feature set, LR achieves comparable performance without the complexity or additional tuning needs of XGBoost.

---

## Code Structure

### challenge/model.py

- `DelayModel.preprocess(data, target_column=None)`:  
  Builds delay if needed and returns either `X` or `(X, y)`.  
  Uses vectorized datetime ops and stable one-hot encoding + reindex.

- `DelayModel.fit(X, y)`:  
  Trains and stores the estimator in `self._model`.

- `DelayModel.predict(X)`:  
  Returns `List[int]`; defensive fallback if called before fit.

---

### challenge/api.py

- Pydantic schemas: `Flight`, `PredictRequest`, `PredictResponse`.

- `/health`: liveness probe.

- `/predict`:  
  Domain validation against the training catalog (valid `OPERA`),  
  strict `TIPOVUELO ∈ {'N','I'}`, `MES ∈ {1..12}`; returns 400 on domain errors.

- Bootstraps the model once at startup using `data/data.csv` → faster requests, consistent behavior.

---

## Packaging & Runtime

### Docker (see Dockerfile):

- `FROM python:3.10-slim`  
- Install `requirements.txt`, copy source code and dataset.  
- Run as non-root user (`appuser`) and fix permissions (`chown -R appuser:appuser /app`).  
- `ENV PYTHONPATH=/app`, `EXPOSE 8080`, `CMD uvicorn challenge.api:app ...`  

---

### .dockerignore

- Excludes virtual environments (`venvs`), caches, coverage artifacts.  
- **Does not exclude** `data/`.

---

## Tests

### Model tests

- `make model-test` → ✅ passed locally (4/4).  
- Coverage > 85% over model module.

---

### API tests

- `make api-test` → ✅ passed locally (4/4).  
- Dependency pin: `anyio<4` to keep FastAPI/Starlette TestClient compatible.

---

## Stress Test (Part III)

**Command:**  
```bash
make stress-test 
```
(Runs Locust headless for 60s, 100 users, spawn 1/s)

**Target:** `http://localhost:8080` (Docker container running locally)

### Results (local)

- **Requests:** 6,644  
- **Failures:** 0 (0.00%)  
- **Throughput:** ~111 req/s sustained  
- **Latency percentiles:**  
  - p50 ≈ 270 ms  
  - p95 ≈ 500 ms  
  - p99 ≈ 530 ms  
  - p99.9 ≈ 1000 ms  
  - max ≈ 1200 ms  

---

### Report attached

- `docs/stress-test.pdf` (recommended for reviewers)  
- `docs/reports/stress-test.html` (optional, interactive)

---

### Test dependencies pins (Locust web stack):
```text
flask==2.0.3  
Werkzeug==2.0.3  
itsdangerous==2.0.1  
jinja2==3.0.3
```
**Rationale:** ensure Flask compatibility for Locust UI imports.

---

## Deploy (Cloud Run) — Ready

Although Cloud Run deploy is ready, it was pending billing enablement at review time.

**Steps:**
```bash
PROJECT=$(gcloud config get-value project)
REGION=us-central1

gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com containerregistry.googleapis.com
gcloud builds submit --tag "gcr.io/$PROJECT/flight-delay-api"
gcloud run deploy flight-delay-api \
  --image "gcr.io/$PROJECT/flight-delay-api" \
  --platform managed --region "$REGION" \
  --allow-unauthenticated --port 8080

```

Then set `STRESS_URL` in the `Makefile` to the Cloud Run URL and re-run:

```bash
make stress-test
```

---

## How to Run Locally

### With Python (venv)

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-test.txt

make model-test && make api-test
```

### Docker

```bash
# Build the image
docker build -t flight-delay-api:latest .

# Run the container
docker run --rm -p 8080:8080 -e PORT=8080 flight-delay-api:latest

# Health check
curl http://localhost:8080/health

# Run stress test (uses http://localhost:8080)
make stress-test

```

---