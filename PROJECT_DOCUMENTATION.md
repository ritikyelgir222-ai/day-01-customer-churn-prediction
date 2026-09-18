# Project Documentation: Telco Customer Churn Prediction
### Technical Documentation & Handover — Phase 12 of the SDLC

This document is the technical documentation and user manual a real handover
package would include. It complements `README.md` (which covers setup and
run commands) by explaining the architecture, data, decisions, and results
in enough depth that someone who did not build this project could maintain
or extend it.

---

## 1. Project Summary

| | |
|---|---|
| **Objective** | Predict which customers are at risk of churning so a retention team can prioritize outreach |
| **Client context** | Telecom company (postpaid customers) |
| **Data source** | IBM Telco Customer Churn dataset — [Kaggle: blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Dataset size** | 7,043 customers, 21 raw columns |
| **Final model** | XGBoost Classifier, 29 engineered features |
| **Test AUC** | 0.8464 |
| **Business framing** | Score customers, target the highest-risk decile with retention offers |

---

## 2. Architecture

```
data/Telco-Customer-Churn.csv
        │
        ▼
data_loader.py ──────► loads raw CSV
        │
        ▼
clean_and_engineer.py ─► fixes data quality issues, drops leaky columns,
        │                collapses redundant categories, encodes,
        │                engineers 3 new features
        ▼
   split.py ───────────► stratified train/val/test split
        │
        ▼
train_model.py ───────► trains baseline (logistic regression),
        │                candidate (random forest), and final
        │                model (XGBoost); evaluates; saves artifacts
        ▼
outputs/churn_model.joblib, feature_columns.joblib, scaler.joblib
        │
        ├──────────────► app.py ─── FastAPI service, /score endpoint,
        │                           reuses clean_and_engineer.py so
        │                           training and serving logic never
        │                           drift apart
        │
        └──────────────► monitor.py ─ PSI drift check + retrain trigger,
                                       compares against the baseline
                                       recall stored in business_validation.json
```

The key design principle: **cleaning and feature engineering logic lives in
exactly one file** (`clean_and_engineer.py`) and is imported by both
`train_model.py` and `app.py`. This avoids "training/serving skew" — a
common real-world bug where the API quietly uses slightly different logic
than training did, causing the model to behave worse in production than in
evaluation for reasons that are hard to trace.

---

## 3. Data Dictionary (raw columns, as received)

| Column | Type | Description | Kept as-is / Transformed / Dropped |
|---|---|---|---|
| `customerID` | string | Unique customer identifier | **Dropped** — no generalizable signal, risk of leakage |
| `gender` | string | Male / Female | Binary-encoded |
| `SeniorCitizen` | int (0/1) | Already binary | Kept as-is |
| `Partner` | string | Yes/No | Binary-encoded |
| `Dependents` | string | Yes/No | Binary-encoded |
| `tenure` | int | Months as a customer | Kept as-is |
| `PhoneService` | string | Yes/No | Binary-encoded |
| `MultipleLines` | string | Yes/No/No phone service | "No phone service" collapsed to "No", then binary-encoded |
| `InternetService` | string | DSL/Fiber optic/No | One-hot encoded (3 columns) |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | string | Yes/No/No internet service | "No internet service" collapsed to "No", then binary-encoded |
| `Contract` | string | Month-to-month/One year/Two year | One-hot encoded (3 columns) |
| `PaperlessBilling` | string | Yes/No | Binary-encoded |
| `PaymentMethod` | string | 4 categories | One-hot encoded (4 columns) |
| `MonthlyCharges` | float | Current monthly bill | Kept as-is |
| `TotalCharges` | string (should be numeric) | Lifetime billed amount | Converted to numeric; 11 blank values (all `tenure == 0`) imputed as 0 |
| `Churn` | string | Yes/No | **Target** — binary-encoded |

**Engineered features (not in the raw data):**

| Feature | Formula | Business rationale |
|---|---|---|
| `num_add_on_services` | Count of Yes across the 6 add-on service columns | More bundled services = higher switching cost = hypothesized lower churn |
| `avg_historical_monthly_charge` | `TotalCharges / tenure` (or `MonthlyCharges` if `tenure == 0`) | Reconstructs what the customer has historically paid per month |
| `charge_deviation` | `MonthlyCharges - avg_historical_monthly_charge` | Proxy for "bill shock" — a sudden increase relative to the customer's own history, a known churn trigger |

Full reasoning for every decision above is inline in `clean_and_engineer.py` — this table is a summary, not a replacement for reading those comments.

---

## 4. Key EDA Findings

From `eda.py`, run against the real dataset:

| Cut | Finding |
|---|---|
| Contract type | Month-to-month churns at **42.7%**, vs. 11.3% for one-year and **2.8%** for two-year |
| Tenure | Churn drops steadily with tenure: 52.9% in months 0–6, down to 9.5% at 49–72 months |
| Internet service | Fiber optic customers churn at **41.9%**, vs. 19.0% for DSL and 7.4% for no internet |
| Payment method | Electronic check customers churn at **45.3%** — by far the highest of the four payment methods |
| Add-on services | Generally declining churn as add-on count increases (21.4% at 0 add-ons down to 5.3% at 6 add-ons), though not perfectly monotonic (1 add-on shows 45.8%, likely a smaller, mixed subgroup) |

These findings directly informed which engineered features were built (e.g. `num_add_on_services` was built specifically because of the add-on pattern above) and confirm the same intuitions that the SDLC template's Phase 6 hypothesized.

---

## 5. Modeling Results

### Experiment log (validation set)

| Model | Val AUC | Val Top-Decile Recall |
|---|---|---|
| Logistic Regression (baseline) | 0.8293 | 0.2740 |
| Random Forest | 0.8304 | 0.2776 |
| **XGBoost (final)** | 0.8256 | 0.2705 |

**Note on model selection:** Random Forest and XGBoost are within ~1 point
of each other on the business metric (top-decile recall) — close enough to
be normal run-to-run noise, not a decisive win either way. XGBoost was
still carried forward as the final model for reasons beyond this single
validation split: it supports incremental training (useful once monthly
retraining is set up), early stopping, and monotonic constraints if the
business later requires certain features to only ever increase or decrease
risk. This is a judgment call, documented rather than hidden — a stricter
process would validate this choice across multiple splits before locking it in.

### Held-out test set (final, unbiased evaluation)

| Metric | Value |
|---|---|
| Test AUC | 0.8464 |
| Precision at top-20% threshold | 0.6745 |
| Recall at top-20% threshold | 0.5107 |
| Top-decile recall | 0.2929 |

### What drives the model (feature importance)

Top 5 by importance:

1. `Contract_Month-to-month` (0.379) — by far the strongest driver
2. `InternetService_Fiber optic` (0.111)
3. `Contract_Two year` (0.111)
4. `InternetService_No` (0.052)
5. `Contract_One year` (0.051)

Contract type alone accounts for roughly half of the model's total decision weight across its three encoded categories — consistent with the EDA finding that contract type had the single largest churn-rate spread of anything examined.

### Business validation

On the 1,057-customer held-out test set:

- The model's top-risk decile (~106 customers) catches an estimated **82 of the actual churners** in that group (top-decile recall: 0.293)
- **Assuming** a 35% success rate for retention offers (an illustrative assumption, not derived from this dataset, since it contains no offer/response history), this implies **~29 prevented churns**
- At the test set's average monthly charge, that's an estimated **$22,698/year** in retained revenue for this segment

This last number is explicitly flagged in `business_validation.json` as depending on an assumed offer-success rate. Before presenting it as a business case, that 35% figure should be replaced with a real number from an A/B test or historical campaign data.

---

## 6. API Reference (Phase 10)

**Base URL (local):** `http://127.0.0.1:8000`

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Browser-based test form (pick a profile, click Score) |
| `/docs` | GET | Auto-generated interactive API docs (Swagger UI) |
| `/health` | GET | Health check, returns `{"status": "ok"}` |
| `/score` | POST | Score a single customer — see request/response shape below |

**Request body (`/score`):** all 19 raw customer fields (same names/casing as the CRM export — see `app.py`'s `CustomerRecord` model).

**Response:**
```json
{
  "churn_risk_score": 0.9526,
  "risk_tier": "High",
  "top_factors": ["Contract_Month-to-month", "Contract_Two year", "InternetService_Fiber optic"]
}
```

`risk_tier` cutoffs: High ≥ 0.5, Medium ≥ 0.25, Low below that — a business judgment call intended to roughly match the top-20% operating threshold used in evaluation, adjustable based on retention team capacity.

---

## 7. Monitoring & Maintenance Plan (Phase 13)

`monitor.py` implements:

- **Feature drift check** via Population Stability Index (PSI) on `tenure`, `MonthlyCharges`, `TotalCharges`, `num_add_on_services`. PSI < 0.1 = stable, 0.1–0.25 = moderate, > 0.25 = significant drift requiring investigation.
- **Performance decay check**: compares current top-decile recall against the baseline stored in `business_validation.json`; triggers a retrain recommendation if recall drops more than 0.05.

**In production**, this script should run monthly against the actual new customer batch (not the held-out test set, which is what it uses here as a stand-in — see the comment at the top of `monitor.py` for why).

---

## 8. Known Limitations (stated for the handover record)

1. **Single snapshot, not longitudinal data.** This dataset has one row per customer with no repeated monthly observations, so the train/test split is a stratified random split rather than the time-based split a production system should use. A live deployment should retrain once multiple months of history are available.
2. **Revenue estimate depends on an assumed retention-offer success rate** (35%), not observed data. Replace with a real campaign response rate before using the dollar figure in a business case.
3. **Per-prediction explanations use global feature importance**, not per-customer SHAP values, to avoid an extra dependency in the demo API. A production version should compute real per-prediction SHAP values before showing "why" explanations to retention agents.
4. **Fairness review not performed.** `gender` and `SeniorCitizen` are included as features. Per the SDLC template's Phase 9 guidance, any model influencing customer-facing decisions should go through a bias/fairness check before deployment — this hasn't been done here.
5. **Model selection margin is narrow.** Random Forest performed comparably to (and marginally better than, on this run) the chosen XGBoost model on the business metric — see Section 5.

---

## 9. File Map (for quick reference)

| File | Phase | Purpose |
|---|---|---|
| `data_loader.py` | 5 | Load raw data, document source |
| `eda.py` | 6 | Churn-rate breakdowns, summary chart |
| `clean_and_engineer.py` | 5 (fixes) + 7 | Cleaning, elimination, encoding, feature engineering — fully commented |
| `split.py` | 7 | Stratified train/val/test split |
| `train_model.py` | 8–9 | Model training, evaluation, business validation |
| `app.py` | 10 | FastAPI scoring service + test form |
| `monitor.py` | 13 | Drift detection, retrain trigger |
| `README.md` | 12 | Setup and run instructions |
| `PROJECT_DOCUMENTATION.md` (this file) | 12 | Technical documentation and handover |
