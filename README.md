# Customer Churn Prediction — Full Project Code (Real Dataset)

Companion code to `churn-prediction-full-sdlc.md` (the Day 1 flagship
project for the 100-day LinkedIn series). This version uses a **real**
dataset instead of synthetic data.

## Data source

**IBM Telco Customer Churn** — 7,043 customers, 21 columns. Same file
published on Kaggle as ["Telco Customer Churn" by blastchar](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
Originally IBM Watson Analytics / Cognos sample data.

The CSV is already included at `data/Telco-Customer-Churn.csv`. If you'd
rather pull it fresh from Kaggle yourself: download
`WA_Fn-UseC_-Telco-Customer-Churn.csv` from the link above and drop it in
at that same path — the schema is identical, so nothing else changes.

## Why every script is commented the way it is

Every cleaning step, elimination, transformation, and modeling choice has
a `WHY` comment next to it in the code — not just what was done, but the
business or statistical reasoning behind it. A few highlights, so you
know what to look for:

- **`customerID` is dropped** — unique identifier, would cause the model
  to overfit/leak rather than generalize.
- **The "No internet service" / "No phone service" categories are
  collapsed to "No"** in 7 columns — they're redundant with
  `InternetService`/`PhoneService`, and keeping them just adds noisy,
  duplicate one-hot columns.
- **11 blank `TotalCharges` values are imputed as 0, not dropped or
  median-filled** — those are all `tenure == 0` customers who genuinely
  haven't been billed yet; 0 is the factually correct value, not a
  statistical guess.
- **`num_add_on_services`, `avg_historical_monthly_charge`, and
  `charge_deviation`** are engineered features with specific business
  hypotheses behind them (bundled customers are stickier; a sudden bill
  increase relative to a customer's own history is a churn trigger) —
  see `clean_and_engineer.py` for the full reasoning.
- **Model selection is metric-driven, and honestly reported** —
  `train_model.py` explicitly notes that Random Forest and XGBoost were
  a near-tie on the business metric in this run, and explains why
  XGBoost was still carried forward (production/retraining practicality),
  rather than silently picking a winner and implying it was a landslide.

## Setup

```bash
pip install -r requirements.txt
```

## Run order

```bash
python data_loader.py         # Phase 5 — loads & inspects the raw CSV
python eda.py                  # Phase 6 — churn-rate breakdowns + outputs/eda_summary.png
python clean_and_engineer.py   # Phase 5 (fixes) + 7 — cleaning, encoding, engineered features
python train_model.py          # Phase 8-9 — baseline + final model, business validation
python monitor.py              # Phase 13 — drift check + retrain-trigger simulation
```

## Serve the model (Phase 10)

```bash
uvicorn app:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d '{
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": 190.0
      }'
```

Expected: a high risk score (~0.95) and `"risk_tier": "High"` — this
profile (new customer, month-to-month, fiber, electronic check) matches
the exact combination the EDA identifies as highest-risk.

## File map

| File | SDLC Phase | What it does |
|---|---|---|
| `data_loader.py` | 5 | Loads the real dataset, documents its source |
| `eda.py` | 6 | Churn-rate breakdowns by contract, tenure, internet type, payment method, add-ons |
| `clean_and_engineer.py` | 5 (fixes) + 7 | All cleaning, eliminations, encoding, and engineered features — **fully commented with reasoning** |
| `split.py` | 7 | Stratified train/val/test split, with reasoning for why stratified (not time-based) |
| `train_model.py` | 8–9 | Baseline (logistic regression) → Random Forest → XGBoost, business-metric evaluation, feature importance |
| `app.py` | 10 | FastAPI scoring service, reuses the exact same cleaning/feature code as training (avoids train/serve skew) |
| `monitor.py` | 13 | PSI-based drift check + recall-drop retrain trigger |

## Outputs produced (in `outputs/`)

- `engineered_data.csv`
- `eda_summary.png`
- `experiment_log.csv` — baseline vs. candidate models
- `business_validation.json` — projected revenue impact (illustrative — see the note field)
- `feature_importance.csv`
- `churn_model.joblib`, `scaler.joblib`, `feature_columns.joblib`
- `model_card.json` — includes explicit known limitations

## Known limitations (stated honestly, not hidden)

- This is a **single snapshot**, not repeated monthly observations per
  customer — so the "time-based split" a real production system should
  use isn't possible here; a stratified random split is used instead
  (reasoning in `split.py`).
- The revenue-impact estimate in `business_validation.json` depends on an
  **assumed** 35% retention-offer success rate, since this dataset has no
  actual offer/response data. Replace with a real campaign response rate
  before treating that number as fact.
- The monitoring script simulates "new data" using the held-out test set,
  since there's no second time period to genuinely compare against — see
  the comment at the top of `monitor.py`.
