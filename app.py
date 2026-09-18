"""
Phase 10: MLOps & Deployment
--------------------------------
A minimal FastAPI service that scores a single customer record using the
raw fields a CRM would actually have on hand — the cleaning and feature
engineering logic from clean_and_engineer.py is reused here rather than
duplicated, so a transformation change only ever needs to happen in one
place (a common and important production discipline: training/serving
skew, where the API silently uses different logic than training, is one
of the most common causes of a model behaving worse in production than
in evaluation).

Run with:  uvicorn app:app --reload
"""

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from clean_and_engineer import clean_data, engineer_features

app = FastAPI(title="Telco Churn Risk Scoring API", version="1.0")

MODEL = joblib.load("outputs/churn_model.joblib")
FEATURE_COLUMNS = joblib.load("outputs/feature_columns.joblib")


class CustomerRecord(BaseModel):
    # Field names and casing intentionally mirror the raw CRM columns —
    # WHY: this means an integration engineer wiring this endpoint up to
    # the actual CRM export doesn't have to remember a second, different
    # naming convention just for this API.
    gender: str  # "Male" | "Female"
    SeniorCitizen: int  # 0 or 1
    Partner: str  # "Yes" | "No"
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str  # "Yes" | "No" | "No phone service"
    InternetService: str  # "DSL" | "Fiber optic" | "No"
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str  # "Month-to-month" | "One year" | "Two year"
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class ScoreResponse(BaseModel):
    churn_risk_score: float
    risk_tier: str
    top_factors: list[str]


def tier_from_score(score: float) -> str:
    # WHY THESE CUTOFFS: chosen to roughly match the 80th-percentile
    # operating threshold used during evaluation (High ~ top 20%), with
    # Medium as a softer watch-list band below that. These are business
    # judgment calls, not statistically derived boundaries — a retention
    # team could reasonably move them based on how many agents they have.
    if score >= 0.5:
        return "High"
    if score >= 0.25:
        return "Medium"
    return "Low"


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# A tiny built-in test form at the root URL.
# WHY THIS EXISTS: the API's real endpoint (/score) only accepts POST
# requests with a JSON body — visiting it directly in a browser sends a
# GET request instead, which correctly returns "405 Method Not Allowed."
# That's not a bug, it's the API behaving correctly. This page exists
# purely so you can test the model by clicking a button instead of using
# curl or Swagger's raw JSON editor.
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head><title>Telco Churn Risk Scoring</title></head>
    <body style="font-family: sans-serif; max-width: 600px; margin: 40px auto;">
        <h2>Telco Churn Risk Scoring — Test Form</h2>
        <p>Fill in a customer profile and click Score. Full API docs at
           <a href="/docs">/docs</a>.</p>
        <form id="scoreForm">
            <label>Tenure (months): <input name="tenure" type="number" value="2"></label><br><br>
            <label>Contract:
                <select name="Contract">
                    <option>Month-to-month</option>
                    <option>One year</option>
                    <option>Two year</option>
                </select>
            </label><br><br>
            <label>Internet Service:
                <select name="InternetService">
                    <option>Fiber optic</option>
                    <option>DSL</option>
                    <option>No</option>
                </select>
            </label><br><br>
            <label>Payment Method:
                <select name="PaymentMethod">
                    <option>Electronic check</option>
                    <option>Mailed check</option>
                    <option>Bank transfer (automatic)</option>
                    <option>Credit card (automatic)</option>
                </select>
            </label><br><br>
            <label>Monthly Charges ($): <input name="MonthlyCharges" type="number" step="0.01" value="95.00"></label><br><br>
            <label>Total Charges ($): <input name="TotalCharges" type="number" step="0.01" value="190.00"></label><br><br>
            <button type="submit">Score this customer</button>
        </form>
        <h3 id="result"></h3>
        <script>
        document.getElementById("scoreForm").addEventListener("submit", async function(e) {
            e.preventDefault();
            const form = new FormData(e.target);
            const payload = {
                gender: "Female", SeniorCitizen: 0, Partner: "No", Dependents: "No",
                tenure: parseInt(form.get("tenure")),
                PhoneService: "Yes", MultipleLines: "No",
                InternetService: form.get("InternetService"),
                OnlineSecurity: "No", OnlineBackup: "No", DeviceProtection: "No",
                TechSupport: "No", StreamingTV: "Yes", StreamingMovies: "Yes",
                Contract: form.get("Contract"),
                PaperlessBilling: "Yes",
                PaymentMethod: form.get("PaymentMethod"),
                MonthlyCharges: parseFloat(form.get("MonthlyCharges")),
                TotalCharges: parseFloat(form.get("TotalCharges"))
            };
            const res = await fetch("/score", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            document.getElementById("result").innerText =
                "Risk score: " + data.churn_risk_score +
                " | Tier: " + data.risk_tier +
                " | Top factors: " + data.top_factors.join(", ");
        });
        </script>
    </body>
    </html>
    """


@app.post("/score", response_model=ScoreResponse)
def score_customer(record: CustomerRecord):
    raw_row = pd.DataFrame([record.model_dump()])
    raw_row["customerID"] = "N/A"  # placeholder so clean_data()'s drop-column step has something to drop
    raw_row["Churn"] = "No"  # placeholder; unused for scoring, but engineer_features() expects the column to exist

    cleaned = clean_data(raw_row)
    engineered = engineer_features(cleaned)

    # WHY REINDEX WITH FEATURE_COLUMNS: a single-row request may not
    # contain every one-hot category seen during training (e.g. this one
    # customer's PaymentMethod one-hot won't produce columns for the
    # other three payment methods). Reindexing against the saved
    # training-time column list fills any missing dummy columns with 0,
    # which is the mathematically correct value for "this category was
    # not present for this row" — and guards against a column-order
    # mismatch silently corrupting predictions.
    X = engineered.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    proba = float(MODEL.predict_proba(X)[0, 1])
    tier = tier_from_score(proba)

    # Per-prediction explanation: a full SHAP call would be more precise,
    # but is left out here to avoid an extra heavyweight dependency for a
    # demo endpoint. Global feature importance is used as a lighter-weight
    # stand-in — acceptable for now, but flagged as a known simplification
    # a production version should replace with real per-prediction SHAP
    # values before being shown to retention agents as "why" explanations.
    importances = pd.Series(MODEL.feature_importances_, index=FEATURE_COLUMNS)
    top_factors = importances.sort_values(ascending=False).head(3).index.tolist()

    return ScoreResponse(
        churn_risk_score=round(proba, 4),
        risk_tier=tier,
        top_factors=top_factors,
    )
