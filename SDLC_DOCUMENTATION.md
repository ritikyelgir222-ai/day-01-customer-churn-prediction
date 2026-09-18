# SDLC Documentation: Customer Churn Prediction
### Telecom Industry | ML | Day 1 of the 100-Day Series
### Filled against the master 14-phase SDLC template, using the real project we built

This follows `sdlc-documentation-template.md` phase by phase. Every number
below is taken directly from the actual pipeline run (`train_model.py`,
`eda.py`, `monitor.py`) on the real IBM Telco Customer Churn dataset — not
illustrative placeholders.

---

## Phase 1: Discovery & Stakeholder Requirement Gathering
**Owner:** Business Analyst / Data Scientist | **Output:** Meeting notes, stakeholder map

- **Stakeholders identified:**
  - Business problem owner: VP of Customer Retention
  - Budget approver: Head of Marketing
  - Day-to-day users: Retention campaign managers
- **Discovery findings:**
  - Current process: blanket retention discounts sent to any customer who complains or shows obvious risk signs, with no way to prioritize who's actually worth targeting
  - Decision this project informs: *who* gets a retention offer, ranked by actual churn risk
  - Data already exists: customer profile, billing, contract, and service-usage data in the CRM/billing systems — no support-ticket or usage-trend data is currently captured, a real constraint discovered once we looked at what fields the dataset actually had
  - Compliance constraints: model must be explainable enough that a retention agent can be told *why* a customer is flagged, not just a black-box score
- **Deliverable — Problem Statement:** "Reduce revenue loss from preventable churn by replacing undifferentiated retention outreach with a targeted, risk-ranked customer list."

---

## Phase 2: Business Requirement Document (BRD)
**Owner:** Business Analyst | **Output:** BRD

- **Business objective:** Prioritize the highest-risk ~20% of customers for retention outreach, rather than treating all customers equally
- **Scope:**
  - In-scope: all postpaid customers with complete billing and contract data (7,043 customers in the dataset used)
  - Out-of-scope: no distinction was made by customer segment (e.g. residential vs. business) — the dataset doesn't carry that field, which is itself a documented constraint, not an oversight
- **Success metrics / KPIs:**
  - Primary: top-decile recall (what fraction of actual churners are caught in the highest-risk 10%) — chosen because it's what a capacity-limited retention team can actually act on
  - Secondary: estimated annualized revenue saved from targeted retention offers
- **Assumptions & constraints:** single-snapshot dataset (no repeated monthly history per customer); no real offer/response data available, so revenue impact had to be estimated using an assumed 35% offer-success rate rather than a measured one
- **Sign-off:** N/A — this is a self-directed portfolio project; in a real engagement this BRD would require sign-off from the VP of Retention before technical work began

---

## Phase 3: Functional & Technical Requirement Document (FRD/TRD)
**Owner:** Data Scientist / Tech Lead | **Output:** FRD/TRD

- **Functional requirements:**
  - System scores a customer given their profile, contract, and billing data
  - Returns a risk score, a risk tier (High/Medium/Low), and the top factors driving that score
  - Exposed as a REST API (`/score`) plus a simple browser-based test form
- **Non-functional requirements:**
  - Response time: sub-second for a single-customer score (confirmed in testing)
  - Training/serving consistency: the exact same cleaning and feature-engineering code (`clean_and_engineer.py`) is imported by both the training script and the API, specifically to prevent training/serving skew
- **Data sources:** a single static CSV (`data/Telco-Customer-Churn.csv`) — the IBM Telco Customer Churn dataset, sourced from Kaggle (blastchar/telco-customer-churn)
- **Integration points:** none in this version — designed as a standalone service; a production version would integrate with a CRM to write scores back and a marketing automation platform to trigger offers

---

## Phase 4: Project Planning
**Owner:** Project/Delivery Manager | **Output:** Project plan, risk register

- **Team & roles:** single-person build (data loading, EDA, feature engineering, modeling, API, monitoring, and documentation all done by the same person/session for this portfolio project — in a real team this would be split across a Data Engineer, Data Scientist, and MLOps Engineer as in the generic template)
- **Build sequence used:**
  1. Data loading & inspection
  2. EDA
  3. Cleaning & feature engineering
  4. Model training & evaluation
  5. API deployment
  6. Monitoring script
  7. Documentation
- **Risk register (real risks that came up during the build, not hypothetical ones):**
  - Risk: `TotalCharges` was stored as text with 11 blank values — discovered during Phase 5, not anticipated in planning. Resolved by tracing the cause (all `tenure == 0` customers) and imputing 0 rather than guessing.
  - Risk: FastAPI's browser-facing root URL returned 404/405 when the user tested it directly in a browser — not a bug, but a real point of confusion resolved by adding a proper test-form route at `/`.

---

## Phase 5: Data Collection & Data Understanding
**Owner:** Data Engineer / Data Scientist | **Output:** Data dictionary, data quality report

- **Data inventory:** one static CSV, 7,043 rows, 21 columns, sourced from Kaggle (see `data_loader.py` for full source documentation)
- **Data dictionary:** see `PROJECT_DOCUMENTATION.md` Section 3 for the full column-by-column dictionary, including what was transformed, collapsed, or dropped and why
- **Data quality report (actual findings):**
  - No duplicate rows
  - `TotalCharges` stored as string with 11 blank values, all corresponding to `tenure == 0` customers — imputed as 0, not dropped, since these are real signed-up customers whose churn behavior matters
  - Class imbalance: 26.54% churn rate — addressed via `class_weight="balanced"` (logistic regression, random forest) and `scale_pos_weight` (XGBoost), not by discarding data
  - Several columns (`MultipleLines` and 6 add-on service columns) carried a redundant "No internet/phone service" category that added no information beyond what `InternetService`/`PhoneService` already captured — collapsed to reduce noisy one-hot columns
- **Access & governance:** dataset is a public, already-anonymized sample; no PII present, no additional governance steps required for this exercise (a real internal dataset would require this step to be substantive, not skipped)

---

## Phase 6: Exploratory Data Analysis (EDA)
**Owner:** Data Scientist | **Output:** EDA notebook/report (`eda.py`, `outputs/eda_summary.png`)

**Actual findings, not illustrative ones:**

- Churn by contract type: **42.7%** (month-to-month) vs. 11.3% (one year) vs. **2.8%** (two year)
- Churn by tenure: 52.9% in months 0–6, declining steadily to 9.5% at 49–72 months
- Churn by internet service: **41.9%** (fiber optic) vs. 19.0% (DSL) vs. 7.4% (no internet)
- Churn by payment method: **45.3%** (electronic check) — clearly the highest of the four methods, more than double the next closest (mailed check, 19.1%)
- Churn by number of add-on services: generally declining from 21.4% (0 add-ons) to 5.3% (6 add-ons)

**Hypotheses carried into modeling** (and confirmed by the final feature importances in Phase 8): contract type, internet service type, and payment method would be the strongest predictors — this held up; contract type alone accounted for ~54% of the final model's total feature importance across its three encoded categories.

---

## Phase 7: Data Preprocessing & Feature Engineering
**Owner:** Data Scientist / ML Engineer | **Output:** `clean_and_engineer.py`

- **Cleaning:** `TotalCharges` converted from text to numeric, blanks imputed as 0 (see Phase 5); `customerID` dropped (unique identifier, would cause leakage/overfitting rather than generalize)
- **Encoding:** binary Yes/No columns mapped to 0/1; multi-category columns (`InternetService`, `Contract`, `PaymentMethod`) one-hot encoded rather than label-encoded, specifically because their categories aren't ordinal (e.g. month-to-month isn't "one step less" risky than two-year — it's a fundamentally different risk regime)
- **Feature engineering (3 new features, each with a stated business hypothesis):**
  1. `num_add_on_services` — bundled customers are more embedded, hypothesized to churn less (confirmed in EDA)
  2. `avg_historical_monthly_charge` — reconstructs a customer's historical average bill from `TotalCharges / tenure`
  3. `charge_deviation` — current bill minus historical average, a "bill shock" proxy
- **Split strategy:** stratified random split (85/15 train+val/test, then 15% of the remainder for validation), *not* time-based — because this dataset is a single cross-sectional snapshot with no repeated monthly observations per customer, a time-based split (the generically "correct" choice per the template) isn't actually possible here. This limitation is stated explicitly rather than glossed over.

---

## Phase 8: Model Development
**Owner:** Data Scientist / ML Engineer | **Output:** `train_model.py`, `outputs/experiment_log.csv`

**Actual experiment log (validation set):**

| Model | Val AUC | Val Top-Decile Recall |
|---|---|---|
| Logistic Regression (baseline) | 0.8293 | 0.2740 |
| Random Forest | 0.8304 | 0.2776 |
| XGBoost (final) | 0.8256 | 0.2705 |

- **Baseline model:** logistic regression, chosen for explainability and to establish a performance floor before trying anything more complex
- **Model selection criterion:** top-decile recall (the business-relevant metric), not raw AUC
- **Honest note (not glossed over):** Random Forest actually performed marginally *better* than XGBoost on this validation run. XGBoost was still selected as the final model for reasons beyond this one split — incremental training support, early stopping, and monotonic constraint support for future business rules — but this is documented as a judgment call on a near-tie, not a clear technical win, exactly as the template's "not just accuracy" guidance intends.

---

## Phase 9: Model Evaluation & Business Validation
**Owner:** Data Scientist | **Output:** `outputs/business_validation.json`, `outputs/feature_importance.csv`

- **Technical metrics (held-out test set, 1,057 customers, never touched before this step):**
  - Test AUC: 0.8464
  - Precision at top-20% threshold: 0.6745
  - Recall at top-20% threshold: 0.5107
  - Top-decile recall: 0.2929
- **Business validation:** the model's top-risk decile catches an estimated 82 of the actual churners in that group. Assuming a 35% retention-offer success rate — explicitly flagged as an *assumption*, since this dataset has no real offer/response data — this implies ~29 prevented churns and an estimated **$22,698/year** in retained revenue for this test segment.
- **Bias/fairness check:** **not performed** in this version. `gender` and `SeniorCitizen` are included as model features. Per the template's Phase 9 guidance, a real deployment influencing customer-facing decisions should not skip this step — it's called out explicitly in the limitations rather than left unmentioned.
- **Explainability:** feature importance computed and reviewed (see Phase 6 — findings matched EDA hypotheses). Per-prediction explanations in the API use global feature importance as a lighter-weight stand-in for true per-prediction SHAP values, a stated simplification, not a hidden one.

---

## Phase 10: MLOps & Deployment
**Owner:** ML Engineer | **Output:** `app.py`

- **Packaging:** FastAPI service exposing `/score` (POST), `/health`, and a browser-based `/` test form
- **Design discipline applied:** the API imports the exact same `clean_and_engineer.py` functions used in training, rather than reimplementing the transformation logic — this was a deliberate choice to prevent training/serving skew, a common real-world production bug
- **CI/CD, containerization:** not implemented in this version (no Docker, no automated pipeline) — noted as a gap versus the generic template rather than presented as done
- **Versioning:** model, scaler, and feature-column list are saved as separate joblib artifacts (`churn_model.joblib`, `scaler.joblib`, `feature_columns.joblib`) alongside a `model_card.json` documenting training row count and known limitations

---

## Phase 11: Testing
**Owner:** QA / Data Scientist | **Output:** manual test log (no formal test suite written)

- **What was actually tested:**
  - Ran the full pipeline end-to-end from raw CSV through model training, confirming no errors and stable output at each stage
  - Verified the API's scoring logic directly and via FastAPI's `TestClient`, using two constructed profiles: a high-risk profile (new, month-to-month, fiber, electronic check → scored 0.9526, correctly tiered "High") and a low-risk profile (long-tenure, two-year, DSL, auto-pay → scored 0.0296, correctly tiered "Low")
  - Confirmed the root `/` test form renders correctly and successfully calls `/score`
- **What was NOT done (stated honestly):** no formal unit test suite, no UAT with an actual retention team, since there is no real stakeholder for this portfolio project. This is a real gap relative to the generic template's Phase 11, not a step quietly skipped without mention.

---

## Phase 12: Documentation & Handover
**Owner:** Data Scientist | **Output:** `README.md`, `PROJECT_DOCUMENTATION.md`, this file

- **Technical documentation:** `PROJECT_DOCUMENTATION.md` covers architecture, full data dictionary, EDA findings, modeling results, API reference, and known limitations
- **User manual:** the `/` test form in `app.py` doubles as a lightweight user manual — no separate dashboard exists
- **Runbook:** not formally written; `README.md`'s run-order instructions serve this purpose informally

---

## Phase 13: Monitoring & Maintenance
**Owner:** MLOps / Data Scientist | **Output:** `monitor.py`

- **Drift monitoring implemented:** Population Stability Index (PSI) on `tenure`, `MonthlyCharges`, `TotalCharges`, `num_add_on_services`
- **Actual result when run:** all four features showed PSI < 0.02 (well within the "stable" range) — expected, since the "new batch" compared against is the held-out test set from the same original snapshot, not genuinely new data collected over time. This is stated explicitly in `monitor.py`'s comments as a simulation limitation, not presented as a real production drift check.
- **Retraining trigger implemented:** compares current top-decile recall against the baseline stored in `business_validation.json`; triggers if recall drops more than 0.05. Confirmed working (0.000 recall change on this run, correctly reporting "no retrain needed").
- **Support SLA:** not applicable — no live production system or support team exists for this portfolio project.

---

## Phase 14: Project Closure & Delivery
**Owner:** N/A (self-directed project) | **Output:** this document, `PROJECT_DOCUMENTATION.md`

- **Closure against original objective:** the stated Phase 2 objective — prioritize the highest-risk ~20% of customers — was met: the model achieves 0.6745 precision and 0.5107 recall at that threshold on real, held-out data.
- **Retrospective:**
  - What went well: the shared cleaning module (`clean_and_engineer.py`) between training and serving avoided a whole category of bugs; documenting the *reasoning* behind each transformation (not just the code) made the eventual "why did you do X" questions trivial to answer
  - What to improve next time: build in a lightweight automated test suite from the start rather than only manual verification; get real offer-response data (or a documented proxy) before quoting a revenue-impact number, even an illustrative one
- **Handover:** this project is packaged as a complete, runnable zip (code + data + trained model + documentation) — no separate maintenance team exists, but everything needed to pick this back up cold is included.

---

## Mapping back to the Day 1 LinkedIn post

| LinkedIn section | Pulled from phases | Real number used |
|---|---|---|
| Client & Problem | 1–2 | Telecom, blanket-discount problem |
| Requirements & Data | 3, 5 | IBM Telco dataset, 7,043 customers |
| Approach | 6–8 | EDA-confirmed drivers → XGBoost |
| Result & Business Impact | 9 | Top-decile recall 0.293, ~$22.7K/year illustrative |
| Path to Production | 10–13 | FastAPI + PSI drift monitoring |
