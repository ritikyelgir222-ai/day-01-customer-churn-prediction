"""
Phases 5 (data quality fixes) & 7 (feature engineering)
------------------------------------------------------------
Every transformation, elimination, and engineered feature below has a
one-line WHY comment next to it. This is deliberately verbose — the
intent is that someone reviewing this file (a stakeholder, a teammate,
or future-you) can audit every decision without having to guess at the
reasoning.

FINAL FEATURE LIST is built at the bottom as FEATURE_COLUMNS.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Columns dropped entirely, and why
# ---------------------------------------------------------------------------
# customerID: a unique identifier with no generalizable signal. Keeping it
#             in the feature set would let a model "memorize" individual
#             customers on the training set and completely fail to
#             generalize — this is a classic leakage trap, so it's dropped
#             before feature engineering and re-attached only for reporting
#             (e.g. handing a scored list back to the CRM).
COLUMNS_TO_DROP = ["customerID"]

# ---------------------------------------------------------------------------
# Columns collapsed from 3 categories down to 2, and why
# ---------------------------------------------------------------------------
# MultipleLines has {"Yes", "No", "No phone service"}. "No phone service"
# is not new information — it's fully implied by PhoneService == "No".
# Keeping it as a separate category would let a tree model split on it
# redundantly and would add a needless one-hot column. We collapse it to
# "No" so the column purely answers "does this customer have multiple
# phone lines," and the "do they have phone service at all" question stays
# with PhoneService, where it belongs.
#
# The same logic applies to OnlineSecurity, OnlineBackup, DeviceProtection,
# TechSupport, StreamingTV, and StreamingMovies, which each have a
# "No internet service" category that's fully implied by
# InternetService == "No". Collapsing these keeps each column focused on
# a single yes/no question ("does the customer have this add-on") instead
# of conflating "doesn't have this add-on" with "can't have this add-on
# because they have no internet at all."
COLUMNS_WITH_REDUNDANT_SERVICE_CATEGORY = [
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Binary Yes/No columns that map straight to 0/1
BINARY_YES_NO_COLUMNS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn",
] + COLUMNS_WITH_REDUNDANT_SERVICE_CATEGORY  # these become binary AFTER collapsing

# Multi-category columns that get one-hot encoded (no ordinal relationship
# between categories, so one-hot is the right call rather than label
# encoding, which would falsely imply an order like DSL < Fiber < No)
ONE_HOT_COLUMNS = ["InternetService", "Contract", "PaymentMethod"]

# Add-on service columns used to build the "num_add_on_services" engineered
# feature below (after collapsing "No internet service" -> "No")
ADD_ON_SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    # -----------------------------------------------------------------
    # Fix 1: TotalCharges is stored as text and has 11 blank values.
    #
    # WHY IT HAPPENS: those 11 customers all have tenure == 0, i.e. they
    # signed up but hadn't been billed for a full cycle yet when this
    # snapshot was taken. The source system left TotalCharges blank
    # rather than 0 for these brand-new accounts.
    #
    # BUSINESS LOGIC APPLIED: a customer with 0 months of tenure has, by
    # definition, not accumulated any historical charges yet, so we
    # impute TotalCharges = 0 for these rows rather than dropping them.
    # Dropping would lose 11 real signed-up customers (a small but non-
    # zero group whose churn behavior is arguably some of the MOST
    # important to model, since new-customer churn is a known risk
    # window). Imputing with the column median would be factually wrong
    # here (it would invent billing history that doesn't exist), so 0 is
    # the logically correct value, not just a statistically convenient one.
    # -----------------------------------------------------------------
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # -----------------------------------------------------------------
    # Fix 2: drop identifier column (see COLUMNS_TO_DROP reasoning above)
    # -----------------------------------------------------------------
    df = df.drop(columns=COLUMNS_TO_DROP)

    # -----------------------------------------------------------------
    # Fix 3: collapse the redundant "No internet/phone service" category
    # down to "No" (see reasoning above). Doing this BEFORE binary-encoding
    # so the subsequent Yes/No -> 1/0 mapping is a clean two-value map.
    # -----------------------------------------------------------------
    for col in COLUMNS_WITH_REDUNDANT_SERVICE_CATEGORY:
        df[col] = df[col].replace(
            {"No internet service": "No", "No phone service": "No"}
        )

    return df


def engineer_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    df = clean_df.copy()

    # -----------------------------------------------------------------
    # Binary-encode all Yes/No columns (including Churn, the target)
    # WHY: models need numeric input; Yes/No has no ordinal meaning so a
    # simple 0/1 map is correct (no need for one-hot on a 2-value column,
    # that would just be a redundant second column with the same info).
    # -----------------------------------------------------------------
    for col in BINARY_YES_NO_COLUMNS:
        df[col] = (df[col] == "Yes").astype(int)

    # gender: encode as binary too. NOTE ON FAIRNESS: gender and
    # SeniorCitizen are demographic attributes. We keep them here because
    # they are legitimate, commonly-used signals in telecom churn modeling
    # (unlike, say, credit or lending decisions, which have stricter
    # regulatory scrutiny around protected attributes) — but per Phase 9
    # of the SDLC doc, any model using these should still go through a
    # bias/fairness check before being used to make customer-facing
    # decisions (e.g. confirming the model isn't systematically
    # under-serving retention offers to any demographic group).
    df["gender"] = (df["gender"] == "Male").astype(int)

    # -----------------------------------------------------------------
    # One-hot encode the multi-category columns
    # WHY one-hot and not label encoding: Contract ("Month-to-month",
    # "One year", "Two year") might look ordinal, but the actual risk
    # relationship isn't linear (month-to-month is dramatically riskier,
    # not just "one step" riskier), so treating it as an unordered
    # categorical and letting the model learn each category's own
    # coefficient/split is more accurate than forcing an artificial
    # numeric order.
    # -----------------------------------------------------------------
    df = pd.get_dummies(df, columns=ONE_HOT_COLUMNS, prefix=ONE_HOT_COLUMNS)
    # convert the resulting True/False dummy columns to 0/1 ints for
    # consistency with the rest of the feature set
    dummy_cols = [c for c in df.columns if any(c.startswith(p + "_") for p in ONE_HOT_COLUMNS)]
    df[dummy_cols] = df[dummy_cols].astype(int)

    # -----------------------------------------------------------------
    # Engineered feature 1: num_add_on_services
    # BUSINESS LOGIC: a customer with more add-on services (security,
    # backup, device protection, tech support, streaming) is more
    # "embedded" in the ecosystem and has a higher switching cost to
    # leave. This is a well-documented pattern in telecom churn — bundled
    # customers churn less — and it compresses 6 correlated binary
    # columns into a single, more stable signal, which also reduces
    # multicollinearity among the individual add-on columns.
    # -----------------------------------------------------------------
    df["num_add_on_services"] = df[ADD_ON_SERVICE_COLUMNS].sum(axis=1)

    # -----------------------------------------------------------------
    # Engineered feature 2: avg_historical_monthly_charge
    # BUSINESS LOGIC: TotalCharges / tenure reconstructs what the
    # customer has been paying on average over their whole relationship.
    # Comparing that to their CURRENT MonthlyCharges (see charge_deviation
    # below) reveals whether their bill just went up or down relative to
    # their own history — a classic churn trigger (e.g. a promotional
    # rate expiring). We guard tenure == 0 to avoid dividing by zero for
    # the brand-new accounts fixed above; for those, current
    # MonthlyCharges is the best available estimate of their "average."
    # -----------------------------------------------------------------
    df["avg_historical_monthly_charge"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"],
    )

    # -----------------------------------------------------------------
    # Engineered feature 3: charge_deviation ("bill shock" proxy)
    # BUSINESS LOGIC: current bill minus historical average. A large
    # positive value means the customer is suddenly paying noticeably
    # more than they're used to — a plausible, explainable churn driver
    # that a raw MonthlyCharges column alone would not capture (a
    # customer paying $90/month who has ALWAYS paid $90/month is in a
    # very different situation from one who just jumped from $60 to $90).
    # -----------------------------------------------------------------
    df["charge_deviation"] = df["MonthlyCharges"] - df["avg_historical_monthly_charge"]

    # -----------------------------------------------------------------
    # NOT engineered, and why: we did NOT drop TotalCharges even though
    # it is mechanically close to tenure * MonthlyCharges (i.e. partially
    # redundant with two other features already in the model). We kept
    # it because charge_deviation and avg_historical_monthly_charge are
    # both DERIVED from it, and tree-based models (our final model) are
    # robust to moderate multicollinearity — they simply won't split on
    # a redundant column if a correlated one already captures the signal.
    # For a linear/logistic baseline this redundancy matters more, which
    # is one reason the baseline is expected to underperform the final
    # model here, not just an accident of tuning.
    # -----------------------------------------------------------------

    return df


def get_feature_columns(engineered_df: pd.DataFrame) -> list:
    """Every column except the target and anything explicitly excluded."""
    exclude = {"Churn"}
    return [c for c in engineered_df.columns if c not in exclude]


if __name__ == "__main__":
    from data_loader import load_raw_data

    raw = load_raw_data()
    cleaned = clean_data(raw)
    engineered = engineer_features(cleaned)
    feature_cols = get_feature_columns(engineered)

    print(f"Raw columns: {len(raw.columns)}  ->  Final feature columns: {len(feature_cols)}")
    print("\nFinal feature columns:")
    for c in feature_cols:
        print(f"  - {c}")

    engineered.to_csv("outputs/engineered_data.csv", index=False)
    print("\nSaved -> outputs/engineered_data.csv")
