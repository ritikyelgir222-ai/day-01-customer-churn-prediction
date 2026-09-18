"""
Phase 5: Data Collection & Data Understanding
------------------------------------------------
DATA SOURCE
-----------
This project uses the IBM "Telco Customer Churn" dataset — the same file
published on Kaggle as "Telco Customer Churn" by user blastchar:
    https://www.kaggle.com/datasets/blastchar/telco-customer-churn

It originates from IBM's own sample data (originally hosted as
"WA_Fn-UseC_-Telco-Customer-Churn.csv" on IBM Watson Analytics / Cognos
sample data, and mirrored in several public IBM GitHub repos, e.g.
IBM/telco-customer-churn-on-icp4d).

It contains 7,043 real-structure (though anonymized/fictional) customer
records for a California-based telecom provider, with demographic,
account, service-usage, and billing fields, plus a Churn label.

If you're following along on Kaggle: download `WA_Fn-UseC_-Telco-Customer-
Churn.csv` from the link above and place it at `data/Telco-Customer-
Churn.csv` — the schema is identical to the file already included in this
project, so nothing else needs to change.

WHY THIS DATASET
-----------------
- It's a real, widely-used churn dataset (not synthetic), so the signals
  we find in EDA are genuine relationships, not ones we baked in ourselves.
- It has the exact structure a telecom retention team would actually have:
  contract type, tenure, billing, and service add-ons — which lines up
  with the "client & problem" framing used throughout the 100-day series.
- It has one well-known, well-documented data quality issue (blank
  TotalCharges for brand-new customers) which is a good, realistic case
  study for Phase 5's "data quality report" step — see `clean_and_engineer.py`.
"""

import pandas as pd

RAW_DATA_PATH = "data/Telco-Customer-Churn.csv"


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {RAW_DATA_PATH}")
    print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")
    print("\nColumn dtypes:")
    print(df.dtypes)
