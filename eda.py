"""
Phase 6: Exploratory Data Analysis
------------------------------------
WHY THESE SPECIFIC CUTS: we don't explore every column exhaustively here —
we deliberately focus on the cuts a retention stakeholder would actually
ask about first (contract type, tenure, and add-on services), because
Phase 6's job is to generate hypotheses that Phase 7/8 can act on, not to
produce every possible chart. Payment method and internet type are checked
next since they're the next most commonly cited churn drivers in telecom,
both in this dataset's own documentation and in the domain literature.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import load_raw_data
from clean_and_engineer import clean_data


def run_eda(df: pd.DataFrame, out_dir: str = "outputs"):
    df = df.copy()
    df["churned"] = (df["Churn"] == "Yes").astype(int)

    print("=== Churn rate by contract type ===")
    print(df.groupby("Contract")["churned"].mean().round(3))
    # WHY WE CHECK THIS FIRST: contract type is the single feature every
    # telecom retention playbook already assumes matters most, so it's
    # the natural first sanity check before trusting anything else in
    # the data.

    print("\n=== Churn rate by tenure bucket ===")
    tenure_bucket = pd.cut(
        df["tenure"], bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6m", "7-12m", "13-24m", "25-48m", "49-72m"],
    )
    print(df.groupby(tenure_bucket, observed=True)["churned"].mean().round(3))
    # WHY: tests the "early tenure is the highest-risk window" hypothesis
    # commonly cited for this exact dataset.

    print("\n=== Churn rate by internet service type ===")
    print(df.groupby("InternetService")["churned"].mean().round(3))
    # WHY: fiber customers pay significantly more on average; checking
    # whether that translates into higher churn (price sensitivity) is a
    # natural follow-up once we see MonthlyCharges distributions.

    print("\n=== Churn rate by payment method ===")
    print(df.groupby("PaymentMethod")["churned"].mean().round(3))
    # WHY: payment friction (manual vs. automatic payment) is a
    # documented churn driver in telecom — customers on manual payment
    # methods have to actively re-commit every billing cycle.

    cleaned = clean_data(df)
    cleaned["churned"] = df["churned"].values
    add_on_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    # NOTE: clean_data() already collapsed "No internet service" -> "No"
    # for these columns, so a simple "== 'Yes'" count is safe here.
    cleaned["num_add_on_services"] = cleaned[add_on_cols].apply(lambda r: (r == "Yes").sum(), axis=1)
    print("\n=== Churn rate by number of add-on services ===")
    print(cleaned.groupby("num_add_on_services")["churned"].mean().round(3))
    # WHY: tests the "bundled customers are stickier" hypothesis that
    # motivates the num_add_on_services engineered feature.

    # ---- Chart: the three strongest cuts, side by side ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    df.groupby("Contract")["churned"].mean().plot(
        kind="bar", ax=axes[0], color="#4C72B0", title="Churn rate by contract type"
    )
    axes[0].set_ylabel("Churn rate")
    axes[0].tick_params(axis="x", rotation=20)

    df.groupby(tenure_bucket, observed=True)["churned"].mean().plot(
        kind="bar", ax=axes[1], color="#55A868", title="Churn rate by tenure"
    )
    axes[1].set_ylabel("Churn rate")

    cleaned.groupby("num_add_on_services")["churned"].mean().plot(
        kind="bar", ax=axes[2], color="#C44E52", title="Churn rate by # add-on services"
    )
    axes[2].set_ylabel("Churn rate")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/eda_summary.png", dpi=120)
    print(f"\nSaved chart -> {out_dir}/eda_summary.png")


if __name__ == "__main__":
    data = load_raw_data()
    run_eda(data)
