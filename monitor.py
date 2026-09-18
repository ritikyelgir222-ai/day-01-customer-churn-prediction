"""
Phase 13: Monitoring & Maintenance
--------------------------------------
WHY PSI (Population Stability Index) specifically: PSI is the standard,
interview-recognizable metric for feature drift in industry because it
gives a single interpretable number per feature with well-established
severity thresholds (< 0.1 stable, 0.1-0.25 moderate shift, > 0.25
significant shift needing action) — as opposed to, say, a raw
Kolmogorov-Smirnov statistic, which is harder to explain to a
non-technical stakeholder in a monitoring dashboard.

WHY THIS SCRIPT SIMULATES A "NEW BATCH" instead of using held-out real
data: this dataset is a single snapshot with no natural second time
period to compare against. Since a real production system compares
THIS MONTH's incoming customer data against the distribution the model
was TRAINED on, we simulate that by re-using the test set as a stand-in
"new batch" — it's genuinely unseen by the model (same reason it's the
right choice for the final evaluation in train_model.py), so it's a
reasonable proxy for "new data arriving after deployment," even though
in reality it's from the same original snapshot.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import joblib

from data_loader import load_raw_data
from clean_and_engineer import clean_data, engineer_features, get_feature_columns
from split import split_data
from train_model import top_decile_recall

RECALL_DROP_THRESHOLD = 0.05  # per SDLC doc Phase 13: retrain if top-decile recall drops more than this


def population_stability_index(expected, actual, bins=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    expected_pct = np.clip(expected_pct, 1e-4, None)
    actual_pct = np.clip(actual_pct, 1e-4, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def run_monitoring_check():
    model = joblib.load("outputs/churn_model.joblib")
    feature_cols = joblib.load("outputs/feature_columns.joblib")

    raw = load_raw_data()
    cleaned = clean_data(raw)
    engineered = engineer_features(cleaned)

    X = engineered[feature_cols]
    y = engineered["Churn"]
    # WHY WE RE-SPLIT HERE with the same random_state as training: this
    # reconstructs the exact same train/test partition used originally,
    # so "reference" (train) and "new batch" (test) are the correct,
    # non-overlapping groups to compare — not two arbitrary random slices.
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print("=== Feature Drift (PSI): train distribution vs. held-out batch ===")
    print("PSI < 0.1: stable | 0.1-0.25: moderate | > 0.25: significant drift\n")
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges", "num_add_on_services"]
    for feat in numeric_features:
        psi = population_stability_index(X_train[feat], X_test[feat])
        flag = "SIGNIFICANT DRIFT" if psi > 0.25 else ("moderate" if psi > 0.1 else "ok")
        print(f"  {feat}: PSI={psi:.4f} [{flag}]")
    # WHY NO DRIFT IS EXPECTED HERE: since train/test came from the same
    # random stratified split of one snapshot, we EXPECT near-zero PSI —
    # this run is really a check that the monitoring code itself works
    # correctly, not a genuine test of real-world drift. A live system
    # would replace X_test here with actual new customer data collected
    # after deployment.

    scores = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, scores)
    recall_top_decile = top_decile_recall(y_test.values, scores)

    print(f"\n=== Performance on held-out batch ===")
    print(f"AUC: {auc:.4f}")
    print(f"Top-decile recall: {recall_top_decile:.4f}")

    # WHY WE COMPARE AGAINST THE ORIGINAL TEST-SET RECALL (not an
    # arbitrary number): this is the same top_decile_recall value
    # train_model.py measured and stored — using it as the baseline
    # means the retrain trigger is measuring genuine performance decay
    # over time, not decay relative to an unrelated benchmark.
    import json
    with open("outputs/business_validation.json") as f:
        baseline_recall = json.load(f)["top_decile_recall"]

    drop = baseline_recall - recall_top_decile
    if drop > RECALL_DROP_THRESHOLD:
        print(f"\n⚠️  RETRAIN TRIGGERED: recall dropped by {drop:.3f} (threshold: {RECALL_DROP_THRESHOLD})")
    else:
        print(f"\n✅ No retrain needed (recall change: {drop:.3f}, threshold: {RECALL_DROP_THRESHOLD})")


if __name__ == "__main__":
    run_monitoring_check()
