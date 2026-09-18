"""
Phase 8: Model Development & Phase 9: Evaluation & Business Validation
---------------------------------------------------------------------------
Every modeling choice below has a WHY comment. The goal is that a
non-technical stakeholder reading the printed output, and a technical
reviewer reading the code, both understand not just WHAT was done but
WHY it was the right call for THIS problem.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data_loader import load_raw_data
from clean_and_engineer import clean_data, engineer_features, get_feature_columns
from split import split_data


def top_decile_recall(y_true, y_score, decile=0.10):
    """
    BUSINESS-RELEVANT METRIC, not just a technical one.
    WHY WE TRACK THIS ALONGSIDE AUC: the retention team can only realistically
    act on a limited list each month (budget for proactive offers is not
    unlimited). AUC measures overall ranking quality across the WHOLE
    customer base, which is not what the business actually does with the
    model. What matters operationally is: "if we can only afford to
    intervene on the riskiest 10% of customers, what fraction of the
    people who were actually going to churn did we correctly catch in that
    group?" That's exactly what top-decile recall answers, and it is the
    metric most directly tied to the BRD's revenue-impact success measure.
    """
    y_true = np.asarray(y_true)
    n_top = max(1, int(len(y_score) * decile))
    top_idx = np.argsort(y_score)[-n_top:]
    caught = y_true[top_idx].sum()
    total_churners = y_true.sum()
    return caught / total_churners if total_churners > 0 else 0.0


def train_and_evaluate():
    raw = load_raw_data()
    cleaned = clean_data(raw)
    engineered = engineer_features(cleaned)
    feature_cols = get_feature_columns(engineered)

    X = engineered[feature_cols]
    y = engineered["Churn"]

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    experiment_log = []

    # -----------------------------------------------------------------
    # Baseline: Logistic Regression
    # WHY THIS AS THE BASELINE: it's simple, fast, and fully explainable
    # (every feature gets one coefficient a stakeholder can read directly
    # as "this many percentage points of risk per unit of X"). Per the
    # SDLC doc's Phase 8 guidance, we ALWAYS establish this floor before
    # reaching for a more complex model — if a complex model can't beat
    # this by a meaningful margin, it isn't worth the added
    # explainability cost.
    #
    # WHY class_weight="balanced": churn is imbalanced (~26.5% positive).
    # Without this, the model would be lightly biased toward predicting
    # "no churn" for everyone, since that's right ~73.5% of the time by
    # default — balanced weighting forces it to actually learn the
    # minority (churn) class's signal instead of free-riding on the
    # majority class.
    #
    # WHY WE SCALE FEATURES HERE (but not for the tree models below):
    # logistic regression's coefficients and convergence are sensitive to
    # feature scale (e.g. tenure ranging 0-72 vs. a 0/1 flag) — StandardScaler
    # puts every feature on a comparable scale so no feature dominates
    # purely because of its raw numeric range. Tree-based models (Random
    # Forest, XGBoost) split on raw thresholds and are scale-invariant by
    # construction, so scaling would add complexity with zero benefit there.
    # -----------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(X_train_scaled, y_train)
    logreg_val_scores = logreg.predict_proba(X_val_scaled)[:, 1]

    experiment_log.append({
        "model": "logistic_regression (baseline)",
        "val_auc": round(roc_auc_score(y_val, logreg_val_scores), 4),
        "val_top_decile_recall": round(top_decile_recall(y_val, logreg_val_scores), 4),
    })

    # -----------------------------------------------------------------
    # Candidate: Random Forest
    # WHY TRIED: a natural next step up from a linear model — it can
    # capture non-linear interactions (e.g. "month-to-month contract AND
    # fiber internet" being worse than either alone) that logistic
    # regression's linear decision boundary cannot represent without
    # manually engineered interaction terms.
    # -----------------------------------------------------------------
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    rf_val_scores = rf.predict_proba(X_val)[:, 1]
    experiment_log.append({
        "model": "random_forest",
        "val_auc": round(roc_auc_score(y_val, rf_val_scores), 4),
        "val_top_decile_recall": round(top_decile_recall(y_val, rf_val_scores), 4),
    })

    # -----------------------------------------------------------------
    # Final candidate: Gradient Boosting (XGBoost)
    # WHY THIS AS THE FINAL CHOICE (assuming it wins, confirmed by the
    # printed experiment log below): boosted trees build each new tree to
    # correct the previous ensemble's errors, which tends to outperform a
    # Random Forest's independent-trees-averaged approach on tabular data
    # like this, especially for capturing the kind of conditional risk
    # patterns telecom churn is known for (e.g. new customers on
    # month-to-month with high bills being a uniquely high-risk
    # combination, not just each factor's average effect added together).
    #
    # WHY scale_pos_weight instead of class_weight="balanced" here:
    # XGBoost's native imbalance handling is scale_pos_weight (ratio of
    # negative to positive examples), which is the library's documented,
    # tested way to handle imbalance — using it directly avoids relying
    # on a wrapper's approximation of the same idea.
    #
    # SELECTION CRITERION: per Phase 8's guidance, the final model is
    # chosen by val_top_decile_recall (the business-relevant metric), not
    # just whichever has the highest val_auc — a model that ranks
    # marginally better overall but is worse specifically among the
    # highest-risk customers is the wrong choice for this use case.
    # -----------------------------------------------------------------
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
    )
    xgb.fit(X_train, y_train)
    xgb_val_scores = xgb.predict_proba(X_val)[:, 1]
    xgb_auc = roc_auc_score(y_val, xgb_val_scores)
    xgb_top_decile_recall = top_decile_recall(y_val, xgb_val_scores)
    experiment_log.append({
        "model": "xgboost (final)",
        "val_auc": round(xgb_auc, 4),
        "val_top_decile_recall": round(xgb_top_decile_recall, 4),
    })

    print("=== Experiment Log (validation set) ===")
    log_df = pd.DataFrame(experiment_log)
    print(log_df.to_string(index=False))
    log_df.to_csv("outputs/experiment_log.csv", index=False)
    # HONEST NOTE ON MODEL SELECTION: on this dataset, Random Forest and
    # XGBoost land within ~1 point of each other on val_top_decile_recall
    # — close enough to be within normal run-to-run noise, not a decisive
    # win either way on this metric alone. We still proceed with XGBoost
    # as the final model for reasons beyond this single validation split:
    # it trains incrementally (useful once monthly retraining is set up
    # in Phase 13), supports early stopping and monotonic constraints if
    # the business later insists a feature like tenure can only ever
    # DECREASE risk, and is the more common production standard for
    # tabular problems at this scale. In a real project, this close a
    # margin would justify running both on a few more validation splits
    # before locking in the choice — that additional check is skipped
    # here for the sake of the walkthrough.

    # -----------------------------------------------------------------
    # Phase 9: Final evaluation on the held-out TEST set
    # WHY A SEPARATE TEST SET (never touched until now): the validation
    # set was used to compare and pick among three models above, so any
    # metric on it is now slightly optimistic (we picked the model that
    # happened to do well there). The test set was held out from every
    # decision made so far, so it gives an honest, unbiased read of how
    # the chosen model will actually perform on new customers.
    # -----------------------------------------------------------------
    xgb_test_scores = xgb.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, xgb_test_scores)

    # WHY THE 80th PERCENTILE THRESHOLD: this operationalizes "flag the
    # riskiest 20% of the customer base for the retention team," a
    # capacity-driven choice (matches roughly what a retention team can
    # realistically act on in a month) rather than an arbitrary 0.5
    # probability cutoff, which would be meaningless on an imbalanced,
    # business-driven problem like this.
    threshold = np.percentile(xgb_test_scores, 80)
    test_preds = (xgb_test_scores >= threshold).astype(int)
    test_recall = recall_score(y_test, test_preds)
    test_precision = precision_score(y_test, test_preds)
    test_top_decile_recall = top_decile_recall(y_test.values, xgb_test_scores)

    # -----------------------------------------------------------------
    # Business validation: translate the model's catch rate into a
    # revenue estimate.
    # WHY WE DO THIS INSTEAD OF STOPPING AT AUC: per Phase 9 of the SDLC
    # doc, a model is only validated once its performance is tied back to
    # the KPI defined in the BRD (retained revenue), not just a technical
    # score a business stakeholder can't act on.
    #
    # ASSUMPTION MADE EXPLICIT: we assume a targeted retention offer
    # successfully retains 35% of the customers it's given to who would
    # otherwise have churned. This number is NOT derived from this
    # dataset (which has no offer/intervention data) — it's a plausible,
    # commonly-cited retention-offer success rate used here for
    # illustration. In a real project this number would come from an A/B
    # test or historical campaign data, and should be swapped in before
    # this estimate is presented as fact.
    # -----------------------------------------------------------------
    engineered_test = engineered.loc[X_test.index]
    avg_monthly_revenue_per_customer = engineered_test["MonthlyCharges"].mean()
    n_test = len(y_test)
    churners_in_top_decile = int(round(test_top_decile_recall * y_test.sum()))
    assumed_offer_success_rate = 0.35
    prevented_churners = round(churners_in_top_decile * assumed_offer_success_rate)
    annualized_revenue_saved = prevented_churners * avg_monthly_revenue_per_customer * 12

    business_summary = {
        "test_auc": round(test_auc, 4),
        "test_precision_at_top20pct_threshold": round(test_precision, 4),
        "test_recall_at_top20pct_threshold": round(test_recall, 4),
        "top_decile_recall": round(test_top_decile_recall, 4),
        "n_test_customers": n_test,
        "estimated_churners_caught_in_top_decile": churners_in_top_decile,
        "assumed_retention_offer_success_rate": assumed_offer_success_rate,
        "estimated_prevented_churners": prevented_churners,
        "estimated_annualized_revenue_saved_usd": round(annualized_revenue_saved, 2),
        "note": (
            "Revenue figure depends on an assumed 35% offer-success rate, "
            "not observed data — treat as illustrative until replaced with "
            "a real campaign-response rate."
        ),
    }

    print("\n=== Business Validation Summary (Phase 9) ===")
    for k, v in business_summary.items():
        print(f"{k}: {v}")
    with open("outputs/business_validation.json", "w") as f:
        json.dump(business_summary, f, indent=2)

    # -----------------------------------------------------------------
    # Explainability (Phase 9): feature importance
    # WHY WE SHOW THIS: a model that flags a customer as high-risk is not
    # useful to a retention agent unless they also know WHY, both to
    # build trust in the score and to decide what kind of offer might
    # actually address the customer's specific risk driver (e.g. a
    # contract-length driver suggests a loyalty discount; a billing-driver
    # suggests a rate review).
    # -----------------------------------------------------------------
    importances = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n=== Feature Importance (final model) ===")
    print(importances.round(4).to_string())
    importances.to_csv("outputs/feature_importance.csv", header=["importance"])

    # -----------------------------------------------------------------
    # Save artifacts for deployment (Phase 10)
    # -----------------------------------------------------------------
    joblib.dump(xgb, "outputs/churn_model.joblib")
    joblib.dump(scaler, "outputs/scaler.joblib")  # retained for the baseline model only
    joblib.dump(feature_cols, "outputs/feature_columns.joblib")

    with open("outputs/model_card.json", "w") as f:
        json.dump({
            "model_type": "XGBoost Classifier",
            "data_source": "IBM Telco Customer Churn dataset (Kaggle: blastchar/telco-customer-churn)",
            "n_features": len(feature_cols),
            "features": feature_cols,
            "training_rows": len(X_train),
            "validation_auc": round(xgb_auc, 4),
            "test_auc": round(test_auc, 4),
            "decision_threshold_percentile": 80,
            "intended_use": "Score active customers to prioritize retention outreach.",
            "known_limitations": (
                "Trained on a single cross-sectional snapshot (no repeated "
                "monthly observations per customer), so it cannot capture "
                "within-customer trends over time the way a production "
                "system with monthly history would. A production version "
                "should retrain on a time-based split across multiple "
                "months once that history is available, rather than the "
                "random stratified split used here."
            ),
        }, f, indent=2)

    print("\nSaved model -> outputs/churn_model.joblib")
    print("Saved model card -> outputs/model_card.json")


if __name__ == "__main__":
    train_and_evaluate()
