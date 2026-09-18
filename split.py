"""
Part of Phase 7: train/validation/test split
------------------------------------------------
WHY A STRATIFIED RANDOM SPLIT (not time-based) here: this dataset is a
single snapshot in time (one row per customer, no repeated monthly
observations), so there is no "past vs. future" axis to split on the way
a real production system would (which SHOULD use a time-based split to
avoid leaking future information backward — see the note in
train_model.py's model card). Given a snapshot, a stratified split is the
correct choice specifically because churn is imbalanced (~26.5% positive):
stratifying on the target ensures the train/val/test sets each preserve
that ~26.5% rate, so none of them is accidentally easier or harder than
the others just from random imbalance in the split itself.
"""

from sklearn.model_selection import train_test_split


def split_data(X, y, test_size=0.15, val_size=0.15, random_state=42):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative_size, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
