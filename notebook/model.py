from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)


# ----------------------------
# Config
# ----------------------------
HORIZON_HOURS = 24
TEST_FRACTION = 0.20
RANDOM_STATE = 42

# Optional: include machine model (categorical) as one-hot features
INCLUDE_MODEL_OHE = True

# Threshold used for confusion matrix + classification report
# (AUC/PR-AUC are threshold-free; threshold impacts operational metrics)
THRESHOLD = 0.50


# ----------------------------
# Paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / f"features_h{HORIZON_HOURS}.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def time_ordered_split(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time order: first (1-test_fraction) for train, last test_fraction for test."""
    df = df.sort_values("datetime").reset_index(drop=True)
    split_idx = int(np.floor((1.0 - test_fraction) * len(df)))
    split_idx = min(max(split_idx, 1), len(df) - 1)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing features file: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    # Basic validation
    required = {"datetime", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Parse datetime
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "label"]).copy()

    # Optional: one-hot encode 'model' if present
    if INCLUDE_MODEL_OHE and "model" in df.columns:
        df["model"] = df["model"].astype(str)
        df = pd.get_dummies(df, columns=["model"], drop_first=True)

    # Define feature set (Spark-style: everything except these identifiers/targets)
    drop_cols = {"label", "datetime", "machineID"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    if not feature_cols:
        raise ValueError("No feature columns found after excluding label/datetime/machineID.")

    # Ensure numeric features
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=feature_cols + ["label"]).copy()

    # Split
    train, test = time_ordered_split(df, TEST_FRACTION)

    X_train = train[feature_cols].to_numpy(dtype=float)
    y_train = train["label"].astype(int).to_numpy()

    X_test = test[feature_cols].to_numpy(dtype=float)
    y_test = test["label"].astype(int).to_numpy()

    # Quick dataset stats
    print(f"Train rows: {len(train):,} | Test rows: {len(test):,}")
    print(f"Train datetime: {train['datetime'].min()} -> {train['datetime'].max()}")
    print(f"Test  datetime: {test['datetime'].min()} -> {test['datetime'].max()}")
    print(f"Positive rate (train): {y_train.mean():.4f} | (test): {y_test.mean():.4f}")

    # Model (fast GBDT)
    clf = HistGradientBoostingClassifier(
        max_depth=5,
        max_iter=80,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
    )

    print("\nTraining model (HistGradientBoostingClassifier)...")
    clf.fit(X_train, y_train)
    print("Training completed.")

    # Probabilities for evaluation
    y_proba = clf.predict_proba(X_test)[:, 1]

    # Metrics
    # AUC requires both classes in y_test
    if len(np.unique(y_test)) < 2:
        auc = float("nan")
        print("WARNING: Test set contains only one class; ROC-AUC is undefined.")
    else:
        auc = roc_auc_score(y_test, y_proba)

    pr_auc = average_precision_score(y_test, y_proba)

    print(f"\nROC-AUC (H={HORIZON_HOURS}h) = {auc:.4f}" if np.isfinite(auc) else f"\nROC-AUC (H={HORIZON_HOURS}h) = NaN")
    print(f"PR-AUC  (H={HORIZON_HOURS}h) = {pr_auc:.4f}")

    # Threshold-based reporting
    y_pred = (y_proba >= THRESHOLD).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    print("\nTest label distribution:")
    print(pd.Series(y_test).value_counts().sort_index().rename_axis("label").to_frame("count"))

    print(f"\nConfusion matrix (threshold={THRESHOLD:.2f}) [ [TN FP] [FN TP] ]:")
    print(cm)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    # ----------------------------
    # Plots
    # ----------------------------
    # 1) ROC curve
    if np.isfinite(auc):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve (H={HORIZON_HOURS}h) | AUC={auc:.4f}")
        roc_path = REPORTS_DIR / f"roc_curve_h{HORIZON_HOURS}.png"
        plt.savefig(roc_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nSaved ROC curve to: {roc_path}")

    # 2) Precision–Recall curve (best for imbalanced classification)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve (H={HORIZON_HOURS}h) | PR-AUC={pr_auc:.4f}")
    pr_path = REPORTS_DIR / f"pr_curve_h{HORIZON_HOURS}.png"
    plt.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved PR curve to: {pr_path}")

    # Optional: simple confusion matrix plot (no seaborn)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix (threshold={THRESHOLD:.2f})")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    cm_path = REPORTS_DIR / f"confusion_matrix_h{HORIZON_HOURS}_t{str(THRESHOLD).replace('.','_')}.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")


if __name__ == "__main__":
    main()