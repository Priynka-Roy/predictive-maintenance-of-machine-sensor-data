from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HORIZON_HOURS = 24
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / f"features_h{HORIZON_HOURS}.csv"

RANDOM_STATE = 42
TEST_FRACTION = 0.20


def main() -> None:
    df = pd.read_csv(DATA_FILE)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "label"]).copy()

    # Explicitly exclude categorical "model" unless you one-hot encode it
    drop_cols = {"label", "datetime", "machineID", "model"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Numeric conversion for safety
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=feature_cols + ["label"]).copy()

    # Time-respecting split (sorted, last 20% as test)
    df = df.sort_values("datetime").reset_index(drop=True)
    split_idx = int(np.floor((1.0 - TEST_FRACTION) * len(df)))
    split_idx = min(max(split_idx, 1), len(df) - 1)

    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    X_train = train[feature_cols].to_numpy(dtype=float)
    y_train = train["label"].astype(int).to_numpy()
    X_test = test[feature_cols].to_numpy(dtype=float)
    y_test = test["label"].astype(int).to_numpy()

    clf = GradientBoostingClassifier(
        n_estimators=80,
        max_depth=5,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else float("nan")
    print(f"AUC (H={HORIZON_HOURS}h) = {auc:.4f}" if np.isfinite(auc) else f"AUC (H={HORIZON_HOURS}h) = NaN")

    print("\nTest label distribution:")
    print(pd.Series(y_test).value_counts().sort_index().rename_axis("label").to_frame("count"))


if __name__ == "__main__":
    main()