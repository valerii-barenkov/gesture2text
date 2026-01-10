# src/ml/predict_one.py
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\..*")

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml.features import extract_features

LANDMARKS = 21
COORDS = ("x", "y", "z")


# Paths / IO
def project_root() -> Path:
    # src/ml/predict_one.py -> repo root
    return Path(__file__).resolve().parents[2]


def read_csv_clean(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    # supports "# ..." header line in frozen files
    df = pd.read_csv(path, comment="#")
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    return df


def load_bundle(model_path: Path) -> tuple[Any, list[str] | None, int | None]:
    """
    Returns:
      - model: estimator/pipeline
      - classes: list[str] or None
      - n_features: int or None
    """
    obj = joblib.load(model_path)

    if isinstance(obj, dict):
        if "model" not in obj:
            raise ValueError("Model bundle is dict but has no key 'model'.")
        model = obj["model"]
        classes = obj.get("classes", None)
        if classes is not None:
            classes = [str(c) for c in classes]
        n_features = obj.get("n_features", None)
        n_features = int(n_features) if n_features is not None else None
        return model, classes, n_features

    # fallback: model saved directly
    model = obj
    n_features = getattr(model, "n_features_in_", None)
    n_features = int(n_features) if n_features is not None else None
    return model, None, n_features


# Features
def required_feature_cols() -> list[str]:
    return [f"{c}{i}" for i in range(LANDMARKS) for c in COORDS]


def build_X(
    df: pd.DataFrame,
    use_last_n: int | None = None,
) -> tuple[np.ndarray, list[int], int]:
    """
    Returns:
      - X: (N_valid, 63)
      - kept_idx: original df indices used for X
      - skipped: number of rows dropped (extract_features -> None)
    """
    need = required_feature_cols()
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing feature columns: {missing[:6]} (total {len(missing)}). "
            f"First columns: {list(df.columns)[:10]}"
        )

    df_use = df.tail(use_last_n).copy() if use_last_n is not None else df

    X_rows: list[np.ndarray] = []
    kept_idx: list[int] = []
    skipped = 0

    for idx, row in df_use.iterrows():
        feat = extract_features(row)  # (63,) or None
        if feat is None:
            skipped += 1
            continue
        X_rows.append(feat)
        kept_idx.append(int(idx))

    if not X_rows:
        raise ValueError("No valid rows after feature extraction (all rows were skipped).")

    X = np.vstack(X_rows).astype(float)

    # safety: replace any NaN/inf if something slips through
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, kept_idx, skipped


def decode_pred(pred: Any, classes: list[str] | None) -> str:
    # Most likely: model trained on strings -> already correct
    if isinstance(pred, (str, np.str_)):
        return str(pred)

    # Sometimes: numeric predictions
    if isinstance(pred, (int, np.integer)):
        if classes is not None:
            i = int(pred)
            if 0 <= i < len(classes):
                return str(classes[i])
        return str(int(pred))

    return str(pred)


# Reporting
def print_top_confusions(y_true: list[str], y_pred: list[str], top_k: int = 15) -> None:
    conf = Counter()
    for t, p in zip(y_true, y_pred):
        if t != p:
            conf[(t, p)] += 1

    print("\n=== TOP CONFUSIONS (true -> pred) ===")
    if not conf:
        print("No confusions")
        return

    for (t, p), cnt in conf.most_common(top_k):
        print(f"{t:>8} -> {p:<8} : {cnt}")


# Main
def main() -> None:
    root = project_root()

    default_model = root / "src" / "data" / "models" / "gesture2text_combined_frozen_best.joblib"
    if not default_model.exists():
        default_model = root / "src" / "data" / "models" / "gesture2text_combined_frozen.joblib"

    default_csv = root / "src" / "data" / "raw" / "user_valeriy" / "samples_frozen_best.csv"
    if not default_csv.exists():
        default_csv = root / "src" / "data" / "raw" / "user_valeriy" / "samples_frozen.csv"

    parser = argparse.ArgumentParser(description="Predict/evaluate using a trained Gesture2Text model bundle.")
    parser.add_argument("--model", type=str, default=str(default_model), help="Path to .joblib model bundle")
    parser.add_argument("--csv", type=str, default=str(default_csv), help="Path to CSV file (supports # comments)")
    parser.add_argument("--row", type=int, default=0, help="Single row index to predict")
    parser.add_argument("--n", type=int, default=None, help="Evaluate on last N rows (batch mode)")
    args = parser.parse_args()

    model_path = Path(args.model)
    csv_path = Path(args.csv)

    print(f"PROJECT_ROOT: {root}")
    print(f"MODEL_PATH:   {model_path}")
    print(f"CSV_PATH:     {csv_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model, classes, n_features = load_bundle(model_path)
    df = read_csv_clean(csv_path)

    # quick sanity checks
    if n_features is not None and n_features != 63:
        print(f"[WARN] Model expects n_features={n_features}, but features.py returns 63.")

    # Batch mode
    if args.n is not None:
        from sklearn.metrics import classification_report

        n = int(args.n)
        df_tail = df.tail(n).copy()

        X, kept_idx, skipped = build_X(df_tail, use_last_n=None)

        y_true: list[str] | None = None
        if "label" in df_tail.columns:
            y_true = df_tail.loc[kept_idx, "label"].astype(str).tolist()

        y_pred_raw = model.predict(X)
        y_pred = [decode_pred(p, classes) for p in y_pred_raw]

        print("\n=== BATCH EVAL ===")
        print(f"Rows requested: {n}")
        print(f"Rows used:      {len(y_pred)}")
        print(f"Rows skipped:   {skipped}")

        if y_true is None or len(y_true) != len(y_pred) or len(y_pred) == 0:
            print("[WARN] Column 'label' not found or length mismatch. Showing first 20 predictions:")
            print(y_pred[:20])
            return

        acc = sum(int(t == p) for t, p in zip(y_true, y_pred)) / len(y_pred)
        print(f"Accuracy (used rows): {acc:.4f}")

        print_top_confusions(y_true, y_pred)

        print("\n=== CLASSIFICATION REPORT ===")
        present_labels = sorted(set(y_true))
        print(
            classification_report(
                y_true,
                y_pred,
                labels=present_labels,
                target_names=present_labels,
                digits=4,
                zero_division=0,
            )
        )
        return

    # Single row mode
    row_idx = int(args.row)
    if row_idx < 0 or row_idx >= len(df):
        raise ValueError(f"Row index out of range: {row_idx} (df size={len(df)})")

    row = df.iloc[row_idx]
    feat = extract_features(row)
    if feat is None:
        raise ValueError(f"Row {row_idx} produced invalid features (extract_features returned None).")

    X = np.asarray(feat, dtype=float).reshape(1, -1)
    if X.shape[1] != 63:
        raise ValueError(f"Feature vector has wrong size: {X.shape[1]} (expected 63).")

    pred_raw = model.predict(X)[0]
    pred_label = decode_pred(pred_raw, classes)

    print("\n=== PREDICTION ===")
    print(f"Predicted: {pred_label} (raw={pred_raw})")

    # show top-3 probabilities if available
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0].astype(float)
        top_idx = np.argsort(proba)[::-1][:3]

        print("\n=== TOP-3 PROBABILITIES ===")
        for i in top_idx:
            if hasattr(model, "classes_"):
                cls_name = decode_pred(model.classes_[i], classes)
            else:
                cls_name = decode_pred(i, classes)
            print(f"{cls_name:>8}: {proba[i]:.4f}")

    # minimal row info
    print("\n=== ROW INFO ===")
    for col in ("label", "handedness", "timestamp", "user"):
        if col in df.columns:
            print(f"{col:>10}: {row.get(col)}")


if __name__ == "__main__":
    main()