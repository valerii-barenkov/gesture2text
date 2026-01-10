# src/ml/train.py
import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.features import extract_features

# suppress noisy sklearn runtime warnings (they don't affect final accuracy here)
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\..*")

LANDMARKS = 21
COORDS = ("x", "y", "z")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_csv_clean(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path, comment="#")
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    return df


def required_columns() -> list[str]:
    # x0,y0,z0..x20,y20,z20
    return [f"{c}{i}" for i in range(LANDMARKS) for c in COORDS]


def make_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    missing = [c for c in required_columns() if c not in df.columns]
    if missing:
        preview = ", ".join(missing[:6])
        raise ValueError(f"Missing landmark columns: {preview} (total {len(missing)})")

    if "label" not in df.columns:
        raise ValueError("CSV must contain 'label' column for training.")

    X_rows: list[np.ndarray] = []
    y_labels: list[str] = []
    skipped = 0

    for _, row in df.iterrows():
        feat = extract_features(row)
        if feat is None:
            skipped += 1
            continue
        X_rows.append(feat)
        y_labels.append(str(row["label"]))

    if not X_rows:
        raise ValueError("No valid rows after feature extraction (all rows were skipped).")

    X = np.vstack(X_rows).astype(float)
    y = np.asarray(y_labels, dtype=str)

    if skipped:
        print(f"[WARN] Skipped rows: {skipped}")

    # extra safety (your features.py already guards this, but keep it)
    if not np.isfinite(X).all():
        print("[WARN] Non-finite values found in X. Replacing with 0.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y


def dataset_paths(root: Path) -> dict[str, Path]:
    return {
        "valeriy": root / "data" / "raw" / "user_valeriy" / "samples.csv",
        "mom": root / "data" / "raw" / "user_mom" / "samples.csv",
    }


def load_dataset(root: Path, dataset: str) -> pd.DataFrame:
    paths = dataset_paths(root)

    if dataset in ("valeriy", "mom"):
        return read_csv_clean(paths[dataset])

    if dataset == "combined":
        parts: list[pd.DataFrame] = []
        for key, p in paths.items():
            df = read_csv_clean(p)
            if "user" not in df.columns:
                df["user"] = f"user_{key}"
            parts.append(df)
        return pd.concat(parts, ignore_index=True)

    raise ValueError(f"Unknown dataset: {dataset}")


def train_pipeline(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(set(y)) > 1 else None,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)

    return pipeline, acc, report


def main() -> None:
    root = project_root()
    default_out = root / "src" / "data" / "models" / "gesture2text_combined_frozen.joblib"

    parser = argparse.ArgumentParser(description="Train Gesture2Text classifier on CSV samples.")
    parser.add_argument(
        "--dataset",
        choices=["valeriy", "mom", "combined"],
        default="combined",
        help="Dataset to use for training",
    )
    parser.add_argument(
        "--out",
        default=str(default_out),
        help="Output .joblib path",
    )
    args = parser.parse_args()

    print("=== TRAIN: PATHS ===")
    print(f"PROJECT_ROOT: {root}")
    print(f"DATASET: {args.dataset}")

    df = load_dataset(root, args.dataset)
    print(f"[INFO] Rows loaded: {len(df)}")

    X, y = make_xy(df)
    classes = sorted({str(c) for c in y})
    print(f"[INFO] X shape: {X.shape} | classes: {classes}")

    model, acc, report = train_pipeline(X, y)

    print("\n=== TRAIN RESULT (hold-out 20%) ===")
    print(f"Accuracy: {acc:.4f}")
    print("\n=== CLASSIFICATION REPORT ===")
    print(report)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": model,
        "classes": classes,
        "n_features": int(X.shape[1]),
    }
    joblib.dump(bundle, out_path)
    print(f"\n[OK] Saved model bundle to: {out_path}")


if __name__ == "__main__":
    main()