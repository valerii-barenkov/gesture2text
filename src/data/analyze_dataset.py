# src/data/analyze_dataset.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def project_root() -> Path:
    # src/data/analyze_dataset.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


def ensure_src_on_path(root: Path) -> None:
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def dataset_paths(root: Path, source: str) -> dict[str, Path]:
    if source == "live":
        return {
            "valeriy": root / "data" / "raw" / "user_valeriy" / "samples.csv",
            "mom": root / "data" / "raw" / "user_mom" / "samples.csv",
        }

    # frozen_best (default)
    return {
        "valeriy": root / "src" / "data" / "raw" / "user_valeriy" / "samples_frozen_best.csv",
        "mom": root / "src" / "data" / "raw" / "user_mom" / "samples_frozen_best.csv",
    }


def read_csv_strict(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path, comment="#")
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    return df


def print_basic_stats(df: pd.DataFrame, title: str) -> None:
    print("\n==============================")
    print(f"DATASET: {title}")
    print("==============================")

    print("\n=== ОБЩАЯ ИНФОРМАЦИЯ ===")
    print(df.info())

    print("\n=== ПЕРВЫЕ 5 СТРОК ===")
    print(df.head())

    if "label" in df.columns:
        print("\n=== РАСПРЕДЕЛЕНИЕ ПО ЖЕСТАМ ===")
        print(df["label"].value_counts())
    else:
        print("\n[!] Колонка 'label' не найдена")

    if "handedness" in df.columns:
        print("\n=== РАСПРЕДЕЛЕНИЕ ПО РУКАМ ===")
        print(df["handedness"].value_counts())
    else:
        print("\n[!] Колонка 'handedness' не найдена")

    print("\n=== ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ (общее число) ===")
    print(int(df.isnull().sum().sum()))


def test_feature_extractor(df: pd.DataFrame) -> None:
    from ml.features import extract_features

    row = df.iloc[0]
    feat = extract_features(row)

    print("\n=== ПРОВЕРКА FEATURE EXTRACTOR ===")
    if feat is None:
        print("[WARN] extract_features вернул None на первой строке (попробуй другую строку)")
        return
    print("Размер вектора признаков:", feat.shape)
    print("Первые 10 значений:", feat[:10])


def main() -> None:
    root = project_root()
    ensure_src_on_path(root)

    parser = argparse.ArgumentParser(
        description="Dataset analyzer for Gesture2Text (valeriy / mom / combined)."
    )
    parser.add_argument(
        "--user",
        choices=["valeriy", "mom", "combined"],
        default="combined",
        help="Какой датасет анализировать",
    )
    parser.add_argument(
        "--source",
        choices=["frozen_best", "live"],
        default="frozen_best",
        help="Откуда читать данные: frozen_best (стабильные) или live (сбор в data/raw)",
    )
    args = parser.parse_args()

    paths = dataset_paths(root, args.source)

    if args.user in ("valeriy", "mom"):
        p = paths[args.user]
        df = read_csv_strict(p)
        print_basic_stats(df, f"{args.user} ({args.source})")
        test_feature_extractor(df)
        return

    parts: list[pd.DataFrame] = []
    for key, p in paths.items():
        df_part = read_csv_strict(p)
        if "user" not in df_part.columns:
            df_part["user"] = f"user_{key}"
        parts.append(df_part)

    df_all = pd.concat(parts, ignore_index=True)

    print_basic_stats(df_all, f"combined ({args.source})")

    if "user" in df_all.columns:
        print("\n=== РАСПРЕДЕЛЕНИЕ ПО ПОЛЬЗОВАТЕЛЯМ ===")
        print(df_all["user"].value_counts())

    test_feature_extractor(df_all)


if __name__ == "__main__":
    main()