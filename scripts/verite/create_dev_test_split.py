import argparse
import json
import re
from pathlib import Path
from collections import Counter

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


DEFAULT_FEATURE_PATH = "data/VERITE/processed/full_usable/verite_stage1_features.jsonl"
DEFAULT_OUT_PATH = "data/VERITE/processed/full_usable/verite_dev_test_split.csv"
DEFAULT_LOG_PATH = "data/VERITE/processed/full_usable/verite_dev_test_split_log.json"


def extract_group_id(image_path):
    image_path = str(image_path or "")
    name = Path(image_path).stem

    # true_0, false_0 -> 0
    m = re.search(r"_(\d+)$", name)
    if m:
        return m.group(1)

    # fallback
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path", default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--out_path", default=DEFAULT_OUT_PATH)
    parser.add_argument("--log_path", default=DEFAULT_LOG_PATH)
    parser.add_argument("--dev_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []
    with open(args.feature_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    df = pd.DataFrame(rows)

    required = ["sample_id", "label", "image_path"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["group_id"] = df["image_path"].apply(extract_group_id)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=args.dev_size,
        random_state=args.seed,
    )

    # GroupShuffleSplit의 test 부분을 dev로 사용
    train_idx, dev_idx = next(
        splitter.split(df, groups=df["group_id"])
    )

    df["split"] = "test"
    df.loc[dev_idx, "split"] = "dev"

    split_df = df[["sample_id", "label", "image_path", "group_id", "split"]].copy()
    split_df = split_df.sort_values("sample_id").reset_index(drop=True)

    # group leakage check
    group_split_counts = split_df.groupby("group_id")["split"].nunique()
    leaked_groups = group_split_counts[group_split_counts > 1].index.tolist()

    split_df.to_csv(args.out_path, index=False)

    log = {
        "feature_path": args.feature_path,
        "out_path": args.out_path,
        "dev_size": args.dev_size,
        "seed": args.seed,
        "total_samples": int(len(split_df)),
        "total_groups": int(split_df["group_id"].nunique()),
        "split_counts": split_df["split"].value_counts().to_dict(),
        "group_counts_by_split": {
            split: int(g["group_id"].nunique())
            for split, g in split_df.groupby("split")
        },
        "label_counts_all": split_df["label"].value_counts().to_dict(),
        "label_counts_by_split": {
            split: g["label"].value_counts().to_dict()
            for split, g in split_df.groupby("split")
        },
        "group_leakage_count": len(leaked_groups),
        "leaked_groups_preview": leaked_groups[:20],
        "note": (
            "Group-aware dev/test split. group_id is extracted from image_path numeric suffix, "
            "so images/true_0.jpg and images/false_0.jpg are assigned to the same split. "
            "Dev is used only for adaptive policy grid search. Test is used for final reporting."
        ),
    }

    with open(args.log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print("saved:", args.out_path)
    print("saved log:", args.log_path)
    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
