import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature_path",
        default="data/CrisisMMD/processed/full_usable/crisismmd_stage1_features.jsonl",
    )
    parser.add_argument(
        "--out_path",
        default="data/CrisisMMD/processed/full_usable/crisismmd_dev_test_split.csv",
    )
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--dev_fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_jsonl(args.feature_path)
    df = pd.DataFrame(rows)

    required = ["sample_id", "tweet_id", "image_id", "event_name", "gold_label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Stratification target:
    # event + label을 함께 고려해서 event별 label 분포가 dev/test에 최대한 유지되도록 함.
    y = (df["event_name"].astype(str) + "__" + df["gold_label"].astype(str)).values

    # Group:
    # 같은 tweet_id에 여러 image가 있을 수 있으므로 tweet_id 기준으로 묶음.
    groups = df["tweet_id"].astype(str).values

    splitter = StratifiedGroupKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.seed,
    )

    folds = list(splitter.split(df, y, groups=groups))
    if args.dev_fold < 0 or args.dev_fold >= len(folds):
        raise ValueError(f"dev_fold must be between 0 and {len(folds)-1}")

    test_idx, dev_idx = folds[args.dev_fold]

    split = pd.Series("test", index=df.index)
    split.iloc[dev_idx] = "dev"

    out = df[["sample_id", "tweet_id", "image_id", "event_name", "gold_label"]].copy()
    out["split"] = split.values

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    dev_groups = set(out.loc[out["split"] == "dev", "tweet_id"].astype(str))
    test_groups = set(out.loc[out["split"] == "test", "tweet_id"].astype(str))
    leaked = sorted(dev_groups & test_groups)

    log = {
        "feature_path": args.feature_path,
        "out_path": str(out_path),
        "split_method": "StratifiedGroupKFold",
        "group_key": "tweet_id",
        "stratify_key": "event_name + gold_label",
        "n_splits": args.n_splits,
        "dev_fold": args.dev_fold,
        "seed": args.seed,
        "total_samples": int(len(out)),
        "total_groups": int(out["tweet_id"].astype(str).nunique()),
        "split_counts": out["split"].value_counts().to_dict(),
        "group_counts_by_split": out.groupby("split")["tweet_id"].nunique().to_dict(),
        "label_counts_all": out["gold_label"].value_counts().to_dict(),
        "label_counts_by_split": {
            s: out[out["split"] == s]["gold_label"].value_counts().to_dict()
            for s in ["dev", "test"]
        },
        "event_counts_by_split": {
            s: out[out["split"] == s]["event_name"].value_counts().to_dict()
            for s in ["dev", "test"]
        },
        "event_label_counts_by_split": {
            s: (
                out[out["split"] == s]
                .groupby(["event_name", "gold_label"])
                .size()
                .astype(int)
                .to_dict()
            )
            for s in ["dev", "test"]
        },
        "group_leakage_count": len(leaked),
        "leaked_groups_preview": leaked[:10],
        "note": (
            "CrisisMMD dev/test split uses tweet_id as group_id to prevent leakage "
            "across multiple images from the same tweet. Stratification uses event_name "
            "and gold_label to preserve disaster-event and label distributions."
        ),
    }

    # JSON does not support tuple keys, so stringify event-label count keys.
    for sp in ["dev", "test"]:
        log["event_label_counts_by_split"][sp] = {
            f"{k[0]}::{k[1]}": int(v)
            for k, v in log["event_label_counts_by_split"][sp].items()
        }

    log_path = out_path.with_name(out_path.stem + "_log.json")
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

    print("saved:", out_path)
    print("saved log:", log_path)
    print(json.dumps(log, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
