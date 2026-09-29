import argparse
import json
from pathlib import Path

import pandas as pd


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature_path",
        default="data/CrisisMMD/processed/full_usable/crisismmd_stage1_features.jsonl",
    )
    parser.add_argument(
        "--split_path",
        default="data/CrisisMMD/processed/full_usable/crisismmd_dev_test_split.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="data/CrisisMMD/processed/full_usable/splits",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.feature_path)
    split_df = pd.read_csv(args.split_path)

    split_map = {
        int(r.sample_id): r.split
        for r in split_df.itertuples(index=False)
    }

    by_split = {"dev": [], "test": []}

    for r in rows:
        sid = int(r["sample_id"])
        sp = split_map.get(sid)
        if sp in by_split:
            by_split[sp].append(r)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = {
        "feature_path": args.feature_path,
        "split_path": args.split_path,
        "out_dir": str(out_dir),
        "splits": {},
    }

    for sp, items in by_split.items():
        out_path = out_dir / f"crisismmd_stage1_features_{sp}.jsonl"
        save_jsonl(items, out_path)

        df = pd.DataFrame(items)
        log["splits"][sp] = {
            "feature_out": str(out_path),
            "rows": len(items),
            "label_counts": df["gold_label"].value_counts().to_dict() if len(df) else {},
            "event_counts": df["event_name"].value_counts().to_dict() if len(df) else {},
        }

    log_path = out_dir / "split_files_log.json"
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

    print("saved log:", log_path)
    print(json.dumps(log, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
