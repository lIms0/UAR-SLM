import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_SPLIT_PATH = "data/VERITE/processed/full_usable/verite_dev_test_split.csv"
DEFAULT_FEATURE_PATH = "data/VERITE/processed/full_usable/verite_stage1_features.jsonl"
DEFAULT_EVIDENCE_PATH = "data/VERITE/processed/full_usable/retrieved_evidence_brave_q4_k10.jsonl"
DEFAULT_OUT_DIR = "data/VERITE/processed/full_usable/splits"


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
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_path", default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--feature_path", default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--evidence_path", default=DEFAULT_EVIDENCE_PATH)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    split_df = pd.read_csv(args.split_path)
    split_map = dict(zip(split_df["sample_id"], split_df["split"]))

    feature_rows = load_jsonl(args.feature_path)
    evidence_rows = load_jsonl(args.evidence_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = {
        "split_path": args.split_path,
        "feature_path": args.feature_path,
        "evidence_path": args.evidence_path,
        "out_dir": str(out_dir),
        "splits": {},
    }

    for split in ["dev", "test"]:
        split_ids = set(split_df.loc[split_df["split"] == split, "sample_id"].tolist())

        split_features = [
            row for row in feature_rows
            if row.get("sample_id") in split_ids
        ]

        split_evidence = [
            row for row in evidence_rows
            if row.get("sample_id") in split_ids
        ]

        feature_out = out_dir / f"verite_stage1_features_{split}.jsonl"
        evidence_out = out_dir / f"retrieved_evidence_brave_q4_k10_{split}.jsonl"

        save_jsonl(split_features, feature_out)
        save_jsonl(split_evidence, evidence_out)

        log["splits"][split] = {
            "feature_out": str(feature_out),
            "evidence_out": str(evidence_out),
            "feature_rows": len(split_features),
            "evidence_rows": len(split_evidence),
            "label_counts": pd.DataFrame(split_features)["label"].value_counts().to_dict(),
        }

    log_path = out_dir / "split_files_log.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print("saved log:", log_path)
    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
