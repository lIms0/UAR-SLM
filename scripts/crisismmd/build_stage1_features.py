import argparse
import json
from pathlib import Path

import pandas as pd


LABELS = ["informative", "not_informative"]


def load_annotations(ann_dir: Path):
    files = sorted([
        p for p in ann_dir.glob("*_final_data.tsv")
        if not p.name.startswith("._")
    ])

    rows = []
    for p in files:
        event_name = p.name.replace("_final_data.tsv", "")
        df = pd.read_csv(p, sep="\t")

        required = ["tweet_id", "image_id", "tweet_text", "image_path", "image_info"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{p} missing columns: {missing}")

        df["event_name"] = event_name
        df["source_file"] = str(p)
        rows.append(df)

    if not rows:
        raise RuntimeError(f"No annotation files found in {ann_dir}")

    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="data/CrisisMMD/CrisisMMD_v2.0",
        help="CrisisMMD_v2.0 root directory",
    )
    parser.add_argument(
        "--out_dir",
        default="data/CrisisMMD/processed/full_usable",
    )
    args = parser.parse_args()

    root = Path(args.root)
    ann_dir = root / "annotations"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_annotations(ann_dir)

    before = len(df)

    # Keep only binary image informativeness labels.
    df = df[df["image_info"].isin(LABELS)].copy()

    # Resolve image path.
    df["relative_image_path"] = df["image_path"].astype(str)
    df["abs_image_path"] = df["relative_image_path"].apply(lambda x: str(root / x))
    df["image_exists"] = df["abs_image_path"].apply(lambda x: Path(x).exists())

    missing_images = int((~df["image_exists"]).sum())
    df = df[df["image_exists"]].copy()

    # Stable sample id.
    df = df.sort_values(["event_name", "tweet_id", "image_id"]).reset_index(drop=True)
    df["sample_id"] = range(len(df))

    rows = []
    for _, r in df.iterrows():
        label = str(r["image_info"])

        item = {
            "dataset": "CrisisMMD",
            "sample_id": int(r["sample_id"]),
            "tweet_id": str(r["tweet_id"]),
            "image_id": str(r["image_id"]),
            "event_name": str(r["event_name"]),
            "text": str(r["tweet_text"]),
            "tweet_text": str(r["tweet_text"]),
            "image_path": str(r["abs_image_path"]),
            "relative_image_path": str(r["relative_image_path"]),
            "image_url": None if pd.isna(r.get("image_url")) else str(r.get("image_url")),
            "gold_label": label,
            "label": label,
            "labels": LABELS,
            "task_name": "image_informativeness",
            "text_info": None if pd.isna(r.get("text_info")) else str(r.get("text_info")),
            "text_info_conf": None if pd.isna(r.get("text_info_conf")) else float(r.get("text_info_conf")),
            "image_info": label,
            "image_info_conf": None if pd.isna(r.get("image_info_conf")) else float(r.get("image_info_conf")),
            "text_human": None if pd.isna(r.get("text_human")) else str(r.get("text_human")),
            "image_human": None if pd.isna(r.get("image_human")) else str(r.get("image_human")),
            "image_damage": None if pd.isna(r.get("image_damage")) else str(r.get("image_damage")),
        }
        rows.append(item)

    out_path = out_dir / "crisismmd_stage1_features.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    log = {
        "root": str(root),
        "annotation_dir": str(ann_dir),
        "out_path": str(out_path),
        "rows_before_label_filter": int(before),
        "rows_after_label_filter": int(len(df) + missing_images),
        "missing_images": missing_images,
        "usable_rows": len(rows),
        "event_counts": df["event_name"].value_counts().to_dict(),
        "label_counts": df["image_info"].value_counts().to_dict(),
        "note": "CrisisMMD stage1 features for binary image informativeness classification.",
    }

    log_path = out_dir / "crisismmd_stage1_features_log.json"
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

    print("saved:", out_path)
    print("saved log:", log_path)
    print(json.dumps(log, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
