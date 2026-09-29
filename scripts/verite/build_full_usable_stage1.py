import json
from pathlib import Path

import pandas as pd


METADATA_PATH = Path("data/VERITE/VERITE.csv")
DATA_ROOT = Path("data/VERITE")
OUTPUT_DIR = Path("data/VERITE/processed/full_usable")

OUTPUT_JSONL = OUTPUT_DIR / "verite_image_filtered.jsonl"
FILTERING_LOG_JSON = OUTPUT_DIR / "verite_image_filtering_log.json"
EXCLUSION_LOG_CSV = OUTPUT_DIR / "verite_exclusion_log_image_missing.csv"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(METADATA_PATH)

    # Rename unclear original index column.
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "sample_id"})
    else:
        df["sample_id"] = range(len(df))

    required_cols = ["sample_id", "caption", "image_path", "label"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    original_samples = len(df)
    label_counts_original = df["label"].value_counts(dropna=False).to_dict()

    kept_records = []
    excluded_records = []

    for _, row in df.iterrows():
        sample_id = row["sample_id"]
        caption = row["caption"]
        rel_image_path = row["image_path"]
        label = row["label"]

        abs_image_path = DATA_ROOT / str(rel_image_path)

        record = {
            "sample_id": int(sample_id),
            "caption": str(caption),
            "claim": str(caption),
            "image_path": str(rel_image_path),
            "abs_image_path": str(abs_image_path),
            "label": str(label),
        }

        if not abs_image_path.exists():
            excluded_records.append({
                "sample_id": int(sample_id),
                "caption": str(caption),
                "image_path": str(rel_image_path),
                "abs_image_path": str(abs_image_path),
                "label": str(label),
                "exclusion_reason": "missing_image",
            })
        else:
            kept_records.append(record)

    kept_df = pd.DataFrame(kept_records)
    excluded_df = pd.DataFrame(excluded_records)

    if len(kept_df) > 0:
        label_counts_after_image_filtering = kept_df["label"].value_counts(dropna=False).to_dict()
    else:
        label_counts_after_image_filtering = {}

    # Save image-filtered dataset as JSONL.
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for rec in kept_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Save exclusion log.
    excluded_df.to_csv(EXCLUSION_LOG_CSV, index=False)

    filtering_log = {
        "metadata_path": str(METADATA_PATH),
        "data_root": str(DATA_ROOT),
        "original_samples": original_samples,
        "label_counts_original": label_counts_original,
        "missing_image_samples": len(excluded_records),
        "label_counts_after_image_filtering": label_counts_after_image_filtering,
        "image_filtered_samples": len(kept_records),
        "excluded_sample_ids": [rec["sample_id"] for rec in excluded_records],
        "exclusion_reason": {
            "missing_image": len(excluded_records)
        },
        "image_filtered_dataset_path": str(OUTPUT_JSONL),
        "image_exclusion_log_path": str(EXCLUSION_LOG_CSV),
    }

    with FILTERING_LOG_JSON.open("w", encoding="utf-8") as f:
        json.dump(filtering_log, f, ensure_ascii=False, indent=2)

    print("Stage 1 image filtering complete")
    print(f"original_samples: {original_samples}")
    print(f"missing_image_samples: {len(excluded_records)}")
    print(f"image_filtered_samples: {len(kept_records)}")
    print(f"saved dataset: {OUTPUT_JSONL}")
    print(f"saved filtering log: {FILTERING_LOG_JSON}")
    print(f"saved exclusion log: {EXCLUSION_LOG_CSV}")


if __name__ == "__main__":
    main()
