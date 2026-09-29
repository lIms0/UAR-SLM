import json
from pathlib import Path
from collections import Counter


DATA_PATH = Path("./data/VERITE/hash_dataset/processed/verite_200_triplet_hash.final.jsonl")

VISUAL_PATH = Path(
    "./outputs/VERITE/hash_dataset/visual_cues/visual_cues_qwen25vl3b_imageonly.final.jsonl"
)

CLIP_PATH = Path(
    "./outputs/VERITE/hash_dataset/clip_scores/clip_scores_vit_base_patch32.jsonl"
)

OUT_PATH = Path(
    "./outputs/VERITE/hash_dataset/features/verite_stage1_features.jsonl"
)

LOG_PATH = Path(
    "./outputs/VERITE/hash_dataset/logs/verite_stage1_features_log.json"
)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_jsonl_by_id(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            data[row["id"]] = row
    return data


def main():
    visual = load_jsonl_by_id(VISUAL_PATH)
    clip = load_jsonl_by_id(CLIP_PATH)

    total = 0
    missing_visual = 0
    missing_clip = 0
    original_visual_parse_failures = 0
    fallback_applied = 0
    visual_cue_unusable = 0
    label_counts = Counter()

    with open(DATA_PATH, "r", encoding="utf-8") as fin, open(OUT_PATH, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            total += 1

            sid = row["id"]
            label_counts[row["label"]] += 1

            v = visual.get(sid)
            c = clip.get(sid)

            if v is None:
                missing_visual += 1
            if c is None:
                missing_clip += 1

            if v is not None and not v.get("parse_success", False):
                original_visual_parse_failures += 1

            if v is not None and v.get("fallback_applied", False):
                fallback_applied += 1

            if v is not None and not v.get("visual_cue_usable", False):
                visual_cue_unusable += 1

            out = {
                "id": sid,
                "sample_id": row.get("sample_id"),
                "dataset": row.get("dataset"),
                "pair_id": row.get("pair_id"),
                "hash_rank": row.get("hash_rank"),
                "row_id": row.get("row_id"),
                "claim": row.get("claim"),
                "label": row.get("label"),
                "image_path": row.get("image_path"),

                "visual_cue_model": v.get("visual_cue_model") if v else None,
                "visual_prompt_type": v.get("visual_prompt_type") if v else None,
                "parsed_visual_cues": v.get("parsed_visual_cues") if v else None,
                "raw_visual_output": v.get("raw_visual_output") if v else None,
                "visual_parse_success": v.get("parse_success", False) if v else False,
                "visual_cue_usable": v.get("visual_cue_usable", False) if v else False,
                "fallback_applied": v.get("fallback_applied", False) if v else False,
                "failure_recorded": v.get("failure_recorded", False) if v else False,
                "failure_type": v.get("failure_type") if v else None,
                "visual_latency_sec": v.get("visual_latency_sec", 0.0) if v else 0.0,

                "clip_model": c.get("clip_model") if c else None,
                "clip_similarity_raw": c.get("clip_similarity_raw") if c else None,
                "clip_similarity_norm": c.get("clip_similarity_norm") if c else None,
                "u_cross": c.get("u_cross") if c else None,
                "clip_latency_sec": c.get("clip_latency_sec", 0.0) if c else 0.0,
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    log = {
        "data_path": str(DATA_PATH),
        "visual_path": str(VISUAL_PATH),
        "clip_path": str(CLIP_PATH),
        "output_path": str(OUT_PATH),
        "total": total,
        "label_counts": dict(label_counts),
        "missing_visual": missing_visual,
        "missing_clip": missing_clip,
        "original_visual_parse_failures": original_visual_parse_failures,
        "fallback_applied": fallback_applied,
        "visual_cue_unusable": visual_cue_unusable
    }

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print("saved:", OUT_PATH)
    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
