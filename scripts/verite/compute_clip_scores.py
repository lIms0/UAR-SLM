import json
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


DATA_PATH = Path("./data/VERITE/processed/full_usable/verite_visual_cue_filtered.jsonl")
DATA_ROOT = Path("./data/VERITE")

OUT_DIR = Path("./data/VERITE/processed/full_usable")
OUT_PATH = OUT_DIR / "verite_with_clip_score.jsonl"
LOG_PATH = OUT_DIR / "clip_score_log.json"

MODEL_NAME = "openai/clip-vit-base-patch32"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_image_path(row):
    if row.get("abs_image_path"):
        return str(row["abs_image_path"])

    rel_image_path = row["image_path"]
    return str(DATA_ROOT / rel_image_path)


def compute_similarity(model, processor, image_path, text, device):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        text=[text],
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    start = time.time()

    with torch.no_grad():
        outputs = model(**inputs)

        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        similarity = (image_embeds @ text_embeds.T).item()

    latency = time.time() - start

    return similarity, latency


def main():
    rows = load_rows(DATA_PATH)
    print("total final usable samples:", len(rows))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    print("loading model:", MODEL_NAME)

    processor = CLIPProcessor.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    model = CLIPModel.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    model.to(device)
    model.eval()

    errors = 0
    total_latency = 0.0
    scored_rows = 0

    label_counts = {}
    try:
        import pandas as pd
        label_counts = pd.DataFrame(rows)["label"].value_counts(dropna=False).to_dict()
    except Exception:
        label_counts = {}

    with open(OUT_PATH, "w", encoding="utf-8") as fout:
        for row in tqdm(rows, desc="Computing CLIP scores"):
            out = dict(row)

            try:
                claim = row.get("claim") or row.get("caption") or ""
                image_abs_path = resolve_image_path(row)

                sim, latency = compute_similarity(
                    model=model,
                    processor=processor,
                    image_path=image_abs_path,
                    text=claim,
                    device=device,
                )

                # CLIP cosine similarity can be negative.
                # This normalized value is only used as a simple cross-modal uncertainty signal.
                sim_norm = max(0.0, min(1.0, (sim + 1.0) / 2.0))
                u_cross = 1.0 - sim_norm

                out.update({
                    "claim": claim,
                    "abs_image_path": image_abs_path,
                    "clip_model": MODEL_NAME,
                    "clip_similarity_raw": round(sim, 6),
                    "clip_similarity_norm": round(sim_norm, 6),
                    "u_cross": round(u_cross, 6),
                    "clip_latency_sec": round(latency, 6),
                    "clip_success": True,
                })

                total_latency += latency
                scored_rows += 1

            except Exception as e:
                errors += 1
                out.update({
                    "clip_model": MODEL_NAME,
                    "clip_success": False,
                    "clip_error": repr(e),
                })

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

    log = {
        "stage": "stage3_clip_score_computation",
        "data_path": str(DATA_PATH),
        "output_path": str(OUT_PATH),
        "model_name": MODEL_NAME,
        "total_samples": len(rows),
        "label_counts": label_counts,
        "scored_rows": scored_rows,
        "errors": errors,
        "total_clip_latency_sec": round(total_latency, 4),
        "average_clip_latency_sec": round(total_latency / scored_rows, 6) if scored_rows else 0.0,
        "note": (
            "CLIP image-claim similarity was computed on the final usable VERITE dataset. "
            "The input dataset already excludes samples with missing images and failed visual cue extraction."
        ),
    }

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print("saved:", OUT_PATH)
    print("saved log:", LOG_PATH)
    print(json.dumps(log, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()