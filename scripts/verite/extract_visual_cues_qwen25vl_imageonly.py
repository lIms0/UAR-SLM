import json
import time
import re
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


DATA_PATH = Path("./data/VERITE/processed/full_usable/verite_image_filtered.jsonl")
DATA_ROOT = Path("./data/VERITE")

OUT_DIR = Path("./data/VERITE/processed/full_usable")
OUT_PATH = OUT_DIR / "visual_cues_qwen25vl3b_imageonly_sample_level.jsonl"
IMAGE_CACHE_PATH = OUT_DIR / "visual_cues_qwen25vl3b_imageonly_image_cache.jsonl"
VISUAL_FILTERED_PATH = OUT_DIR / "verite_visual_cue_filtered.jsonl"
FAILURE_LOG_PATH = OUT_DIR / "verite_exclusion_log_visual_cue_failure.csv"
LOG_PATH = OUT_DIR / "visual_cues_qwen25vl3b_imageonly_log.json"

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_prompt():
    return """
You are extracting visual cues from an image for a fact-checking experiment.

Describe only what is directly visible in the image.
Do not use external knowledge.
Do not infer the event name, exact location, date, political group, or person identity unless it is clearly visible as text or a recognizable visual sign in the image.
If the location, date, event, or identity is not clearly visible, write "unknown".
Do not use the file name or metadata.

Return only valid JSON.
Do not wrap the answer in markdown code fences.

Use the following keys:
{
  "generated_caption": "short factual description of the image",
  "visible_objects": ["object1", "object2"],
  "scene": "main visible scene or setting",
  "people_activity": "what people appear to be doing, or unknown",
  "visible_text": "any readable text in the image, or empty string",
  "location_cues": "visible location cues, or unknown",
  "time_cues": "visible time/date cues, or unknown",
  "uncertain_visual_points": ["things that cannot be confirmed from the image alone"]
}
""".strip()


def clean_json_text(text):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    return text.strip()


def parse_visual_json(raw_text):
    cleaned = clean_json_text(raw_text)

    try:
        parsed = json.loads(cleaned)
        return parsed, True
    except Exception:
        return {
            "generated_caption": "",
            "visible_objects": [],
            "scene": "",
            "people_activity": "",
            "visible_text": "",
            "location_cues": "",
            "time_cues": "",
            "uncertain_visual_points": [],
            "parse_error": True,
            "raw_text": raw_text,
        }, False


def is_usable_visual_cue(parsed_cues):
    if not isinstance(parsed_cues, dict):
        return False

    generated_caption = str(parsed_cues.get("generated_caption", "")).strip()
    scene = str(parsed_cues.get("scene", "")).strip()
    visible_objects = parsed_cues.get("visible_objects", [])

    has_caption = len(generated_caption) > 0
    has_scene = len(scene) > 0
    has_objects = isinstance(visible_objects, list) and len(visible_objects) > 0

    return has_caption or has_scene or has_objects


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_existing_image_cache(path):
    cache = {}
    if not path.exists():
        return cache

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            image_key = row.get("image_path")
            if image_key:
                cache[image_key] = row

    return cache


def resolve_image_path(row):
    if row.get("abs_image_path"):
        return str(row["abs_image_path"])

    rel_image_path = row["image_path"]
    return str(DATA_ROOT / rel_image_path)


def run_image(model, processor, image_key, image_abs_path):
    Image.open(image_abs_path).convert("RGB")

    prompt = build_prompt()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_abs_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    start = time.time()

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    latency = time.time() - start

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    raw_output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    parsed_output, parse_success = parse_visual_json(raw_output)
    usable_visual_cue = parse_success and is_usable_visual_cue(parsed_output)

    return {
        "image_path": image_key,
        "image_abs_path": image_abs_path,
        "visual_cue_model": MODEL_NAME,
        "visual_prompt_type": "image_only_no_claim",
        "raw_visual_output": raw_output,
        "parsed_visual_cues": parsed_output,
        "parse_success": parse_success,
        "usable_visual_cue": usable_visual_cue,
        "visual_latency_sec": round(latency, 4),
    }


def main():
    rows = load_rows(DATA_PATH)
    print("total image-filtered samples:", len(rows))

    image_key_to_abs = {}
    for row in rows:
        image_key = row["image_path"]
        image_abs_path = resolve_image_path(row)
        image_key_to_abs[image_key] = image_abs_path

    unique_image_paths = sorted(image_key_to_abs.keys())
    print("unique images:", len(unique_image_paths))

    image_cache = load_existing_image_cache(IMAGE_CACHE_PATH)
    print("already cached images:", len(image_cache))

    print("loading model:", MODEL_NAME)

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )

    model.eval()

    model_calls = 0
    errors = 0
    parse_failures = 0
    unusable_visual_cues = 0
    total_visual_latency = 0.0

    with open(IMAGE_CACHE_PATH, "a", encoding="utf-8") as cache_out:
        for image_key in tqdm(unique_image_paths, desc="Extracting image-level visual cues"):
            if image_key in image_cache:
                continue

            image_abs_path = image_key_to_abs[image_key]

            try:
                result = run_image(model, processor, image_key, image_abs_path)
                image_cache[image_key] = result

                cache_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                cache_out.flush()

                model_calls += 1
                total_visual_latency += result.get("visual_latency_sec", 0.0)

                if not result.get("parse_success", False):
                    parse_failures += 1

                if not result.get("usable_visual_cue", False):
                    unusable_visual_cues += 1

            except Exception as e:
                err = {
                    "image_path": image_key,
                    "image_abs_path": image_abs_path,
                    "visual_cue_model": MODEL_NAME,
                    "visual_prompt_type": "image_only_no_claim",
                    "error": repr(e),
                    "raw_visual_output": None,
                    "parsed_visual_cues": None,
                    "parse_success": False,
                    "usable_visual_cue": False,
                    "visual_latency_sec": 0.0,
                }

                image_cache[image_key] = err
                cache_out.write(json.dumps(err, ensure_ascii=False) + "\n")
                cache_out.flush()

                model_calls += 1
                errors += 1

    sample_level_rows = []
    visual_filtered_rows = []
    failure_rows = []

    for row in rows:
        image_key = row["image_path"]
        cue = image_cache.get(image_key, {})

        parse_success = bool(cue.get("parse_success", False))
        usable_visual_cue = bool(cue.get("usable_visual_cue", False))

        out = {
            "sample_id": row.get("sample_id"),
            "caption": row.get("caption"),
            "claim": row.get("claim", row.get("caption")),
            "image_path": row.get("image_path"),
            "abs_image_path": row.get("abs_image_path", resolve_image_path(row)),
            "label": row.get("label"),
            "visual_cue_model": cue.get("visual_cue_model", MODEL_NAME),
            "visual_prompt_type": cue.get("visual_prompt_type", "image_only_no_claim"),
            "raw_visual_output": cue.get("raw_visual_output"),
            "parsed_visual_cues": cue.get("parsed_visual_cues"),
            "parse_success": parse_success,
            "usable_visual_cue": usable_visual_cue,
            "visual_latency_sec": cue.get("visual_latency_sec", 0.0),
            "visual_model_call_unit": "unique_image",
        }

        if "error" in cue:
            out["error"] = cue["error"]

        sample_level_rows.append(out)

        if parse_success and usable_visual_cue:
            visual_filtered_rows.append(out)
        else:
            reason = "visual_cue_failure"
            if "error" in cue:
                reason = "visual_cue_runtime_error"
            elif not parse_success:
                reason = "visual_cue_parse_failure"
            elif not usable_visual_cue:
                reason = "visual_cue_unusable"

            failure_rows.append({
                "sample_id": row.get("sample_id"),
                "caption": row.get("caption"),
                "image_path": row.get("image_path"),
                "abs_image_path": row.get("abs_image_path", resolve_image_path(row)),
                "label": row.get("label"),
                "exclusion_reason": reason,
                "error": cue.get("error"),
                "raw_visual_output": cue.get("raw_visual_output"),
            })

    with open(OUT_PATH, "w", encoding="utf-8") as fout:
        for row in sample_level_rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(VISUAL_FILTERED_PATH, "w", encoding="utf-8") as fout:
        for row in visual_filtered_rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    import pandas as pd
    pd.DataFrame(failure_rows).to_csv(FAILURE_LOG_PATH, index=False)

    original_label_counts = {}
    image_filtered_label_counts = {}
    final_label_counts = {}

    try:
        import pandas as pd
        original_df = pd.read_csv("data/VERITE/VERITE.csv")
        original_label_counts = original_df["label"].value_counts(dropna=False).to_dict()
    except Exception:
        original_label_counts = {}

    try:
        image_filtered_label_counts = pd.DataFrame(rows)["label"].value_counts(dropna=False).to_dict()
    except Exception:
        image_filtered_label_counts = {}

    try:
        final_label_counts = pd.DataFrame(visual_filtered_rows)["label"].value_counts(dropna=False).to_dict()
    except Exception:
        final_label_counts = {}

    log = {
        "stage": "stage2_visual_cue_extraction",
        "data_path": str(DATA_PATH),
        "output_path_sample_level": str(OUT_PATH),
        "image_cache_path": str(IMAGE_CACHE_PATH),
        "visual_filtered_dataset_path": str(VISUAL_FILTERED_PATH),
        "visual_failure_log_path": str(FAILURE_LOG_PATH),
        "model_name": MODEL_NAME,
        "visual_prompt_type": "image_only_no_claim",
        "original_samples": 1014,
        "label_counts_original": original_label_counts,
        "image_filtered_samples": len(rows),
        "label_counts_after_image_filtering": image_filtered_label_counts,
        "visual_cue_failure_samples": len(failure_rows),
        "label_counts_after_visual_cue_filtering": final_label_counts,
        "final_usable_samples_after_visual_cue_filtering": len(visual_filtered_rows),
        "excluded_sample_ids_visual_cue": [r["sample_id"] for r in failure_rows],
        "exclusion_reason_visual_cue": pd.DataFrame(failure_rows)["exclusion_reason"].value_counts(dropna=False).to_dict() if failure_rows else {},
        "total_unique_images": len(unique_image_paths),
        "already_cached_images_at_start": len(load_existing_image_cache(IMAGE_CACHE_PATH)) - model_calls,
        "new_model_calls": model_calls,
        "errors_for_new_calls": errors,
        "parse_failures_for_new_calls": parse_failures,
        "unusable_visual_cues_for_new_calls": unusable_visual_cues,
        "total_visual_latency_sec_for_new_calls": round(total_visual_latency, 4),
        "average_latency_sec_for_new_calls": round(total_visual_latency / model_calls, 4) if model_calls else 0.0,
        "note": "Visual cues were extracted once per unique image using image-only prompts and then mapped back to sample-level rows. Claim text was not used in the visual cue prompt.",
    }

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print("saved sample-level visual cues:", OUT_PATH)
    print("saved image-level cache:", IMAGE_CACHE_PATH)
    print("saved visual-cue-filtered dataset:", VISUAL_FILTERED_PATH)
    print("saved visual cue failure log:", FAILURE_LOG_PATH)
    print("saved log:", LOG_PATH)
    print(json.dumps(log, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()