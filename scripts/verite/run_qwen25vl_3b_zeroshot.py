import argparse
import json
import re
import time
from pathlib import Path
from collections import Counter

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


LABELS = ["true", "miscaptioned", "out-of-context"]

DEFAULT_DATA_PATH = "./data/VERITE/processed/full_usable/verite_stage1_features.jsonl"
DEFAULT_OUT_DIR = "./outputs/VERITE/full_usable/main_comparison/vlm_zeroshot"
DEFAULT_LOG_DIR = "./outputs/VERITE/full_usable/main_comparison/logs"

MODEL_MAP = {
    "qwen25vl_3b": "Qwen/Qwen2.5-VL-3B-Instruct",
}


def load_rows(path, limit=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def get_model_name(model_tag):
    if model_tag in MODEL_MAP:
        return MODEL_MAP[model_tag]
    return model_tag


def normalize_label(label):
    if label is None:
        return None

    x = str(label).strip().lower()
    x = x.replace("_", "-")

    aliases = {
        "truthful": "true",
        "real": "true",
        "match": "true",
        "matched": "true",

        "mis-captioned": "miscaptioned",
        "mis captioned": "miscaptioned",
        "wrong-caption": "miscaptioned",
        "wrong caption": "miscaptioned",

        "out of context": "out-of-context",
        "out-of-context": "out-of-context",
        "out-of context": "out-of-context",
        "ooc": "out-of-context",
        "different-context": "out-of-context",
        "different context": "out-of-context",
    }

    if x in aliases:
        return aliases[x]

    if x in LABELS:
        return x

    return None


def clean_json_text(text):
    text = str(text or "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    return text.strip()


def parse_prediction(raw_text):
    cleaned = clean_json_text(raw_text)

    try:
        obj = json.loads(cleaned)
        pred_label = normalize_label(obj.get("label"))
        confidence = obj.get("confidence", None)

        try:
            confidence = float(confidence)
        except Exception:
            confidence = None

        if confidence is not None:
            confidence = max(0.0, min(1.0, confidence))

        return {
            "pred_label": pred_label,
            "confidence": confidence,
            "reason": obj.get("reason", ""),
            "parse_success": pred_label in LABELS,
            "parse_mode": "json",
        }
    except Exception:
        pass

    text = str(raw_text or "").lower()
    found = []

    if "out-of-context" in text or "out of context" in text or "ooc" in text:
        found.append("out-of-context")
    if "miscaptioned" in text or "mis-captioned" in text or "mis captioned" in text or "wrong caption" in text:
        found.append("miscaptioned")

    first_line = text.splitlines()[0].strip() if text.splitlines() else text.strip()
    if first_line.startswith("true"):
        found.append("true")

    found = list(dict.fromkeys(found))

    if len(found) == 1:
        return {
            "pred_label": found[0],
            "confidence": None,
            "reason": "",
            "parse_success": True,
            "parse_mode": "fallback_text",
        }

    return {
        "pred_label": None,
        "confidence": None,
        "reason": "",
        "parse_success": False,
        "parse_mode": "failed",
    }


def build_prompt(claim):
    return f"""
You are a strict image-caption verification classifier.

Given the image and the claim/caption, choose exactly one label from:
true
miscaptioned
out-of-context

Definitions:
true = the image and the claim/caption are factually consistent.
miscaptioned = the image is related to the claim/caption, but an important detail is wrong, unsupported, or misleading.
out-of-context = the image is reused in a different context from the claim/caption.

Use the image and the claim/caption only.
Do not use web search or external retrieved evidence.

Return only valid JSON with exactly these keys:
{{
  "label": "true | miscaptioned | out-of-context",
  "confidence": 0.0,
  "reason": "one short sentence"
}}

Confidence must be a number between 0 and 1.
Do not include markdown.
Do not include any text outside JSON.

Claim/Caption:
{claim}
""".strip()


def resolve_image_path(row):
    if row.get("abs_image_path"):
        return str(row["abs_image_path"])
    return str(Path("./data/VERITE") / str(row.get("image_path")))


def make_output_paths(model_tag, run_name, out_dir, log_dir):
    out_dir = Path(out_dir)
    log_dir = Path(log_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        run_name = f"{model_tag}_zeroshot"

    out_path = out_dir / f"{run_name}.jsonl"
    log_path = log_dir / f"{run_name}_log.json"

    return out_path, log_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--model_tag", default="qwen25vl_3b")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--log_dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    model_name = get_model_name(args.model_tag)
    out_path, log_path = make_output_paths(args.model_tag, args.run_name, args.out_dir, args.log_dir)

    rows = load_rows(args.data_path, limit=args.limit)

    print("data_path:", args.data_path)
    print("model_tag:", args.model_tag)
    print("model_name:", model_name)
    print("samples:", len(rows))
    print("out_path:", out_path)

    processor = AutoProcessor.from_pretrained(
        model_name,
        local_files_only=args.local_files_only,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=args.local_files_only,
    )

    model.eval()

    errors = 0
    parse_failures = 0
    total_latency = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    gold_counts = Counter()
    pred_counts = Counter()

    with open(out_path, "w", encoding="utf-8") as fout:
        for row in tqdm(rows, desc=f"VLM zero-shot {args.model_tag}"):
            sid = row.get("sample_id", row.get("id"))
            claim = row.get("claim", row.get("caption", ""))
            image_path = resolve_image_path(row)

            try:
                Image.open(image_path).convert("RGB")

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": build_prompt(claim)},
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
                input_tokens = int(inputs["input_ids"].shape[-1])

                start = time.time()

                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )

                latency = time.time() - start

                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]

                output_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                output_tokens = int(generated_ids_trimmed[0].shape[-1])

                parsed = parse_prediction(output_text)
                pred_label = parsed.get("pred_label")
                confidence = parsed.get("confidence")
                reason = parsed.get("reason", "")
                parse_success = parsed.get("parse_success", False)

                if not parse_success:
                    parse_failures += 1

                gold_label = row.get("label")
                gold_counts[gold_label] += 1
                pred_counts[pred_label] += 1

                out = {
                    "id": sid,
                    "sample_id": row.get("sample_id"),
                    "pair_id": row.get("pair_id"),
                    "method": "qwen25vl_3b_zeroshot",
                    "model_tag": args.model_tag,
                    "model_path": model_name,

                    "gold_label": gold_label,
                    "pred_label": pred_label,
                    "confidence": confidence,
                    "parse_success": parse_success,
                    "parse_mode": parsed.get("parse_mode"),
                    "reason": reason,
                    "raw_output": output_text,

                    "image_path": row.get("image_path"),
                    "abs_image_path": image_path,

                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "inference_latency_sec": round(latency, 4),

                    "retrieval_calls_used": 0,
                    "used_evidence_count": 0,
                    "graph_nodes": 0,
                    "graph_edges": 0,
                    "graph_expansion_steps": 0,
                }

                total_latency += latency
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens

            except Exception as e:
                errors += 1
                out = {
                    "id": sid,
                    "sample_id": row.get("sample_id"),
                    "pair_id": row.get("pair_id"),
                    "method": "qwen25vl_3b_zeroshot",
                    "model_tag": args.model_tag,
                    "model_path": model_name,
                    "gold_label": row.get("label"),
                    "image_path": row.get("image_path"),
                    "abs_image_path": image_path,
                    "error": repr(e),
                }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

    denom = max(len(rows), 1)

    log = {
        "method": "qwen25vl_3b_zeroshot",
        "model_tag": args.model_tag,
        "model_path": model_name,
        "data_path": args.data_path,
        "output_path": str(out_path),
        "samples": len(rows),
        "errors": errors,
        "parse_failures": parse_failures,
        "total_latency_sec": round(total_latency, 4),
        "avg_latency_sec": round(total_latency / denom, 4),
        "avg_input_tokens": round(total_input_tokens / denom, 2),
        "avg_output_tokens": round(total_output_tokens / denom, 2),
        "retrieval_calls_used": 0,
        "used_evidence_count": 0,
        "graph_nodes": 0,
        "graph_edges": 0,
        "graph_expansion_steps": 0,
        "gold_counts": dict(gold_counts),
        "pred_counts": dict(pred_counts),
        "note": "Zero-shot VLM baseline. The model directly observes the raw image and claim/caption. No retrieval is used.",
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print("saved:", out_path)
    print("saved log:", log_path)
    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()