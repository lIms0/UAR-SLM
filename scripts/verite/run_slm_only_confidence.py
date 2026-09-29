import argparse
import json
import re
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from confidence_parsing import parse_confidence_prediction


DEFAULT_DATA_PATH = "./data/VERITE/processed/full_usable/verite_stage1_features.jsonl"
DEFAULT_OUT_DIR = "./outputs/VERITE/full_usable/main_comparison/slm_only"
DEFAULT_LOG_DIR = "./outputs/VERITE/full_usable/main_comparison/logs"

MODEL_MAP = {
    "llama32_1b": "./models/local_models/llama32_1b",
    "llama32_3b": "./models/local_models/llama32_3b",
    "llama31_8b": "./models/local_models/llama31_8b",
    "gemma2_9b": "./models/local_models/gemma2_9b",
}

LABELS = ["true", "miscaptioned", "out-of-context"]


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


def cue_to_text(row):
    cues = row.get("parsed_visual_cues") or {}

    if not isinstance(cues, dict):
        cues = {}

    return "\n".join([
        f"Generated image caption: {cues.get('generated_caption', '')}",
        f"Visible objects: {cues.get('visible_objects', [])}",
        f"Scene: {cues.get('scene', '')}",
        f"People activity: {cues.get('people_activity', '')}",
        f"Visible text: {cues.get('visible_text', '')}",
        f"Location cues: {cues.get('location_cues', '')}",
        f"Time cues: {cues.get('time_cues', '')}",
        f"Uncertain visual points: {cues.get('uncertain_visual_points', [])}",
        f"Visual cue fallback applied: {row.get('fallback_applied', False)}",
        f"Visual cue failure type: {row.get('failure_type', None)}",
    ])


def build_prompt(row):
    visual_text = cue_to_text(row)

    return f"""
You are a strict image-caption verification classifier.

Choose exactly one label from:
true
miscaptioned
out-of-context

Definitions:
true = the main visible scene supports the claim.
miscaptioned = the image is related to the claim, but an important detail is wrong, unsupported, or suspicious.
out-of-context = the image shows a clearly different situation from the claim.

Use only:
1. the provided visual cues,
2. the CLIP score.

Do not use external knowledge.

Return only valid JSON with exactly these keys:
{{
  "label": "true | miscaptioned | out-of-context",
  "confidence": 0.0,
  "reason": "one short sentence"
}}

Confidence must be a number between 0 and 1.
Do not include markdown.
Do not include any text outside JSON.

Claim:
{row.get("claim", "")}

Visual cues:
{visual_text}

CLIP:
raw={row.get("clip_similarity_raw")}
normalized={row.get("clip_similarity_norm")}
u_cross={row.get("u_cross")}
""".strip()


def clean_json_text(text):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    return text.strip()


def parse_prediction(text):
    if text is None:
        return None, "", False

    raw = str(text).strip()
    cleaned = raw.lower().strip()

    # Use only the first generated line if the model produces extra text.
    lines = [x.strip() for x in cleaned.splitlines() if x.strip()]
    first_line = lines[0] if lines else cleaned

    first_line = first_line.replace('"', '').replace("'", "")
    first_line = first_line.replace(".", "").replace(",", "").replace(":", "")
    first_line = first_line.replace("_", "-")
    first_line = " ".join(first_line.split())

    aliases = {
        "true": "true",
        "truthful": "true",
        "real": "true",
        "match": "true",
        "matched": "true",

        "miscaptioned": "miscaptioned",
        "mis-captioned": "miscaptioned",
        "mis captioned": "miscaptioned",
        "wrong-caption": "miscaptioned",
        "wrong caption": "miscaptioned",

        "out-of-context": "out-of-context",
        "out of context": "out-of-context",
        "out-context": "out-of-context",
        "ooc": "out-of-context",
    }

    if first_line in aliases:
        return aliases[first_line], "", True

    # Robust fallback: search the whole output, but do not over-accept explanations.
    found = []

    if "out-of-context" in cleaned or "out of context" in cleaned or "ooc" in cleaned:
        found.append("out-of-context")

    if "miscaptioned" in cleaned or "mis-captioned" in cleaned or "mis captioned" in cleaned or "wrong caption" in cleaned:
        found.append("miscaptioned")

    # true is accepted only if it appears as the first generated label.
    if first_line.startswith("true"):
        found.append("true")

    found = list(dict.fromkeys(found))

    if len(found) == 1:
        return found[0], raw, False

    return None, raw, False

def load_rows(path, limit=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def get_model_path(model_tag):
    if model_tag in MODEL_MAP:
        return MODEL_MAP[model_tag]

    p = Path(model_tag)
    if p.exists():
        return str(p)

    raise ValueError(f"Unknown model_tag or path: {model_tag}")


def make_output_paths(model_tag, run_name, out_dir, log_dir):
    out_dir = Path(out_dir)
    log_dir = Path(log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        run_name = f"slm_only_{model_tag}"

    out_path = out_dir / f"{run_name}.jsonl"
    log_path = log_dir / f"{run_name}_log.json"

    return out_path, log_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--model_tag", required=True, help="llama32_1b, llama32_3b, llama31_8b, gemma2_9b, or local path")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--log_dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_input_tokens", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    model_path = get_model_path(args.model_tag)
    out_path, log_path = make_output_paths(args.model_tag, args.run_name, args.out_dir, args.log_dir)

    rows = load_rows(args.data_path, limit=args.limit)

    print("data_path:", args.data_path)
    print("model_tag:", args.model_tag)
    print("model_path:", model_path)
    print("samples:", len(rows))
    print("out_path:", out_path)

    print("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )

    model.eval()

    errors = 0
    parse_failures = 0
    total_latency = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for row in tqdm(rows, desc=f"SLM-only-confidence {args.model_tag}"):
            prompt = build_prompt(row)

            messages = [{"role": "user", "content": prompt}]

            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                text = prompt

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_tokens
            )

            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            input_tokens = int(inputs["input_ids"].shape[-1])

            start = time.time()

            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )

                latency = time.time() - start

                generated = outputs[0][input_tokens:]
                output_text = tokenizer.decode(generated, skip_special_tokens=True)
                output_tokens = int(generated.shape[-1])

                parsed = parse_confidence_prediction(output_text)
                pred_label = parsed.get('pred_label')
                confidence = parsed.get('confidence')
                reason = parsed.get('reason', '')
                parse_success = parsed.get('parse_success', False)

                if not parse_success:
                    parse_failures += 1

                out = {
                    "id": row.get("id"),
                    "sample_id": row.get("sample_id"),
                    "pair_id": row.get("pair_id"),
                    "method": "slm_only_confidence",
                    "model_tag": args.model_tag,
                    "model_path": model_path,
                    "gold_label": row.get("label"),
                    "pred_label": pred_label,
                    "confidence": confidence,
                    "parse_success": parse_success,
                    "parse_mode": parsed.get("parse_mode"),
                    "reason": reason,
                    "raw_output": output_text,

                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "inference_latency_sec": round(latency, 4),

                    "retrieval_calls_used": 0,
                    "used_evidence_count": 0,
                    "graph_nodes": 0,
                    "graph_edges": 0,
                    "graph_expansion_steps": 0,

                    "clip_similarity_norm": row.get("clip_similarity_norm"),
                    "u_cross": row.get("u_cross"),

                    "fallback_applied": row.get("fallback_applied", False),
                    "visual_parse_success": row.get("visual_parse_success", False),
                    "failure_type": row.get("failure_type"),
                }

                total_latency += latency
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens

            except Exception as e:
                errors += 1
                out = {
                    "id": row.get("id"),
                    "sample_id": row.get("sample_id"),
                    "pair_id": row.get("pair_id"),
                    "method": "slm_only_confidence",
                    "model_tag": args.model_tag,
                    "model_path": model_path,
                    "gold_label": row.get("label"),
                    "error": repr(e),
                }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

    denom = max(len(rows), 1)
    log = {
        "method": "slm_only_confidence",
        "model_tag": args.model_tag,
        "model_path": model_path,
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
        "note": "SLM-only does not use retrieval or graph reasoning. Retrieval and graph costs are recorded as zero.",
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print("saved:", out_path)
    print("saved log:", log_path)
    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
