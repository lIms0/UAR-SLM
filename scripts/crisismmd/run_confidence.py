import argparse
import json
import re
import time
from pathlib import Path
from collections import Counter

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


LABELS = ["informative", "not_informative"]

MODEL_MAP = {
    "llama32_1b": "./models/local_models/llama32_1b",
    "llama32_3b": "./models/local_models/llama32_3b",
    "llama31_8b": "./models/local_models/llama31_8b",
    "gemma2_9b": "./models/local_models/gemma2_9b",
}


def normalize_label(x):
    if x is None:
        return None

    x = str(x).strip().lower()
    x = x.replace("-", "_").replace(" ", "_")

    aliases = {
        "informative": "informative",
        "info": "informative",
        "useful": "informative",
        "relevant": "informative",

        "not_informative": "not_informative",
        "notinformative": "not_informative",
        "non_informative": "not_informative",
        "uninformative": "not_informative",
        "not_useful": "not_informative",
        "irrelevant": "not_informative",
    }

    return aliases.get(x)


def clamp_confidence(x):
    try:
        v = float(x)
    except Exception:
        return None

    if v > 1.0 and v <= 100.0:
        v = v / 100.0

    if v < 0:
        v = 0.0
    if v > 1:
        v = 1.0

    return v


def clean_json_text(text):
    text = str(text or "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        return text[start:end + 1].strip()

    return text


def parse_output(text):
    raw = str(text or "").strip()

    cleaned = clean_json_text(raw)
    try:
        obj = json.loads(cleaned)
        label = normalize_label(obj.get("label") or obj.get("pred_label"))
        confidence = clamp_confidence(obj.get("confidence"))
        reason = str(obj.get("reason") or "").strip()
        evidence_support = str(obj.get("evidence_support") or "").strip().lower()

        if label in LABELS and confidence is not None:
            return {
                "pred_label": label,
                "confidence": confidence,
                "reason": reason,
                "evidence_support": evidence_support,
                "parse_success": True,
                "parse_mode": "json",
            }
    except Exception:
        pass

    label = None
    lower = raw.lower()

    for pat in [
        r'"label"\s*:\s*"([^"]+)"',
        r"'label'\s*:\s*'([^']+)'",
        r'label\s*[:=]\s*([a-zA-Z_\- ]+)',
    ]:
        m = re.search(pat, raw, flags=re.IGNORECASE)
        if m:
            label = normalize_label(m.group(1))
            if label in LABELS:
                break

    if label not in LABELS:
        if "not informative" in lower or "not_informative" in lower or "uninformative" in lower:
            label = "not_informative"
        elif "informative" in lower:
            label = "informative"

    confidence = None
    for pat in [
        r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)',
        r"'confidence'\s*:\s*([0-9]*\.?[0-9]+)",
        r'confidence\s*[:=]\s*([0-9]*\.?[0-9]+)',
    ]:
        m = re.search(pat, raw, flags=re.IGNORECASE)
        if m:
            confidence = clamp_confidence(m.group(1))
            break

    evidence_support = ""
    for pat in [
        r'"evidence_support"\s*:\s*"([^"]+)"',
        r"'evidence_support'\s*:\s*'([^']+)'",
        r'evidence_support\s*[:=]\s*([a-zA-Z_\-]+)',
    ]:
        m = re.search(pat, raw, flags=re.IGNORECASE)
        if m:
            evidence_support = m.group(1).strip().lower()
            break

    reason = ""
    m = re.search(r'"reason"\s*:\s*"([^"]*)', raw, flags=re.IGNORECASE | re.DOTALL)
    if m:
        reason = m.group(1).strip()

    return {
        "pred_label": label,
        "confidence": confidence,
        "reason": reason,
        "evidence_support": evidence_support,
        "parse_success": bool(label in LABELS and confidence is not None),
        "parse_mode": "fallback",
    }


def load_jsonl(path, limit=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def load_jsonl_by_id(path):
    data = {}
    if path is None:
        return data

    path = Path(path)
    if not path.exists():
        print("[WARN] evidence file does not exist:", path)
        return data

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                sid = r.get("sample_id", r.get("id"))
                if sid is not None:
                    data[str(sid)] = r

    return data


def get_model_path(model_tag):
    if model_tag in MODEL_MAP:
        return MODEL_MAP[model_tag]

    p = Path(model_tag)
    if p.exists():
        return str(p)

    raise ValueError(f"Unknown model_tag or path: {model_tag}")


def format_evidence(evidence_row, max_evidence=5, max_chars_per_evidence=450):
    evidence = []
    if evidence_row:
        evidence = evidence_row.get("evidence") or []

    if not evidence:
        return "No retrieved evidence."

    lines = []
    for i, ev in enumerate(evidence[:max_evidence], start=1):
        title = ev.get("title") or ""
        snippet = ev.get("snippet") or ""
        url = ev.get("url") or ""
        query = ev.get("query") or ""
        query_type = ev.get("query_type") or ""

        text = f"[{i}] Title: {title}\nSnippet: {snippet}\nURL: {url}\nQuery type: {query_type}\nQuery: {query}"
        text = text.strip()

        if len(text) > max_chars_per_evidence:
            text = text[:max_chars_per_evidence] + "..."

        lines.append(text)

    return "\n\n".join(lines)


def build_prompt(row, mode, evidence_row=None, max_evidence=5):
    event_name = row.get("event_name", "")
    tweet_text = row.get("tweet_text") or row.get("text") or ""

    base = f"""
You are a strict disaster image informativeness classifier.

Classify whether the image attached to a disaster-related tweet is informative for disaster understanding or response.

Choose exactly one label from:
informative
not_informative

Definitions:
informative = the post likely contains useful situational information about the disaster, damage, affected people, rescue, relief, infrastructure, or relevant event conditions.
not_informative = the post is not useful for disaster understanding or response, such as jokes, opinions, generic comments, irrelevant content, or unclear information.

Use only the provided tweet text, event name, and retrieved evidence if available.
Do not use hidden labels or dataset annotations.
Do not infer from any field named image_info, text_info, image_human, image_damage, or gold_label.

Return only valid JSON with exactly these keys:
{{
  "label": "informative | not_informative",
  "confidence": 0.0,
  "reason": "maximum 8 words"
}}

Confidence must be a number between 0 and 1.
Do not include markdown.
Do not include text outside JSON.

Event:
{event_name}

Tweet:
{tweet_text}
""".strip()

    if mode == "always_rag":
        evidence_text = format_evidence(evidence_row, max_evidence=max_evidence)
        base += f"""

Retrieved evidence:
{evidence_text}

For evidence_support, decide whether the retrieved evidence supports the selected label.

Return only valid JSON with exactly these keys:
{{
  "label": "informative | not_informative",
  "confidence": 0.0,
  "evidence_support": "supports | contradicts | insufficient",
  "reason": "maximum 8 words"
}}
""".strip()

    return base


def make_paths(mode, model_tag, run_name, out_dir, log_dir):
    out_dir = Path(out_dir)
    log_dir = Path(log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        run_name = f"{model_tag}_{mode}"

    return out_dir / f"{run_name}.jsonl", log_dir / f"{run_name}_log.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["slm_only", "always_rag"])
    parser.add_argument("--feature_path", required=True)
    parser.add_argument("--evidence_path", default=None)
    parser.add_argument("--model_tag", required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--out_dir", default="outputs/CrisisMMD/full_usable/main_comparison")
    parser.add_argument("--log_dir", default="outputs/CrisisMMD/full_usable/main_comparison/logs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_input_tokens", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_evidence", type=int, default=5)
    args = parser.parse_args()

    model_path = get_model_path(args.model_tag)

    method_out_dir = Path(args.out_dir) / args.mode
    out_path, log_path = make_paths(args.mode, args.model_tag, args.run_name, method_out_dir, args.log_dir)

    rows = load_jsonl(args.feature_path, limit=args.limit)
    evidence_by_id = load_jsonl_by_id(args.evidence_path) if args.mode == "always_rag" else {}

    print("mode:", args.mode)
    print("feature_path:", args.feature_path)
    print("evidence_path:", args.evidence_path)
    print("model_tag:", args.model_tag)
    print("model_path:", model_path)
    print("samples:", len(rows))
    print("evidence rows:", len(evidence_by_id))
    print("out_path:", out_path)

    print("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()

    errors = 0
    parse_failures = 0
    missing_evidence_rows = 0

    total_latency = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    total_retrieval_calls = 0
    total_used_evidence = 0
    total_retrieval_latency = 0.0

    gold_counts = Counter()
    pred_counts = Counter()

    with out_path.open("w", encoding="utf-8") as fout:
        for row in tqdm(rows, desc=f"CrisisMMD {args.mode} {args.model_tag}"):
            sid = row.get("sample_id", row.get("id"))
            sid_key = str(sid)

            evidence_row = {}
            if args.mode == "always_rag":
                evidence_row = evidence_by_id.get(sid_key)
                if evidence_row is None:
                    missing_evidence_rows += 1
                    evidence_row = {}

            available_evidence = evidence_row.get("evidence") or []
            available_evidence_count = int(
                evidence_row.get("available_evidence_count", evidence_row.get("retrieved_evidence_count", 0)) or 0
            )
            used_evidence_count = min(len(available_evidence), args.max_evidence) if args.mode == "always_rag" else 0
            retrieval_calls_used = int(evidence_row.get("retrieval_calls_for_cache", 0) or 0) if args.mode == "always_rag" else 0
            retrieval_latency_sec_for_cache = float(evidence_row.get("retrieval_latency_sec_for_cache", 0.0) or 0.0) if args.mode == "always_rag" else 0.0

            prompt = build_prompt(
                row=row,
                mode=args.mode,
                evidence_row=evidence_row,
                max_evidence=args.max_evidence,
            )

            try:
                messages = [{"role": "user", "content": prompt}]
                try:
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                    text = prompt

                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_input_tokens,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                input_tokens = int(inputs["input_ids"].shape[-1])

                start = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                latency = time.time() - start
                generated = outputs[0][input_tokens:]
                output_text = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                output_tokens = int(generated.shape[-1])

                parsed = parse_output(output_text)
                pred_label = parsed.get("pred_label")
                confidence = parsed.get("confidence")
                reason = parsed.get("reason", "")
                evidence_support = parsed.get("evidence_support", "")
                parse_success = parsed.get("parse_success", False)

                if not parse_success:
                    parse_failures += 1

                gold_label = row.get("gold_label", row.get("label"))
                gold_counts[gold_label] += 1
                pred_counts[pred_label] += 1

                out = {
                    "id": sid,
                    "sample_id": row.get("sample_id"),
                    "tweet_id": row.get("tweet_id"),
                    "image_id": row.get("image_id"),
                    "event_name": row.get("event_name"),
                    "method": args.mode + "_confidence",
                    "model_tag": args.model_tag,
                    "model_path": model_path,

                    "gold_label": gold_label,
                    "pred_label": pred_label,
                    "confidence": confidence,
                    "evidence_support": evidence_support,
                    "parse_success": parse_success,
                    "parse_mode": parsed.get("parse_mode"),
                    "reason": reason,
                    "raw_output": output_text,

                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "inference_latency_sec": round(latency, 4),

                    "available_evidence_count": available_evidence_count,
                    "retrieval_calls_used": retrieval_calls_used,
                    "used_evidence_count": used_evidence_count,
                    "retrieval_latency_sec_for_cache": retrieval_latency_sec_for_cache,

                    "clip_similarity_norm": row.get("clip_similarity_norm"),
                    "u_cross": row.get("u_cross"),

                    "tweet_text": row.get("tweet_text") or row.get("text"),
                }

                total_latency += latency
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                total_retrieval_calls += retrieval_calls_used
                total_used_evidence += used_evidence_count
                total_retrieval_latency += retrieval_latency_sec_for_cache

            except Exception as e:
                errors += 1
                out = {
                    "id": sid,
                    "sample_id": row.get("sample_id"),
                    "method": args.mode + "_confidence",
                    "model_tag": args.model_tag,
                    "gold_label": row.get("gold_label", row.get("label")),
                    "error": repr(e),
                }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

    denom = max(len(rows), 1)

    log = {
        "method": args.mode + "_confidence",
        "model_tag": args.model_tag,
        "model_path": model_path,
        "feature_path": args.feature_path,
        "evidence_path": args.evidence_path,
        "output_path": str(out_path),
        "samples": len(rows),
        "evidence_rows": len(evidence_by_id),
        "missing_evidence_rows": missing_evidence_rows,
        "errors": errors,
        "parse_failures": parse_failures,
        "total_latency_sec": round(total_latency, 4),
        "avg_latency_sec": round(total_latency / denom, 4),
        "avg_input_tokens": round(total_input_tokens / denom, 2),
        "avg_output_tokens": round(total_output_tokens / denom, 2),
        "total_retrieval_calls_used": total_retrieval_calls,
        "avg_retrieval_calls_used": round(total_retrieval_calls / denom, 4),
        "total_used_evidence_count": total_used_evidence,
        "avg_used_evidence_count": round(total_used_evidence / denom, 4),
        "total_retrieval_latency_sec_for_cache": round(total_retrieval_latency, 4),
        "gold_counts": dict(gold_counts),
        "pred_counts": dict(pred_counts),
        "note": "CrisisMMD confidence-aware experiment. Prompt avoids gold/annotation-like fields such as image_info, text_info, image_human, and image_damage.",
    }

    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    print("saved:", out_path)
    print("saved log:", log_path)
    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
