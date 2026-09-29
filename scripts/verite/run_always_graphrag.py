import argparse
import json
import re
import time
from pathlib import Path
from collections import Counter

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


DEFAULT_FEATURE_PATH = "./data/VERITE/processed/full_usable/verite_stage1_features.jsonl"
DEFAULT_EVIDENCE_PATH = "./data/VERITE/processed/full_usable/retrieved_evidence_brave_q4_k10.jsonl"
DEFAULT_OUT_DIR = "./outputs/VERITE/full_usable/main_comparison/always_graphrag"
DEFAULT_LOG_DIR = "./outputs/VERITE/full_usable/main_comparison/logs"

MODEL_MAP = {
    "llama32_1b": "./models/local_models/llama32_1b",
    "llama32_3b": "./models/local_models/llama32_3b",
    "llama31_8b": "./models/local_models/llama31_8b",
    "gemma2_9b": "./models/local_models/gemma2_9b",
}

LABELS = ["true", "miscaptioned", "out-of-context"]


STOPWORDS = {
    "image", "photo", "picture", "photograph", "the", "and", "with", "during",
    "near", "from", "this", "that", "into", "onto", "there", "their", "about",
    "have", "has", "had", "were", "was", "are", "is", "for", "of", "to", "in",
    "on", "at", "by", "an", "a", "as", "it", "its", "be", "been", "being",
    "shows", "showing", "show", "claim", "fact", "check"
}


def load_jsonl_by_sample_id(path):
    path = Path(path)
    data = {}

    if not path.exists():
        print(f"[WARN] file does not exist: {path}")
        return data

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row.get("sample_id", row.get("id"))
            if sid is not None:
                data[str(sid)] = row

    return data


def load_feature_rows(path, limit=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def cue_to_text(row):
    cues = row.get("parsed_visual_cues") or {}

    if not isinstance(cues, dict):
        cues = {}

    return "\n".join([
        f"Generated image caption: {cues.get('generated_caption', row.get('generated_caption', ''))}",
        f"Visible objects: {cues.get('visible_objects', row.get('visible_objects', []))}",
        f"Scene: {cues.get('scene', row.get('scene', ''))}",
        f"People activity: {cues.get('people_activity', row.get('people_activity', ''))}",
        f"Visible text: {cues.get('visible_text', row.get('visible_text', ''))}",
        f"Location cues: {cues.get('location_cues', row.get('location_cues', ''))}",
        f"Time cues: {cues.get('time_cues', row.get('time_cues', ''))}",
        f"Uncertain visual points: {cues.get('uncertain_visual_points', row.get('uncertain_visual_points', []))}",
    ])


def tokenize(text):
    text = (text or "").lower()
    text = re.sub(r'["“”]', " ", text)
    toks = re.findall(r"[a-z0-9][a-z0-9\-]+", text)
    return [t for t in toks if t not in STOPWORDS and len(t) > 2]


def truncate_text(text, max_chars):
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def select_evidence(evidence_row, max_evidence=5):
    evidence = evidence_row.get("evidence") or []
    if not evidence:
        return []

    # Prefer evidence with higher keyword overlap and relevance score, but keep stable ordering as fallback.
    ranked = []
    for i, ev in enumerate(evidence):
        score = ev.get("relevance_score", 0.0) or 0.0
        overlap = ev.get("keyword_overlap_count", 0) or 0
        ranked.append((overlap, score, -i, ev))

    ranked.sort(reverse=True)
    return [x[-1] for x in ranked[:max_evidence]]


def build_evidence_graph(evidence_row, max_evidence=5, edge_min_overlap=2, max_chars_per_node=360):
    selected = select_evidence(evidence_row, max_evidence=max_evidence)

    nodes = []
    for i, ev in enumerate(selected, start=1):
        title = ev.get("title") or ""
        snippet = ev.get("snippet") or ""
        url = ev.get("url") or ""
        domain = ev.get("domain") or ""
        query_type = ev.get("query_type") or ""
        query = ev.get("query") or ""

        node_text = f"{title} {snippet}"
        node_terms = sorted(set(tokenize(node_text)))

        nodes.append({
            "node_id": f"E{i}",
            "title": title,
            "snippet": snippet,
            "url": url,
            "domain": domain,
            "query_type": query_type,
            "query": query,
            "relevance_score": ev.get("relevance_score"),
            "keyword_overlap_count": ev.get("keyword_overlap_count"),
            "matched_query_terms": ev.get("matched_query_terms", []),
            "node_terms": node_terms,
        })

    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            terms_i = set(nodes[i]["node_terms"])
            terms_j = set(nodes[j]["node_terms"])
            shared = sorted(terms_i & terms_j)

            if len(shared) >= edge_min_overlap:
                edges.append({
                    "source": nodes[i]["node_id"],
                    "target": nodes[j]["node_id"],
                    "edge_type": "lexical_overlap",
                    "shared_terms": shared[:12],
                    "weight": len(shared),
                })

    lines = []
    if not nodes:
        graph_text = "No evidence graph could be constructed because no evidence was available."
    else:
        lines.append("Evidence graph nodes:")
        for n in nodes:
            node_line = (
                f"{n['node_id']}. Title: {truncate_text(n['title'], 160)}\n"
                f"Snippet: {truncate_text(n['snippet'], max_chars_per_node)}\n"
                f"Source: {n['domain'] or n['url']}\n"
                f"Query type: {n['query_type']}"
            )
            lines.append(node_line)

        if edges:
            lines.append("\nEvidence graph edges:")
            for e in edges:
                shared = ", ".join(e["shared_terms"][:8])
                lines.append(
                    f"{e['source']} -- {e['target']} | shared terms: {shared}"
                )
        else:
            lines.append("\nEvidence graph edges: No strong lexical-overlap edges were found.")

        graph_text = "\n\n".join(lines)

    graph = {
        "nodes": nodes,
        "edges": edges,
        "graph_construction": {
            "node_unit": "retrieved_evidence",
            "edge_rule": "lexical_overlap_between_evidence_title_and_snippet",
            "edge_min_overlap": edge_min_overlap,
            "max_evidence": max_evidence,
        },
    }

    return graph, graph_text


def build_prompt(feature_row, evidence_graph_text):
    visual_text = cue_to_text(feature_row)

    return f"""
You are a strict image-caption verification classifier.

Classify the claim using:
1. image-derived visual cues,
2. CLIP image-claim similarity,
3. an evidence graph built from retrieved web evidence.

Choose exactly one label from:
true
miscaptioned
out-of-context

Definitions:
true = the claim is supported by the visual cues and evidence graph.
miscaptioned = the image is related to the claim, but an important detail is wrong, unsupported, or suspicious.
out-of-context = the image shows a clearly different situation from the claim, or the evidence graph suggests the image belongs to a different context.

Rules:
- Use only the provided visual cues, CLIP score, and evidence graph.
- Do not use external knowledge beyond the provided evidence graph.
- Do not explain.
- Do not write a sentence.
- Output exactly one label only.

Claim:
{feature_row.get("claim", "")}

Visual cues:
{visual_text}

CLIP:
raw={feature_row.get("clip_similarity_raw")}
normalized={feature_row.get("clip_similarity_norm")}
u_cross={feature_row.get("u_cross")}

Evidence graph:
{evidence_graph_text}

Label:
""".strip()


def parse_prediction(text):
    if text is None:
        return None, "", False

    raw = str(text).strip()
    cleaned = raw.lower().strip()

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

    found = []

    if "out-of-context" in cleaned or "out of context" in cleaned or "ooc" in cleaned:
        found.append("out-of-context")

    if "miscaptioned" in cleaned or "mis-captioned" in cleaned or "mis captioned" in cleaned or "wrong caption" in cleaned:
        found.append("miscaptioned")

    if first_line.startswith("true"):
        found.append("true")

    found = list(dict.fromkeys(found))

    if len(found) == 1:
        return found[0], raw, False

    return None, raw, False


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
        run_name = f"always_graphrag_{model_tag}"

    out_path = out_dir / f"{run_name}.jsonl"
    log_path = log_dir / f"{run_name}_log.json"

    return out_path, log_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path", default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--evidence_path", default=DEFAULT_EVIDENCE_PATH)
    parser.add_argument("--model_tag", required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--log_dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_input_tokens", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument("--max_evidence", type=int, default=5)
    parser.add_argument("--edge_min_overlap", type=int, default=2)
    args = parser.parse_args()

    model_path = get_model_path(args.model_tag)
    out_path, log_path = make_output_paths(
        args.model_tag,
        args.run_name,
        args.out_dir,
        args.log_dir,
    )

    feature_rows = load_feature_rows(args.feature_path, limit=args.limit)
    evidence_by_id = load_jsonl_by_sample_id(args.evidence_path)

    print("feature_path:", args.feature_path)
    print("evidence_path:", args.evidence_path)
    print("model_tag:", args.model_tag)
    print("model_path:", model_path)
    print("samples:", len(feature_rows))
    print("evidence rows:", len(evidence_by_id))
    print("out_path:", out_path)

    print("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
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

    total_retrieval_calls_used = 0
    total_used_evidence_count = 0
    total_graph_nodes = 0
    total_graph_edges = 0
    total_graph_expansion_steps = 0

    gold_counts = Counter()
    pred_counts = Counter()

    with open(out_path, "w", encoding="utf-8") as fout:
        for row in tqdm(feature_rows, desc=f"Always-GraphRAG {args.model_tag}"):
            sid = row.get("sample_id", row.get("id"))
            sid_key = str(sid) if sid is not None else ""
            evidence_row = evidence_by_id.get(sid_key)

            if evidence_row is None:
                missing_evidence_rows += 1
                evidence_row = {}

            graph, graph_text = build_evidence_graph(
                evidence_row=evidence_row,
                max_evidence=args.max_evidence,
                edge_min_overlap=args.edge_min_overlap,
            )

            graph_nodes = len(graph["nodes"])
            graph_edges = len(graph["edges"])
            graph_expansion_steps = 1 if graph_nodes > 0 else 0
            used_evidence_count = graph_nodes

            retrieval_calls_used = int(evidence_row.get("retrieval_calls_for_cache", 0) or 0)
            available_evidence_count = int(
                evidence_row.get("available_evidence_count", evidence_row.get("retrieved_evidence_count", 0)) or 0
            )

            prompt = build_prompt(
                feature_row=row,
                evidence_graph_text=graph_text,
            )

            try:
                messages = [{"role": "user", "content": prompt}]

                try:
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
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

                inference_latency = time.time() - start

                generated = outputs[0][input_tokens:]
                output_text = tokenizer.decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                output_tokens = int(generated.shape[-1])

                pred_label, reason, parse_success = parse_prediction(output_text)

                if not parse_success:
                    parse_failures += 1

                gold_label = row.get("label")
                gold_counts[gold_label] += 1
                pred_counts[pred_label] += 1

                out = {
                    "id": sid,
                    "sample_id": row.get("sample_id"),
                    "pair_id": row.get("pair_id"),
                    "method": "always_graphrag",
                    "model_tag": args.model_tag,
                    "model_path": model_path,

                    "gold_label": gold_label,
                    "pred_label": pred_label,
                    "parse_success": parse_success,
                    "raw_output": output_text,
                    "reason": reason,

                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "inference_latency_sec": round(inference_latency, 4),

                    "available_pool_name": evidence_row.get("available_pool_name"),
                    "available_retrieval_backend": evidence_row.get("available_retrieval_backend", evidence_row.get("retrieval_backend")),
                    "available_max_queries": evidence_row.get("available_max_queries"),
                    "available_max_results_per_query": evidence_row.get("available_max_results_per_query"),
                    "available_evidence_count": available_evidence_count,

                    "retrieval_calls_used": retrieval_calls_used,
                    "used_evidence_count": used_evidence_count,

                    "graph_nodes": graph_nodes,
                    "graph_edges": graph_edges,
                    "graph_expansion_steps": graph_expansion_steps,
                    "evidence_graph": graph,

                    "clip_similarity_norm": row.get("clip_similarity_norm"),
                    "u_cross": row.get("u_cross"),
                }

                total_latency += inference_latency
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens

                total_retrieval_calls_used += retrieval_calls_used
                total_used_evidence_count += used_evidence_count
                total_graph_nodes += graph_nodes
                total_graph_edges += graph_edges
                total_graph_expansion_steps += graph_expansion_steps

            except Exception as e:
                errors += 1
                out = {
                    "id": sid,
                    "sample_id": row.get("sample_id"),
                    "method": "always_graphrag",
                    "model_tag": args.model_tag,
                    "model_path": model_path,
                    "gold_label": row.get("label"),
                    "error": repr(e),
                }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

    denom = max(len(feature_rows), 1)

    log = {
        "method": "always_graphrag",
        "model_tag": args.model_tag,
        "model_path": model_path,
        "feature_path": args.feature_path,
        "evidence_path": args.evidence_path,
        "output_path": str(out_path),
        "samples": len(feature_rows),
        "evidence_rows": len(evidence_by_id),
        "missing_evidence_rows": missing_evidence_rows,
        "errors": errors,
        "parse_failures": parse_failures,

        "total_latency_sec": round(total_latency, 4),
        "avg_latency_sec": round(total_latency / denom, 4),
        "avg_input_tokens": round(total_input_tokens / denom, 2),
        "avg_output_tokens": round(total_output_tokens / denom, 2),

        "total_retrieval_calls_used": total_retrieval_calls_used,
        "avg_retrieval_calls_used": round(total_retrieval_calls_used / denom, 4),
        "total_used_evidence_count": total_used_evidence_count,
        "avg_used_evidence_count": round(total_used_evidence_count / denom, 4),

        "total_graph_nodes": total_graph_nodes,
        "avg_graph_nodes": round(total_graph_nodes / denom, 4),
        "total_graph_edges": total_graph_edges,
        "avg_graph_edges": round(total_graph_edges / denom, 4),
        "total_graph_expansion_steps": total_graph_expansion_steps,
        "avg_graph_expansion_steps": round(total_graph_expansion_steps / denom, 4),

        "max_evidence": args.max_evidence,
        "edge_min_overlap": args.edge_min_overlap,

        "gold_counts": dict(gold_counts),
        "pred_counts": dict(pred_counts),

        "note": (
            "Always-GraphRAG constructs a lightweight evidence graph for every sample using retrieved evidence from the offline q4_k10 cache. "
            "Evidence items are graph nodes, and lexical-overlap relations are graph edges. "
            "This is an evidence graph, not an external knowledge graph."
        ),
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print("saved:", out_path)
    print("saved log:", log_path)
    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
