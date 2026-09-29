import json
from pathlib import Path

import pandas as pd


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def to_map(rows):
    return {str(r.get("sample_id", r.get("id"))): r for r in rows}


def pick(row, keys, default=None):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def main():
    merged_path = Path("outputs/VERITE/full_usable/main_comparison/diagnostics/llama32_3b_dev_method_complementarity_merged.csv")
    slm_path = Path("outputs/VERITE/full_usable/main_comparison/slm_only/llama32_3b_slm_only_dev.jsonl")
    rag_path = Path("outputs/VERITE/full_usable/main_comparison/always_rag/llama32_3b_always_rag_dev.jsonl")
    graphrag_path = Path("outputs/VERITE/full_usable/main_comparison/always_graphrag/llama32_3b_always_graphrag_dev.jsonl")

    df = pd.read_csv(merged_path)

    slm = to_map(load_jsonl(slm_path))
    rag = to_map(load_jsonl(rag_path))
    graph = to_map(load_jsonl(graphrag_path))

    rows = []
    for _, r in df.iterrows():
        sid = str(r["sample_id"])
        s = slm[sid]
        rg = rag[sid]
        gr = graph[sid]

        item = r.to_dict()
        item.update({
            "clip_similarity_norm": pick(s, ["clip_similarity_norm"], 0.0),
            "u_cross": pick(s, ["u_cross"], 0.0),
            "slm_total_tokens": pick(s, ["total_tokens"], 0.0),
            "rag_available_evidence_count": pick(rg, ["available_evidence_count"], 0.0),
            "rag_used_evidence_count": pick(rg, ["used_evidence_count"], 0.0),
            "rag_retrieval_latency_sec_for_cache": pick(rg, ["retrieval_latency_sec_for_cache"], 0.0),
            "graph_nodes": pick(gr, ["graph_nodes"], 0.0),
            "graph_edges": pick(gr, ["graph_edges"], 0.0),
            "graph_expansion_steps": pick(gr, ["graph_expansion_steps"], 0.0),
        })
        rows.append(item)

    out = pd.DataFrame(rows)
    out_path = Path("outputs/VERITE/full_usable/main_comparison/diagnostics/llama32_3b_dev_selector_feature_table.csv")
    out.to_csv(out_path, index=False)

    groups = {
        "rag_helpful": out[out["slm_wrong_rag_correct"] == 1],
        "graphrag_helpful": out[out["slm_wrong_graphrag_correct"] == 1],
        "rag_harmful": out[out["slm_correct_rag_wrong"] == 1],
        "graphrag_harmful": out[out["slm_correct_graphrag_wrong"] == 1],
        "direct_good": out[(out["slm_only_correct"] == 1) & (out["always_rag_correct"] == 1)],
        "none_correct": out[out["none_correct"] == 1],
    }

    feature_cols = [
        "clip_similarity_norm",
        "u_cross",
        "rag_available_evidence_count",
        "rag_used_evidence_count",
        "rag_retrieval_latency_sec_for_cache",
        "graph_nodes",
        "graph_edges",
        "graph_expansion_steps",
    ]

    print("saved:", out_path)
    for name, sub in groups.items():
        print("\n====", name, "====")
        print("n:", len(sub))
        print("gold:", sub["gold_label"].value_counts().to_dict())
        print("slm_pred:", sub["slm_only_pred"].value_counts().to_dict())
        print("rag_pred:", sub["always_rag_pred"].value_counts().to_dict())
        print(sub[feature_cols].describe().loc[["mean", "std", "min", "50%", "max"]].to_string())


if __name__ == "__main__":
    main()
