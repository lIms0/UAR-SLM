import argparse
import json
import re
from pathlib import Path
from itertools import product
from collections import Counter

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


LABELS = ["true", "miscaptioned", "out-of-context"]

STOPWORDS = {
    "image", "photo", "picture", "photograph", "the", "and", "with", "during",
    "near", "from", "this", "that", "into", "onto", "there", "their", "about",
    "have", "has", "had", "were", "was", "are", "is", "for", "of", "to", "in",
    "on", "at", "by", "an", "a", "as", "it", "its", "be", "been", "being",
    "shows", "showing", "show", "claim", "fact", "check"
}


def load_jsonl(path):
    rows = []
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def by_sample_id(rows):
    return {
        str(row.get("sample_id", row.get("id"))): row
        for row in rows
        if row.get("sample_id", row.get("id")) is not None
    }


def tokenize(text):
    text = (text or "").lower()
    text = re.sub(r'["“”]', " ", text)
    toks = re.findall(r"[a-z0-9][a-z0-9\-]+", text)
    return [t for t in toks if t not in STOPWORDS and len(t) > 2]


def clamp01(x):
    try:
        x = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, x))


def safe_mean(values):
    vals = []
    for v in values:
        try:
            if v is not None:
                vals.append(float(v))
        except Exception:
            pass
    return sum(vals) / len(vals) if vals else 0.0


def select_evidence(evidence_row, max_evidence=5):
    evidence = evidence_row.get("evidence") or []
    ranked = []

    for i, ev in enumerate(evidence):
        score = ev.get("relevance_score", 0.0) or 0.0
        overlap = ev.get("keyword_overlap_count", 0) or 0
        ranked.append((overlap, score, -i, ev))

    ranked.sort(reverse=True)
    return [x[-1] for x in ranked[:max_evidence]]


def compute_evidence_conflict_proxy(evidence_row, max_evidence=5):
    selected = select_evidence(evidence_row, max_evidence=max_evidence)

    if len(selected) < 2:
        return 0.0

    term_sets = []
    for ev in selected:
        text = (ev.get("title") or "") + " " + (ev.get("snippet") or "")
        term_sets.append(set(tokenize(text)))

    jaccards = []
    for i in range(len(term_sets)):
        for j in range(i + 1, len(term_sets)):
            a = term_sets[i]
            b = term_sets[j]
            if not a or not b:
                continue
            jaccards.append(len(a & b) / max(len(a | b), 1))

    if not jaccards:
        return 0.0

    avg_jaccard = sum(jaccards) / len(jaccards)

    # 낮은 overlap일수록 근거들이 서로 다른 맥락일 가능성이 높다고 보는 proxy
    return clamp01(1.0 - avg_jaccard)


def compute_retrieval_benefit_proxy(evidence_row, max_evidence=5):
    selected = select_evidence(evidence_row, max_evidence=max_evidence)

    if not selected:
        return 0.0

    rel_scores = []
    overlap_scores = []

    for ev in selected:
        rel_scores.append(clamp01(ev.get("relevance_score", 0.0) or 0.0))
        overlap = ev.get("keyword_overlap_count", 0) or 0
        overlap_scores.append(min(float(overlap) / 5.0, 1.0))

    top_rel = max(rel_scores) if rel_scores else 0.0
    avg_overlap = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0
    availability = min(len(selected) / float(max_evidence), 1.0)

    return clamp01(0.5 * top_rel + 0.3 * avg_overlap + 0.2 * availability)


def compute_decision_score(feature_row, evidence_row, alpha, beta, gamma, max_evidence=5):
    available_evidence_count = int(
        evidence_row.get(
            "available_evidence_count",
            evidence_row.get("retrieved_evidence_count", 0)
        ) or 0
    )

    u_total = clamp01(feature_row.get("u_cross"))

    if available_evidence_count == 0:
        return 0.0, {
            "U_total": u_total,
            "C_evidence": 0.0,
            "B_retrieval": 0.0,
            "R_budget": 0.0,
            "available_evidence_count": 0,
        }

    c_evidence = compute_evidence_conflict_proxy(evidence_row, max_evidence=max_evidence)
    b_retrieval = compute_retrieval_benefit_proxy(evidence_row, max_evidence=max_evidence)

    # 현재는 offline cache를 사용하므로 budget은 남아 있다고 간주
    r_budget = 1.0

    score = r_budget * (
        alpha * u_total +
        beta * c_evidence +
        gamma * b_retrieval
    )

    signals = {
        "U_total": u_total,
        "C_evidence": c_evidence,
        "B_retrieval": b_retrieval,
        "R_budget": r_budget,
        "available_evidence_count": available_evidence_count,
    }

    return clamp01(score), signals


def choose_action(score, t_rag, t_graph):
    if score < t_rag:
        return "direct"
    if score < t_graph:
        return "rag"
    return "graphrag"


def metric_summary(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )

    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="weighted", zero_division=0
    )

    p_cls, r_cls, f1_cls, support_cls = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0
    )

    return {
        "accuracy": acc,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "f1_true": f1_cls[0],
        "f1_miscaptioned": f1_cls[1],
        "f1_out_of_context": f1_cls[2],
        "recall_out_of_context": r_cls[2],
    }


def get_action_row(action, sid, direct_rows, rag_rows, graph_rows):
    if action == "direct":
        return direct_rows.get(sid)
    if action == "rag":
        return rag_rows.get(sid)
    if action == "graphrag":
        return graph_rows.get(sid)
    return direct_rows.get(sid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path", required=True)
    parser.add_argument("--evidence_path", required=True)
    parser.add_argument("--direct_result_path", required=True)
    parser.add_argument("--rag_result_path", required=True)
    parser.add_argument("--graphrag_result_path", required=True)
    parser.add_argument("--out_dir", default="outputs/VERITE/full_usable/main_comparison/adaptive_grid")
    parser.add_argument("--tag", default="llama32_1b_dev")
    parser.add_argument("--max_evidence", type=int, default=5)

    parser.add_argument("--lambda_retrieval", type=float, default=0.005)
    parser.add_argument("--lambda_graph", type=float, default=0.005)
    parser.add_argument("--lambda_tokens", type=float, default=0.0005)
    args = parser.parse_args()

    feature_rows = load_jsonl(args.feature_path)
    evidence_rows = by_sample_id(load_jsonl(args.evidence_path))
    direct_rows = by_sample_id(load_jsonl(args.direct_result_path))
    rag_rows = by_sample_id(load_jsonl(args.rag_result_path))
    graph_rows = by_sample_id(load_jsonl(args.graphrag_result_path))

    weight_grid = []
    for a in range(0, 11):
        for b in range(0, 11 - a):
            c = 10 - a - b
            weight_grid.append((a / 10.0, b / 10.0, c / 10.0))

    t_rag_grid = [0.30, 0.35, 0.40, 0.45, 0.50]
    t_graph_grid = [0.50, 0.55, 0.60, 0.65, 0.70]

    candidates = []

    for alpha, beta, gamma in weight_grid:
        for t_rag in t_rag_grid:
            for t_graph in t_graph_grid:
                if t_graph <= t_rag:
                    continue

                y_true = []
                y_pred = []
                selected_rows = []
                action_counts = Counter()

                for feat in feature_rows:
                    sid = str(feat.get("sample_id"))
                    ev = evidence_rows.get(sid, {})

                    score, signals = compute_decision_score(
                        feature_row=feat,
                        evidence_row=ev,
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma,
                        max_evidence=args.max_evidence,
                    )

                    action = choose_action(score, t_rag=t_rag, t_graph=t_graph)
                    action_row = get_action_row(
                        action,
                        sid,
                        direct_rows,
                        rag_rows,
                        graph_rows,
                    )

                    if action_row is None:
                        continue

                    gold = action_row.get("gold_label")
                    pred = action_row.get("pred_label")

                    if gold not in LABELS or pred not in LABELS:
                        continue

                    y_true.append(gold)
                    y_pred.append(pred)
                    selected_rows.append(action_row)
                    action_counts[action] += 1

                if not y_true:
                    continue

                metrics = metric_summary(y_true, y_pred)

                avg_retrieval = safe_mean([r.get("retrieval_calls_used") for r in selected_rows])
                avg_graph_nodes = safe_mean([r.get("graph_nodes") for r in selected_rows])
                avg_tokens = safe_mean([r.get("total_tokens") for r in selected_rows])

                objective = (
                    metrics["macro_f1"]
                    - args.lambda_retrieval * avg_retrieval
                    - args.lambda_graph * avg_graph_nodes
                    - args.lambda_tokens * (avg_tokens / 1000.0)
                )

                candidates.append({
                    "tag": args.tag,
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "t_rag": t_rag,
                    "t_graph": t_graph,
                    "objective": objective,
                    **metrics,
                    "avg_retrieval_calls_used": avg_retrieval,
                    "avg_graph_nodes": avg_graph_nodes,
                    "avg_total_tokens": avg_tokens,
                    "action_direct": action_counts.get("direct", 0),
                    "action_rag": action_counts.get("rag", 0),
                    "action_graphrag": action_counts.get("graphrag", 0),
                    "valid_samples": len(y_true),
                })

    if not candidates:
        raise RuntimeError("No grid-search candidates were evaluated.")

    df = pd.DataFrame(candidates)
    df = df.sort_values(
        ["objective", "macro_f1", "avg_retrieval_calls_used", "avg_graph_nodes", "avg_total_tokens"],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)

    best = df.iloc[0].to_dict()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = out_dir / f"{args.tag}_adaptive_grid_candidates.csv"
    best_path = out_dir / f"{args.tag}_adaptive_best_config.json"

    df.to_csv(candidates_path, index=False)

    with best_path.open("w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print("saved candidates:", candidates_path)
    print("saved best:", best_path)

    print("\n===== BEST CONFIG =====")
    print(json.dumps(best, ensure_ascii=False, indent=2))

    print("\n===== TOP 10 =====")
    show_cols = [
        "alpha", "beta", "gamma", "t_rag", "t_graph",
        "objective", "macro_f1", "accuracy",
        "avg_retrieval_calls_used", "avg_graph_nodes", "avg_total_tokens",
        "action_direct", "action_rag", "action_graphrag",
    ]
    print(df[show_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
