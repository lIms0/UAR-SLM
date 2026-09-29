import argparse
import json
import re
from pathlib import Path
from collections import Counter

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


LABELS = ["informative", "not_informative"]


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def to_map(rows):
    return {str(r.get("sample_id", r.get("id"))): r for r in rows}


def clean_text(x):
    x = str(x or "").lower()
    x = re.sub(r"http\S+", " ", x)
    x = re.sub(r"@\w+", " ", x)
    x = re.sub(r"#", " ", x)
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def tokenize(x):
    stop = {
        "rt", "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
        "is", "are", "was", "were", "this", "that", "it", "as", "by", "from",
        "at", "be", "been", "has", "have", "had", "you", "your", "we", "our",
    }
    return [t for t in clean_text(x).split() if len(t) >= 3 and t not in stop]


def evidence_text(ev_row, topk=5):
    evs = ev_row.get("evidence") or []
    parts = []
    for ev in evs[:topk]:
        parts.append(str(ev.get("title") or ""))
        parts.append(str(ev.get("snippet") or ""))
    return " ".join(parts)


def overlap_features(tweet, ev_text):
    tw = set(tokenize(tweet))
    ev = set(tokenize(ev_text))
    if not tw:
        return 0, 0.0
    inter = tw & ev
    return len(inter), len(inter) / len(tw)


def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)

    p_cls, r_cls, f1_cls, support_cls = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="weighted", zero_division=0
    )

    return {
        "accuracy": acc,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "f1_informative": f1_cls[0],
        "f1_not_informative": f1_cls[1],
        "recall_informative": r_cls[0],
        "recall_not_informative": r_cls[1],
        "precision_informative": p_cls[0],
        "precision_not_informative": p_cls[1],
        "support_informative": int(support_cls[0]),
        "support_not_informative": int(support_cls[1]),
    }


def build_feature_table(slm_rows, rag_rows, evidence_rows):
    slm = to_map(slm_rows)
    rag = to_map(rag_rows)
    evmap = to_map(evidence_rows)

    rows = []

    for sid in sorted(set(slm) & set(rag), key=lambda x: int(x) if x.isdigit() else x):
        s = slm[sid]
        r = rag[sid]
        ev = evmap.get(sid, {})

        gold = s.get("gold_label")
        sp = s.get("pred_label")
        rp = r.get("pred_label")

        if gold not in LABELS or sp not in LABELS or rp not in LABELS:
            continue

        tweet = s.get("tweet_text") or ""
        ev_text = evidence_text(ev, topk=5)
        ov_count, ov_ratio = overlap_features(tweet, ev_text)

        rows.append({
            "sample_id": sid,
            "gold_label": gold,
            "slm_pred": sp,
            "rag_pred": rp,
            "slm_confidence": safe_float(s.get("confidence"), 0.0),
            "rag_confidence": safe_float(r.get("confidence"), 0.0),
            "confidence_delta": safe_float(r.get("confidence"), 0.0) - safe_float(s.get("confidence"), 0.0),
            "evidence_support": r.get("evidence_support", ""),
            "available_evidence_count": safe_float(r.get("available_evidence_count"), 0.0),
            "used_evidence_count": safe_float(r.get("used_evidence_count"), 0.0),
            "retrieval_calls_used": safe_float(r.get("retrieval_calls_used"), 0.0),
            "tweet_evidence_overlap_count": ov_count,
            "tweet_evidence_overlap_ratio": ov_ratio,
            "slm_total_tokens": safe_float(s.get("total_tokens"), 0.0),
            "rag_total_tokens": safe_float(r.get("total_tokens"), 0.0),
            "slm_latency": safe_float(s.get("inference_latency_sec"), 0.0),
            "rag_latency": safe_float(r.get("inference_latency_sec"), 0.0),
        })

    return pd.DataFrame(rows)


def simulate(df, cfg):
    y_true = []
    y_pred = []
    out_rows = []

    for _, row in df.iterrows():
        gold = row["gold_label"]
        slm_pred = row["slm_pred"]
        rag_pred = row["rag_pred"]

        final_pred = slm_pred
        action = "direct"
        rag_executed = 0
        rag_accepted = 0

        # RAG 실행 후보: SLM confidence가 낮은 경우
        if row["slm_confidence"] < cfg["t_run"]:
            rag_executed = 1

            if slm_pred == rag_pred:
                final_pred = slm_pred
                action = "rag_same_label"
            else:
                # CrisisMMD 핵심: not_informative -> informative flip만 선별적으로 허용
                is_flip_to_info = (slm_pred == "not_informative" and rag_pred == "informative")

                accept = True

                if cfg["only_flip_to_informative"]:
                    accept = accept and is_flip_to_info

                accept = accept and (row["rag_confidence"] >= cfg["t_accept"])
                accept = accept and (row["confidence_delta"] >= cfg["min_delta"])
                accept = accept and (row["available_evidence_count"] >= cfg["min_available_evidence"])
                accept = accept and (row["retrieval_calls_used"] >= cfg["min_retrieval_calls"])
                accept = accept and (row["tweet_evidence_overlap_ratio"] >= cfg["min_overlap_ratio"])
                accept = accept and (row["tweet_evidence_overlap_count"] >= cfg["min_overlap_count"])

                if accept:
                    final_pred = rag_pred
                    action = "rag_accept"
                    rag_accepted = 1
                else:
                    final_pred = slm_pred
                    action = "rag_reject"

        y_true.append(gold)
        y_pred.append(final_pred)

        out_rows.append({
            "sample_id": row["sample_id"],
            "gold_label": gold,
            "pred_label": final_pred,
            "slm_pred": slm_pred,
            "rag_pred": rag_pred,
            "slm_confidence": row["slm_confidence"],
            "rag_confidence": row["rag_confidence"],
            "confidence_delta": row["confidence_delta"],
            "available_evidence_count": row["available_evidence_count"],
            "retrieval_calls_used": row["retrieval_calls_used"],
            "tweet_evidence_overlap_count": row["tweet_evidence_overlap_count"],
            "tweet_evidence_overlap_ratio": row["tweet_evidence_overlap_ratio"],
            "adaptive_action": action,
            "rag_executed": rag_executed,
            "rag_accepted": rag_accepted,
        })

    metrics = compute_metrics(y_true, y_pred)
    metrics.update(cfg)
    metrics["valid_samples"] = len(out_rows)
    metrics["rag_executed_count"] = int(sum(x["rag_executed"] for x in out_rows))
    metrics["rag_accepted_count"] = int(sum(x["rag_accepted"] for x in out_rows))
    metrics["rag_execution_rate"] = metrics["rag_executed_count"] / max(len(out_rows), 1)
    metrics["rag_accept_rate"] = metrics["rag_accepted_count"] / max(len(out_rows), 1)
    metrics["action_counts"] = dict(Counter(x["adaptive_action"] for x in out_rows))

    return metrics, out_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slm_path", required=True)
    parser.add_argument("--rag_path", required=True)
    parser.add_argument("--evidence_path", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--cost_lambda", type=float, default=0.05)
    parser.add_argument("--out_dir", default="outputs/CrisisMMD/full_usable/main_comparison/adaptive_confidence_grid")
    args = parser.parse_args()

    slm_rows = load_jsonl(args.slm_path)
    rag_rows = load_jsonl(args.rag_path)
    evidence_rows = load_jsonl(args.evidence_path)

    df = build_feature_table(slm_rows, rag_rows, evidence_rows)

    t_run_values = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90, 1.01]
    t_accept_values = [0.0, 0.5, 0.7, 0.85, 0.9, 0.95]
    min_delta_values = [-1.0, -0.3, 0.0, 0.1, 0.3]
    min_available_values = [0, 18, 20, 22, 24, 26, 28, 30]
    min_calls_values = [0, 2, 3, 4]
    min_overlap_ratio_values = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8]
    min_overlap_count_values = [0, 3, 4, 5, 6, 7]

    candidates = []
    best = None
    best_rows = None

    for t_run in t_run_values:
        for t_accept in t_accept_values:
            for min_delta in min_delta_values:
                for min_available in min_available_values:
                    for min_calls in min_calls_values:
                        for min_ov_ratio in min_overlap_ratio_values:
                            for min_ov_count in min_overlap_count_values:
                                for only_flip in [True, False]:
                                    cfg = {
                                        "t_run": t_run,
                                        "t_accept": t_accept,
                                        "min_delta": min_delta,
                                        "min_available_evidence": min_available,
                                        "min_retrieval_calls": min_calls,
                                        "min_overlap_ratio": min_ov_ratio,
                                        "min_overlap_count": min_ov_count,
                                        "only_flip_to_informative": only_flip,
                                    }

                                    metrics, rows = simulate(df, cfg)
                                    objective = metrics["macro_f1"] - args.cost_lambda * metrics["rag_execution_rate"]
                                    metrics["objective"] = objective
                                    metrics["cost_lambda"] = args.cost_lambda
                                    candidates.append(metrics)

                                    if best is None or (
                                        metrics["objective"],
                                        metrics["macro_f1"],
                                        metrics["accuracy"],
                                        -metrics["rag_execution_rate"],
                                    ) > (
                                        best["objective"],
                                        best["macro_f1"],
                                        best["accuracy"],
                                        -best["rag_execution_rate"],
                                    ):
                                        best = metrics
                                        best_rows = rows

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_df = pd.DataFrame(candidates)
    cand_df["action_counts_json"] = cand_df["action_counts"].apply(lambda x: json.dumps(x, ensure_ascii=False))
    cand_df = cand_df.drop(columns=["action_counts"])

    cand_path = out_dir / f"{args.tag}_flip_aware_candidates.csv"
    best_path = out_dir / f"{args.tag}_flip_aware_best.json"
    best_rows_path = out_dir / f"{args.tag}_flip_aware_best_rows.csv"
    feature_path = out_dir / f"{args.tag}_flip_aware_feature_table.csv"

    cand_df.sort_values(["objective", "macro_f1", "accuracy"], ascending=False).to_csv(cand_path, index=False)
    best_path.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(best_rows).to_csv(best_rows_path, index=False)
    df.to_csv(feature_path, index=False)

    print("saved candidates:", cand_path)
    print("saved best:", best_path)
    print("saved best rows:", best_rows_path)
    print("saved feature table:", feature_path)

    print("\n===== BEST =====")
    print(json.dumps(best, indent=2, ensure_ascii=False))

    show_cols = [
        "objective", "accuracy", "macro_f1", "weighted_f1",
        "f1_informative", "f1_not_informative",
        "recall_informative", "recall_not_informative",
        "rag_executed_count", "rag_accepted_count", "rag_execution_rate",
        "t_run", "t_accept", "min_available_evidence",
        "min_retrieval_calls", "min_overlap_ratio", "min_overlap_count",
        "only_flip_to_informative",
    ]
    print("\n===== TOP 30 =====")
    print(cand_df.sort_values(["objective", "macro_f1", "accuracy"], ascending=False)[show_cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
