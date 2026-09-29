import argparse
import json
from pathlib import Path
from collections import Counter

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix


LABELS = ["true", "miscaptioned", "out-of-context"]


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def to_map(rows):
    out = {}
    for r in rows:
        sid = str(r.get("sample_id", r.get("id")))
        out[sid] = r
    return out


def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def allow_support(mode, support):
    support = str(support or "").lower().strip()
    if mode == "any":
        return True
    if mode == "supports_only":
        return support == "supports"
    if mode == "supports_or_contradicts":
        return support in {"supports", "contradicts"}
    return True


def compute_metrics(y_true, y_pred):
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
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1": f1_macro,
        "weighted_precision": p_weighted,
        "weighted_recall": r_weighted,
        "weighted_f1": f1_weighted,
        "f1_true": f1_cls[0],
        "f1_miscaptioned": f1_cls[1],
        "f1_out_of_context": f1_cls[2],
        "recall_true": r_cls[0],
        "recall_miscaptioned": r_cls[1],
        "recall_out_of_context": r_cls[2],
        "support_true": int(support_cls[0]),
        "support_miscaptioned": int(support_cls[1]),
        "support_out_of_context": int(support_cls[2]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slm_path", required=True)
    parser.add_argument("--rag_path", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--t_run", type=float, required=True)
    parser.add_argument("--t_accept", type=float, default=0.0)
    parser.add_argument("--support_mode", default="any")
    parser.add_argument("--out_dir", default="outputs/VERITE/full_usable/main_comparison/selective_confidence")
    args = parser.parse_args()

    slm_rows = load_jsonl(args.slm_path)
    rag_rows = load_jsonl(args.rag_path)

    slm_map = to_map(slm_rows)
    rag_map = to_map(rag_rows)

    common_ids = sorted(set(slm_map.keys()) & set(rag_map.keys()), key=lambda x: int(x) if x.isdigit() else x)

    out_rows = []
    y_true = []
    y_pred = []

    for sid in common_ids:
        s = slm_map[sid]
        r = rag_map[sid]

        gold = s.get("gold_label")
        slm_pred = s.get("pred_label")
        rag_pred = r.get("pred_label")

        if gold not in LABELS or slm_pred not in LABELS or rag_pred not in LABELS:
            continue

        slm_conf = safe_float(s.get("confidence"), 0.0)
        rag_conf = safe_float(r.get("confidence"), 0.0)
        support = r.get("evidence_support")

        final_pred = slm_pred
        action = "direct"
        rag_executed = False
        rag_accepted = False

        if slm_conf < args.t_run:
            rag_executed = True

            same_label = slm_pred == rag_pred
            accept_by_conf = rag_conf >= args.t_accept
            accept_by_support = allow_support(args.support_mode, support)

            if same_label:
                final_pred = slm_pred
                action = "rag_same_label"
            elif accept_by_conf and accept_by_support:
                final_pred = rag_pred
                action = "rag_accept"
                rag_accepted = True
            else:
                final_pred = slm_pred
                action = "rag_reject"

        y_true.append(gold)
        y_pred.append(final_pred)

        out_rows.append({
            "sample_id": sid,
            "method": "selective_confidence_rag",
            "gold_label": gold,
            "pred_label": final_pred,
            "slm_pred": slm_pred,
            "slm_confidence": slm_conf,
            "rag_pred": rag_pred,
            "rag_confidence": rag_conf,
            "evidence_support": support,
            "adaptive_action": action,
            "rag_executed": int(rag_executed),
            "rag_accepted": int(rag_accepted),
            "retrieval_calls_used": int(r.get("retrieval_calls_used", 0) or 0) if rag_executed else 0,
            "used_evidence_count": int(r.get("used_evidence_count", 0) or 0) if rag_executed else 0,
            "input_tokens": int(s.get("input_tokens", 0) or 0) + (int(r.get("input_tokens", 0) or 0) if rag_executed else 0),
            "output_tokens": int(s.get("output_tokens", 0) or 0) + (int(r.get("output_tokens", 0) or 0) if rag_executed else 0),
            "total_tokens": int(s.get("total_tokens", 0) or 0) + (int(r.get("total_tokens", 0) or 0) if rag_executed else 0),
            "inference_latency_sec": float(s.get("inference_latency_sec", 0) or 0) + (float(r.get("inference_latency_sec", 0) or 0) if rag_executed else 0),
            "correct": int(final_pred == gold),
        })

    metrics = compute_metrics(y_true, y_pred)
    action_counts = Counter(x["adaptive_action"] for x in out_rows)

    metrics.update({
        "tag": args.tag,
        "valid_samples": len(out_rows),
        "t_run": args.t_run,
        "t_accept": args.t_accept,
        "support_mode": args.support_mode,
        "rag_executed_count": int(sum(x["rag_executed"] for x in out_rows)),
        "rag_accepted_count": int(sum(x["rag_accepted"] for x in out_rows)),
        "rag_execution_rate": sum(x["rag_executed"] for x in out_rows) / max(len(out_rows), 1),
        "rag_accept_rate": sum(x["rag_accepted"] for x in out_rows) / max(len(out_rows), 1),
        "action_counts": dict(action_counts),
        "avg_retrieval_calls_used": sum(x["retrieval_calls_used"] for x in out_rows) / max(len(out_rows), 1),
        "avg_used_evidence_count": sum(x["used_evidence_count"] for x in out_rows) / max(len(out_rows), 1),
        "avg_total_tokens": sum(x["total_tokens"] for x in out_rows) / max(len(out_rows), 1),
        "avg_inference_latency_sec": sum(x["inference_latency_sec"] for x in out_rows) / max(len(out_rows), 1),
    })

    report = classification_report(y_true, y_pred, labels=LABELS, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_path = out_dir / f"{args.tag}_rows.jsonl"
    summary_path = out_dir / f"{args.tag}_summary.json"
    report_path = out_dir / f"{args.tag}_classification_report.json"
    cm_path = out_dir / f"{args.tag}_confusion_matrix.json"

    with rows_path.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    cm_path.write_text(json.dumps({"labels": LABELS, "matrix": cm}, indent=2, ensure_ascii=False), encoding="utf-8")

    print("saved rows:", rows_path)
    print("saved summary:", summary_path)
    print("saved report:", report_path)
    print("saved cm:", cm_path)
    print("\n===== SUMMARY =====")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
