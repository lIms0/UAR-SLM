import argparse
import json
from pathlib import Path
from collections import Counter

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

LABELS = ["informative", "not_informative"]


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def infer_method(path, rows):
    if rows and rows[0].get("method"):
        return rows[0]["method"]
    return Path(path).stem


def get_gold(row):
    return row.get("gold_label") or row.get("label")


def get_pred(row):
    return row.get("pred_label") or row.get("prediction") or row.get("pred")


def safe_mean(xs):
    vals = []
    for x in xs:
        try:
            if x is not None:
                vals.append(float(x))
        except Exception:
            pass
    return sum(vals) / len(vals) if vals else 0.0


def summarize(path):
    rows = load_jsonl(path)
    method = infer_method(path, rows)

    valid = []
    invalid = []

    for r in rows:
        gold = get_gold(r)
        pred = get_pred(r)

        if gold in LABELS and pred in LABELS:
            valid.append(r)
        else:
            invalid.append(r)

    y_true = [get_gold(r) for r in valid]
    y_pred = [get_pred(r) for r in valid]

    if not valid:
        return {
            "method": method,
            "path": str(path),
            "rows": len(rows),
            "valid_rows": 0,
            "invalid_rows": len(invalid),
        }, {}, []

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

    report = classification_report(
        y_true, y_pred, labels=LABELS, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()

    summary = {
        "method": method,
        "path": str(path),
        "rows": len(rows),
        "valid_rows": len(valid),
        "invalid_rows": len(invalid),

        "accuracy": acc,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1": f1_macro,
        "weighted_precision": p_weighted,
        "weighted_recall": r_weighted,
        "weighted_f1": f1_weighted,

        "f1_informative": f1_cls[0],
        "f1_not_informative": f1_cls[1],
        "recall_informative": r_cls[0],
        "recall_not_informative": r_cls[1],
        "precision_informative": p_cls[0],
        "precision_not_informative": p_cls[1],

        "support_informative": int(support_cls[0]),
        "support_not_informative": int(support_cls[1]),

        "gold_counts": dict(Counter(y_true)),
        "pred_counts": dict(Counter(y_pred)),

        "avg_retrieval_calls_used": safe_mean([r.get("retrieval_calls_used", 0) for r in valid]),
        "avg_used_evidence_count": safe_mean([r.get("used_evidence_count", 0) for r in valid]),
        "avg_total_tokens": safe_mean([r.get("total_tokens", 0) for r in valid]),
        "avg_inference_latency_sec": safe_mean([r.get("inference_latency_sec", 0) for r in valid]),
    }

    if any("rag_executed" in r for r in valid):
        summary["rag_execution_rate"] = safe_mean([r.get("rag_executed", 0) for r in valid])
        summary["rag_accept_rate"] = safe_mean([r.get("rag_accepted", 0) for r in valid])

    return summary, report, cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--result_paths", nargs="+", required=True)
    parser.add_argument("--out_dir", default="outputs/CrisisMMD/full_usable/main_comparison/eval")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    reports = {}
    matrices = {}

    for p in args.result_paths:
        summary, report, cm = summarize(p)
        key = summary["method"]
        if key in reports:
            key = Path(p).stem
            summary["method"] = key

        summaries.append(summary)
        reports[key] = report
        matrices[key] = {
            "labels": LABELS,
            "matrix": cm,
            "rows_are_gold_columns_are_pred": True,
        }

    df = pd.DataFrame(summaries)

    summary_csv = out_dir / f"{args.tag}_summary.csv"
    summary_json = out_dir / f"{args.tag}_summary.json"
    report_json = out_dir / f"{args.tag}_classification_reports.json"
    cm_json = out_dir / f"{args.tag}_confusion_matrices.json"

    df.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    report_json.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    cm_json.write_text(json.dumps(matrices, indent=2, ensure_ascii=False), encoding="utf-8")

    print("saved:", summary_csv)
    print("saved:", summary_json)
    print("saved:", report_json)
    print("saved:", cm_json)

    cols = [
        "method", "accuracy", "macro_f1", "weighted_f1",
        "f1_informative", "f1_not_informative",
        "recall_informative", "recall_not_informative",
        "avg_retrieval_calls_used", "avg_used_evidence_count",
        "avg_total_tokens", "avg_inference_latency_sec",
        "rag_execution_rate", "rag_accept_rate",
    ]
    cols = [c for c in cols if c in df.columns]

    print("\n===== SUMMARY =====")
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
