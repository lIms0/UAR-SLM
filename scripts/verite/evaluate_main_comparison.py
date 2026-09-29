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

LABELS = ["true", "miscaptioned", "out-of-context"]


def load_jsonl(path):
    path = Path(path)
    rows = []

    if not path.exists():
        print(f"[WARN] missing file: {path}")
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def safe_mean(values):
    vals = []
    for v in values:
        try:
            if v is not None:
                vals.append(float(v))
        except Exception:
            pass
    return sum(vals) / len(vals) if vals else 0.0


def infer_run_name(path):
    return Path(path).stem


def evaluate_rows(rows, result_path):
    valid = [
        r for r in rows
        if r.get("gold_label") in LABELS and r.get("pred_label") in LABELS
    ]

    invalid = [
        r for r in rows
        if not (r.get("gold_label") in LABELS and r.get("pred_label") in LABELS)
    ]

    if not valid:
        return None, None, None

    y_true = [r["gold_label"] for r in valid]
    y_pred = [r["pred_label"] for r in valid]

    method = valid[0].get("method", "unknown")
    model_tag = valid[0].get("model_tag", "unknown")
    run_name = infer_run_name(result_path)

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

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    action_counts = Counter(
        r.get("adaptive_action")
        for r in valid
        if r.get("adaptive_action")
    )

    summary = {
        "run_name": run_name,
        "result_path": str(result_path),
        "method": method,
        "model_tag": model_tag,

        "samples": len(rows),
        "valid_predictions": len(valid),
        "invalid_predictions": len(invalid),

        "accuracy": acc,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1": f1_macro,
        "weighted_precision": p_weighted,
        "weighted_recall": r_weighted,
        "weighted_f1": f1_weighted,

        "precision_true": p_cls[0],
        "precision_miscaptioned": p_cls[1],
        "precision_out_of_context": p_cls[2],

        "recall_true": r_cls[0],
        "recall_miscaptioned": r_cls[1],
        "recall_out_of_context": r_cls[2],

        "f1_true": f1_cls[0],
        "f1_miscaptioned": f1_cls[1],
        "f1_out_of_context": f1_cls[2],

        "support_true": int(support_cls[0]),
        "support_miscaptioned": int(support_cls[1]),
        "support_out_of_context": int(support_cls[2]),

        "avg_input_tokens": safe_mean([r.get("input_tokens") for r in valid]),
        "avg_output_tokens": safe_mean([r.get("output_tokens") for r in valid]),
        "avg_total_tokens": safe_mean([r.get("total_tokens") for r in valid]),
        "avg_inference_latency_sec": safe_mean([r.get("inference_latency_sec") for r in valid]),

        "avg_retrieval_calls_used": safe_mean([r.get("retrieval_calls_used") for r in valid]),
        "avg_used_evidence_count": safe_mean([r.get("used_evidence_count") for r in valid]),
        "avg_graph_nodes": safe_mean([r.get("graph_nodes") for r in valid]),
        "avg_graph_edges": safe_mean([r.get("graph_edges") for r in valid]),
        "avg_graph_expansion_steps": safe_mean([r.get("graph_expansion_steps") for r in valid]),

        "adaptive_direct_count": action_counts.get("direct", 0),
        "adaptive_rag_count": action_counts.get("rag", 0),
        "adaptive_graphrag_count": action_counts.get("graphrag", 0),
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
        output_dict=True,
    )

    confusion = {
        "run_name": run_name,
        "method": method,
        "model_tag": model_tag,
        "labels": LABELS,
        "matrix": cm.tolist(),
        "rows_are_gold_columns_are_pred": True,
    }

    return summary, report, confusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_paths", nargs="+", required=True)
    parser.add_argument("--out_dir", default="outputs/VERITE/full_usable/main_comparison/eval")
    parser.add_argument("--tag", default="main_comparison")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    reports = {}
    confusions = {}

    for path in args.result_paths:
        rows = load_jsonl(path)
        if not rows:
            continue

        summary, report, confusion = evaluate_rows(rows, path)

        if summary is None:
            print(f"[WARN] no valid predictions: {path}")
            continue

        summaries.append(summary)
        reports[summary["run_name"]] = report
        confusions[summary["run_name"]] = confusion

    if not summaries:
        raise RuntimeError("No result files were evaluated.")

    df = pd.DataFrame(summaries)

    preferred_cols = [
        "model_tag", "method", "run_name",
        "samples", "valid_predictions", "invalid_predictions",
        "accuracy", "macro_f1", "weighted_f1",
        "precision_true", "recall_true", "f1_true",
        "precision_miscaptioned", "recall_miscaptioned", "f1_miscaptioned",
        "precision_out_of_context", "recall_out_of_context", "f1_out_of_context",
        "avg_retrieval_calls_used", "avg_used_evidence_count",
        "avg_graph_nodes", "avg_graph_edges", "avg_graph_expansion_steps",
        "avg_input_tokens", "avg_output_tokens", "avg_total_tokens",
        "avg_inference_latency_sec",
        "adaptive_direct_count", "adaptive_rag_count", "adaptive_graphrag_count",
        "result_path",
    ]

    cols = [c for c in preferred_cols if c in df.columns]
    df = df[cols]

    summary_csv = out_dir / f"{args.tag}_summary.csv"
    summary_json = out_dir / f"{args.tag}_summary.json"
    reports_json = out_dir / f"{args.tag}_classification_reports.json"
    confusions_json = out_dir / f"{args.tag}_confusion_matrices.json"

    df.to_csv(summary_csv, index=False)

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    with reports_json.open("w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)

    with confusions_json.open("w", encoding="utf-8") as f:
        json.dump(confusions, f, ensure_ascii=False, indent=2)

    print("saved:", summary_csv)
    print("saved:", summary_json)
    print("saved:", reports_json)
    print("saved:", confusions_json)

    print("\n===== SUMMARY =====")
    show_cols = [
        "model_tag", "method", "accuracy", "macro_f1", "weighted_f1",
        "f1_true", "f1_miscaptioned", "f1_out_of_context",
        "recall_out_of_context",
        "avg_retrieval_calls_used", "avg_used_evidence_count",
        "avg_graph_nodes", "avg_graph_edges", "avg_total_tokens",
        "avg_inference_latency_sec",
        "adaptive_direct_count", "adaptive_rag_count", "adaptive_graphrag_count",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    print(df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
