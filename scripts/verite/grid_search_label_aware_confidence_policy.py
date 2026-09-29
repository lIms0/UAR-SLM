import argparse
import json
from pathlib import Path
from collections import Counter

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


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
        "f1_true": f1_cls[0],
        "f1_miscaptioned": f1_cls[1],
        "f1_out_of_context": f1_cls[2],
        "recall_true": r_cls[0],
        "recall_miscaptioned": r_cls[1],
        "recall_out_of_context": r_cls[2],
    }


def support_ok(mode, support):
    support = str(support or "").lower().strip()
    if mode == "any":
        return True
    if mode == "supports_only":
        return support == "supports"
    if mode == "contradicts_only":
        return support == "contradicts"
    if mode == "supports_or_contradicts":
        return support in {"supports", "contradicts"}
    return True


def label_ok(mode, rag_pred):
    if mode == "any":
        return True
    if mode == "not_out_of_context":
        return rag_pred != "out-of-context"
    if mode == "true_only":
        return rag_pred == "true"
    if mode == "miscaptioned_only":
        return rag_pred == "miscaptioned"
    if mode == "true_or_miscaptioned":
        return rag_pred in {"true", "miscaptioned"}
    return True


def simulate(slm_rows, rag_rows, cfg):
    slm_map = to_map(slm_rows)
    rag_map = to_map(rag_rows)
    common_ids = sorted(set(slm_map) & set(rag_map), key=lambda x: int(x) if x.isdigit() else x)

    y_true, y_pred = [], []
    out_rows = []

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
        delta = rag_conf - slm_conf
        support = r.get("evidence_support")

        final_pred = slm_pred
        action = "direct"
        rag_executed = 0
        rag_accepted = 0

        if slm_conf < cfg["t_run"]:
            rag_executed = 1

            same_label = slm_pred == rag_pred

            if same_label:
                final_pred = slm_pred
                action = "rag_same_label"
            else:
                accept = True
                accept = accept and (rag_conf >= cfg["t_accept"])
                accept = accept and (delta >= cfg["min_delta"])
                accept = accept and support_ok(cfg["support_mode"], support)
                accept = accept and label_ok(cfg["rag_label_mode"], rag_pred)

                if cfg["protect_slm_ooc"] and slm_pred == "out-of-context" and rag_pred != "out-of-context":
                    accept = False

                if cfg["protect_slm_miscap"] and slm_pred == "miscaptioned" and rag_pred == "true":
                    # RAG가 true로 끌고 가는 경우가 harmful할 수 있어 더 엄격하게 제한
                    accept = accept and (rag_conf >= cfg["miscap_to_true_min_conf"])

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
            "sample_id": sid,
            "gold_label": gold,
            "slm_pred": slm_pred,
            "rag_pred": rag_pred,
            "slm_confidence": slm_conf,
            "rag_confidence": rag_conf,
            "delta": delta,
            "evidence_support": support,
            "pred_label": final_pred,
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
    parser.add_argument("--tag", required=True)
    parser.add_argument("--cost_lambda", type=float, default=0.05)
    parser.add_argument("--out_dir", default="outputs/VERITE/full_usable/main_comparison/adaptive_confidence_grid")
    args = parser.parse_args()

    slm_rows = load_jsonl(args.slm_path)
    rag_rows = load_jsonl(args.rag_path)

    t_values = [round(x / 20, 2) for x in range(0, 21)]
    delta_values = [-1.0, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2]
    support_modes = ["any", "supports_only", "supports_or_contradicts"]
    rag_label_modes = ["any", "not_out_of_context", "true_or_miscaptioned", "true_only"]

    candidates = []
    best = None
    best_rows = None

    for t_run in t_values:
        for t_accept in t_values:
            for min_delta in delta_values:
                for support_mode in support_modes:
                    for rag_label_mode in rag_label_modes:
                        for protect_slm_ooc in [False, True]:
                            for protect_slm_miscap in [False, True]:
                                for miscap_to_true_min_conf in [0.0, 0.5, 0.7, 0.85]:
                                    cfg = {
                                        "t_run": t_run,
                                        "t_accept": t_accept,
                                        "min_delta": min_delta,
                                        "support_mode": support_mode,
                                        "rag_label_mode": rag_label_mode,
                                        "protect_slm_ooc": protect_slm_ooc,
                                        "protect_slm_miscap": protect_slm_miscap,
                                        "miscap_to_true_min_conf": miscap_to_true_min_conf,
                                    }

                                    metrics, rows = simulate(slm_rows, rag_rows, cfg)

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

    cand_path = out_dir / f"{args.tag}_label_aware_candidates.csv"
    best_path = out_dir / f"{args.tag}_label_aware_best.json"
    best_rows_path = out_dir / f"{args.tag}_label_aware_best_rows.csv"

    cand_df.sort_values(["objective", "macro_f1", "accuracy"], ascending=False).to_csv(cand_path, index=False)
    best_path.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(best_rows).to_csv(best_rows_path, index=False)

    print("saved candidates:", cand_path)
    print("saved best:", best_path)
    print("saved best rows:", best_rows_path)
    print("\n===== BEST =====")
    print(json.dumps(best, indent=2, ensure_ascii=False))

    show_cols = [
        "objective", "accuracy", "macro_f1", "weighted_f1",
        "f1_true", "f1_miscaptioned", "f1_out_of_context",
        "recall_out_of_context",
        "rag_executed_count", "rag_accepted_count", "rag_execution_rate",
        "t_run", "t_accept", "min_delta", "support_mode", "rag_label_mode",
        "protect_slm_ooc", "protect_slm_miscap", "miscap_to_true_min_conf",
    ]
    print("\n===== TOP 20 =====")
    print(cand_df.sort_values(["objective", "macro_f1", "accuracy"], ascending=False)[show_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
