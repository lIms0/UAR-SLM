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


def allow_support(mode, support):
    support = str(support or "").lower().strip()

    if mode == "any":
        return True
    if mode == "supports_only":
        return support == "supports"
    if mode == "supports_or_contradicts":
        return support in {"supports", "contradicts"}
    if mode == "not_insufficient":
        return support != "insufficient"
    return True


def simulate_policy(slm_rows, rag_rows, t_run, t_accept, support_mode, accept_same_label=True):
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

        # 기본은 direct
        final_pred = slm_pred
        action = "direct"
        rag_executed = False
        rag_accepted = False

        # SLM confidence가 낮은 샘플만 RAG 실행 대상
        if slm_conf < t_run:
            rag_executed = True

            same_label = (slm_pred == rag_pred)
            accept_by_conf = rag_conf >= t_accept
            accept_by_support = allow_support(support_mode, support)

            if same_label and accept_same_label:
                # 같은 label이면 RAG를 실행했더라도 최종 label은 동일
                final_pred = slm_pred
                action = "rag_same_label"
                rag_accepted = False
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
            "gold_label": gold,
            "slm_pred": slm_pred,
            "slm_confidence": slm_conf,
            "rag_pred": rag_pred,
            "rag_confidence": rag_conf,
            "evidence_support": support,
            "final_pred": final_pred,
            "action": action,
            "rag_executed": int(rag_executed),
            "rag_accepted": int(rag_accepted),
            "correct": int(final_pred == gold),
        })

    metrics = compute_metrics(y_true, y_pred)
    action_counts = Counter(x["action"] for x in out_rows)

    metrics.update({
        "valid_samples": len(out_rows),
        "t_run": t_run,
        "t_accept": t_accept,
        "support_mode": support_mode,
        "accept_same_label": accept_same_label,
        "rag_executed_count": int(sum(x["rag_executed"] for x in out_rows)),
        "rag_accepted_count": int(sum(x["rag_accepted"] for x in out_rows)),
        "rag_execution_rate": sum(x["rag_executed"] for x in out_rows) / max(len(out_rows), 1),
        "rag_accept_rate": sum(x["rag_accepted"] for x in out_rows) / max(len(out_rows), 1),
        "action_counts": dict(action_counts),
    })

    return metrics, out_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slm_path", required=True)
    parser.add_argument("--rag_path", required=True)
    parser.add_argument("--tag", default="llama32_3b_dev_confidence")
    parser.add_argument("--cost_lambda", type=float, default=0.02)
    parser.add_argument("--out_dir", default="outputs/VERITE/full_usable/main_comparison/adaptive_confidence_grid")
    args = parser.parse_args()

    slm_rows = load_jsonl(args.slm_path)
    rag_rows = load_jsonl(args.rag_path)

    t_values = [round(x / 20, 2) for x in range(0, 21)]  # 0.00, 0.05, ..., 1.00
    support_modes = ["any", "supports_only", "supports_or_contradicts"]

    candidates = []
    best = None
    best_rows = None

    for t_run in t_values:
        for t_accept in t_values:
            for support_mode in support_modes:
                metrics, rows = simulate_policy(
                    slm_rows=slm_rows,
                    rag_rows=rag_rows,
                    t_run=t_run,
                    t_accept=t_accept,
                    support_mode=support_mode,
                    accept_same_label=True,
                )

                # 1차 objective:
                # macro-F1 우선, 같은 성능이면 RAG 실행률이 낮은 정책 선호
                objective = metrics["macro_f1"] - args.cost_lambda * metrics["rag_execution_rate"]
                metrics["objective"] = objective
                metrics["cost_lambda"] = args.cost_lambda

                candidates.append(metrics)

                if best is None:
                    best = metrics
                    best_rows = rows
                else:
                    if (metrics["objective"], metrics["macro_f1"], -metrics["rag_execution_rate"]) > (
                        best["objective"], best["macro_f1"], -best["rag_execution_rate"]
                    ):
                        best = metrics
                        best_rows = rows

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_df = pd.DataFrame(candidates)

    # action_counts는 dict라 csv에서 보기 어렵기 때문에 문자열로 저장
    cand_df["action_counts_json"] = cand_df["action_counts"].apply(lambda x: json.dumps(x, ensure_ascii=False))
    cand_df = cand_df.drop(columns=["action_counts"])

    cand_path = out_dir / f"{args.tag}_selective_confidence_candidates.csv"
    best_path = out_dir / f"{args.tag}_selective_confidence_best.json"
    best_rows_path = out_dir / f"{args.tag}_selective_confidence_best_rows.csv"

    cand_df.sort_values(
        ["objective", "macro_f1", "accuracy"],
        ascending=False
    ).to_csv(cand_path, index=False)

    best_path.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(best_rows).to_csv(best_rows_path, index=False)

    print("saved candidates:", cand_path)
    print("saved best:", best_path)
    print("saved best rows:", best_rows_path)

    print("\n===== BEST CONFIG =====")
    print(json.dumps(best, indent=2, ensure_ascii=False))

    print("\n===== TOP 10 =====")
    show_cols = [
        "t_run", "t_accept", "support_mode", "objective",
        "accuracy", "macro_f1", "weighted_f1",
        "f1_true", "f1_miscaptioned", "f1_out_of_context",
        "recall_out_of_context",
        "rag_executed_count", "rag_accepted_count",
        "rag_execution_rate", "rag_accept_rate",
    ]
    print(
        cand_df.sort_values(["objective", "macro_f1", "accuracy"], ascending=False)[show_cols]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
