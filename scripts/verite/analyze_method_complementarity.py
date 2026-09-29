import argparse
import json
from pathlib import Path
from collections import Counter

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix


LABELS = ["true", "miscaptioned", "out-of-context"]


def load_jsonl(path):
    rows = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_rows(rows, method_name):
    out = {}
    for r in rows:
        sid = str(r.get("sample_id", r.get("id")))
        gold = r.get("gold_label") or r.get("label") or r.get("gold")
        pred = r.get("pred_label") or r.get("prediction") or r.get("pred") or r.get("answer")

        if gold not in LABELS or pred not in LABELS:
            continue

        out[sid] = {
            "sample_id": sid,
            f"{method_name}_gold": gold,
            f"{method_name}_pred": pred,
            f"{method_name}_correct": int(gold == pred),
            f"{method_name}_input_tokens": r.get("input_tokens", 0),
            f"{method_name}_output_tokens": r.get("output_tokens", 0),
            f"{method_name}_total_tokens": r.get("total_tokens", 0),
            f"{method_name}_retrieval_calls_used": r.get("retrieval_calls_used", 0),
            f"{method_name}_used_evidence_count": r.get("used_evidence_count", 0),
            f"{method_name}_graph_nodes": r.get("graph_nodes", 0),
            f"{method_name}_graph_edges": r.get("graph_edges", 0),
            f"{method_name}_latency_sec": r.get("inference_latency_sec", 0),
        }
    return out


def safe_mean(xs):
    vals = []
    for x in xs:
        try:
            vals.append(float(x))
        except Exception:
            pass
    return sum(vals) / len(vals) if vals else 0.0


def compute_metrics(y_true, y_pred, prefix):
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
        f"{prefix}_accuracy": acc,
        f"{prefix}_macro_precision": p_macro,
        f"{prefix}_macro_recall": r_macro,
        f"{prefix}_macro_f1": f1_macro,
        f"{prefix}_weighted_precision": p_weighted,
        f"{prefix}_weighted_recall": r_weighted,
        f"{prefix}_weighted_f1": f1_weighted,
        f"{prefix}_f1_true": f1_cls[0],
        f"{prefix}_f1_miscaptioned": f1_cls[1],
        f"{prefix}_f1_out_of_context": f1_cls[2],
        f"{prefix}_recall_out_of_context": r_cls[2],
    }


def choose_oracle_pred(row, methods, priority):
    gold = row["gold_label"]

    # 정답을 맞힌 method가 있으면 priority 순서대로 선택
    for m in priority:
        if row.get(f"{m}_pred") == gold:
            return row[f"{m}_pred"], m

    # 아무도 못 맞히면 priority 첫 번째 method의 예측을 사용
    fallback = priority[0]
    return row[f"{fallback}_pred"], fallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slm_path", required=True)
    parser.add_argument("--rag_path", required=True)
    parser.add_argument("--graphrag_path", required=True)
    parser.add_argument(
        "--out_dir",
        default="outputs/VERITE/full_usable/main_comparison/diagnostics",
    )
    parser.add_argument("--tag", default="llama32_3b_test")
    args = parser.parse_args()

    method_paths = {
        "slm_only": args.slm_path,
        "always_rag": args.rag_path,
        "always_graphrag": args.graphrag_path,
    }

    maps = {
        m: normalize_rows(load_jsonl(p), m)
        for m, p in method_paths.items()
    }

    common_ids = set.intersection(*[set(x.keys()) for x in maps.values()])
    common_ids = sorted(common_ids, key=lambda x: int(x) if x.isdigit() else x)

    merged = []
    for sid in common_ids:
        row = {"sample_id": sid}
        gold_values = []

        for m in method_paths:
            row.update(maps[m][sid])
            gold_values.append(maps[m][sid][f"{m}_gold"])

        if len(set(gold_values)) != 1:
            row["gold_label"] = gold_values[0]
            row["gold_mismatch"] = 1
        else:
            row["gold_label"] = gold_values[0]
            row["gold_mismatch"] = 0

        row["slm_only_correct"] = int(row["slm_only_pred"] == row["gold_label"])
        row["always_rag_correct"] = int(row["always_rag_pred"] == row["gold_label"])
        row["always_graphrag_correct"] = int(row["always_graphrag_pred"] == row["gold_label"])

        row["any_correct"] = int(
            row["slm_only_correct"] or row["always_rag_correct"] or row["always_graphrag_correct"]
        )
        row["all_correct"] = int(
            row["slm_only_correct"] and row["always_rag_correct"] and row["always_graphrag_correct"]
        )
        row["none_correct"] = int(not row["any_correct"])

        row["slm_wrong_rag_correct"] = int((not row["slm_only_correct"]) and row["always_rag_correct"])
        row["slm_wrong_graphrag_correct"] = int((not row["slm_only_correct"]) and row["always_graphrag_correct"])
        row["slm_wrong_any_augmented_correct"] = int(
            (not row["slm_only_correct"]) and (row["always_rag_correct"] or row["always_graphrag_correct"])
        )

        row["slm_correct_rag_wrong"] = int(row["slm_only_correct"] and (not row["always_rag_correct"]))
        row["slm_correct_graphrag_wrong"] = int(row["slm_only_correct"] and (not row["always_graphrag_correct"]))
        row["slm_correct_any_augmented_wrong"] = int(
            row["slm_only_correct"] and ((not row["always_rag_correct"]) or (not row["always_graphrag_correct"]))
        )

        oracle_pred, oracle_action = choose_oracle_pred(
            row,
            methods=["slm_only", "always_rag", "always_graphrag"],
            priority=["slm_only", "always_rag", "always_graphrag"],
        )
        row["oracle_pred"] = oracle_pred
        row["oracle_action"] = oracle_action

        # augmented를 우선 선택하는 oracle도 함께 계산
        oracle_aug_pred, oracle_aug_action = choose_oracle_pred(
            row,
            methods=["slm_only", "always_rag", "always_graphrag"],
            priority=["always_rag", "always_graphrag", "slm_only"],
        )
        row["oracle_augmented_first_pred"] = oracle_aug_pred
        row["oracle_augmented_first_action"] = oracle_aug_action

        merged.append(row)

    df = pd.DataFrame(merged)

    y_true = df["gold_label"].tolist()

    summary = {
        "tag": args.tag,
        "samples_common": int(len(df)),
        "gold_mismatch_count": int(df["gold_mismatch"].sum()),
        "paths": method_paths,
        "correctness_counts": {
            "slm_only_correct": int(df["slm_only_correct"].sum()),
            "always_rag_correct": int(df["always_rag_correct"].sum()),
            "always_graphrag_correct": int(df["always_graphrag_correct"].sum()),
            "any_correct_oracle": int(df["any_correct"].sum()),
            "all_correct": int(df["all_correct"].sum()),
            "none_correct": int(df["none_correct"].sum()),
            "slm_wrong_rag_correct": int(df["slm_wrong_rag_correct"].sum()),
            "slm_wrong_graphrag_correct": int(df["slm_wrong_graphrag_correct"].sum()),
            "slm_wrong_any_augmented_correct": int(df["slm_wrong_any_augmented_correct"].sum()),
            "slm_correct_rag_wrong": int(df["slm_correct_rag_wrong"].sum()),
            "slm_correct_graphrag_wrong": int(df["slm_correct_graphrag_wrong"].sum()),
            "slm_correct_any_augmented_wrong": int(df["slm_correct_any_augmented_wrong"].sum()),
        },
        "oracle_action_counts_slm_first": dict(Counter(df["oracle_action"])),
        "oracle_action_counts_augmented_first": dict(Counter(df["oracle_augmented_first_action"])),
    }

    # method별 metric
    for m in ["slm_only", "always_rag", "always_graphrag"]:
        y_pred = df[f"{m}_pred"].tolist()
        summary.update(compute_metrics(y_true, y_pred, m))

    summary.update(compute_metrics(y_true, df["oracle_pred"].tolist(), "oracle_slm_first"))
    summary.update(compute_metrics(y_true, df["oracle_augmented_first_pred"].tolist(), "oracle_augmented_first"))

    # label별 complementarity
    label_rows = []
    for label in LABELS:
        sub = df[df["gold_label"] == label]
        item = {
            "gold_label": label,
            "support": int(len(sub)),
            "slm_only_correct": int(sub["slm_only_correct"].sum()),
            "always_rag_correct": int(sub["always_rag_correct"].sum()),
            "always_graphrag_correct": int(sub["always_graphrag_correct"].sum()),
            "any_correct_oracle": int(sub["any_correct"].sum()),
            "slm_wrong_rag_correct": int(sub["slm_wrong_rag_correct"].sum()),
            "slm_wrong_graphrag_correct": int(sub["slm_wrong_graphrag_correct"].sum()),
            "slm_wrong_any_augmented_correct": int(sub["slm_wrong_any_augmented_correct"].sum()),
            "slm_correct_rag_wrong": int(sub["slm_correct_rag_wrong"].sum()),
            "slm_correct_graphrag_wrong": int(sub["slm_correct_graphrag_wrong"].sum()),
        }
        label_rows.append(item)

    label_df = pd.DataFrame(label_rows)

    # 패턴별 개수
    pattern_counts = Counter()
    for _, r in df.iterrows():
        pattern = (
            f"S{int(r['slm_only_correct'])}"
            f"_R{int(r['always_rag_correct'])}"
            f"_G{int(r['always_graphrag_correct'])}"
        )
        pattern_counts[pattern] += 1

    pattern_df = pd.DataFrame([
        {"pattern": k, "count": v}
        for k, v in sorted(pattern_counts.items())
    ])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_csv = out_dir / f"{args.tag}_method_complementarity_merged.csv"
    summary_json = out_dir / f"{args.tag}_method_complementarity_summary.json"
    label_csv = out_dir / f"{args.tag}_method_complementarity_by_label.csv"
    pattern_csv = out_dir / f"{args.tag}_method_complementarity_patterns.csv"

    df.to_csv(merged_csv, index=False)
    label_df.to_csv(label_csv, index=False)
    pattern_df.to_csv(pattern_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("saved:", merged_csv)
    print("saved:", summary_json)
    print("saved:", label_csv)
    print("saved:", pattern_csv)

    print("\n===== ORACLE / COMPLEMENTARITY SUMMARY =====")
    keys = [
        "samples_common",
        "slm_only_accuracy",
        "always_rag_accuracy",
        "always_graphrag_accuracy",
        "oracle_slm_first_accuracy",
        "oracle_augmented_first_accuracy",
        "slm_only_macro_f1",
        "always_rag_macro_f1",
        "always_graphrag_macro_f1",
        "oracle_slm_first_macro_f1",
        "oracle_augmented_first_macro_f1",
    ]
    for k in keys:
        print(f"{k}: {summary.get(k)}")

    print("\ncorrectness_counts:")
    print(json.dumps(summary["correctness_counts"], indent=2, ensure_ascii=False))

    print("\noracle_action_counts_slm_first:")
    print(json.dumps(summary["oracle_action_counts_slm_first"], indent=2, ensure_ascii=False))

    print("\nby label:")
    print(label_df.to_string(index=False))

    print("\npatterns:")
    print(pattern_df.to_string(index=False))


if __name__ == "__main__":
    main()
