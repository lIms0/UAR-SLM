import argparse
import json
import random
from pathlib import Path
from collections import Counter

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix


LABELS = ["true", "miscaptioned", "out-of-context"]
VALID_SET = set(LABELS)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_sid(row):
    return str(row.get("sample_id", row.get("id", row.get("pair_id"))))


def to_map(rows):
    return {get_sid(r): r for r in rows}


def norm_label(x):
    if x is None:
        return None
    x = str(x).strip().lower().replace("_", "-")
    aliases = {
        "true": "true",
        "miscaptioned": "miscaptioned",
        "mis-captioned": "miscaptioned",
        "mis captioned": "miscaptioned",
        "out-of-context": "out-of-context",
        "out of context": "out-of-context",
        "ooc": "out-of-context",
    }
    return aliases.get(x, x)


def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def support_ok(mode, support):
    support = str(support or "").strip().lower()

    if mode == "any":
        return True
    if mode == "supports_only":
        return support == "supports"
    if mode == "supports_or_contradicts":
        return support in {"supports", "contradicts"}

    return True


def infer_retrieval_count_from_selective(rows):
    if not rows:
        return 0

    if "rag_executed" in rows[0]:
        return sum(int(r.get("rag_executed", 0) or 0) for r in rows)

    if "retrieval_calls_used" in rows[0]:
        return sum(1 for r in rows if safe_float(r.get("retrieval_calls_used"), 0.0) > 0)

    if "adaptive_action" in rows[0]:
        return sum(1 for r in rows if str(r.get("adaptive_action", "")).startswith("rag"))

    raise ValueError("Cannot infer retrieval count from selective reference file.")


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)

    f1_macro = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )[2]

    f1_weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="weighted", zero_division=0
    )[2]

    f1_each = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0
    )[2]

    return {
        "accuracy": acc,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "f1_true": f1_each[0],
        "f1_miscaptioned": f1_each[1],
        "f1_out_of_context": f1_each[2],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slm_path", required=True)
    parser.add_argument("--rag_path", required=True)
    parser.add_argument("--selective_ref_path", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--t_accept", type=float, default=0.0)
    parser.add_argument("--support_mode", default="any")
    parser.add_argument("--out_dir", default="outputs/VERITE/full_usable/main_comparison/random_trigger_rag")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    slm_rows = load_jsonl(args.slm_path)
    rag_rows = load_jsonl(args.rag_path)
    ref_rows = load_jsonl(args.selective_ref_path)

    slm_map = to_map(slm_rows)
    rag_map = to_map(rag_rows)
    ref_map = to_map(ref_rows)

    common_ids = sorted(
        set(slm_map) & set(rag_map) & set(ref_map),
        key=lambda x: int(x) if x.isdigit() else x,
    )

    retrieval_count = infer_retrieval_count_from_selective([ref_map[sid] for sid in common_ids])
    retrieval_count = min(retrieval_count, len(common_ids))

    triggered_ids = set(rng.sample(common_ids, retrieval_count))

    out_rows = []
    y_true = []
    y_pred = []

    for sid in common_ids:
        s = slm_map[sid]
        r = rag_map[sid]

        gold = norm_label(s.get("gold_label", s.get("label")))
        slm_pred = norm_label(s.get("pred_label"))
        rag_pred = norm_label(r.get("pred_label"))

        if gold not in VALID_SET or slm_pred not in VALID_SET or rag_pred not in VALID_SET:
            continue

        rag_executed = sid in triggered_ids
        rag_accepted = False
        action = "direct"
        final_pred = slm_pred

        if rag_executed:
            same_label = slm_pred == rag_pred
            rag_conf = safe_float(r.get("confidence"), 0.0)
            ev_support = r.get("evidence_support")

            if same_label:
                final_pred = slm_pred
                action = "rag_same_label"
            elif rag_conf >= args.t_accept and support_ok(args.support_mode, ev_support):
                final_pred = rag_pred
                rag_accepted = True
                action = "rag_accept"
            else:
                final_pred = slm_pred
                action = "rag_reject"

        y_true.append(gold)
        y_pred.append(final_pred)

        out_rows.append({
            "sample_id": sid,
            "method": "random_trigger_rag_exact_count",
            "gold_label": gold,
            "pred_label": final_pred,
            "slm_pred": slm_pred,
            "slm_confidence": safe_float(s.get("confidence"), 0.0),
            "rag_pred": rag_pred,
            "rag_confidence": safe_float(r.get("confidence"), 0.0),
            "evidence_support": r.get("evidence_support"),
            "adaptive_action": action,
            "rag_executed": int(rag_executed),
            "rag_accepted": int(rag_accepted),
            "seed": args.seed,
            "retrieval_count_matched_to_selective_perf": retrieval_count,
            "retrieval_rate_matched_to_selective_perf": retrieval_count / max(len(common_ids), 1),
            "retrieval_calls_used": int(r.get("retrieval_calls_used", 0) or 0) if rag_executed else 0,
            "used_evidence_count": int(r.get("used_evidence_count", 0) or 0) if rag_executed else 0,
            "input_tokens": int(s.get("input_tokens", 0) or 0) + (int(r.get("input_tokens", 0) or 0) if rag_executed else 0),
            "output_tokens": int(s.get("output_tokens", 0) or 0) + (int(r.get("output_tokens", 0) or 0) if rag_executed else 0),
            "total_tokens": int(s.get("total_tokens", 0) or 0) + (int(r.get("total_tokens", 0) or 0) if rag_executed else 0),
            "correct": int(final_pred == gold),
        })

    metrics = compute_metrics(y_true, y_pred)
    metrics.update({
        "tag": args.tag,
        "seed": args.seed,
        "valid_samples": len(out_rows),
        "retrieval_count": int(sum(x["rag_executed"] for x in out_rows)),
        "retrieval_rate": sum(x["rag_executed"] for x in out_rows) / max(len(out_rows), 1),
        "rag_accepted_count": int(sum(x["rag_accepted"] for x in out_rows)),
        "rag_accept_rate": sum(x["rag_accepted"] for x in out_rows) / max(len(out_rows), 1),
        "avg_retrieval_calls_used": sum(x["retrieval_calls_used"] for x in out_rows) / max(len(out_rows), 1),
        "avg_total_tokens": sum(x["total_tokens"] for x in out_rows) / max(len(out_rows), 1),
        "action_counts": dict(Counter(x["adaptive_action"] for x in out_rows)),
    })

    report = classification_report(y_true, y_pred, labels=LABELS, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_path = out_dir / f"{args.tag}_seed{args.seed}_rows.jsonl"
    summary_path = out_dir / f"{args.tag}_seed{args.seed}_summary.json"
    report_path = out_dir / f"{args.tag}_seed{args.seed}_classification_report.json"
    cm_path = out_dir / f"{args.tag}_seed{args.seed}_confusion_matrix.json"

    with rows_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    cm_path.write_text(json.dumps({"labels": LABELS, "matrix": cm}, indent=2, ensure_ascii=False), encoding="utf-8")

    print("saved rows:", rows_path)
    print("saved summary:", summary_path)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
