import argparse
import json
from pathlib import Path
from collections import Counter

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix


LABELS = ["true", "miscaptioned", "out-of-context"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", required=True)
    parser.add_argument("--out_path", default=None)
    args = parser.parse_args()

    pred_path = Path(args.pred_path)

    y_true = []
    y_pred = []
    parse_success = 0
    errors = 0

    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            if "error" in row:
                errors += 1
                continue

            gold = row.get("gold_label")
            pred = row.get("pred_label")

            y_true.append(gold)
            y_pred.append(pred)

            if row.get("parse_success", False):
                parse_success += 1

    total = len(y_true)

    acc = accuracy_score(y_true, y_pred) if total else 0.0
    p, r, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0
    )

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    report = {
        "pred_path": str(pred_path),
        "total": total,
        "errors": errors,
        "parse_success": parse_success,
        "parse_success_rate": parse_success / total if total else 0.0,
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "gold_counts": dict(Counter(y_true)),
        "pred_counts": dict(Counter(y_pred)),
        "per_class": {
            label: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
                "support": int(support[i])
            }
            for i, label in enumerate(LABELS)
        },
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            zero_division=0
        )
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.out_path:
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("saved:", out_path)


if __name__ == "__main__":
    main()
