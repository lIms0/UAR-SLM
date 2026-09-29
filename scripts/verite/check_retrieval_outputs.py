import argparse
import json
from pathlib import Path
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="./outputs/VERITE/hash_dataset/retrieval/retrieved_evidence_ddg.jsonl")
    args = parser.parse_args()

    path = Path(args.path)

    total = 0
    zero = 0
    errors = 0
    label_counts = Counter()
    evidence_counts = Counter()
    retrieval_calls = Counter()

    examples_zero = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            total += 1

            label_counts[row.get("label")] += 1

            n = row.get("retrieved_evidence_count", 0)
            evidence_counts[n] += 1

            c = row.get("retrieval_calls", 0)
            retrieval_calls[c] += 1

            if n == 0:
                zero += 1
                if len(examples_zero) < 10:
                    examples_zero.append({
                        "id": row.get("id"),
                        "label": row.get("label"),
                        "claim": row.get("claim"),
                        "queries": row.get("queries"),
                    })

            if row.get("retrieval_errors"):
                errors += 1

    report = {
        "path": str(path),
        "total_samples": total,
        "zero_evidence_samples": zero,
        "samples_with_retrieval_errors": errors,
        "label_counts": dict(label_counts),
        "evidence_count_distribution": dict(evidence_counts),
        "retrieval_calls_distribution": dict(retrieval_calls),
        "zero_evidence_examples": examples_zero,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
