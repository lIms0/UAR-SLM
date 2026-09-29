import argparse
import json
from pathlib import Path
from collections import Counter

from confidence_parsing import parse_confidence_prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    in_path = Path(args.input_path)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)

            parsed = parse_confidence_prediction(r.get("raw_output", ""))

            r["pred_label"] = parsed.get("pred_label")
            r["confidence"] = parsed.get("confidence")
            r["evidence_support"] = parsed.get("evidence_support", r.get("evidence_support", ""))
            r["reason"] = parsed.get("reason", r.get("reason", ""))
            r["parse_success"] = parsed.get("parse_success", False)
            r["parse_mode"] = parsed.get("parse_mode")

            rows.append(r)

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("saved:", out_path)
    print("rows:", len(rows))
    print("parse:", Counter(r.get("parse_success") for r in rows))
    print("pred:", Counter(r.get("pred_label") for r in rows))
    confs = [r.get("confidence") for r in rows if isinstance(r.get("confidence"), (int, float))]
    print("confidence count:", len(confs))
    if confs:
        print("confidence min/avg/max:", min(confs), sum(confs)/len(confs), max(confs))


if __name__ == "__main__":
    main()
