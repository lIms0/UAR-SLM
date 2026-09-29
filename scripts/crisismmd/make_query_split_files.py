import argparse
import json
from pathlib import Path

import pandas as pd


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_queries(row):
    queries = row.get("queries", [])
    out = []

    for i, q in enumerate(queries):
        if isinstance(q, dict):
            query_text = q.get("query", "")
            query_type = q.get("query_type", f"q{i+1}")
        else:
            query_text = str(q)
            query_type = f"q{i+1}"

        if query_text.strip():
            out.append({
                "query": query_text.strip(),
                "query_type": query_type,
            })

    row["queries"] = out
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query_path",
        default="data/CrisisMMD/processed/full_usable/crisismmd_retrieval_queries_q4.jsonl",
    )
    parser.add_argument(
        "--split_path",
        default="data/CrisisMMD/processed/full_usable/crisismmd_dev_test_split.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="data/CrisisMMD/processed/full_usable/splits",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.query_path)
    rows = [normalize_queries(r) for r in rows]

    split_df = pd.read_csv(args.split_path)

    split_map = {
        int(r.sample_id): r.split
        for r in split_df.itertuples(index=False)
    }

    by_split = {"dev": [], "test": []}

    for r in rows:
        sid = int(r["sample_id"])
        sp = split_map.get(sid)
        if sp in by_split:
            by_split[sp].append(r)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = {
        "query_path": args.query_path,
        "split_path": args.split_path,
        "out_dir": str(out_dir),
        "splits": {},
        "note": "Queries are normalized to dict format: {'query': text, 'query_type': qN}, matching VERITE retrieval script format.",
    }

    for sp, items in by_split.items():
        out_path = out_dir / f"crisismmd_retrieval_queries_q4_{sp}.jsonl"
        save_jsonl(items, out_path)

        query_counts = [len(x.get("queries", [])) for x in items]
        log["splits"][sp] = {
            "query_out": str(out_path),
            "rows": len(items),
            "avg_queries_per_sample": sum(query_counts) / len(query_counts) if query_counts else 0,
            "min_queries": min(query_counts) if query_counts else 0,
            "max_queries": max(query_counts) if query_counts else 0,
            "estimated_query_calls": int(sum(query_counts)),
        }

    log_path = out_dir / "query_split_files_log.json"
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

    print("saved log:", log_path)
    print(json.dumps(log, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
