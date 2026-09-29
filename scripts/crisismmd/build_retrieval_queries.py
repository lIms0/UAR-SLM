import argparse
import json
import re
from pathlib import Path


EVENT_NAME_MAP = {
    "california_wildfires": "California wildfires 2017",
    "hurricane_harvey": "Hurricane Harvey 2017",
    "hurricane_irma": "Hurricane Irma 2017",
    "hurricane_maria": "Hurricane Maria 2017",
    "iraq_iran_earthquake": "Iraq Iran earthquake 2017",
    "mexico_earthquake": "Mexico earthquake 2017",
    "srilanka_floods": "Sri Lanka floods 2017",
}


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


def clean_text(text):
    text = str(text)

    # Remove URLs.
    text = re.sub(r"http\S+", " ", text)

    # Remove RT marker and @mentions. They often hurt web search quality.
    text = re.sub(r"\bRT\b", " ", text)
    text = re.sub(r"@\w+", " ", text)

    # Keep hashtag words but remove the # symbol.
    text = text.replace("#", " ")

    # Remove mojibake / non-ASCII artifacts from old Twitter text.
    text = text.encode("ascii", errors="ignore").decode("ascii", errors="ignore")

    # Keep simple search-friendly characters.
    text = re.sub(r"[^A-Za-z0-9\s\-']", " ", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def shorten(text, max_words=18):
    words = text.split()
    return " ".join(words[:max_words])


def build_queries(row):
    event_name = row.get("event_name", "")
    event_text = EVENT_NAME_MAP.get(event_name, event_name.replace("_", " "))

    tweet_text = clean_text(row.get("tweet_text") or row.get("text") or "")
    short_tweet = shorten(tweet_text, max_words=18)

    image_human = row.get("image_human")
    image_damage = row.get("image_damage")

    queries = []

    # q1: event + tweet 핵심 문장
    if short_tweet:
        queries.append(f"{event_text} {short_tweet}")

    # q2: event 중심 검색
    queries.append(f"{event_text} disaster damage victims relief")

    # q3: image_human 정보가 있으면 humanitarian category 반영
    if image_human and image_human != "None" and image_human != "not_humanitarian":
        human_text = str(image_human).replace("_", " ")
        queries.append(f"{event_text} {human_text}")

    # q4: image_damage 정보가 있으면 damage category 반영
    if image_damage and image_damage != "None":
        damage_text = str(image_damage).replace("_", " ")
        queries.append(f"{event_text} {damage_text}")

    # 중복 제거, 최대 4개
    dedup = []
    seen = set()
    for q in queries:
        q = re.sub(r"\s+", " ", q).strip()
        if q and q.lower() not in seen:
            dedup.append(q)
            seen.add(q.lower())

    return dedup[:4]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature_path",
        default="data/CrisisMMD/processed/full_usable/crisismmd_stage1_features.jsonl",
    )
    parser.add_argument(
        "--out_path",
        default="data/CrisisMMD/processed/full_usable/crisismmd_retrieval_queries_q4.jsonl",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.feature_path)

    out_rows = []
    query_counts = []

    for r in rows:
        queries = build_queries(r)
        query_counts.append(len(queries))

        out_rows.append({
            "dataset": "CrisisMMD",
            "sample_id": int(r["sample_id"]),
            "tweet_id": str(r["tweet_id"]),
            "image_id": str(r["image_id"]),
            "event_name": str(r["event_name"]),
            "gold_label": str(r["gold_label"]),
            "tweet_text": r.get("tweet_text", ""),
            "image_path": r.get("image_path", ""),
            "queries": queries,
        })

    save_jsonl(out_rows, args.out_path)

    log = {
        "feature_path": args.feature_path,
        "out_path": args.out_path,
        "rows": len(out_rows),
        "avg_queries_per_sample": sum(query_counts) / len(query_counts) if query_counts else 0,
        "min_queries": min(query_counts) if query_counts else 0,
        "max_queries": max(query_counts) if query_counts else 0,
        "note": "CrisisMMD retrieval queries use event name, cleaned tweet text, humanitarian category, and damage category.",
    }

    log_path = Path(args.out_path).with_name(Path(args.out_path).stem + "_log.json")
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

    print("saved:", args.out_path)
    print("saved log:", log_path)
    print(json.dumps(log, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
