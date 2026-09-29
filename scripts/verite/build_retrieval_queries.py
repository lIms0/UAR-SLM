import argparse
import json
import re
from pathlib import Path
from collections import Counter


DEFAULT_FEATURE_PATH = "./data/VERITE/processed/full_usable/verite_stage1_features.jsonl"
DEFAULT_OUT_PATH = "./data/VERITE/processed/full_usable/verite_retrieval_queries_q4.jsonl"
DEFAULT_LOG_PATH = "./data/VERITE/processed/full_usable/verite_retrieval_queries_q4_log.json"

MAX_QUERIES_PER_SAMPLE = 4


def clean_text(text):
    text = text or ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_image_prefix(claim):
    claim = clean_text(claim)
    claim = re.sub(r"^Image of\s+", "", claim, flags=re.I)
    claim = re.sub(r"^Photo of\s+", "", claim, flags=re.I)
    claim = re.sub(r"^Picture of\s+", "", claim, flags=re.I)
    return claim.strip()


def shorten_query(q, max_words=28):
    words = clean_text(q).split()
    return " ".join(words[:max_words])


def get_visual_caption(row):
    cues = row.get("parsed_visual_cues") or {}
    if not isinstance(cues, dict):
        return ""
    return clean_text(cues.get("generated_caption", ""))


def get_visible_text(row):
    cues = row.get("parsed_visual_cues") or {}
    if not isinstance(cues, dict):
        return ""
    return clean_text(cues.get("visible_text", ""))


STOPWORDS = {
    "image", "photo", "picture", "of", "a", "an", "the", "in", "on", "at", "during",
    "near", "from", "to", "with", "and", "or", "for", "by", "is", "are", "was",
    "were", "this", "that", "these", "those", "showing", "shows"
}


def extract_years(text):
    return re.findall(r"\b(?:19|20)\d{2}\b", text or "")


def extract_quoted_terms(text):
    text = clean_text(text)

    # simple capitalized phrase extraction
    phrases = re.findall(r"\b[A-Z][A-Za-z0-9\-\u00C0-\u017F]+(?:\s+[A-Z][A-Za-z0-9\-\u00C0-\u017F]+)*", text)

    cleaned = []
    for p in phrases:
        pp = p.strip()
        if len(pp) <= 2:
            continue
        if pp.lower() in {"image", "photo", "picture"}:
            continue
        cleaned.append(pp)

    # keep order and deduplicate
    out = []
    seen = set()
    for p in cleaned:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            out.append(p)

    return out


def extract_content_terms(text, max_terms=8):
    text = clean_text(text)
    tokens = re.findall(r"[A-Za-z0-9\u00C0-\u017F\-]+", text)

    terms = []
    for tok in tokens:
        low = tok.lower()
        if low in STOPWORDS:
            continue
        if len(low) <= 2:
            continue
        terms.append(tok)

    # keep order and deduplicate
    out = []
    seen = set()
    for t in terms:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)

    return out[:max_terms]


def quote_phrase(x):
    x = clean_text(x)
    if not x:
        return ""
    if " " in x:
        return f'"{x}"'
    return x


def build_keyword_query(claim):
    claim = remove_image_prefix(claim)
    years = extract_years(claim)
    phrases = extract_quoted_terms(claim)
    terms = extract_content_terms(claim)

    selected = []

    # 우선 고유명사/장소/조직 후보
    for p in phrases[:5]:
        selected.append(quote_phrase(p))

    # 그다음 핵심 일반 단어
    for t in terms:
        if t.lower() not in " ".join(selected).lower():
            selected.append(t)

    # 연도 추가
    for y in years:
        if y not in selected:
            selected.append(y)

    return shorten_query(" ".join(selected), max_words=18)


def build_queries(row):
    claim = clean_text(row.get("claim", ""))
    claim_no_prefix = remove_image_prefix(claim)
    visual_caption = get_visual_caption(row)
    visible_text = get_visible_text(row)

    queries = []

    keyword_query = build_keyword_query(claim)

    if keyword_query:
        queries.append({
            "query_type": "claim_keywords",
            "query": keyword_query,
        })

    if keyword_query:
        queries.append({
            "query_type": "claim_factcheck",
            "query": shorten_query(keyword_query + " fact check image", max_words=22),
        })

    # 정확 문장 검색은 너무 일반 단어에 끌릴 수 있으므로 3순위로 둠
    if claim_no_prefix:
        queries.append({
            "query_type": "claim_phrase",
            "query": shorten_query('"' + claim_no_prefix + '"', max_words=32),
        })

    # OCR이 있을 때만 visible_text query 사용
    if visible_text and visible_text.lower() not in ["unknown", "none", ""]:
        queries.append({
            "query_type": "visible_text",
            "query": shorten_query('"' + visible_text + '"', max_words=20),
        })

    # visual caption은 너무 일반적이면 제외하고, 마지막 보조 query로만 사용
    if visual_caption and len(visual_caption.split()) >= 5:
        generic_words = {"crowd", "people", "image", "photo", "scene"}
        cap_terms = set(w.lower().strip(".,") for w in visual_caption.split())
        if not cap_terms.issubset(generic_words):
            queries.append({
                "query_type": "visual_caption",
                "query": shorten_query(visual_caption, max_words=18),
            })

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for q in queries:
        qq = q["query"].strip()
        key = qq.lower()
        if qq and key not in seen:
            seen.add(key)
            deduped.append(q)

    return deduped[:MAX_QUERIES_PER_SAMPLE]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path", default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--out_path", default=DEFAULT_OUT_PATH)
    parser.add_argument("--log_path", default=DEFAULT_LOG_PATH)
    args = parser.parse_args()

    feature_path = Path(args.feature_path)
    out_path = Path(args.out_path)
    log_path = Path(args.log_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    label_counts = Counter()
    query_count_dist = Counter()

    with open(feature_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            total += 1
            label_counts[row.get("label")] += 1

            queries = build_queries(row)
            query_count_dist[len(queries)] += 1

            out = {
                "id": row.get("id"),
                "sample_id": row.get("sample_id"),
                "pair_id": row.get("pair_id"),
                "label": row.get("label"),
                "claim": row.get("claim"),
                "image_path": row.get("image_path"),
                "queries": queries,
                "query_count": len(queries),
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    log = {
        "feature_path": str(feature_path),
        "out_path": str(out_path),
        "total_samples": total,
        "label_counts": dict(label_counts),
        "max_queries_per_sample": MAX_QUERIES_PER_SAMPLE,
        "query_count_distribution": dict(query_count_dist),
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print("saved:", out_path)
    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
