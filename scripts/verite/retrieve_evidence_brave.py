import argparse
import json
import os
import re
import time
from pathlib import Path
from collections import Counter
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv


DEFAULT_QUERY_PATH = "./data/VERITE/processed/full_usable/verite_retrieval_queries_q4.jsonl"
DEFAULT_OUT_PATH = "./data/VERITE/processed/full_usable/retrieved_evidence_brave_q4_k10.jsonl"
DEFAULT_LOG_PATH = "./data/VERITE/processed/full_usable/retrieved_evidence_brave_q4_k10_log.json"

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


BAD_DOMAINS = [
    "wordow.com",
    "wordreference.com",
    "redkiwiapp.com",
    "namu.wiki",
    "engram.us",
    "wiktionary.org",
    "dictionary.com",
    "cambridge.org",
    "merriam-webster.com",
    "collinsdictionary.com",
    "reverso.net",
    "bab.la",
    "vocabulary.com",
    "translate.google",
    "google.co.jp",
    "support.microsoft.com",
    "microsoft.com",
    "play.google.com",
    "wooribank.com",
    "wooricard.com",
    "sogou.com",
    "zhihu.com",
    "windowsforum.com",
    "howtogeek.com",
    "intel.com",
    "blackbox.ai",
]


BAD_TEXT_TERMS = [
    "dictionary",
    "meaning",
    "translation",
    "translator",
    "뜻",
    "의미",
    "영어 사전",
    "번역",
    "download",
    "driver",
    "dark mode",
    "windows",
    "microsoft support",
    "google 翻訳",
]


def load_project_env():
    project_root = Path(__file__).resolve().parents[2]
    dotenv_path = project_root / ".env"
    load_dotenv(dotenv_path=dotenv_path)


def get_domain(url):
    try:
        netloc = urlparse(url or "").netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def has_cjk(text):
    text = text or ""
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            return True
        if "\u3040" <= ch <= "\u30ff":
            return True
        if "\uac00" <= ch <= "\ud7a3":
            return True
    return False


def tokenize_for_overlap(text):
    text = text or ""
    text = text.lower()
    text = re.sub(r'["“”]', " ", text)
    toks = re.findall(r"[a-z0-9][a-z0-9\-]+", text)

    stop = {
        "image", "photo", "picture", "fact", "check",
        "the", "and", "with", "during", "near", "from",
        "this", "that", "into", "onto", "a", "an",
        "in", "on", "at", "of", "to", "for", "by",
        "is", "are", "was", "were"
    }

    return [t for t in toks if t not in stop and len(t) > 2]


def evidence_relevance_score(query, title, snippet):
    q_terms = tokenize_for_overlap(query)
    text_terms = set(tokenize_for_overlap((title or "") + " " + (snippet or "")))

    if not q_terms:
        return 0.0, 0, []

    matched = [t for t in q_terms if t in text_terms]
    score = len(matched) / max(len(set(q_terms)), 1)

    return score, len(matched), matched


def reject_reason(item, query, min_overlap=2):
    title = item.get("title") or ""
    snippet = item.get("snippet") or ""
    url = item.get("url") or ""
    domain = get_domain(url)
    joined = f"{title} {snippet}".lower()

    if any(bad in domain for bad in BAD_DOMAINS):
        return "bad_domain"

    if any(term.lower() in joined for term in BAD_TEXT_TERMS):
        return "bad_title_or_snippet"

    if has_cjk(title + " " + snippet):
        return "non_english_result"

    _, n_match, _ = evidence_relevance_score(query, title, snippet)

    if n_match < min_overlap:
        return f"low_keyword_overlap:{n_match}"

    return None


def load_existing_ids(path):
    done = set()
    if not path.exists():
        return done

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                sid = row.get("sample_id", row.get("id"))
                if sid is not None:
                    done.add(str(sid))
            except Exception:
                continue

    return done


def search_brave(query, api_key, count=5, country="US", search_lang="en", min_overlap=2):
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }

    params = {
        "q": query,
        "count": min(max(count * 3, count), 20),
        "country": country,
        "search_lang": search_lang,
        "safesearch": "moderate",
        "text_decorations": "false",
    }

    response = requests.get(
        BRAVE_ENDPOINT,
        headers=headers,
        params=params,
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Brave API error: status={response.status_code}, body={response.text[:500]}"
        )

    data = response.json()
    raw_results = data.get("web", {}).get("results", []) or []

    results = []
    rejected = []

    for r in raw_results:
        item = {
            "title": r.get("title"),
            "url": r.get("url"),
            "snippet": r.get("description"),
            "source": "brave",
            "age": r.get("age"),
            "language": r.get("language"),
            "family_friendly": r.get("family_friendly"),
        }

        reason = reject_reason(item, query, min_overlap=min_overlap)

        if reason is not None:
            item["reject_reason"] = reason
            rejected.append(item)
            continue

        score, n_match, matched = evidence_relevance_score(
            query,
            item.get("title"),
            item.get("snippet"),
        )

        item["relevance_score"] = round(score, 4)
        item["keyword_overlap_count"] = n_match
        item["matched_query_terms"] = matched

        results.append(item)

        if len(results) >= count:
            break

    return results, rejected


def dedup_evidence(evidence):
    seen = set()
    out = []

    for e in evidence:
        url = e.get("url") or ""
        title = e.get("title") or ""
        key = url.strip().lower() or title.strip().lower()

        if not key:
            continue

        if key in seen:
            continue

        seen.add(key)
        out.append(e)

    return out


def main():
    load_project_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--query_path", default=DEFAULT_QUERY_PATH)
    parser.add_argument("--out_path", default=DEFAULT_OUT_PATH)
    parser.add_argument("--log_path", default=DEFAULT_LOG_PATH)
    parser.add_argument("--max_results_per_query", type=int, default=10)
    parser.add_argument("--max_queries_per_sample", type=int, default=4)
    parser.add_argument("--sleep_sec", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--country", default="US")
    parser.add_argument("--search_lang", default="en")
    parser.add_argument("--min_overlap", type=int, default=2)
    args = parser.parse_args()

    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "BRAVE_API_KEY is missing. Put BRAVE_API_KEY=... in /home/oem/UAR-SLM/.env"
        )

    query_path = Path(args.query_path)
    out_path = Path(args.out_path)
    log_path = Path(args.log_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = load_existing_ids(out_path) if args.resume else set()

    rows = []
    with open(query_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if args.limit is not None and len(rows) >= args.limit:
                break

    processed = 0
    skipped = 0
    errors = 0
    zero_evidence = 0
    total_retrieval_calls = 0
    total_latency = 0.0
    evidence_count_dist = Counter()
    rejection_reason_counts = Counter()

    mode = "a" if args.resume else "w"

    with open(out_path, mode, encoding="utf-8") as fout:
        for row in rows:
            sid = str(row.get("sample_id", row.get("id")))

            if sid in done_ids:
                skipped += 1
                continue

            processed += 1
            sample_start = time.time()
            all_evidence = []
            retrieval_errors = []
            rejected_total = []

            queries = row.get("queries", [])[:args.max_queries_per_sample]

            for qi, q in enumerate(queries):
                query = q.get("query", "")
                query_type = q.get("query_type", "")

                if not query:
                    continue

                call_start = time.time()

                try:
                    results, rejected = search_brave(
                        query=query,
                        api_key=api_key,
                        count=args.max_results_per_query,
                        country=args.country,
                        search_lang=args.search_lang,
                        min_overlap=args.min_overlap,
                    )

                    total_retrieval_calls += 1

                    for rr in rejected[:20]:
                        rejection_reason_counts[rr.get("reject_reason")] += 1
                        rejected_total.append({
                            "query": query,
                            "query_type": query_type,
                            "query_rank": qi + 1,
                            "rejected_title": rr.get("title"),
                            "rejected_url": rr.get("url"),
                            "reject_reason": rr.get("reject_reason"),
                        })

                    for rank, e in enumerate(results, start=1):
                        e["query"] = query
                        e["query_type"] = query_type
                        e["query_rank"] = qi + 1
                        e["rank"] = rank
                        e["retrieval_latency_sec"] = round(time.time() - call_start, 4)
                        all_evidence.append(e)

                except Exception as e:
                    errors += 1
                    retrieval_errors.append({
                        "query": query,
                        "query_type": query_type,
                        "error": repr(e),
                    })

                time.sleep(args.sleep_sec)

            all_evidence = dedup_evidence(all_evidence)
            evidence_count = len(all_evidence)
            evidence_count_dist[evidence_count] += 1

            if evidence_count == 0:
                zero_evidence += 1

            latency = time.time() - sample_start
            total_latency += latency

            out = {
                "id": sid,
                "sample_id": row.get("sample_id"),
                "pair_id": row.get("pair_id"),
                "label": row.get("label"),
                "claim": row.get("claim"),
                "queries": queries,

                "available_max_queries": args.max_queries_per_sample,
                "available_max_results_per_query": args.max_results_per_query,
                "available_evidence_count": evidence_count,
                "available_retrieval_backend": "brave",
                "available_pool_name": "brave_q4_k10_maximum_evidence_cache",

                "evidence": all_evidence,
                "retrieved_evidence_count": evidence_count,

                "retrieval_calls_for_cache": len(queries),
                "retrieval_backend": "brave",
                "retrieval_latency_sec_for_cache": round(latency, 4),
                "retrieval_errors": retrieval_errors,
                "rejected_evidence_preview": rejected_total[:20],
                "rejected_evidence_count": len(rejected_total),
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

    log = {
        "stage": "stage6_brave_q4_k10_maximum_evidence_cache_construction",
        "query_path": str(query_path),
        "out_path": str(out_path),
        "available_pool_name": "brave_q4_k10_maximum_evidence_cache",
        "available_retrieval_backend": "brave",
        "processed_samples": processed,
        "skipped_samples": skipped,
        "errors": errors,
        "zero_evidence_samples": zero_evidence,
        "total_retrieval_calls_for_cache": total_retrieval_calls,
        "total_latency_sec_for_cache": round(total_latency, 4),
        "avg_latency_sec_for_cache_per_sample": round(total_latency / processed, 4) if processed else 0.0,
        "evidence_count_distribution": dict(evidence_count_dist),
        "rejection_reason_counts": dict(rejection_reason_counts),
        "available_max_results_per_query": args.max_results_per_query,
        "available_max_queries": args.max_queries_per_sample,
        "country": args.country,
        "search_lang": args.search_lang,
        "min_overlap": args.min_overlap,
        "note": (
            "This file is an offline maximum evidence candidate cache. "
            "q4_k10 means up to 4 queries per sample and up to 10 filtered results per query. "
            "This is not the method-level used cost. "
            "Later methods must separately record actual retrieval calls used, used evidence count, "
            "graph nodes, graph edges, tokens, and inference latency."
        ),
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print("saved:", out_path)
    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
