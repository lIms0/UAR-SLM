import json
import re
from pathlib import Path
from collections import Counter

import pandas as pd


slm_path = Path("outputs/CrisisMMD/full_usable/main_comparison/slm_only/llama32_3b_slm_only_confidence_dev.jsonl")
rag_path = Path("outputs/CrisisMMD/full_usable/main_comparison/always_rag/llama32_3b_always_rag_confidence_dev.jsonl")
evidence_path = Path("data/CrisisMMD/processed/full_usable/splits/retrieved_evidence_brave_q4_k10_dev.jsonl")


def load_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def to_map(rows):
    return {str(r.get("sample_id", r.get("id"))): r for r in rows}


def clean_text(x):
    x = str(x or "").lower()
    x = re.sub(r"http\S+", " ", x)
    x = re.sub(r"@\w+", " ", x)
    x = re.sub(r"#", " ", x)
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def tokenize(x):
    stop = {
        "rt", "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
        "is", "are", "was", "were", "this", "that", "it", "as", "by", "from",
        "at", "be", "been", "has", "have", "had", "you", "your", "we", "our"
    }
    return [t for t in clean_text(x).split() if len(t) >= 3 and t not in stop]


def evidence_text(ev_row, topk=5):
    evs = ev_row.get("evidence") or []
    parts = []
    for ev in evs[:topk]:
        parts.append(str(ev.get("title") or ""))
        parts.append(str(ev.get("snippet") or ""))
    return " ".join(parts)


def overlap_features(tweet, ev_text):
    tw = set(tokenize(tweet))
    ev = set(tokenize(ev_text))
    if not tw:
        return 0, 0.0
    inter = tw & ev
    return len(inter), len(inter) / len(tw)


def has_any(text, words):
    t = clean_text(text)
    return int(any(w in t for w in words))


slm = to_map(load_jsonl(slm_path))
rag = to_map(load_jsonl(rag_path))
evmap = to_map(load_jsonl(evidence_path))

rows = []

for sid in sorted(set(slm) & set(rag)):
    s = slm[sid]
    r = rag[sid]
    ev = evmap.get(sid, {})

    gold = s.get("gold_label")
    sp = s.get("pred_label")
    rp = r.get("pred_label")

    tweet = s.get("tweet_text") or ""
    ev_text = evidence_text(ev, topk=5)
    ov_count, ov_ratio = overlap_features(tweet, ev_text)

    slm_correct = int(sp == gold)
    rag_correct = int(rp == gold)

    group = "other"
    if (not slm_correct) and rag_correct:
        group = "rag_helpful"
    elif slm_correct and (not rag_correct):
        group = "rag_harmful"
    elif slm_correct and rag_correct:
        group = "both_correct"
    elif (not slm_correct) and (not rag_correct):
        group = "both_wrong"

    text_clean = clean_text(tweet)

    rows.append({
        "sample_id": sid,
        "gold_label": gold,
        "slm_pred": sp,
        "rag_pred": rp,
        "group": group,

        "slm_confidence": s.get("confidence"),
        "rag_confidence": r.get("confidence"),
        "confidence_delta": (r.get("confidence") or 0) - (s.get("confidence") or 0),
        "evidence_support": r.get("evidence_support"),

        "tweet_len_chars": len(tweet),
        "tweet_len_words": len(tokenize(tweet)),
        "tweet_url_count": len(re.findall(r"http\S+", tweet)),
        "tweet_mention_count": len(re.findall(r"@\w+", tweet)),
        "tweet_has_rt": int(str(tweet).strip().lower().startswith("rt ")),
        "tweet_has_photo_word": has_any(tweet, ["photo", "photos", "picture", "video", "watch"]),
        "tweet_has_damage_word": has_any(tweet, ["damage", "damaged", "destroyed", "death", "dead", "killed", "injured", "missing", "evacuated", "rescue", "relief"]),
        "tweet_has_emotion_word": has_any(tweet, ["pray", "prayers", "terrifying", "sad", "omg", "wow", "crazy", "thoughts"]),

        "available_evidence_count": r.get("available_evidence_count"),
        "used_evidence_count": r.get("used_evidence_count"),
        "retrieval_calls_used": r.get("retrieval_calls_used"),
        "tweet_evidence_overlap_count": ov_count,
        "tweet_evidence_overlap_ratio": ov_ratio,
    })

df = pd.DataFrame(rows)

out_path = Path("outputs/CrisisMMD/full_usable/main_comparison/diagnostics/rag_flip_feature_table_dev.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)

print("saved:", out_path)
print("rows:", len(df))

focus = df[
    (df["slm_pred"] == "not_informative") &
    (df["rag_pred"] == "informative")
].copy()

print("\n===== flip subset: SLM not_informative -> RAG informative =====")
print("n:", len(focus))
print("group:", focus["group"].value_counts().to_dict())
print("gold:", focus["gold_label"].value_counts().to_dict())

features = [
    "slm_confidence",
    "rag_confidence",
    "confidence_delta",
    "tweet_len_chars",
    "tweet_len_words",
    "tweet_url_count",
    "tweet_mention_count",
    "tweet_has_rt",
    "tweet_has_photo_word",
    "tweet_has_damage_word",
    "tweet_has_emotion_word",
    "available_evidence_count",
    "used_evidence_count",
    "retrieval_calls_used",
    "tweet_evidence_overlap_count",
    "tweet_evidence_overlap_ratio",
]

for g in ["rag_helpful", "rag_harmful"]:
    sub = focus[focus["group"] == g]
    print("\n====", g, "====")
    print("n:", len(sub))
    print("gold:", sub["gold_label"].value_counts().to_dict())
    print("support:", sub["evidence_support"].value_counts().to_dict())
    if len(sub) > 0:
        print(sub[features].describe().loc[["mean", "std", "min", "50%", "max"]].to_string())

print("\n===== top words in helpful/harmful tweets =====")
for g in ["rag_helpful", "rag_harmful"]:
    sub = focus[focus["group"] == g]
    cnt = Counter()
    for t in sub["sample_id"]:
        tweet = slm[str(t)].get("tweet_text") or ""
        cnt.update(tokenize(tweet))
    print("\n", g)
    print(cnt.most_common(40))
