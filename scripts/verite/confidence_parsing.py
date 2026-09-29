import json
import re

LABELS = ["true", "miscaptioned", "out-of-context"]


def normalize_label(label):
    if label is None:
        return None

    x = str(label).strip().lower()
    x = x.replace("_", "-")

    aliases = {
        "truthful": "true",
        "real": "true",
        "match": "true",
        "matched": "true",
        "supported": "true",

        "mis-captioned": "miscaptioned",
        "mis captioned": "miscaptioned",
        "wrong-caption": "miscaptioned",
        "wrong caption": "miscaptioned",

        "out of context": "out-of-context",
        "out-of-context": "out-of-context",
        "out-context": "out-of-context",
        "ooc": "out-of-context",
        "different-context": "out-of-context",
        "different context": "out-of-context",
    }

    if x in aliases:
        return aliases[x]

    if x in LABELS:
        return x

    return None


def clean_json_text(text):
    text = str(text or "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```json\\s*", "", text)
        text = re.sub(r"^```\\s*", "", text)
        text = re.sub(r"\\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        return text[start:end + 1].strip()

    return text


def clamp_confidence(x):
    try:
        v = float(x)
    except Exception:
        return None

    if v > 1.0 and v <= 100.0:
        v = v / 100.0

    if v < 0.0:
        v = 0.0
    if v > 1.0:
        v = 1.0

    return v


def parse_confidence_prediction(text):
    raw = str(text or "").strip()

    # 1) JSON 우선 파싱
    cleaned_json = clean_json_text(raw)
    try:
        obj = json.loads(cleaned_json)
        label = normalize_label(obj.get("label") or obj.get("pred_label"))
        confidence = clamp_confidence(obj.get("confidence"))
        reason = str(obj.get("reason") or "").strip()
        evidence_support = str(obj.get("evidence_support") or "").strip().lower()

        if label in LABELS and confidence is not None:
            return {
                "pred_label": label,
                "confidence": confidence,
                "reason": reason,
                "evidence_support": evidence_support,
                "parse_success": True,
                "parse_mode": "json",
                "raw_output": raw,
            }
    except Exception:
        pass

    lower = raw.lower()

    # 2) JSON-like partial output fallback
    label = None
    label_patterns = [
        r'"label"\s*:\s*"([^"]+)"',
        r"'label'\s*:\s*'([^']+)'",
        r'label\s*[:=]\s*([a-zA-Z_\- ]+)',
        r'"pred_label"\s*:\s*"([^"]+)"',
    ]

    for pat in label_patterns:
        m = re.search(pat, raw, flags=re.IGNORECASE)
        if m:
            label = normalize_label(m.group(1).strip())
            if label in LABELS:
                break

    if label not in LABELS:
        # order matters: out-of-context before true
        if "out-of-context" in lower or "out of context" in lower or "ooc" in lower:
            label = "out-of-context"
        elif "miscaptioned" in lower or "mis-captioned" in lower or "mis captioned" in lower or "wrong caption" in lower:
            label = "miscaptioned"
        elif re.search(r'\btrue\b', lower):
            label = "true"

    confidence = None
    conf_patterns = [
        r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)',
        r"'confidence'\s*:\s*([0-9]*\.?[0-9]+)",
        r'confidence\s*[:=]\s*([0-9]*\.?[0-9]+)',
    ]

    for pat in conf_patterns:
        m = re.search(pat, raw, flags=re.IGNORECASE)
        if m:
            confidence = clamp_confidence(m.group(1))
            break

    evidence_support = ""
    support_patterns = [
        r'"evidence_support"\s*:\s*"([^"]+)"',
        r"'evidence_support'\s*:\s*'([^']+)'",
        r'evidence_support\s*[:=]\s*([a-zA-Z_\-]+)',
    ]

    for pat in support_patterns:
        m = re.search(pat, raw, flags=re.IGNORECASE)
        if m:
            evidence_support = m.group(1).strip().lower()
            break

    reason = ""
    reason_patterns = [
        r'"reason"\s*:\s*"([^"]*)',
        r"'reason'\s*:\s*'([^']*)",
        r'reason\s*[:=]\s*(.+)',
    ]

    for pat in reason_patterns:
        m = re.search(pat, raw, flags=re.IGNORECASE | re.DOTALL)
        if m:
            reason = m.group(1).strip()
            break

    return {
        "pred_label": label,
        "confidence": confidence,
        "reason": reason,
        "evidence_support": evidence_support,
        "parse_success": bool(label in LABELS and confidence is not None),
        "parse_mode": "fallback",
        "raw_output": raw,
    }

