import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


LABELS = ["true", "miscaptioned", "out-of-context"]
VALID_SET = set(LABELS)


def normalize_label(x):
    if x is None:
        return "invalid"

    x = str(x).strip().lower().replace("_", "-")

    aliases = {
        "true": "true",
        "truthful": "true",
        "real": "true",

        "miscaptioned": "miscaptioned",
        "mis-captioned": "miscaptioned",
        "mis captioned": "miscaptioned",

        "out-of-context": "out-of-context",
        "out of context": "out-of-context",
        "ooc": "out-of-context",

        "invalid": "invalid",
    }

    return aliases.get(x, x if x in VALID_SET else "invalid")


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_sid(row):
    return str(row.get("sample_id", row.get("id", row.get("pair_id"))))


def load_method(paths):
    runs = []

    for p in paths:
        rows = load_jsonl(p)
        m = {}

        for r in rows:
            sid = get_sid(r)
            gold = normalize_label(r.get("gold_label", r.get("label")))
            pred = normalize_label(r.get("pred_label", r.get("prediction")))

            if gold not in VALID_SET:
                continue

            calls = float(r.get("retrieval_calls_used", 0) or 0)
            tokens = float(r.get("total_tokens", 0) or 0)

            m[sid] = {
                "gold": gold,
                "pred": pred,
                "calls": calls,
                "tokens": tokens,
                "invalid": int(pred == "invalid"),
            }

        runs.append(m)

    return runs


def compute_one(run, ids):
    y_true = [run[i]["gold"] for i in ids]
    y_pred = [run[i]["pred"] for i in ids]

    acc = accuracy_score(y_true, y_pred)

    macro = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )[2]

    weighted = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        average="weighted",
        zero_division=0,
    )[2]

    f1_each = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        average=None,
        zero_division=0,
    )[2]

    return {
        "accuracy": acc,
        "macro_f1": macro,
        "weighted_f1": weighted,
        "f1_true": f1_each[0],
        "f1_miscaptioned": f1_each[1],
        "f1_out_of_context": f1_each[2],
        "avg_calls": float(np.mean([run[i]["calls"] for i in ids])),
        "avg_tokens": float(np.mean([run[i]["tokens"] for i in ids])),
        "invalid_count": int(sum(run[i]["invalid"] for i in ids)),
        "invalid_rate": float(np.mean([run[i]["invalid"] for i in ids])),
    }


def compute_group(runs, ids):
    vals = [compute_one(r, ids) for r in runs]
    out = {}

    for k in vals[0].keys():
        arr = np.array([v[k] for v in vals], dtype=float)
        out[k] = float(arr.mean())
        out[k + "_std"] = float(arr.std(ddof=0)) if len(arr) > 1 else 0.0

    out["num_runs"] = len(runs)
    out["num_samples"] = len(ids)
    return out


def bootstrap_group_macro_f1(runs, ids, sampled_idx):
    sampled_ids = [ids[i] for i in sampled_idx]
    return compute_group(runs, sampled_ids)["macro_f1"]


def paired_bootstrap(runs_a, runs_b, ids, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(ids)
    deltas = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a = bootstrap_group_macro_f1(runs_a, ids, idx)
        b = bootstrap_group_macro_f1(runs_b, ids, idx)
        deltas.append(a - b)

    deltas = np.array(deltas)

    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])

    p_one = (np.sum(deltas <= 0) + 1) / (len(deltas) + 1)
    p_two = 2 * min(
        (np.sum(deltas <= 0) + 1) / (len(deltas) + 1),
        (np.sum(deltas >= 0) + 1) / (len(deltas) + 1),
    )

    return {
        "delta_macro_f1": float(deltas.mean()),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_one_sided_A_gt_B": float(p_one),
        "p_two_sided": float(min(p_two, 1.0)),
        "n_boot": n_boot,
    }


def parse_method_arg(s):
    name, paths = s.split("=", 1)
    paths = [p.strip() for p in paths.split(",") if p.strip()]
    return name.strip(), paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", action="append", required=True)
    parser.add_argument("--compare", action="append", default=[])
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    methods = {}
    paths_record = {}

    for item in args.method:
        name, paths = parse_method_arg(item)
        methods[name] = load_method(paths)
        paths_record[name] = paths

    common = None
    for runs in methods.values():
        for r in runs:
            ids = set(r.keys())
            common = ids if common is None else common & ids

    ids = sorted(common, key=lambda x: int(x) if x.isdigit() else x)

    print("Common samples:", len(ids))

    summary = {
        "tag": args.tag,
        "num_common_samples": len(ids),
        "methods": {},
        "comparisons": {},
        "paths": paths_record,
    }

    for name, runs in methods.items():
        summary["methods"][name] = compute_group(runs, ids)

    for comp in args.compare:
        a, b = [x.strip() for x in comp.split(">", 1)]
        summary["comparisons"][f"{a} > {b}"] = paired_bootstrap(
            methods[a],
            methods[b],
            ids,
            args.n_boot,
            args.seed,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{args.tag}_summary.json"
    md_path = out_dir / f"{args.tag}_summary.md"

    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append(f"# {args.tag}")
    lines.append("")
    lines.append(f"- Common samples: {len(ids)}")
    lines.append(f"- Bootstrap samples: {args.n_boot}")
    lines.append("")
    lines.append("## Main metrics")
    lines.append("")
    lines.append("| Method | Runs | Acc. | Macro-F1 | W-F1 | F1-T | F1-M | F1-OOC | Calls | Tokens | Invalid |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for name, m in summary["methods"].items():
        lines.append(
            f"| {name} | {m['num_runs']} | "
            f"{m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['weighted_f1']:.4f} | "
            f"{m['f1_true']:.4f} | {m['f1_miscaptioned']:.4f} | {m['f1_out_of_context']:.4f} | "
            f"{m['avg_calls']:.4f} | {m['avg_tokens']:.2f} | {m['invalid_count']:.0f} |"
        )

    lines.append("")
    lines.append("## Paired bootstrap significance")
    lines.append("")
    lines.append("| Comparison | ΔMacro-F1 | 95% CI | p(one-sided A>B) | p(two-sided) |")
    lines.append("|---|---:|---:|---:|---:|")

    for name, r in summary["comparisons"].items():
        lines.append(
            f"| {name} | {r['delta_macro_f1']:.4f} | "
            f"[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}] | "
            f"{r['p_one_sided_A_gt_B']:.4f} | {r['p_two_sided']:.4f} |"
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("saved json:", json_path)
    print("saved md:", md_path)
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
