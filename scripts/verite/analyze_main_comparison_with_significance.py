import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


LABELS = ["true", "miscaptioned", "out-of-context"]


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_sample_id(row):
    return str(row.get("sample_id", row.get("id", row.get("pair_id"))))


def normalize_label(x):
    if x is None:
        return None

    x = str(x).strip().lower()
    x = x.replace("_", "-")

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
    }

    return aliases.get(x, x)


def load_method_group(paths):
    """
    하나의 method가 여러 seed 파일을 가질 수 있음.
    반환값:
    [
      {sample_id: {"gold": ..., "pred": ..., "calls": ..., "tokens": ...}},
      ...
    ]
    """
    group = []

    for path in paths:
        rows = load_jsonl(path)
        m = {}

        for r in rows:
            sid = get_sample_id(r)

            gold = normalize_label(r.get("gold_label", r.get("label")))
            pred = normalize_label(r.get("pred_label", r.get("prediction")))

            if gold not in LABELS or pred not in LABELS:
                continue

            calls = float(r.get("retrieval_calls_used", 0) or 0)
            tokens = float(r.get("total_tokens", 0) or 0)

            m[sid] = {
                "gold": gold,
                "pred": pred,
                "calls": calls,
                "tokens": tokens,
            }

        group.append(m)

    return group


def macro_f1(y_true, y_pred):
    return precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )[2]


def weighted_f1(y_true, y_pred):
    return precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        average="weighted",
        zero_division=0,
    )[2]


def per_class_f1(y_true, y_pred):
    return precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        average=None,
        zero_division=0,
    )[2]


def compute_metrics_for_one_map(m, sample_ids):
    y_true = [m[sid]["gold"] for sid in sample_ids]
    y_pred = [m[sid]["pred"] for sid in sample_ids]

    f1_each = per_class_f1(y_true, y_pred)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "weighted_f1": weighted_f1(y_true, y_pred),
        "f1_true": f1_each[0],
        "f1_miscaptioned": f1_each[1],
        "f1_out_of_context": f1_each[2],
        "avg_calls": float(np.mean([m[sid]["calls"] for sid in sample_ids])),
        "avg_tokens": float(np.mean([m[sid]["tokens"] for sid in sample_ids])),
    }


def compute_group_metrics(group_maps, sample_ids):
    """
    seed가 여러 개면 metric을 seed별로 계산한 뒤 평균.
    Random-Trigger RAG 같은 경우에 사용.
    """
    metrics_list = [compute_metrics_for_one_map(m, sample_ids) for m in group_maps]

    out = {}
    for key in metrics_list[0].keys():
        vals = [x[key] for x in metrics_list]
        out[key] = float(np.mean(vals))
        out[key + "_std"] = float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0

    out["num_runs"] = len(group_maps)
    out["num_samples"] = len(sample_ids)

    return out


def bootstrap_metric(group_maps, sample_ids, indices):
    sampled_ids = [sample_ids[i] for i in indices]
    return compute_group_metrics(group_maps, sampled_ids)["macro_f1"]


def paired_bootstrap(group_a, group_b, sample_ids, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(sample_ids)

    deltas = []

    for _ in range(n_boot):
        indices = rng.integers(0, n, size=n)
        a = bootstrap_metric(group_a, sample_ids, indices)
        b = bootstrap_metric(group_b, sample_ids, indices)
        deltas.append(a - b)

    deltas = np.array(deltas)

    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])

    # A가 B보다 크다는 방향의 one-sided p-value
    p_one_sided = (np.sum(deltas <= 0) + 1) / (len(deltas) + 1)

    # 차이가 0이 아니라는 two-sided p-value
    p_two_sided = 2 * min(
        (np.sum(deltas <= 0) + 1) / (len(deltas) + 1),
        (np.sum(deltas >= 0) + 1) / (len(deltas) + 1),
    )
    p_two_sided = min(float(p_two_sided), 1.0)

    return {
        "delta_macro_f1_mean": float(np.mean(deltas)),
        "delta_macro_f1_std": float(np.std(deltas, ddof=0)),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_one_sided_A_gt_B": float(p_one_sided),
        "p_two_sided": float(p_two_sided),
        "n_boot": n_boot,
    }


def parse_method_arg(method_arg):
    """
    형식:
    MethodName=path1,path2,path3
    """
    name, paths_str = method_arg.split("=", 1)
    paths = [p.strip() for p in paths_str.split(",") if p.strip()]
    return name.strip(), paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        help="MethodName=path1,path2,... 형태로 입력",
    )
    parser.add_argument("--compare", action="append", default=[])
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    methods = {}
    raw_paths = {}

    for item in args.method:
        name, paths = parse_method_arg(item)
        raw_paths[name] = paths
        methods[name] = load_method_group(paths)

    # 모든 method와 모든 seed 파일에 공통으로 존재하는 sample_id만 사용
    common_ids = None

    for name, group in methods.items():
        for m in group:
            ids = set(m.keys())
            if common_ids is None:
                common_ids = ids
            else:
                common_ids &= ids

    sample_ids = sorted(common_ids, key=lambda x: int(x) if str(x).isdigit() else str(x))

    print("Common samples:", len(sample_ids))

    summary = {
        "tag": args.tag,
        "num_common_samples": len(sample_ids),
        "methods": {},
        "comparisons": {},
        "paths": raw_paths,
    }

    for name, group in methods.items():
        metrics = compute_group_metrics(group, sample_ids)
        summary["methods"][name] = metrics

    for comp in args.compare:
        if ">" in comp:
            a, b = [x.strip() for x in comp.split(">", 1)]
        elif "," in comp:
            a, b = [x.strip() for x in comp.split(",", 1)]
        else:
            raise ValueError(f"Invalid compare format: {comp}")

        if a not in methods or b not in methods:
            raise ValueError(f"Unknown method in comparison: {comp}")

        result = paired_bootstrap(
            methods[a],
            methods[b],
            sample_ids,
            n_boot=args.n_boot,
            seed=args.seed,
        )

        summary["comparisons"][f"{a} > {b}"] = result

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{args.tag}_summary_with_significance.json"
    md_path = out_dir / f"{args.tag}_summary_with_significance.md"

    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append(f"# {args.tag}")
    lines.append("")
    lines.append(f"- Common samples: {len(sample_ids)}")
    lines.append(f"- Bootstrap samples: {args.n_boot}")
    lines.append("")

    lines.append("## Main metrics")
    lines.append("")
    lines.append("| Method | Runs | Acc. | Macro-F1 | W-F1 | F1-T | F1-M | F1-OOC | Calls | Tokens |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for name, m in summary["methods"].items():
        lines.append(
            f"| {name} | {m['num_runs']} | "
            f"{m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['weighted_f1']:.4f} | "
            f"{m['f1_true']:.4f} | {m['f1_miscaptioned']:.4f} | {m['f1_out_of_context']:.4f} | "
            f"{m['avg_calls']:.4f} | {m['avg_tokens']:.2f} |"
        )

    lines.append("")
    lines.append("## Paired bootstrap significance")
    lines.append("")
    lines.append("| Comparison | ΔMacro-F1 | 95% CI | p(one-sided A>B) | p(two-sided) |")
    lines.append("|---|---:|---:|---:|---:|")

    for name, r in summary["comparisons"].items():
        lines.append(
            f"| {name} | {r['delta_macro_f1_mean']:.4f} | "
            f"[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}] | "
            f"{r['p_one_sided_A_gt_B']:.4f} | {r['p_two_sided']:.4f} |"
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("saved json:", out_path)
    print("saved md:", md_path)
    print("\n".join(lines))


if __name__ == "__main__":
    main()