"""Cap09 replay analysis: proposed methods vs PBDOO / BDOO / Vanilla-block.

Primary metric: normalized regret = avg_regret_per_round / num_nodes,
aggregated as the sample mean over the 10 seeds of each configuration.
Compares the corrected cap09 results (complete/) against the archived
historical results (archived_runs/) and writes a Markdown report.
"""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

COMPLETE = Path("/root/replay_cap09/complete")
ARCHIVED = Path("/root/PDOM_cap09_replay_inputs/archived_runs")
REPORT = Path("/root/replay_cap09/analysis_report.md")

FILES = [f"{e}_alpha{a}_runs.csv" for e in ("impact_epsilon_and_T", "impact_topology") for a in (0, 1)]
PROPOSED = {"0.0": "PTF-PBDOO-cor1", "1.0": "PTF-PBDOO-strong"}
PBDOO = {"0.0": "PBDOO-cor1", "1.0": "PBDOO-strong"}
VANILLA = {"0.0": "PTF-PBDOO-fixed-block-optimized-eta-cor1", "1.0": "PTF-PBDOO-fixed-block-strong"}
BDOO = "BDOO"


def load_configs(path: Path):
    """Return {(algo, T, eps, n, topo): {"regrets": [...], "norm_mean": float, "std": float}}."""
    groups = defaultdict(list)
    with open(path, newline="") as h:
        for r in csv.DictReader(h):
            key = (r["algorithm"], int(r["T"]), r["epsilon"], int(r["num_nodes"]), r["topology"])
            groups[key].append(float(r["avg_regret_per_round"]))
    out = {}
    for (algo, T, eps, n, topo), regrets in groups.items():
        out[(algo, T, eps, n, topo)] = {
            "norm_mean": (sum(regrets) / len(regrets)) / n,
            "std": statistics.stdev(regrets) if len(regrets) > 1 else 0.0,
            "n_runs": len(regrets),
        }
    return out


def compare(configs, alpha):
    """Per-config comparison of the proposed method against each comparator.

    Returns {comparator_name: [ {"cfg": cfg, "prop": float, "comp": float,
    "win": bool, "reduction": float} ]} with reduction = (comp-prop)/comp.
    """
    prop_algo = PROPOSED[alpha]
    comps = {"BDOO": BDOO, "PBDOO": PBDOO[alpha], "Vanilla-block": VANILLA[alpha]}
    result = {name: [] for name in comps}
    for (algo, T, eps, n, topo), pv in configs.items():
        if algo != prop_algo:
            continue
        for name, calgo in comps.items():
            if calgo == BDOO:
                ck = (calgo, T, "", n, topo)  # BDOO is epsilon-independent, logged once per T
            else:
                ck = (calgo, T, eps, n, topo)
            cv = configs.get(ck)
            if cv is None:
                continue
            p, c = pv["norm_mean"], cv["norm_mean"]
            result[name].append({
                "cfg": (T, eps, n, topo), "prop": p, "comp": c,
                "win": p < c, "reduction": (c - p) / c if c else float("nan"),
            })
    return result


def summarize(entries):
    wins = sum(1 for e in entries if e["win"])
    reds = [e["reduction"] for e in entries]
    return {
        "wins": wins, "total": len(entries),
        "max": max(reds), "mean": sum(reds) / len(reds),
        "median": statistics.median(reds),
    }


def breakdown(entries, idx):
    """Group comparison entries by config component idx (0=T, 1=eps, 3=topo)."""
    groups = defaultdict(list)
    for e in entries:
        groups[e["cfg"][idx]].append(e)
    return {k: summarize(v) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def pct(x):
    return f"{100 * x:.2f}%"


def table_summary(comp_res):
    lines = ["| 对比 | 胜出配置数 | 胜出比例 | 最大 regret 降幅 | 平均降幅 | 中位降幅 |",
             "|---|---|---|---|---|---|"]
    for name, entries in comp_res.items():
        s = summarize(entries)
        lines.append(f"| vs {name} | {s['wins']}/{s['total']} | {pct(s['wins'] / s['total'])} "
                     f"| {pct(s['max'])} | {pct(s['mean'])} | {pct(s['median'])} |")
    return lines


def table_breakdown(comp_res, idx, label):
    lines = [f"| {label} | 对比 | 胜出 | 平均降幅 | 中位降幅 |", "|---|---|---|---|---|"]
    for name, entries in comp_res.items():
        for k, s in breakdown(entries, idx).items():
            lines.append(f"| {k} | vs {name} | {s['wins']}/{s['total']} | {pct(s['mean'])} | {pct(s['median'])} |")
    return lines


def main():
    new = {f: load_configs(COMPLETE / f) for f in FILES}
    old = {f: load_configs(ARCHIVED / f) for f in FILES}
    out = []
    out.append("# PDOM cap09 定向复跑：实验结果分析报告")
    out.append("")
    out.append("- 主指标：normalized regret = avg_regret_per_round / num_nodes（每配置 10 个 seed 的样本均值）")
    out.append("- 修正口径：BDOO / PBDOO 采用 query-feasible-cap09-v2 计分（观测反馈前对实际可行查询计分，0.9 探索半径上限）")
    out.append("- 本文算法：PTF-PBDOO-cor1（凸，alpha=0）与 PTF-PBDOO-strong（强凸，alpha=1）；对比 PBDOO、BDOO 与 Vanilla-block 基线")
    out.append("")

    advantage_change = []
    for alpha, alpha_name in (("0.0", "凸情形 (alpha=0)"), ("1.0", "强凸情形 (alpha=1)")):
        out.append(f"## {alpha_name}")
        for exp in ("impact_epsilon_and_T", "impact_topology"):
            f = f"{exp}_alpha{int(float(alpha))}_runs.csv"
            res_new = compare(new[f], alpha)
            res_old = compare(old[f], alpha)
            out.append(f"### {exp}（修正后）")
            out += table_summary(res_new)
            out.append("")
            idx_label = [(0, "T"), (1, "epsilon")] if exp == "impact_epsilon_and_T" else [(0, "T"), (3, "topology")]
            for idx, label in idx_label:
                out.append(f"按 {label} 分组：")
                out += table_breakdown(res_new, idx, label)
                out.append("")
            out.append(f"### {exp}：与历史结果的优势变化（修正后 − 历史）")
            out.append("| 对比 | 历史胜出率 | 修正后胜出率 | 历史平均降幅 | 修正后平均降幅 | 平均降幅变化（百分点） |")
            out.append("|---|---|---|---|---|---|")
            for name in res_new:
                so, sn = summarize(res_old[name]), summarize(res_new[name])
                delta = 100 * (sn["mean"] - so["mean"])
                out.append(f"| vs {name} | {pct(so['wins'] / so['total'])} | {pct(sn['wins'] / sn['total'])} "
                           f"| {pct(so['mean'])} | {pct(sn['mean'])} | {delta:+.2f} |")
                advantage_change.append((alpha_name, exp, name, delta))
            out.append("")

    # PBDOO low-SD phenomenon
    out.append("## PBDOO 的 seed 间样本标准差（低 SD 现象）")
    out.append("")
    out.append("| 文件 | PBDOO SD 中位数 | 本文算法 SD 中位数 | BDOO SD 中位数 | PBDOO SD < 本文算法 SD 的配置占比 |")
    out.append("|---|---|---|---|---|")
    for f in FILES:
        alpha = "0.0" if "alpha0" in f else "1.0"
        pa, ba = PBDOO[alpha], PROPOSED[alpha]
        cfgs = new[f]
        pbd_std = [v["std"] for k, v in cfgs.items() if k[0] == pa]
        prop_std = {(k[1], k[2], k[3], k[4]): v["std"] for k, v in cfgs.items() if k[0] == PROPOSED[alpha]}
        bdoo_std = [v["std"] for k, v in cfgs.items() if k[0] == BDOO]
        lower = sum(1 for k, v in cfgs.items()
                    if k[0] == pa and v["std"] < prop_std.get((k[1], k[2], k[3], k[4]), float("inf")))
        tot = len(pbd_std)
        out.append(f"| {f} | {statistics.median(pbd_std):.4f} | {statistics.median(prop_std.values()):.4f} "
                   f"| {statistics.median(bdoo_std):.4f} | {lower}/{tot} ({pct(lower / tot)}) |")
    out.append("")

    REPORT.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
