"""Produce checked manuscript tables/figures from complete revision evaluations.

Use --traditional-only while training is in progress; the default requires all
27 learned checkpoints, all seeds and every declared comparison condition.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/hopfield/revision_comparisons"
FIGURES = ROOT / "Manuscript/figures"
GENERATED = ROOT / "Manuscript/generated"
SEEDS = [42, 123, 1024]
CONFIGS = ["1", "2", "3", "1-2", "1-3", "2-3", "1-2-3"]
KEYS = ["method", "controlled_nodes", "scenario", "mismatch", "noise_std"]
METRICS = ["mae", "rmse", "steady_l1", "final_l1", "energy_u", "saturation_fraction", "mean_post_pulse_error"]


def read_all():
    frames = [pd.read_csv(p, dtype={"controlled_nodes": str}) for p in sorted((OUT / "evaluation").glob("*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    assert not df.duplicated(KEYS+["controller_seed", "test_seed"]).any()
    for _, group in df.groupby(KEYS+["controller_seed"]):
        assert sorted(group.test_seed.tolist()) == list(range(100,120))
    assert np.isfinite(df[["mae", "energy_u", "steady_l1"]].to_numpy()).all()
    assert (df.energy_u >= 0).all()
    for _, group in df.groupby(["scenario","mismatch","noise_std","test_seed"]):
        assert group.initial_x.nunique() == 1 and group.initial_y.nunique() == 1
    df.to_csv(OUT / "raw_metrics.csv", index=False, encoding="utf-8-sig")
    return df


def summaries(df):
    seedmeans = df.groupby(KEYS+["controller_seed"], as_index=False)[METRICS].mean()
    seedmeans.to_csv(OUT / "per_seed_metrics.csv", index=False, encoding="utf-8-sig")
    rows = []
    for key, group in seedmeans.groupby(KEYS):
        item = dict(zip(KEYS, key))
        item["num_controller_seeds"] = len(group)
        item["num_test_initial_conditions"] = 20
        for metric in METRICS:
            values = group[metric].dropna().to_numpy()
            if len(values):
                item[metric+"_mean"] = float(values.mean())
                item[metric+"_seed_sd"] = float(values.std(ddof=1)) if len(values)>1 else np.nan
        rows.append(item)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    return seedmeans, summary


def traditional_audit(df):
    old = pd.read_csv(ROOT / "logs/hopfield/traditional_baselines_k8_final/raw_metrics.csv")
    old.loc[(old.scenario=="mismatch") & (old.mismatch==0), "scenario"] = "nominal"
    old = old[~((old.scenario=="noise") & (old.noise_std==0))].copy()
    keys = ["method","node","controller_seed","test_seed","scenario","mismatch","noise_std"]
    baselines = df[df.node.isin([2,3]) & df.method.isin(["PPO","Linear","LQR","SMC"])]
    joined = old.merge(baselines, on=keys, suffixes=("_old","_new"), validate="one_to_one")
    assert len(joined) == len(old) == 2160
    differences = {}
    for metric in METRICS:
        delta = (joined[metric+"_old"]-joined[metric+"_new"]).abs().max()
        assert delta < 1e-9, (metric,delta)
        differences[metric] = float(delta)
    audit = dict(evaluated_episodes=len(df), learned_checkpoints=int(df[df.method.isin(["PPO","SAC"])][["method","controlled_nodes","controller_seed"]].drop_duplicates().shape[0]), reproduced_archived_episodes=len(joined), max_absolute_differences=differences,
                 state_guard_hits=int(df.state_guard_hits.sum()),
                 same_initial_conditions_within_scenario=True,
                 metric_samples=401, control_updates=400, steady_samples=81, post_pulse_samples=201)
    (OUT / "result_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    # Keep the existing figure layout, but clarify the fixed-gain legend.
    import finalize_hopfield_traditional_baselines as oldplots
    old_summary = list(csv.DictReader((ROOT / "logs/hopfield/traditional_baselines_k8_final/metrics_summary.csv").open(encoding="utf-8-sig")))
    oldplots.plot_mismatch(old_summary, FIGURES / "mismatch_methods")
    oldplots.plot_pulse(old_summary, FIGURES / "pulse_methods")
    oldplots.plot_noise(old_summary, FIGURES / "noise_methods")
    print("Archived baseline metrics reproduced exactly", flush=True)


def seed_pair(summary, method, config, scenario, metric):
    selected = summary[(summary.method==method)&(summary.controlled_nodes==config)&(summary.scenario==scenario)]
    if scenario == "mismatch":
        selected = selected[selected.mismatch==2]
    if scenario == "noise":
        selected = selected[selected.noise_std==.05]
    assert len(selected)==1
    row = selected.iloc[0]
    return float(row[metric+"_mean"]), float(row[metric+"_seed_sd"])


def fmt(pair, digits=4):
    mean, sd = pair
    return f"${mean:.{digits}f} \\pm {sd:.{digits}f}$"


def save_figure(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES/f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def sac_outputs(seedmeans, summary):
    scenarios = ["nominal","mismatch","pulse","noise"]
    names = ["Nominal",r"Mismatch ($\eta=2$)","Impulse",r"Noise ($\sigma=.05$)"]
    lines = [r"\begin{table}[pos=htbp,width=\linewidth]", r"\centering\small",
             r"\setlength{\tabcolsep}{3pt}",
             r"\caption{PPO and SAC at the same budget of 401,408 environment transitions. Values are mean $\pm$ sample SD across three training-seed means, each obtained from 20 common test initial conditions. $E_{\mathrm{post}}$ is reported only for the impulse scenario.}",
             r"\label{tab:sac_comparison}",r"\begin{tabular}{@{}cllccc@{}}",r"\toprule",
             r"Node & Scenario & Method & MAE & $E_u$ & $E_{\mathrm{post}}$ \\",r"\midrule"]
    for config in ("2","3"):
        for scenario, name in zip(scenarios,["Nominal",r"$\eta=2$","Impulse",r"$\sigma=0.05$"]):
            for method in ("PPO","SAC"):
                post = fmt(seed_pair(summary,method,config,scenario,"mean_post_pulse_error")) if scenario=="pulse" else "--"
                lines.append(f"{config} & {name} & {method} & {fmt(seed_pair(summary,method,config,scenario,'mae'))} & {fmt(seed_pair(summary,method,config,scenario,'energy_u'),2)} & {post} \\\\")
        if config=="2":
            lines.append(r"\midrule")
    lines += [r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    (GENERATED / "sac_table.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")
    fig, axes = plt.subplots(2,2,figsize=(9,6),constrained_layout=True)
    for column, config in enumerate(("2","3")):
        for row, metric in enumerate(("mae","energy_u")):
            axis = axes[row,column]
            for method,offset,color in (("PPO",-.17,"#c43d3d"),("SAC",.17,"#7560a9")):
                pairs = [seed_pair(summary,method,config,s,metric) for s in scenarios]
                axis.bar(np.arange(4)+offset,[v[0] for v in pairs],.32,
                         yerr=[v[1] for v in pairs],capsize=3,label=method,color=color)
            axis.set_xticks(np.arange(4), names,rotation=18,ha="right")
            axis.set_ylabel("MAE" if metric=="mae" else r"Total control energy $E_u$")
            axis.grid(axis="y",linestyle=":",alpha=.35)
            if row==0:
                axis.set_title(f"Node {config}")
                axis.legend(frameon=False)
    save_figure(fig,"ppo_sac_matched_budget")


def multi_outputs(seedmeans, summary):
    lines = [r"\begin{table}[pos=htbp,width=\linewidth]",r"\centering\small",
             r"\setlength{\tabcolsep}{4pt}",
             r"\caption{Nominal PPO ablation over all nonempty actuator subsets. Each configuration uses three final policies and 20 common test initial conditions per policy. Values are mean $\pm$ SD across training-seed means; energy is summed over all active inputs.}",
             r"\label{tab:multi_input_comparison}",r"\begin{tabular}{@{}ccccc@{}}",r"\toprule",
             r"Controlled nodes & Inputs & MAE & Steady L1 error & $E_u$ \\",r"\midrule"]
    fig, axes = plt.subplots(1,2,figsize=(9,3.6),constrained_layout=True)
    labels = []
    colors={1:"#3075b4",2:"#d88024",3:"#368348"}
    for index, config in enumerate(CONFIGS):
        count = len(config.split("-"))
        label = r"$\{"+config.replace("-",",")+r"\}$"
        labels.append(label)
        lines.append(f"{label} & {count} & {fmt(seed_pair(summary,'PPO',config,'nominal','mae'))} & {fmt(seed_pair(summary,'PPO',config,'nominal','steady_l1'))} & {fmt(seed_pair(summary,'PPO',config,'nominal','energy_u'),2)} \\\\")
        selected = seedmeans[(seedmeans.method=="PPO")&(seedmeans.controlled_nodes==config)&(seedmeans.scenario=="nominal")]
        for axis,metric in zip(axes,("mae","energy_u")):
            values = selected.sort_values("controller_seed")[metric].to_numpy()
            axis.scatter(index+np.linspace(-.13,.13,len(values)),values,color=colors[count],s=28,alpha=.75)
            axis.scatter(index,np.mean(values),color="black",marker="_",s=220,zorder=4)
    lines += [r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    (GENERATED / "multi_input_table.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")
    for axis, ylabel in zip(axes,("MAE",r"Total control energy $E_u$")):
        axis.set_xticks(range(7),labels)
        axis.set_xlabel("Controlled nodes")
        axis.set_ylabel(ylabel)
        axis.set_yscale("log")
        axis.grid(axis="y",linestyle=":",alpha=.35)
    save_figure(fig,"multi_input_matched_budget")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--traditional-only",action="store_true")
    parser.add_argument("--multi-only",action="store_true")
    args=parser.parse_args()
    GENERATED.mkdir(parents=True,exist_ok=True)
    df=read_all()
    seedmeans,summary=summaries(df)
    traditional_audit(df)
    if args.traditional_only:
        return
    required = [("PPO",CONFIGS)] if args.multi_only else [("PPO",CONFIGS),("SAC",["2","3"])]
    for method, configs in required:
        for config in configs:
            group=df[(df.method==method)&(df.controlled_nodes==config)&(df.scenario=="nominal")]
            assert sorted(group.controller_seed.unique())==sorted(SEEDS), (method,config,"incomplete")
    if not args.multi_only:
        sac_outputs(seedmeans,summary)
    multi_outputs(seedmeans,summary)
    print(summary[(summary.scenario=="nominal")|(summary.method=="SAC")].to_string(index=False))


if __name__=="__main__":
    main()
