"""Repair historical L1-versus-component-MAE labels using traceable data.

Observation data are reused, restricted to the same three training seeds and
ten archived test initial conditions. The old raw files remain unchanged.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"Code"))
from env.continuous_hopfield_env import ContinuousHopfieldEnv
OUT=ROOT/"logs/hopfield/revision_comparisons"
FIG=ROOT/"Manuscript/figures"
GEN=ROOT/"Manuscript/generated"
SEEDS=[42,123,1024]


def main():
    torch.set_num_threads(1)
    GEN.mkdir(exist_ok=True)
    frames=[pd.read_csv(OUT/f"evaluation/PPO_nodes{node}_seed{seed}.csv") for node in (1,2,3) for seed in SEEDS]
    df=pd.concat(frames)
    summary=df[df.scenario=="nominal"].groupby("node")[["mae","rmse","final_l1"]].mean()
    rows=[f"Node {node} & {r.mae:.4f} & {r.rmse:.4f} & {r.final_l1:.4f} \\\\" for node,r in summary.iterrows()]
    (GEN/"node_selection_rows.tex").write_text("\n".join(rows)+"\n",encoding="utf-8")
    figure,axis=plt.subplots(figsize=(6.5,3.5),constrained_layout=True)
    for node,color in ((1,"#376ab3"),(2,"#3a9150"),(3,"#c83939")):
        cache=OUT/f"node{node}_l1_curves.npy"
        if cache.exists():
            curves=np.load(cache)
        else:
            curves=[]
            for seed in SEEDS:
                model=PPO.load(ROOT/f"models/hopfield/ppo_continuous_hopfield_node{node}_final_seed_{seed}.zip",device="cpu")
                seed_curves=[]
                for test_seed in range(100,120):
                    env=ContinuousHopfieldEnv(pinning_node=node-1)
                    obs,_=env.reset(seed=test_seed)
                    values=[np.abs(env.statey-env.statex).sum()]
                    for _ in range(400):
                        action,_=model.predict(obs,deterministic=True)
                        obs,*_=env.step(action)
                        values.append(np.abs(env.statey-env.statex).sum())
                    seed_curves.append(values)
                    env.close()
                curves.append(seed_curves)
            curves=np.asarray(curves)
            np.save(cache,curves)
        means=curves.mean(axis=1)
        mean=means.mean(axis=0)
        sd=means.std(axis=0,ddof=1)
        time=np.arange(401)*.05
        axis.plot(time,mean,color=color,label=f"Node {node}")
        axis.fill_between(time,np.maximum(mean-sd,1e-8),mean+sd,color=color,alpha=.15)
    axis.set(yscale="log",xlabel="Time (s)",ylabel=r"$\|e(t)\|_1$")
    axis.legend(frameon=False)
    axis.grid(linestyle=":",alpha=.35)
    for ext in ("pdf","png"):
        figure.savefig(FIG/f"node_selection_audited.{ext}",dpi=220,bbox_inches="tight")
    plt.close(figure)
    fig,axes=plt.subplots(1,2,figsize=(9,3.4),constrained_layout=True)
    obs_rows=[]
    rows=[]
    for axis,node in zip(axes,(3,2)):
        for dim,color in ((6,"#3a9150"),(4,"#d28a29"),(2,"#376ab3")):
            data=pd.read_csv(ROOT/f"logs/hopfield/node{node}_obs{dim}_step_by_step_data.csv")
            data=data[data.train_seed.isin(SEEDS)&data.test_seed.isin(range(100,110))].copy()
            assert len(data)==3*10*401
            assert data.groupby(["train_seed","test_seed"]).step.nunique().eq(401).all()
            means=data.groupby(["train_seed","step"]).error_l1.mean().unstack("step").loc[SEEDS]
            mean=means.mean(axis=0).to_numpy()
            sd=means.std(axis=0,ddof=1).to_numpy()
            axis.plot(time,mean,color=color,label=f"{dim}D")
            axis.fill_between(time,means.min(axis=0).to_numpy(),means.max(axis=0).to_numpy(),color=color,alpha=.15)
            mae=float(data.error_l1.mean()/3.)
            final=float(data[data.step==400].error_l1.mean())
            rows.append(f"Node {node}, {dim}D & {mae:.4f} & {final:.4f} \\\\")
            obs_rows.append(dict(node=node,observation_dim=dim,mae=mae,final_l1=final,training_seeds="42;123;1024",test_seeds="100:109"))
        axis.set(yscale="log",xlabel="Time (s)",ylabel=r"$\|e(t)\|_1$",title=f"Node {node}")
        axis.legend(frameon=False)
        axis.grid(linestyle=":",alpha=.35)
    for ext in ("pdf","png"):
        fig.savefig(FIG/f"observation_comparison_audited.{ext}",dpi=220,bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(obs_rows).to_csv(OUT/"observation_metrics_audited.csv",index=False,encoding="utf-8-sig")
    (GEN/"observation_rows.tex").write_text("\n".join(rows)+"\n",encoding="utf-8")
    print(pd.DataFrame(obs_rows).to_string(index=False))


if __name__=="__main__":
    main()
