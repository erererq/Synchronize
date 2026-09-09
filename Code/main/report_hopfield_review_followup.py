"""Validate the follow-up files and generate the manuscript LQR table."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'logs/hopfield/review_followup'


def read(path):
    with path.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def main():
    selected=json.loads((OUT/'lqr_frozen_selection.json').read_text())
    old=read(ROOT/'logs/hopfield/revision_comparisons/metrics_summary.csv')
    new=read(OUT/'lqr_test_summary.csv')
    raw=read(OUT/'lqr_test_raw.csv')
    assert len(raw)==320 and len(new)==16
    assert len({(r['node'],r['R'],r['scenario'],r['test_seed']) for r in raw})==320
    worst=0.
    tex=[r'\begin{table}[pos=htbp,width=\linewidth]',r'\centering',r'\small',
         r'\setlength{\tabcolsep}{3pt}',
         r'\caption{LQR weight sensitivity on held-out tests. Each entry is a mean over 20 initial conditions; PPO additionally averages three training seeds. The selected $R=0.01$ is frozen using separate validation seeds. MAE covers the full horizon, including the impulse scenario.}',
         r'\label{tab:lqr_weight_sensitivity}',r'\begin{tabular}{@{}clrrrrrr@{}}',r'\toprule',
         r' & & \multicolumn{2}{c}{PPO} & \multicolumn{2}{c}{LQR, $R=1$} & \multicolumn{2}{c}{LQR, $R=0.01$} \\',
         r'Node & Scenario & MAE & $E_u$ & MAE & $E_u$ & MAE & $E_u$ \\',r'\midrule']
    for node in (2,3):
        assert float(selected[str(node)])==.01
        for scenario,label in [('nominal','Nominal'),('mismatch',r'$\eta=2$'),('pulse','Impulse'),('noise',r'$\sigma=0.05$')]:
            def historical(method):
                return next(r for r in old if r['method']==method and r['controlled_nodes']==str(node)
                    and r['scenario']==scenario and (scenario!='mismatch' or float(r['mismatch'])==2.)
                    and (scenario!='noise' or float(r['noise_std'])==.05))
            ppo=historical('PPO')
            default=next(r for r in new if int(r['node'])==node and r['scenario']==scenario and float(r['R'])==1.)
            tuned=next(r for r in new if int(r['node'])==node and r['scenario']==scenario and float(r['R'])==.01)
            archived=historical('LQR')
            for key in ('mae','energy_u','steady_l1'):
                worst=max(worst,abs(float(default[key])-float(archived[key+'_mean'])))
            tex.append(f"{node} & {label} & {float(ppo['mae_mean']):.4f} & {float(ppo['energy_u_mean']):.2f} & {float(default['mae']):.4f} & {float(default['energy_u']):.2f} & {float(tuned['mae']):.4f} & {float(tuned['energy_u']):.2f} \\\\")
        if node==2:
            tex.append(r'\midrule')
    assert worst<1e-12, worst
    tex.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    (ROOT/'Manuscript/generated/lqr_weight_table.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
    guard=json.loads((OUT/'long_rollout_audit.json').read_text())
    assert guard['trajectories']==60 and guard['state_guard_hits']==0 and guard['max_residual_difference']==0
    mc=read(OUT/'mcle_steps.csv')
    assert len(mc)==12
    assert all((float(r['mcle'])<0)==(int(r['node'])==3) for r in mc)
    summary=dict(lqr_test_episodes=len(raw),lqr_validation_episodes=len(read(OUT/'lqr_validation_raw.csv')),
                 lqr_R1_max_metric_difference=worst, long_rollout=guard,
                 mcle_all_steps_signs_consistent=True)
    (OUT/'followup_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
