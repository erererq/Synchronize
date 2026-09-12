"""Report all same-dimension observation results without model selection."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'logs/hopfield/observation_identity'


def main():
    d=json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    rows=d['rows']
    assert len(rows)==180
    assert len({(r['node'],r['subset'],r['training_seed'],r['initial_seed']) for r in rows})==180
    pre=json.loads((OUT/'preflight.json').read_text())
    prior={(r['node'],r['subset'],r['training_seed'],r['initial_seed']):r for r in pre['rows']}
    for r in rows:
        key=(r['node'],r['subset'],r['training_seed'],r['initial_seed'])
        if key in prior:
            for m in ['mae','rmse','steady_l1','final_l1','energy']:
                assert r[m]==prior[key][m]
    curves=np.load(OUT/'results_curves.npz')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(8.4,3.3),sharey=True,layout='constrained')
    palette={'full6':'#555555','original4':'#ce7b28','alternative4':'#2475a8'}
    for ax,node in zip(axes,[2,3]):
        for subset in ['full6','original4','alternative4']:
            values=[]
            for seed in d['protocol']['training_seeds']:
                v=np.mean([curves[f'{node}_{subset}_{seed}_{initial}'] for initial in d['protocol']['test_initial_seeds']],axis=0)
                values.append(v)
                ax.plot(np.arange(401)*.05,np.maximum(v,1e-5),color=palette[subset],alpha=.25,lw=.7)
            name=('6D: all coordinates' if subset=='full6' else
                  f'4D: coordinates 1,{node}' if subset=='original4' else '4D: coordinates 2,3')
            ax.plot(np.arange(401)*.05,np.maximum(np.mean(values,axis=0),1e-5),color=palette[subset],label=name,lw=1.8)
        ax.set(title=f'Actuation at Node {node}',xlabel='Time',yscale='log',ylim=(1e-4,20),xlim=(0,20))
        ax.grid(alpha=.18,which='major')
        ax.legend(fontsize=8,loc='upper right')
    axes[0].set_ylabel('Mean synchronization error (L1)')
    fig.savefig(ROOT/'Manuscript/figures/observation_identity.png',dpi=240)
    plt.close(fig)
    md=['# 同维度不同观测子集：完整结果','',
        '六个新增PPO模型均从头训练完成并全部保留。原4D和6D模型重新评价，前后各指标与预检查逐项相同；没有根据测试结果选择模型。',
        '', '每个配置3个训练重复，每个模型10个共同初值；预算名义40万步，代码记录实际401408。只改变4维观测的坐标组成；原环境动力学、奖励、动作限幅和PPO设置不变。',
        '', '|节点|观测|MAE均值±SD|末段L1|总能耗|','|---|---|---:|---:|---:|']
    for s in d['summary']:
        m=s['mae'];md.append(f"|{s['node']}|{s['subset']}|{m['mean']:.4f} ± {m['std']:.4f}|{s['steady_l1']['mean']:.4f}|{s['energy']['mean']:.2f}|")
    md.extend(['','误差条/SD以3个训练重复各自的测试均值为单位，不把30条轨迹当成30次独立训练。图中细线为各训练重复均值，粗线为三者平均；曲线为L1，不是MAE。',
        '', '限制：同维度实验能够减少“输入维度不同”的混淆，但仍包含有限预算下策略优化的影响，不能证明某个坐标在理论上必不可少或某个子集最优。原测试初值已经在前期检查中使用，本轮是预先固定设置的跟进验证，不是全新盲测集。',
        '', '精确种子、模型哈希、逐回合指标见results.json；完整误差曲线见results_curves.npz。未修改原始模型。'])
    (OUT/'RESULTS_EXPLAINED.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('\n'.join(md),flush=True)


if __name__=='__main__':main()
