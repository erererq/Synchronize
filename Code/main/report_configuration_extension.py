"""Audit and summarize all predeclared configuration extensions."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'logs/hopfield/configuration_extension'
SEEDS=[42,123,1024]

def stats(rows):
    d={}
    for key in ['mae','tail_l1','energy']:
        means=[float(np.mean([r[key] for r in rows if r['seed']==s])) for s in SEEDS]
        d[key]=dict(mean=float(np.mean(means)),sd=float(np.std(means,ddof=1)),runs=means)
    return d

def main():
    paired=json.loads((OUT/'paired_results.json').read_text())
    transfer=json.loads((OUT/'transfer_results.json').read_text())
    sensitivity=json.loads((OUT/'sensitivity.json').read_text())
    assert len(paired)==360 and len(transfer)==960 and len(sensitivity)==54
    assert len({(r['node'],r['subset'],r['seed'],r['initial'],r['eta']) for r in paired+transfer})==1320
    assert not {r['initial'] for r in paired}&{r['initial'] for r in transfer}
    states={};W=np.array([[-1.4,1.2,-7],[1.1,0,2.8],[.8,-2,4]],dtype=np.float32)
    for r in paired+transfer:
        initial=r['initial'];st=tuple(r['initial_state'])
        assert initial not in states or states[initial]==st
        states[initial]=st
        wr=np.asarray(r['response_weights']);assert wr[1,1]==0
        assert np.max(np.abs((wr-W)[W!=0]/W[W!=0]))<=r['eta']+1e-6
    ps=[];ts=[]
    for n in [2,3]:
        for s in ['original4','alternative4','missing4']:
            ps.append(dict(node=n,subset=s,**stats([r for r in paired if r['node']==n and r['subset']==s])))
        for eta in [0,.01,.03,.05]:
            rr=[r for r in transfer if r['node']==n and r['eta']==eta];tol=.1 if n==2 else .05
            rates=[float(np.mean([r['tail_l1']<=tol and r['guard_hits']==0 for r in rr if r['seed']==s])) for s in SEEDS]
            ts.append(dict(node=n,eta=eta,tolerance=tol,rates=rates,**stats(rr)))
    summary=dict(paired=ps,transfer=ts,guard_hits=sum(r['guard_hits'] for r in paired+transfer),sensitivity=sensitivity)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# 三项贡献补强：结果与边界','',
    '## 1. 完整成对4维观测比较','',
    '补充Node 2观测O13、Node 3观测O12，每组3次训练，名义40万步，全部最终模型保留。将六种节点—成对观测组合在20个新的共同初值上重新评价。不是所有任意4坐标组合，也不是全部控制节点；此处完整限定为Node 2/3与三个状态—误差成对子集。',
    '', '|控制节点|观测|MAE（训练均值±SD）|末段L1|能耗|','|---|---|---:|---:|---:|']
    pair=lambda n,s:('23' if s=='alternative4' else ('13' if n==2 else '12') if s=='missing4' else '12' if n==2 else '13')
    for r in ps:lines.append(f"|{r['node']}|O{pair(r['node'],r['subset'])}|{r['mae']['mean']:.4f} ± {r['mae']['sd']:.4f}|{r['tail_l1']['mean']:.4f}|{r['energy']['mean']:.2f}|")
    lines+=['','## 2. 选择规则敏感性','',
    '仅重新汇总原8候选的20初值验证数据；没有把新增配置偷偷放入旧选择实验，也没有用已看过的测试集宣称新规则独立验证。比较80%、90%、100%的逐训练达标要求与维度优先/能耗优先。以下列出所有组合。',
    '', '|允许节点|误差容限|达标要求|维度优先|能耗优先|','|---|---:|---:|---|---|']
    for scope in ['any','node2','node3']:
        for tol in [.01,.05,.1]:
            for rate in [.8,.9,1.]:
                chosen=[]
                for p in ['dimension','energy']:
                    c=next(r['selected'] for r in sensitivity if r['scope']==scope and r['tolerance']==tol and r['rate']==rate and r['priority']==p)
                    chosen.append('无' if c is None else f"Node {c['node']} {c['subset']}")
                lines.append(f"|{scope}|{tol}|{rate:.0%}|{'|'.join(chosen)}|")
    lines+=['','## 3. 已选配置的冻结策略失配测试','',
    '保留原规则选出的Node 2 O23和Node 3 O13，容限分别0.10、0.05。40个新初值，各3次训练，参数只修改响应系统：W_r=W⊙(1+ηU)，U逐元素均匀分布于[-1,1]，η为0、1%、3%、5%。相同初值跨幅度使用同一失配方向；驱动系统保持不变，零权重保持零。既不重新训练，也不重新选配置。',
    '', '|节点|失配上限|三次训练达标率|末段L1|能耗|','|---|---:|---|---:|---:|']
    for r in ts:lines.append(f"|{r['node']}|{r['eta']:.0%}|{'/'.join(f'{v:.1%}' for v in r['rates'])}|{r['tail_l1']['mean']:.4f}|{r['energy']['mean']:.2f}|")
    lines+=['','不能把名义配置通过新初值验证，等同于参数变化后仍达标。这里测试的是固定策略和配置的联合迁移表现，不能把失败完全归因于观测位置，也不证明在其他网络参数点重新训练仍有相同排序。',
    '', f"数据检查：360条成对观测评价、960条失配评价；实际初始状态保持配对，两组初值不重叠；状态保护边界触发总数{summary['guard_hits']}。模型与逐轨迹信息保留在原始JSON中。", 
    '', '所有设置见protocol.json，运行脚本为Code/main/extend_configuration_evidence.py。完整性检查及报告脚本为Code/main/report_configuration_extension.py。原模型与原8候选选择结果没有覆盖。']
    (OUT/'EXPLANATION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(8.4,3.1),layout='constrained')
    colors={'12':'#d88729','13':'#25849b','23':'#805ea9'}
    for ax,metric,label in zip(axes,['mae','energy'],['Full-horizon MAE','Applied energy']):
        for n in [2,3]:
            rr=sorted([r for r in ps if r['node']==n],key=lambda r:pair(n,r['subset']))
            for j,r in enumerate(rr):
                x=n+(.23*(j-1));name=pair(n,r['subset']);values=r[metric]['runs']
                ax.scatter(x+np.array([-.025,0,.025]),values,s=17,color=colors[name],label=f'Coordinates {name[0]},{name[1]}' if n==2 else None)
                ax.plot([x-.065,x+.065],[r[metric]['mean']]*2,color='black',lw=1.6)
        ax.set(xticks=[2,3],xticklabels=['Node 2','Node 3'],ylabel=label,yscale='log',xlim=(1.5,3.5))
        ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.savefig(ROOT/'Manuscript/figures/paired_observation_complete.png',dpi=240)
    plt.close(fig)
    print(json.dumps(dict(paired=ps,transfer=ts),indent=2),flush=True)

if __name__=='__main__':main()
