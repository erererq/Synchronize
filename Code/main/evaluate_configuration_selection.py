"""Freeze a validation-only choice, then evaluate it on held-out initial states."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='1'
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'logs/hopfield/configuration_selection'
SEEDS=[42,123,1024]
SUBSETS={'local2':2,'original4':4,'alternative4':4,'full6':6}

def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def checkpoint(node,subset,seed):
    if subset=='alternative4':
        return ROOT/f'models/hopfield/observation_identity/PPO_node{node}_obs23_seed{seed}.zip'
    prefix={'local2':'ppo_obs2','original4':'ppo_obs4','full6':'ppo'}[subset]
    return ROOT/f'models/hopfield/{prefix}_continuous_hopfield_node{node}_final_seed_{seed}.zip'

def evaluate(job):
    import torch
    from stable_baselines3 import PPO
    from Code.env.continuous_hopfield_env import ContinuousHopfieldEnv
    torch.set_num_threads(1)
    node,subset,initials,split=job
    indices={'local2':[node-1,node+2], 'original4':[0,node-1,3,node+2],
             'alternative4':[1,2,4,5],'full6':list(range(6))}[subset]
    class AuditedEnv(ContinuousHopfieldEnv):
        def _rk4_step(self,state,W_matrix,B,control_force=0.0):
            result=super()._rk4_step(state,W_matrix,B,control_force)
            self.guard_hits+=int(np.any(np.abs(result)>=10))
            return result
    rows=[]
    for seed in SEEDS:
        path=checkpoint(node,subset,seed)
        model=PPO.load(path,device='cpu')
        assert model.observation_space.shape==(len(indices),)
        assert model.seed==seed and model.num_timesteps==401408
        assert model.n_steps==512 and model.batch_size==128 and model.n_envs==4
        assert model.ent_coef==.001 and model.gamma==.99
        envs=[AuditedEnv(pinning_node=node-1) for _ in initials]
        obs=[];errors=[];start=[]
        for env,initial in zip(envs,initials):
            env.guard_hits=0
            o,_=env.reset(seed=initial)
            obs.append(o[indices]);errors.append([env.statey-env.statex])
            start.append(np.concatenate([env.statex,env.statey]).tolist())
        energy=np.zeros(len(envs))
        for k in range(400):
            actions,_=model.predict(np.asarray(obs),deterministic=True)
            assert np.isfinite(actions).all() and np.max(np.abs(actions))<=1
            obs=[]
            for i,(env,a) in enumerate(zip(envs,actions)):
                o,_,terminated,truncated,_=env.step(a)
                assert not terminated and (not truncated or k==399)
                assert np.isfinite(o).all()
                obs.append(o[indices]);errors[i].append(env.statey-env.statex)
                energy[i]+=float(a[0]*env.scale)**2*env.dt*env.rk4_steps
        for i,env in enumerate(envs):
            e=np.asarray(errors[i],dtype=float);tail=np.abs(e[320:]).sum(axis=1)
            rows.append(dict(node=node,subset=subset,seed=seed,initial=initials[i],split=split,
                initial_state=start[i],mae=float(np.abs(e).mean()),tail_l1=float(tail.mean()),
                tail_max_l1=float(tail.max()),energy=float(energy[i]),guard_hits=env.guard_hits))
            env.close()
    print(f'{split}: Node {node} {subset} complete',flush=True)
    return rows

def summary(rows,tolerance):
    rates=[float(np.mean([r['tail_l1']<=tolerance and r['guard_hits']==0 for r in rows if r['seed']==s])) for s in SEEDS]
    return dict(rates=rates,eligible=min(rates)>=.9,mean_rate=float(np.mean(rates)),
                tail_l1=float(np.mean([r['tail_l1'] for r in rows])),
                energy=float(np.mean([r['energy'] for r in rows])),
                guard_hits=sum(r['guard_hits'] for r in rows))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'validation.json').exists(), 'Use a new version directory; do not overwrite a completed experiment.'
    configs=[(n,s) for n in [2,3] for s in SUBSETS]
    val=list(range(910000,910020));test=list(range(920000,920040))
    assert not set(val)&set(test)
    hashes={str(checkpoint(n,s,k).relative_to(ROOT)):hashlib.sha256(checkpoint(n,s,k).read_bytes()).hexdigest()
            for n,s in configs for k in SEEDS}
    protocol=dict(validation_initials=val,test_initials=test,training_seeds=SEEDS,model_hashes=hashes,
        tolerances=[.01,.05,.1],minimum_rate_per_training_run=.9,horizon=20,tail_window=[16,20],
        success='mean tail L1 <= tolerance and no state-guard hit; not maximum error or certified synchronization',
        ranking='lowest observation dimension, then validation mean energy, then node and subset name',
        scope='nominal fixed network; candidate-only selection; frozen PPO checkpoints',
        threshold_origin='illustrative sensitivity levels informed by prior experiments, fixed before this validation; not application-derived',
        no_test_reselection=True)
    if (OUT/'protocol.json').exists():
        assert json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))==protocol
    else:
        save('protocol.json',protocol)
    validation=sum(map(evaluate,[(n,s,val,'validation') for n,s in configs]),[])
    save('validation.json',validation)
    choices=[]
    for scope in ['any','node2','node3']:
        for tol in protocol['tolerances']:
            stats=[]
            for n,s in configs:
                if scope!='any' and scope!=f'node{n}':continue
                rr=[r for r in validation if r['node']==n and r['subset']==s]
                stats.append(dict(node=n,subset=s,dimension=SUBSETS[s],**summary(rr,tol)))
            eligible=[a for a in stats if a['eligible']]
            chosen=min(eligible,key=lambda a:(a['dimension'],a['energy'],a['node'],a['subset'])) if eligible else None
            choices.append(dict(scope=scope,tolerance=tol,candidates=stats,selected=chosen))
    save('frozen_choices.json',choices)
    selected=sorted({(c['selected']['node'],c['selected']['subset']) for c in choices if c['selected']})
    heldout=sum(map(evaluate,[(n,s,test,'test') for n,s in selected]),[])
    save('test.json',heldout)
    for choice in choices:
        c=choice['selected']
        choice['test']=summary([r for r in heldout if r['node']==c['node'] and r['subset']==c['subset']],choice['tolerance']) if c else None
    save('results.json',choices)
    # Verify common states across candidates and disjoint realized states across splits.
    for rows in [validation,heldout]:
        states={}
        for r in rows:
            initial=r['initial'];state=tuple(r['initial_state'])
            assert initial not in states or states[initial]==state
            states[initial]=state
    assert not {tuple(r['initial_state']) for r in validation}&{tuple(r['initial_state']) for r in heldout}
    report(choices)

def report(choices):
    lines=['# 配置选择规则：实现、验证与解释','',
    '## 做了什么','',
    '冻结已有PPO模型，不重新训练。比较Node 2、Node 3各自的2维、原4维、新4维、6维，共8个配置，每个配置保留3次训练。用20个新初值选择，再用另外40个初值验证被选配置。先保存frozen_choices.json，再运行测试；测试失败不重新选择。',
    '', '## 规则怎么理解','',
    '一次轨迹达标：最后4个时间单位（t=16至20，81个采样点）的平均L1同步误差不超过容限，且未碰到状态保护边界。每个训练重复各自至少90%的验证轨迹达标，配置才可入选。先选择观测维度最少的可行配置，再按验证平均能耗从低到高选择；并列按节点编号和配置名确定。没有可行配置就报告无方案。',
    '', '误差容限0.01、0.05、0.10是参考已有结果制定的示例敏感性设置，在本轮新验证前固定；不是应用提出的要求，也不是理论同步阈值。达标率是有限样本频率，不是总体成功率保证。末段平均达标不表示每个时刻误差都达标。',
    '', '分别考虑可以自由选择Node 2/3、只能控制Node 2、只能控制Node 3三类条件。自由选择不包括Node 1，因此不是全网络最优选择。',
    '', '|允许位置|容限|选中配置|验证各训练达标率|测试各训练达标率|测试末段L1|测试能耗|测试是否仍满足同一规则|',
    '|---|---:|---|---|---|---:|---:|---|']
    for r in choices:
        c=r['selected'];t=r['test']
        if not c:
            lines.append(f"|{r['scope']}|{r['tolerance']}|无可行配置|—|—|—|—|未测试|")
        else:
            rates=lambda v:'/'.join(f'{x:.0%}' for x in v)
            lines.append(f"|{r['scope']}|{r['tolerance']}|Node {c['node']} {c['subset']}|{rates(c['rates'])}|{rates(t['rates'])}|{t['tail_l1']:.4f}|{t['energy']:.2f}|{'是' if t['eligible'] else '否'}|")
    lines+=['','配置名：local2={xi,ei}；original4={x1,xi,e1,ei}；alternative4={x2,x3,e2,e3}；full6=全部状态与误差。这里i是控制节点。观测均使用原来的除以5归一化。',
    '', '## 能成为怎样的贡献','',
    '从“配置之间存在差异”进一步到“根据精度要求和执行器限制筛选候选配置，并在独立初值上验证”。选择本身是简单的约束筛选，不是新RL算法；价值取决于实际验证结果，而不是规则听起来复杂。',
    '', '不能称为最优传感器配置、全局最优联合设计或严格鲁棒保证。仅针对同一网络、同一初值分布和名义工况；没有参数失配和扰动验证。观测维度不等于实体传感器数量。三个训练重复用于检查可重复性，不选择其中最好的一个；未检验新训练种子的泛化。',
    '', '## 文件与复现','',
    'protocol.json保存测试前固定的设置与模型哈希；validation.json和test.json保留逐轨迹结果及实际初始状态；frozen_choices.json保存测试前选择；results.json保存完整筛选与测试汇总。源程序为Code/main/evaluate_configuration_selection.py。原模型、原实验与加密章节未修改。']
    lines+=['','## 全部候选的验证达标率（按三次训练分别列出）','',
        '|节点|配置|容限0.01|容限0.05|容限0.10|平均末段L1|平均能耗|',
        '|---|---|---|---|---|---:|---:|']
    free=[r for r in choices if r['scope']=='any']
    for c in free[0]['candidates']:
        matches=[next(a for a in r['candidates'] if a['node']==c['node'] and a['subset']==c['subset']) for r in free]
        rates=['/'.join(f'{v:.0%}' for v in a['rates']) for a in matches]
        lines.append(f"|{c['node']}|{c['subset']}|{'|'.join(rates)}|{c['tail_l1']:.4f}|{c['energy']:.2f}|")
    lines+=['','## 本轮结果具体说明什么','',
        '1. 可以选Node 2或Node 3时，0.05和0.10两档均选Node 3原4维观测{x1,x3,e1,e3}。每次训练在40个独立测试初值上均全部达标，末段L1总体均值0.0086，平均能耗5.03。这是已固定策略在新初值上的验证，不是新的训练种子验证。',
        '2. 如果位置受限只能用Node 2，0.10档选新4维观测{x2,x3,e2,e3}，三次训练各40条测试轨迹均达标，末段L1均值0.0282，能耗8.11。0.05档不入选，因为验证时有一次训练只有75%达标，虽然三次合计达标率超过90%。不能用平均表现掩盖这次训练。',
        '3. 0.01档没有配置通过验证筛选。Node 3原4维整体平均末段误差虽然低于0.01，但部分训练的逐轨迹达标率不足。这正是“平均误差低”与“每次训练都有较高达标率”的区别。',
        '4. Node 3的2维配置在0.05及0.10档达标率都是100%/0%/100%，并非2维一定不行，而是现有三个训练重复并不一致。本规则因此不选择它，不能据此证明4维是理论最小维数。',
        '5. Node 2在0.10档选4维而非6维，首先是因为预先规定维度优先；4维验证能耗11.76并不低于6维的10.93。选择偏好会影响结论，不能把这一选择说成所有指标都更优。不同初值组之间的能耗均值也不宜直接当作算法优劣比较。',
        '', '本轮共480条验证轨迹、240条测试轨迹，没有触及±10状态保护边界。不同配置使用相同的实际初始状态，验证与测试实际状态不重叠。各训练共享测试初值，因此不能把120条结果当作120个独立初值，或把三次训练说成总体可靠性证明。',
        '', '## 写论文时建议怎么说','',
        '建议称为“基于精度约束的候选配置筛选及独立初值验证”，作为观测—执行器匹配实验的延伸，而不是独立的新算法贡献。可写：在预设示例误差容限下，根据训练重复间的达标一致性筛选可行配置，并按反馈维度和控制能耗排序；选择在独立初值上验证，同时明确报告无可行配置的情形。',
        '', '当前未自动写入论文，以免将探索性示例容限误当成已经确定的实际应用要求；原正文及加密章节保持不变。若后续修改容限、排序或模型，应作为新版本并另用未参与调整的测试初值；不能沿用本轮测试集继续声称盲测。']
    (OUT/'EXPLANATION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
