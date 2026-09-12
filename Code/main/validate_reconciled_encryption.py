"""Exhaustive mapping checks plus paired PPO reconciliation experiments."""
import sys
import json
import itertools
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from Code.encry import photo_encry_checked as b
from Code.encry import photo_encry_reconciled as r


def main():
    tables=set()
    for indices in itertools.product(range(1,9),repeat=4):
        t=r.lane_table(indices)
        inv=r.lane_table(indices,True)
        assert np.array_equal(inv[t],np.arange(256))
        assert not np.any(t==np.arange(256))
        tables.add(t.tobytes())
    assert len(tables)==4096
    rng=np.random.default_rng(20260909)
    synthetic=rng.uniform(-5,5,(4,256))
    rules,helper=r.enroll(synthetic)
    helper=json.loads(json.dumps(helper))
    for offset in [-0.49,0.49]:
        assert np.array_equal(r.reconcile(synthetic+offset,helper),rules)
    for offset in [-0.51,0.51]:
        try:
            r.reconcile(synthetic+offset,helper)
        except ValueError:
            pass
        else:
            raise AssertionError('Out-of-bound synthetic case not rejected.')
    # Known limitation: modulo aliases separated by 8 widths are accepted.
    assert np.array_equal(r.reconcile(synthetic+8,helper),rules)
    rows=[]
    for node in [2,3]:
        for training_seed in [42,123,1024]:
            model=ROOT/f'models/hopfield/ppo_continuous_hopfield_node{node}_final_seed_{training_seed}.zip'
            for seed in [100,101,102]:
                a,c=b.generate_streams(model,node,seed,1500,256)
                for size in [64,128]:
                    plain=rng.integers(0,256,(size,size,3),dtype=np.uint8)
                    n=b.block_count(plain.shape)
                    master=np.asarray(a)[:,:n]
                    response=np.asarray(c)[:,:n]
                    cipher,meta=r.encrypt(plain,master,'validation-only')
                    meta=json.loads(json.dumps(meta))
                    accepted=True
                    try:
                        recovered=r.decrypt(cipher,response,meta,'validation-only')
                        exact=bool(np.array_equal(plain,recovered))
                    except ValueError:
                        accepted=False
                        exact=False
                    sender_rules,_=r.enroll(master)
                    direct=b.quantize_streams(response)
                    def candidate(rules):
                        return r.transform(cipher,rules,8,True)^b.mask(plain.shape,meta['pixel_sum'],'validation-only')[:,:,None]
                    fixed=tuple(np.ones(n,dtype=np.uint8) for _ in range(4))
                    # Bypass agreement check for a non-tautological mapping ablation.
                    fixed_exact=bool(np.array_equal(plain,candidate(fixed)))
                    unrelated=rng.uniform(-5,5,response.shape)
                    try:
                        r.reconcile(unrelated,meta['helper'])
                        unrelated_accepted=True
                    except ValueError:
                        unrelated_accepted=False
                    max_error=float(np.max(np.abs(master.astype(float)-response.astype(float))))
                    if max_error < 0.49:
                        assert accepted and exact
                    rows.append(dict(node=node,training_seed=training_seed,initial_seed=seed,size=size,
                        max_four_stream_error=max_error,within_half_bin=max_error<0.5,
                        agreement_accepted=accepted,exact_recovery=exact,
                        direct_new_mapping_exact=bool(np.array_equal(plain,candidate(direct))),
                        fixed_rule_exact_bypassing_check=fixed_exact,unrelated_accepted=unrelated_accepted,
                        sender_unique=[len(np.unique(s)) for s in sender_rules],
                        helper_json_bytes=len(json.dumps(meta['helper'],separators=(',',':')).encode()),
                        image_bytes=int(plain.nbytes)))
                print(f'node {node}, model {training_seed}, initial {seed} done',flush=True)
    summary=[]
    for node in [2,3]:
        for size in [64,128]:
            items=[x for x in rows if x['node']==node and x['size']==size]
            summary.append(dict(node=node,size=size,trials=len(items),
                exact=sum(x['exact_recovery'] for x in items),
                direct_exact=sum(x['direct_new_mapping_exact'] for x in items),
                fixed_exact=sum(x['fixed_rule_exact_bypassing_check'] for x in items),
                unrelated_accepted=sum(x['unrelated_accepted'] for x in items),
                mean_unique=float(np.mean([x['sender_unique'] for x in items])),
                max_error=max(x['max_four_stream_error'] for x in items),
                helper_bytes_mean=float(np.mean([x['helper_json_bytes'] for x in items]))))
    result=dict(protocol={'delta':1.0,'burn_in':1500,'nodes':[2,3],'model_seeds':[42,123,1024],
        'initial_seeds':[100,101,102],'sizes':[64,128],'paired_prefixes':True,'nominal_only':True},
        exhaustive={'unique_maps':4096,'bytes_each':256,'fixed_points':0,'roundtrip':'pass',
        'synthetic_plus_minus_049':'pass','synthetic_plus_minus_051':'rejected','modulo_8_alias':'accepted_known_limitation'},
        summary=summary,rows=rows)
    target=ROOT/'logs/hopfield/reconciled_encryption'
    target.mkdir(parents=True,exist_ok=True)
    (target/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
