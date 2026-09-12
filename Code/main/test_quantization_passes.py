"""Paired quantizer ablation on fixed PPO trajectories; no retries/training."""
import sys
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from Code.encry import photo_encry_checked as c


def quantize(raw,passes):
    return c.quantize_streams(raw,filter_passes=passes)


def main():
    output=ROOT/'logs/hopfield/quantization_pass_ablation'
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    rng=np.random.default_rng(20260908)
    images={size:rng.integers(0,256,(size,size,3),dtype=np.uint8) for size in [64,128]}
    for node in [2,3]:
        for training_seed in [42,123,1024]:
            model=ROOT/f'models/hopfield/ppo_continuous_hopfield_node{node}_final_seed_{training_seed}.zip'
            for initial_seed in [100,101,102]:
                raw_master,raw_slave=c.generate_streams(model,node,initial_seed,1500,256)
                for size,image in images.items():
                    n=c.block_count(image.shape)
                    a=tuple(x[:n] for x in raw_master)
                    b=tuple(x[:n] for x in raw_slave)
                    for passes in [0,1,3,5,15]:
                        master,slave=quantize(a,passes),quantize(b,passes)
                        cipher,meta=c.encrypt_image(image,master,'ablation-test-only')
                        recovered=c.decrypt_image(cipher,slave,meta,'ablation-test-only')
                        fixed=tuple(np.ones(n,dtype=np.uint8) for _ in range(4))
                        wrong=c.decrypt_image(cipher,fixed,meta,'ablation-test-only')
                        diag=c.rule_diagnostics(master,slave,n)
                        effective=np.mean([np.array_equal(c.dna_table([s[i] for s in master]),c.dna_table([s[i] for s in slave])) for i in range(n)])
                        rows.append(dict(node=node,training_seed=training_seed,initial_seed=initial_seed,size=size,passes=passes,
                            exact_recovery=bool(np.array_equal(image,recovered)),bit_error_rate=float(np.unpackbits(image^recovered).mean()),
                            fixed_rule_exact_recovery=bool(np.array_equal(image,wrong)),effective_map_match=float(effective),
                            **diag))
                print(f'Completed node={node}, model_seed={training_seed}, initial_seed={initial_seed}',flush=True)
    summary=[]
    for node in [2,3]:
        for size in [64,128]:
            for passes in [0,1,3,5,15]:
                r=[x for x in rows if x['node']==node and x['size']==size and x['passes']==passes]
                summary.append(dict(node=node,size=size,passes=passes,trials=len(r),
                    exact_count=sum(x['exact_recovery'] for x in r),
                    all_rules_equal_count=sum(x['all_rules_equal'] for x in r),
                    mean_rule_mismatch=float(np.mean([x['rule_mismatch_rate'] for x in r])),
                    mean_master_unique=float(np.mean([x['master_unique_per_stream'] for x in r])),
                    constant_master_stream_fraction=float(np.mean([np.array(x['master_unique_per_stream'])==1 for x in r])),
                    mean_identity_block_fraction=float(np.mean([x['master_identity_block_fraction'] for x in r])),
                    fixed_rule_exact_count=sum(x['fixed_rule_exact_recovery'] for x in r)))
    result=dict(protocol={'burn_in':1500,'block_size':8,'training_seeds':[42,123,1024],'initial_seeds':[100,101,102],
        'note':'Nominal deterministic PPO, paired trajectories. Sizes share prefixes; trials are not independent across sizes or passes. Fixed supplied password; synthetic images. Not a security test.'},summary=summary,rows=rows)
    (output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
