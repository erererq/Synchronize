"""Validate legacy image-transform reversibility; not a security evaluation."""
from pathlib import Path
import json
import runpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

def main():
    m = runpy.run_path(str(ROOT / 'Code/encry/photo_encry'), run_name='alignment_test')
    rules = m['coding_rules']
    # Exhaust every byte and every four-rule composition independently.
    maps = {}
    for i, rule in rules.items():
        assert set(rule.values()) == set('ACGT')
        for j, decode in rules.items():
            inverse = {v: k for k, v in decode.items()}
            maps[i, j] = np.array([int(''.join(inverse[rule[f'{b:08b}'[k:k+2]]] for k in range(0,8,2)),2) for b in range(256)])
    checks = 0
    for a in rules:
        for b in rules:
            for c in rules:
                for d in rules:
                    original = np.arange(256)
                    cipher = maps[c,d][maps[a,b][original]]
                    restored = maps[b,a][maps[d,c][cipher]]
                    assert np.array_equal(original,restored)
                    checks += 1
    rng = np.random.default_rng(20260908)
    cases = []
    for h,w in [(8,8),(16,16),(17,25),(32,32),(64,64)]:
        for seed in range(4):
            plain = rng.integers(0,256,(h,w,3),dtype=np.uint8)
            n = ((h+7)//8)*((w+7)//8)
            streams = tuple(rng.integers(1,9,n,dtype=np.uint8) for _ in range(4))
            for channel in range(3):
                blocks = m['split_into_blocks'](plain[:,:,channel])
                assert len(blocks) == n
                assert np.array_equal(m['reshape_blocks_to_channel'](blocks,h,w),plain[:,:,channel])
            cipher, metadata = m['encrypt'](streams,plain,'alignment-test-only')
            _, recovered = m['decrypt'](streams,cipher,metadata,'alignment-test-only')
            assert metadata[0] == str(int(plain.sum()))
            assert np.array_equal(plain,recovered), (h,w,seed)
            seq = m['logistic_map'](3.9999,metadata[1],h*w-1)
            expected = (np.round(np.asarray(seq)*10000)%256).astype(np.uint8).reshape(h,w)
            assert np.array_equal(expected,m['reshape_sequence_to_Q'](seq,h,w))
            cases.append(dict(height=h,width=w,case=seed,exact_recovery=True))
    out = dict(rule_rows={i:[r[k] for k in ['00','01','10','11']] for i,r in rules.items()},
               exhaustive_four_rule_compositions=checks,bytes_per_composition=256,
               full_pipeline_cases=cases,
               scope='Fixed matching rule streams; no PPO loading, quantizer reliability or security certification.')
    target = ROOT/'logs/hopfield/encryption_structure_audit/alignment_validation.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(f'PASS: {checks} compositions x 256 bytes; {len(cases)} image round trips. {target}')

if __name__ == '__main__':
    main()
