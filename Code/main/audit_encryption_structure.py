"""Small deterministic diagnostics of the current image transform, no training."""
from pathlib import Path
import json
import runpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    module = runpy.run_path(str(ROOT/'Code/encry/photo_encry'), run_name='audit_module')
    plain = np.random.default_rng(20260908).integers(1,255,(16,16,3),dtype=np.uint8)
    ones = tuple(np.ones(4,dtype=np.uint8) for _ in range(4))
    eights = tuple(np.full(4,8,dtype=np.uint8) for _ in range(4))
    cipher, metadata = module['encrypt'](ones,plain,'audit-only-example')
    other_cipher, _ = module['encrypt'](eights,plain,'audit-only-example')
    status, recovered = module['decrypt'](eights,cipher,metadata,'audit-only-example')
    changed = plain.copy()
    changed[0,0,0] += 1
    changed[0,1,0] -= 1
    changed_cipher, _ = module['encrypt'](ones,changed,'audit-only-example')
    wrong_status, wrong_image = module['decrypt'](ones,cipher,metadata,'different-audit-example')
    output = dict(
        image_shape=list(plain.shape),
        different_rule_streams_same_cipher=bool(np.array_equal(cipher,other_cipher)),
        different_rule_streams_exact_recovery=bool(np.array_equal(plain,recovered)),
        success_flag=bool(status),
        equal_plain_sums=bool(plain.sum()==changed.sum()),
        changed_plain_components=int(np.count_nonzero(plain!=changed)),
        changed_cipher_components=int(np.count_nonzero(cipher!=changed_cipher)),
        wrong_password_success_flag=bool(wrong_status),
        wrong_password_exact_recovery=bool(np.array_equal(plain,wrong_image)),
        metadata_contains_logistic_initial_value=bool(len(metadata)==2 and isinstance(metadata[1],float)),
        interpretation='Structural diagnostics under fixed supplied rule streams, not a complete cryptanalysis.'
    )
    target=ROOT/'logs/hopfield/encryption_structure_audit'
    target.mkdir(parents=True,exist_ok=True)
    (target/'diagnostics.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
