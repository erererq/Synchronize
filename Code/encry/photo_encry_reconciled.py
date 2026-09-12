"""Experimental centered reconciliation and noncancelling DNA substitution.

Not a secure sketch, fuzzy extractor, or authenticated encryption scheme.
Helper data must be accounted for; no confidentiality claim is made for it.
"""
import hashlib
import itertools
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from Code.encry import photo_encry_checked as base

# Eight distinct derangements of four DNA symbols. Each lane uses one rule.
PERMS = tuple(p for p in itertools.permutations(range(4))
              if all(p[i] != i for i in range(4)))[:8]


def continuous(streams):
    a = np.asarray(streams,dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != 4 or a.shape[1] == 0 or not np.isfinite(a).all():
        raise ValueError('Expected four equal-length finite streams.')
    if np.max(np.abs(a)) > 1e12:
        raise ValueError('Values outside supported numerical range.')
    return a


def commitment(rules):
    return hashlib.sha256(np.asarray(rules,dtype=np.uint8).tobytes()).hexdigest()


def enroll(master, delta=1.0):
    a = continuous(master)
    if not np.isfinite(delta) or not 1e-6 <= delta <= 1e6:
        raise ValueError('Invalid quantization width.')
    scaled = a/delta
    integers = np.floor(scaled+0.5).astype(np.int64)
    rules = tuple((integers%8+1).astype(np.uint8))
    helper = {'version':2,'delta':float(delta),'offsets':(scaled-integers).tolist(),
              'rule_digest':commitment(rules)}
    return rules,helper


def reconcile(response, helper):
    a = continuous(response)
    if helper.get('version') != 2:
        raise ValueError('Unsupported helper version.')
    delta = helper['delta']
    if not np.isfinite(delta) or not 1e-6 <= delta <= 1e6:
        raise ValueError('Invalid quantization width.')
    offsets = np.asarray(helper['offsets'],dtype=np.float64)
    if offsets.shape != a.shape or not np.isfinite(offsets).all() or np.any(np.abs(offsets)>0.5000001):
        raise ValueError('Invalid centering offsets.')
    integers = np.floor(a/delta-offsets+0.5).astype(np.int64)
    rules = tuple((integers%8+1).astype(np.uint8))
    if commitment(rules) != helper['rule_digest']:
        raise ValueError('Rule agreement failed; no candidate image released.')
    return rules


def lane_table(indices, inverse=False):
    indices = np.asarray(indices)
    if indices.shape != (4,) or not np.issubdtype(indices.dtype,np.integer) or np.any((indices<1)|(indices>8)):
        raise ValueError('Four integer indices in 1..8 required.')
    values = np.arange(256,dtype=np.uint8)
    out = np.zeros(256,dtype=np.uint8)
    for index,shift in zip(indices,[6,4,2,0]):
        permutation = np.asarray(PERMS[int(index)-1],dtype=np.uint8)
        if inverse:
            permutation = np.argsort(permutation).astype(np.uint8)
        out |= permutation[(values>>shift)&3]<<shift
    return out


def transform(image,rules,p=8,inverse=False):
    base.validate_image(image)
    rules = base.validate_rules(rules,base.block_count(image.shape,p))
    out = np.empty_like(image)
    b = 0
    for i in range(0,image.shape[0],p):
        for j in range(0,image.shape[1],p):
            table = lane_table([r[b] for r in rules],inverse)
            out[i:i+p,j:j+p] = table[image[i:i+p,j:j+p]]
            b += 1
    return out


def encrypt(plain,master,password,p=8,delta=1.0):
    base.validate_image(plain)
    rules,helper = enroll(master,delta)
    base.validate_rules(rules,base.block_count(plain.shape,p))
    pixel_sum = str(int(plain.sum(dtype=np.uint64)))
    cipher = transform(plain^base.mask(plain.shape,pixel_sum,password)[:,:,None],rules,p)
    return cipher, {'version':2,'shape':list(plain.shape),'block_size':p,
                    'pixel_sum':pixel_sum,'helper':helper}


def decrypt(cipher,response,metadata,password):
    base.validate_image(cipher)
    if metadata.get('version') != 2 or metadata.get('shape') != list(cipher.shape):
        raise ValueError('Metadata mismatch.')
    rules = reconcile(response,metadata['helper'])
    return transform(cipher,rules,metadata['block_size'],True)^base.mask(cipher.shape,metadata['pixel_sum'],password)[:,:,None]


def main():
    import argparse
    import getpass
    import json
    import re
    import cv2
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--node',type=int,choices=[1,2,3],required=True)
    parser.add_argument('--seed',type=int,default=100)
    parser.add_argument('--burn-in',type=int,default=1500)
    args=parser.parse_args()
    if args.output.exists() or not args.model.is_file() or args.burn_in<0:
        parser.error('Use a new output directory, an existing model, and nonnegative burn-in.')
    marker=re.search(r'node([123])',args.model.name.lower())
    if marker and int(marker.group(1))!=args.node:
        parser.error('Node and model filename disagree.')
    plain=cv2.imread(str(args.image),cv2.IMREAD_UNCHANGED)
    base.validate_image(plain)
    password=getpass.getpass('Experiment password (not stored): ')
    if not password:
        parser.error('Nonempty password required.')
    args.output.mkdir(parents=True,exist_ok=False)
    report={'version':2,'node':args.node,'seed':args.seed,'burn_in':args.burn_in,
            'delta':1.0,'block_size':8,'attempts':1,'deterministic':True,
            'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),
            'scope':'Functional prototype; helper leakage and authentication unresolved.'}
    code=2
    try:
        n=base.block_count(plain.shape)
        master,response=base.generate_streams(args.model,args.node,args.seed,args.burn_in,n)
        cipher,meta=encrypt(plain,master,password)
        # Exercise the actual serialized helper data, not shared sender arrays.
        text=json.dumps(meta,indent=2)
        meta=json.loads(text)
        (args.output/'metadata.json').write_text(text,encoding='utf-8')
        base.write_lossless(args.output/'cipher.png',cipher)
        report['helper_json_bytes']=len(json.dumps(meta['helper']).encode())
        report['max_four_stream_error']=float(np.max(np.abs(np.asarray(master,dtype=float)-np.asarray(response,dtype=float))))
        sender_rules,_=enroll(master)
        report['sender_unique']=[len(np.unique(s)) for s in sender_rules]
        try:
            recovered=decrypt(cipher,response,meta,password)
        except ValueError as exc:
            report.update(status='agreement_rejected',error=str(exc),exact_recovery=False)
            code=1
        else:
            exact=bool(np.array_equal(plain,recovered))
            base.write_lossless(args.output/'recovered.png',recovered)
            report.update(status='recovered' if exact else 'recovery_failed',exact_recovery=exact)
            fixed=tuple(np.ones(n,dtype=np.uint8) for _ in range(4))
            candidate=transform(cipher,fixed,8,True)^base.mask(plain.shape,meta['pixel_sum'],password)[:,:,None]
            report['fixed_rule_exact_bypassing_check']=bool(np.array_equal(plain,candidate))
            code=0 if exact else 1
    except Exception as exc:
        report.update(status='error',error=str(exc))
        code=2
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return code


if __name__=='__main__':
    raise SystemExit(main())
