"""Reproducible image-transform experiment, NOT a secure communication protocol.

The legacy mask and DNA substitutions are retained. Direct quantization is the
diagnostic default; legacy repeated filtering is an explicit comparison option.
No training, retry filtering, embedded password, or legacy-script import.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RULES = ('AGCT', 'ACGT', 'TGCA', 'TCGA', 'CATG', 'GATC', 'CTAG', 'GTAC')


def validate_image(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError('Image must be a uint8 array.')
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError('Only nonempty three-channel BGR images are supported.')


def block_count(shape, block_size=8):
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError('block_size must be a positive integer.')
    return ((shape[0]+block_size-1)//block_size)*((shape[1]+block_size-1)//block_size)


def validate_rules(streams, count):
    if len(streams) != 4:
        raise ValueError('Exactly four rule streams are required.')
    result = []
    for stream in streams:
        a = np.asarray(stream)
        if a.ndim != 1 or len(a) != count or not np.issubdtype(a.dtype, np.integer):
            raise ValueError('Each rule stream must contain one integer per block.')
        if np.any((a < 1) | (a > 8)):
            raise ValueError('DNA rule indices must be in 1..8.')
        result.append(a.astype(np.uint8))
    return tuple(result)


def kalman_pass(a):
    estimate, error = float(a[0]), 1e-2
    out = []
    for measurement in a:
        error += 1e-5
        gain = error/(error+1e-2)
        estimate += gain*(float(measurement)-estimate)
        error *= 1-gain
        out.append(estimate)
    return np.asarray(out)


def quantize_streams(streams, filter_passes=0):
    """Diagnostic quantizer; neither mode guarantees matching response rules."""
    if not isinstance(filter_passes, int) or not 0 <= filter_passes <= 15:
        raise ValueError('filter_passes must be an integer in 0..15.')
    if len(streams) != 4:
        raise ValueError('Exactly four continuous streams are required.')
    result = []
    for stream in streams:
        a = np.asarray(stream)
        if a.ndim != 1 or not len(a) or not np.all(np.isfinite(a)):
            raise ValueError('Continuous streams must be nonempty finite vectors.')
        q = (np.round(a)%8+1).astype(np.uint8)
        for _ in range(filter_passes):
            q = (kalman_pass(q)%8+1).astype(np.uint8)
        result.append(q)
    validate_rules(result, len(result[0]))
    return tuple(result)


def legacy_quantize(streams):
    """Explicit historical comparison only, not the experiment default."""
    return quantize_streams(streams, filter_passes=15)


def mask(shape, pixel_sum, password):
    if not isinstance(password, str) or not password:
        raise ValueError('A nonempty password is required; it is never logged.')
    if not isinstance(pixel_sum, str) or not pixel_sum.isascii() or not pixel_sum.isdecimal():
        raise ValueError('pixel_sum must be a decimal ASCII string.')
    digest = hashlib.sha256((password+pixel_sum).encode('utf-8')).digest()
    z = int.from_bytes(digest, 'big')/(2**256)
    if z == 0:
        z = 0.123456789
    elif z == 1:
        z = 0.987654321
    sequence = np.empty(shape[0]*shape[1])
    for i in range(len(sequence)):
        sequence[i] = z
        z = 3.9999*z*(1-z)
    return (np.round(sequence*10000)%256).astype(np.uint8).reshape(shape[:2])


def dna_table(indices, inverse=False):
    a,b,c,d = map(int, indices)
    pairs = [(a,b),(c,d)] if not inverse else [(d,c),(b,a)]
    table = np.arange(256, dtype=np.uint8)
    for encode, decode in pairs:
        symbol_map = np.array([RULES[decode-1].index(s) for s in RULES[encode-1]], dtype=np.uint8)
        table = sum((symbol_map[(table >> shift)&3] << shift) for shift in [6,4,2,0])
    return table


def substitute(image, streams, block_size, inverse=False):
    out = np.empty_like(image)
    b = 0
    for i in range(0,image.shape[0],block_size):
        for j in range(0,image.shape[1],block_size):
            table = dna_table([s[b] for s in streams],inverse)
            out[i:i+block_size,j:j+block_size] = table[image[i:i+block_size,j:j+block_size]]
            b += 1
    return out


def encrypt_image(plain, streams, password, block_size=8):
    validate_image(plain)
    streams = validate_rules(streams,block_count(plain.shape,block_size))
    pixel_sum = str(int(plain.sum(dtype=np.uint64)))
    q = mask(plain.shape,pixel_sum,password)
    cipher = substitute(plain ^ q[:,:,None],streams,block_size)
    # No floating-point seed, password, or rule streams in metadata.
    return cipher, {'version':1,'shape':list(plain.shape),'block_size':block_size,'pixel_sum':pixel_sum}


def decrypt_image(cipher, streams, metadata, password):
    """Return a candidate image, never an unauthenticated success flag."""
    validate_image(cipher)
    if metadata.get('version') != 1 or metadata.get('shape') != list(cipher.shape):
        raise ValueError('Metadata version or image shape mismatch.')
    p = metadata['block_size']
    streams = validate_rules(streams,block_count(cipher.shape,p))
    q = mask(cipher.shape,metadata['pixel_sum'],password)
    return substitute(cipher,streams,p,inverse=True) ^ q[:,:,None]


def rule_diagnostics(master, slave, count):
    master, slave = validate_rules(master,count), validate_rules(slave,count)
    return {
        'all_rules_equal':all(np.array_equal(a,b) for a,b in zip(master,slave)),
        'rule_mismatch_rate':float(np.mean(np.concatenate([a != b for a,b in zip(master,slave)]))),
        'master_unique_per_stream':[int(len(np.unique(a))) for a in master],
        'slave_unique_per_stream':[int(len(np.unique(a))) for a in slave],
        'master_identity_block_fraction':float(np.mean([np.array_equal(dna_table([s[b] for s in master]),np.arange(256)) for b in range(count)])),
    }


def generate_streams(model_path, node, seed, burn_in, count):
    if node not in [1,2,3] or burn_in < 0 or count < 1:
        raise ValueError('Invalid node, burn-in, or sample count.')
    sys.path.insert(0,str(ROOT)) if str(ROOT) not in sys.path else None
    from stable_baselines3 import PPO
    from Code.env.continuous_hopfield_env import ContinuousHopfieldEnv
    env = ContinuousHopfieldEnv(pinning_node=node-1)
    # Explicit evaluation horizon: never step beyond a reported truncation.
    env.max_steps = burn_in+count
    try:
        model = PPO.load(str(model_path),env=env,device='cpu')
        obs,_ = env.reset(seed=seed)
        drive,response = [],[]
        for step in range(burn_in+count):
            action,_ = model.predict(obs,deterministic=True)
            obs,_,terminated,truncated,_ = env.step(action)
            if not np.all(np.isfinite(obs)):
                raise RuntimeError('Nonfinite environment observation.')
            if terminated or (truncated and step+1 < burn_in+count):
                raise RuntimeError('Environment ended before requested horizon.')
            if step >= burn_in:
                drive.append(env.statex.copy())
                response.append(env.statey.copy())
        # Retain the original generator's float32 fourth-stream arithmetic.
        def four(values):
            a = np.asarray(values,dtype=np.float32)
            return a[:,0],a[:,1],a[:,2],a[:,0]*a[:,1]+a[:,2]
        return four(drive),four(response)
    finally:
        env.close()


def write_lossless(path,image):
    if path.suffix.lower() != '.png':
        raise ValueError('This experiment writes lossless PNG only.')
    if not cv2.imwrite(str(path),image):
        raise IOError(f'Cannot save {path}')
    loaded = cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if not np.array_equal(image,loaded):
        raise IOError(f'Saved image failed bytewise readback: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='New directory; existing paths are refused.')
    parser.add_argument('--node',type=int,choices=[1,2,3],required=True)
    parser.add_argument('--model',type=Path,required=True,help='PPO checkpoint trained for the selected node.')
    parser.add_argument('--seed',type=int,default=100)
    parser.add_argument('--burn-in',type=int,default=1500)
    parser.add_argument('--block-size',type=int,default=8)
    parser.add_argument('--filter-passes',type=int,choices=range(16),default=0,
                        help='Default 0: direct rounding/modulo; 15 reproduces legacy filtering. Neither guarantees agreement.')
    args = parser.parse_args()
    image = cv2.imread(str(args.image.resolve()),cv2.IMREAD_UNCHANGED)
    validate_image(image)
    count = block_count(image.shape,args.block_size)
    if args.burn_in < 0 or not args.model.is_file():
        parser.error('Nonnegative burn-in and an existing checkpoint are required.')
    # Filenames are a useful guard, not proof of checkpoint training provenance.
    import re
    named_node = re.search(r'node([123])',args.model.name.lower())
    if named_node and int(named_node.group(1)) != args.node:
        parser.error('Checkpoint filename and --node disagree.')
    if args.output.exists():
        parser.error('Output already exists. Use a new directory to preserve previous runs.')
    password = getpass.getpass('Experiment password (not stored): ')
    if not password:
        parser.error('Empty password is not accepted.')
    args.output.mkdir(parents=True,exist_ok=False)
    report = {'status':'running','node':args.node,'seed':args.seed,'burn_in':args.burn_in,
              'block_size':args.block_size,'sample_count':count,'model':str(args.model.resolve()),
              'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),
              'image_sha256':hashlib.sha256(image.tobytes()).hexdigest(),
              'deterministic':True,'attempts':1,'quantization':'direct' if args.filter_passes == 0 else 'legacy_filtered',
              'filter_passes':args.filter_passes,
              'scope':'Functional experiment, not authentication or security certification.'}
    code = 2
    try:
        raw_master,raw_slave = generate_streams(args.model,args.node,args.seed,args.burn_in,count)
        master = quantize_streams(raw_master,args.filter_passes)
        slave = quantize_streams(raw_slave,args.filter_passes)
        report.update(rule_diagnostics(master,slave,count))
        cipher,metadata = encrypt_image(image,master,password,args.block_size)
        recovered = decrypt_image(cipher,slave,metadata,password)
        report['exact_recovery'] = bool(np.array_equal(image,recovered))
        report['changed_component_rate'] = float(np.mean(image != recovered))
        report['bit_error_rate'] = float(np.unpackbits(image ^ recovered).mean())
        report['status'] = 'recovered' if report['exact_recovery'] else 'recovery_failed'
        fixed = tuple(np.ones(count,dtype=np.uint8) for _ in range(4))
        fixed_candidate = decrypt_image(cipher,fixed,metadata,password)
        report['fixed_rule_exact_recovery'] = bool(np.array_equal(image,fixed_candidate))
        report['warnings'] = ['Diagnostic prototype: discrete agreement and DNA cancellation remain unresolved.']
        if report['fixed_rule_exact_recovery']:
            report['warnings'].append('Fixed all-one receiver rules also recover this image; this sample does not establish a need for synchronized receiver rules.')
        if any(n == 1 for n in report['master_unique_per_stream']+report['slave_unique_per_stream']):
            report['warnings'].append('At least one rule stream is constant; recovery is not evidence of synchronization necessity.')
        write_lossless(args.output/'cipher.png',cipher)
        write_lossless(args.output/'recovered.png',recovered)
        (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        code = 0 if report['exact_recovery'] else 1
    except Exception as exc:
        report.update(status='error',error_type=type(exc).__name__,error=str(exc))
    finally:
        (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
