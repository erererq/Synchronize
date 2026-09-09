"""Replay the existing MCLE computation only; do not execute its plotting code."""
import contextlib
import io
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
source=ROOT/'Code/main/check_hop filed_chaos.py'
text=source.read_text(encoding='utf-8')
prefix=text.split('# 绘图',1)[0]
if prefix==text:
    raise RuntimeError('Cannot identify computation/plot boundary.')
scope={'__file__':str(source),'__name__':'mcle_audit'}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(prefix,str(source),'exec'),scope)
result={'source':str(source),'unmodified_calculation':True,
        'node_mcle':{str(i):float(scope[f'max_cle_{i}']) for i in [1,2,3]},
        'note':'Original state RK4, variational Euler at updated state, one-interval burn-in denominator mismatch retained. Does not execute plots or modify source.'}
target=ROOT/'logs/hopfield/mcle_order_audit'
target.mkdir(parents=True,exist_ok=True)
(target/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
