import time as _time
import numpy as _np
try:
    _x0 = mcc.x0s[0]; _FLI = mcc.FLIs[0]
    _params = dict(zip(mcc.par_names, mcc.samples[0,0,:]))
    for _ in range(100): _FLI.get_lnlikelihood(_x0)
    for _ in range(3): mcc.flm.recompute_FastLike(mcc.FLI_swap, _x0, _params)
    print('warm-up done', flush=True)
    _NP=100000; _reps=[]
    for _ in range(7):
        _t=_time.perf_counter()
        for _ in range(_NP): _FLI.get_lnlikelihood(_x0)
        _reps.append((_time.perf_counter()-_t)/_NP)
    print('LOKI PROJECTION: median %.4g ms (min %.4g, max %.4g)'%(1e3*_np.median(_reps),1e3*min(_reps),1e3*max(_reps)), flush=True)
    _NS=50; _reps2=[]
    for _ in range(7):
        _t=_time.perf_counter()
        for _ in range(_NS): mcc.flm.recompute_FastLike(mcc.FLI_swap, _x0, _params)
        _reps2.append((_time.perf_counter()-_t)/_NS)
    print('LOKI SHAPE: median %.4g ms (min %.4g, max %.4g)'%(1e3*_np.median(_reps2),1e3*min(_reps2),1e3*max(_reps2)), flush=True)
    _tp,_ts=_np.median(_reps),_np.median(_reps2)
    print('implied per-iter cost (block 1000): %.4g ms'%(1e3*((_ts+999*_tp)/1000)), flush=True)
except AttributeError as e:
    print('ATTR MISMATCH:', e, flush=True)
    print([a for a in dir(mcc) if not a.startswith('__')], flush=True)
