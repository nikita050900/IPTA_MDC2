#!/usr/bin/env python
# fig2_unmasked_point.py: 95% UL on log10 Mc from the full (unmasked) Run D chain,
# same estimator and block bootstrap as final_paper_analysis.py (q95, 20 tau blocks,
# 300 realisations, rng seed 11). Writes fig2_unmasked_point.json next to the log.
import numpy as np, h5py, json, time, os
H5 = os.environ['PROJECT'] + '/projects/IPTA_MDC2/h5_files/'
PP = os.environ['PROJECT'] + '/projects/IPTA_MDC2/post_processing/'
Q = 0.95; NBOOT = 300
rng = np.random.default_rng(11)
with open(PP + 'sec6_ess_4core.json') as f:
    ESSJ = json.load(f)
tauD = ESSJ['D']['tau_by_param'].get('0_log10_mc', ESSJ['D']['tau_max'])

def boot(vals, pos, Lblk, stat, nboot=NBOOT):
    bid = pos // max(int(Lblk), 1)
    o = np.argsort(bid, kind='stable'); bs = bid[o]; vs = vals[o]
    edges = np.searchsorted(bs, np.unique(bs)); groups = np.split(vs, edges[1:])
    base = stat(vals); res = np.empty(nboot)
    for k in range(nboot):
        pick = rng.integers(0, len(groups), len(groups))
        res[k] = stat(np.concatenate([groups[i] for i in pick]))
    return base, res.std(ddof=1), len(groups)

t0 = time.time()
fn = H5 + 'G2D1_narrow_UL_4core_UNMASKED_outfile.h5'
with h5py.File(fn, 'r') as f:
    pn = [p.decode() if isinstance(p, bytes) else str(p) for p in f['par_names'][:]]
    j = pn.index('0_log10_mc')
    ds = f['samples_cold']; N = ds.shape[1]
    mc = np.empty(N, np.float64)
    CH = 5_000_000
    for a in range(0, N, CH):
        b = min(a + CH, N); mc[a:b] = ds[0, a:b, j]
print('read %d samples in %.0fs, tau_mc=%.0f' % (N, time.time() - t0, tauD), flush=True)
ul, se, nb = boot(mc, np.arange(N), 20 * tauD, lambda v: np.quantile(v, Q))
out = {'file': os.path.basename(fn), 'N': int(N), 'tau_mc': float(tauD), 'blocks': int(nb),
       'UL': float(ul), 'ULerr': float(se), 'nboot': NBOOT, 'seed': 11}
print(json.dumps(out, indent=1), flush=True)
with open(PP + 'fig2_unmasked_point.json', 'w') as f:
    json.dump(out, f, indent=1)
print('done in %.0fs' % (time.time() - t0))
