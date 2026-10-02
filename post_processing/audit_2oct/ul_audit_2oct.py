# ul_audit_2oct.py : upper limit uncertainty audit for the CQG paper (2 Oct 2026)
import numpy as np, h5py, json, time, sys
from scipy.stats import gaussian_kde
try:
    import emcee
except Exception:
    emcee = None
H5 = '/scratch/na00078/projects/IPTA_MDC2/h5_files/'
OUTD = '/scratch/na00078/projects/IPTA_MDC2/post_processing/audit_2oct/'
LOG = open(OUTD + 'ul_audit.log', 'a')
def log(*a):
    s = time.strftime('%H:%M:%S ') + ' '.join(str(x) for x in a); print(s); LOG.write(s + '\n'); LOG.flush()
Q = 0.95; TARGET = 75.4; TOL = 0.01; NBOOT = 300
mpc = 3.086e22; c = 299792458.0; Tsun = 1.327124400e20 / c**3
CONST = np.log10(2 * Tsun**(5/3) * np.pi**(2/3) * c / mpc)   # log10 dL[Mpc] = CONST + 5/3 mc + 2/3 f - h
DF = 1 / (14.962 * 365.25 * 86400.0)
LF_EM = np.log10(3.7e-9)
TAU = dict(C=dict(mc=11136.91, h=403.90), D=dict(mc=9384.82, h=12.30), L=dict(mc=46849.0), M=dict(mc=21358.0))
CHUNK = 5_000_000
OUT = {}

def read(fn, names):
    with h5py.File(H5 + fn, 'r') as f:
        pn = [p.decode() if isinstance(p, bytes) else str(p) for p in f['par_names'][:]]
        ds = f['samples_cold']; N = ds.shape[1]
        idx = [pn.index(n) for n in names]; lo, hi = min(idx), max(idx) + 1
        out = {n: np.empty(N, np.float64) for n in names}
        for a in range(0, N, CHUNK):
            b = min(a + CHUNK, N); blk = ds[0, a:b, lo:hi]
            for n, i in zip(names, idx): out[n][a:b] = blk[:, i - lo]
    log('read', fn, names, N); return out, N

def groups_of(vals, pos, L):
    bid = pos // max(int(L), 1)
    o = np.argsort(bid, kind='stable'); bs, vs = bid[o], vals[o]
    edges = np.searchsorted(bs, np.unique(bs)); return np.split(vs, edges[1:])

def boot(vals, pos, L, rng, nboot=NBOOT):
    g = groups_of(vals, pos, L); res = np.empty(nboot)
    for k in range(nboot):
        pick = rng.integers(0, len(g), len(g))
        res[k] = np.quantile(np.concatenate([g[i] for i in pick]), Q)
    return float(np.quantile(vals, Q)), float(res.std(ddof=1)), len(g)

def eq5(logv, n_eff):
    v = 10**logv; ul = np.quantile(v, Q)
    sub = v if len(v) < 500_000 else v[::max(1, len(v)//500_000)]
    f = gaussian_kde(sub).evaluate([ul])[0]
    err = np.sqrt(Q*(1-Q)) / (f * np.sqrt(max(n_eff, 1.0)))
    return float(np.log10(ul)), float(err / (ul * np.log(10)))

def tau_of(x):
    if emcee is None: return np.nan
    th = 10 if len(x) > 5_000_000 else 1
    return th * float(emcee.autocorr.integrated_time(x[::th], quiet=True)[0])

def perbin(tag, lf, lm, pos, N, Lblk, rho_fig, rng):
    f = 10**lf
    edges = np.arange(f.min(), f.max() + DF, DF)
    k = np.digitize(f, edges) - 1
    rows = []
    log(tag, 'per bin: f_lo f_hi n states blocks UL errBoot errFigEq5 UL_h1 UL_h2 z_half')
    for b in range(len(edges) - 1):
        m = k == b; n = int(m.sum())
        if n == 0: continue
        p = pos[m]; x = lm[m]
        st = int(len(np.unique(np.c_[x, lf[m]], axis=0)))
        ul, se, nb = boot(x, p, Lblk, rng) if n > 1 else (float(x[0]), np.nan, 1)
        ulf, ef = eq5(x, n * rho_fig) if n >= 2 else (np.nan, np.nan)
        x1 = x[p < N//2]; x2 = x[p >= N//2]
        u1 = float(np.quantile(x1, Q)) if x1.size else np.nan
        u2 = float(np.quantile(x2, Q)) if x2.size else np.nan
        z = (u1 - u2) / (2 * se) if (se and se > 0) else np.nan
        r = dict(f_lo=edges[b]*1e9, f_hi=edges[b+1]*1e9, n=n, states=st, blocks=nb, UL=ul, err_boot=se,
                 err_fig=ef, UL_h1=u1, UL_h2=u2, z_half=z, n1=int(x1.size), n2=int(x2.size), plotted_fig=n >= 50)
        rows.append(r)
        log(tag, '%.2f %.2f %d %d %d %.4f %.4f %.4f %.4f %.4f %.2f' % (r['f_lo'], r['f_hi'], n, st, nb, ul, se, ef, u1, u2, z))
    return rows

def which():
    return sys.argv[1:] if len(sys.argv) > 1 else ['ENT', 'D', 'M', 'C', 'L']

for job in which():
  try:
    rng = np.random.default_rng(11)
    if job == 'ENT':
        for tag, fn in (('I', 'core_single_MDC2_DS1.h5'), ('H', 'core_single_MDC2_DS1_varyfgw.h5')):
            with h5py.File(H5 + fn, 'r') as fh:
                pn = [p.decode() if isinstance(p, bytes) else str(p) for p in fh['params'][:]]
                ch = fh['chain'][3000:, :]
            mc = ch[:, pn.index('log10_mc')]
            lf = ch[:, pn.index('log10_fgw')] if 'log10_fgw' in pn else np.full_like(mc, LF_EM)
            h = CONST + (5/3)*mc + (2/3)*lf - np.log10(TARGET)
            n = len(mc); tmc = tau_of(mc); th = tau_of(h); pos = np.arange(n)
            r = dict(N=n, tau_mc=tmc, tau_h=th, ESS_mc=n/tmc)
            for nm, x in (('mc', mc), ('h', h)):
                r[nm+'_eq5_taumc'] = eq5(x, n/tmc); r[nm+'_eq5_tauown'] = eq5(x, n/(tmc if nm=='mc' else th))
                r[nm+'_boot_20taumc'] = boot(x, pos, 20*tmc, rng)
            OUT['ENT_'+tag] = r; log('ENT', tag, json.dumps(r))
    elif job in ('D', 'C'):
        fn = {'D': 'G2D1_narrow_UL_4core.h5', 'C': 'G2D1_broad_UL_4core.h5'}[job]
        C, N = read(fn, ['0_log10_fgw', '0_log10_h', '0_log10_mc'])
        lf = C['0_log10_fgw'] if job == 'C' else np.full(N, LF_EM)
        ldl = CONST + (5/3)*C['0_log10_mc'] + (2/3)*lf - C['0_log10_h']
        idx = np.flatnonzero(np.abs(10**ldl - TARGET) < TOL*TARGET); del ldl
        lm = C['0_log10_mc'][idx]; lh = C['0_log10_h'][idx]; lfm = lf[idx]; del C, lf
        t = TAU[job]
        r = dict(N=N, Nsurv=int(idx.size), distinct_mc=int(len(np.unique(lm))),
                 distinct_states=int(len(np.unique(np.c_[lm, lfm], axis=0))),
                 blocks_20taumc=int(len(np.unique(idx // int(20*t['mc'])))),
                 gap_median=float(np.median(np.diff(idx))))
        r['mc_boot_20taumc'] = boot(lm, idx, 20*t['mc'], rng)
        r['h_boot_20taumc'] = boot(lh, idx, 20*t['mc'], rng)
        r['h_boot_20tauh'] = boot(lh, idx, 20*t['h'], rng)
        r['mc_eq5_raw'] = eq5(lm, idx.size); r['h_eq5_raw'] = eq5(lh, idx.size)
        tre = tau_of(lm); r['tau_reindexed_mc'] = tre; r['rho_fig'] = min(1.0, (idx.size/tre)/idx.size)
        log(job, json.dumps(r))
        if job == 'C':
            r['bins'] = perbin('C', lfm, lm, idx, N, 20*t['mc'], r['rho_fig'], rng)
        OUT[job] = r
    elif job in ('M', 'L'):
        fn = {'M': 'G2D2_fixed_UL_loki_100M_lastTOA_ntol_10_4core.h5', 'L': 'G2D2_broad_UL_loki_100M_lastTOA_4core.h5'}[job]
        C, N = read(fn, ['0_log10_dist', '0_log10_fgw', '0_log10_mc'])
        lf = C['0_log10_fgw'] if job == 'L' else np.full(N, LF_EM)
        lm = C['0_log10_mc']; ld = C['0_log10_dist']
        lh = CONST + (5/3)*lm + (2/3)*lf - ld
        inwin = np.abs(10**ld - TARGET) <= TOL*TARGET*1.0000001
        t = TAU[job]; pos = np.arange(N)
        r = dict(N=N, frac_in_1pc_window=float(inwin.mean()), dist_range=[float(10**ld.min()), float(10**ld.max())])
        tm = tau_of(lm); th = tau_of(lh); r['tau_mc_recomputed'] = tm; r['tau_h'] = th
        r['mc_eq5_taumc'] = eq5(lm, N/t['mc']); r['h_eq5_taumc'] = eq5(lh, N/t['mc']); r['h_eq5_tauh'] = eq5(lh, N/th)
        TH = 100  # thin for the bootstrap only; blocks are 20 tau_mc >> 100 steps long
        r['mc_boot_20taumc'] = boot(lm[::TH], pos[::TH], 20*t['mc'], rng)
        r['h_boot_20taumc'] = boot(lh[::TH], pos[::TH], 20*t['mc'], rng)
        if job == 'L':
            r['distinct_states'] = int(len(np.unique(np.c_[lm, lf], axis=0)))
        log(job, json.dumps(r))
        if job == 'L':
            r['bins'] = perbin('L', lf[::TH], lm[::TH], pos[::TH], N, 20*t['mc'], TH/t['mc'], rng)
        OUT[job] = r
    json.dump(OUT, open(OUTD + 'ul_audit_%s.json' % '_'.join(which()), 'w'), indent=1, default=float)
  except Exception as e:
    import traceback; log('FAIL', job, traceback.format_exc())
log('=== DONE', which())
