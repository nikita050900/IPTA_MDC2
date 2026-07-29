#!/usr/bin/env python3
import numpy as np, h5py, emcee
from scipy.stats import gaussian_kde
from enterprise_extensions import model_utils

H5 = '/scratch/na00078/projects/IPTA_MDC2/h5_files/'
megaparsec = 3.086e22; c = 299792458.0; Tsun = 1.327124400e20/c**3
Q = 0.95; F_INJ = np.log10(3.7e-9)
CHUNK = 2_000_000

def cols_from_raw(fn, names):
    with h5py.File(H5+fn, 'r') as h:
        pn = [p.decode() if isinstance(p, bytes) else p for p in h['par_names'][:]]
        idx = [pn.index(n) for n in names]
        d = h['samples_cold']; N = d.shape[1]
        out = np.empty((N, len(idx)), dtype=np.float64)
        for a in range(0, N, CHUNK):
            b = min(a+CHUNK, N)
            out[a:b] = d[0, a:b, idx]
    return [out[:, j] for j in range(len(idx))]

def cols_from_outfile(fn, names):
    with h5py.File(H5+fn, 'r') as f:
        pn = [p.decode() if isinstance(p, bytes) else p for p in f['par_names'][:]]
        sc = f['samples_cold'][0, :, :]
        return pn, [sc[:, pn.index(n)].astype(np.float64) for n in names]

def tau_of(x):
    th = 10 if len(x) > 5_000_000 else 1
    return th*float(emcee.autocorr.integrated_time(x[::th], quiet=True)[0])

def ul_err(log_samp, n_eff):
    v = 10**log_samp
    ul = np.quantile(v, Q)
    f = gaussian_kde(v if len(v) < 500_000 else v[::max(1, len(v)//500_000)]).evaluate([ul])[0]
    err = np.sqrt(Q*(1-Q))/(f*np.sqrt(n_eff))
    return np.log10(ul), err/(ul*np.log(10))

def loki_h0(fn):
    pn, _ = cols_from_outfile(fn, [])
    names = ['0_log10_mc', '0_log10_dist']
    fc = [n for n in pn if 'freq' in n.lower() or 'fgw' in n.lower()]
    _, cols = cols_from_outfile(fn, names + fc[:1])
    mc, dist = cols[0], cols[1]
    fq = cols[2] if fc else np.full_like(mc, F_INJ)
    if fc and np.all(fq == fq[0]): fq = np.full_like(mc, F_INJ)
    h0 = np.log10(2.0)+(5/3)*(mc+np.log10(Tsun))+(2/3)*(np.log10(np.pi)+fq)+np.log10(c)-(dist+np.log10(megaparsec))
    return mc, h0

print('========== 5.1 Table 5, loki B10 ==========', flush=True)
for tag, fn in [('N broad', 'G2D2_broad_detect_loki_100M_lastTOA_4core_outfile.h5'),
                ('O fixed', 'G2D2_fixed_detect_loki_100M_lastTOA_4core_outfile.h5')]:
    mc, h0 = loki_h0(fn)
    print(tag, 'log10 h0 range %.3f to %.3f' % (h0.min(), h0.max()), flush=True)
    b, e = model_utils.bayes_fac(samples=h0, logAmax=-11)
    print(tag, 'B10 full 1e8: %.1f +/- %.1f' % (b, e), flush=True)
    del mc, h0

print('========== 5.2 Runs A and B ==========', flush=True)
for tag, fn in [('A broad', 'G2D1_broad_detect_4core.h5'),
                ('B fixed', 'G2D1_narrow_detect_4core.h5')]:
    h, gwb = cols_from_raw(fn, ['0_log10_h', 'gwb_log10_A'])
    b, e = model_utils.bayes_fac(samples=h, logAmax=-11)
    print('%s  B10 %.3f +/- %.3f   log10 AGWB median %.3f' % (tag, b, e, np.median(gwb)), flush=True)
    del h, gwb

print('========== 5.3 Table 6, limits with Eq 7 errors ==========', flush=True)
for tag, fn in [('QuickCW broad (C)', 'G2D1_broad_UL_4core_outfile.h5'),
                ('QuickCW fixed (D)', 'G2D1_narrow_UL_4core_outfile.h5')]:
    _, (mc, h) = cols_from_outfile(fn, ['0_log10_mc', '0_log10_h'])
    n = len(mc)
    umc, emc = ul_err(mc, n); uh, eh = ul_err(h, n)
    print('%s  n=%d  Mc UL %.3f +/- %.3f   h0 UL %.3f +/- %.3f' % (tag, n, umc, emc, uh, eh), flush=True)
for tag, fn in [('loki broad (L)', 'G2D2_broad_UL_loki_100M_lastTOA_4core_outfile.h5'),
                ('loki fixed (M)', 'G2D2_fixed_UL_loki_100M_lastTOA_ntol_10_4core_outfile.h5')]:
    mc, h0 = loki_h0(fn)
    ess = len(mc)/tau_of(mc)
    umc, emc = ul_err(mc, ess); uh, eh = ul_err(h0, ess)
    print('%s  N=%d ESS=%.0f  Mc UL %.3f +/- %.3f   h0 UL %.3f +/- %.3f' % (tag, len(mc), ess, umc, emc, uh, eh), flush=True)
    del mc, h0

print('========== 5.4 injected sky position, Run G ==========', flush=True)
import healpy as hp
_, (ct, ph) = cols_from_outfile('G2D2_detect_allsky_4core_outfile.h5', ['0_cos_gwtheta', '0_gwphi'])
nside = 32
pix = hp.ang2pix(nside, np.arccos(ct), ph)
cnt = np.bincount(pix, minlength=hp.nside2npix(nside)).astype(float)
p = cnt/cnt.sum()
inj = hp.ang2pix(nside, 0.6387905062299246, 3.3335788713091694)
lev = p[p >= p[inj]].sum()
print('credible level containing injection: %.1f%% (inside 68%% region: %s)' % (100*lev, lev <= 0.68), flush=True)
print('=== DONE ===', flush=True)
