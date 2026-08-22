import numpy as np, h5py, emcee, time
H5 = '/scratch/na00078/projects/IPTA_MDC2/h5_files/'
OUT = 'scratch/projects/IPTA_MDC2/post_processing/bence_ess_check_output.txt'
L = []
def log(s):
    L.append(str(s))
    with open(OUT,'w') as f: f.write('\n'.join(L)+'\n')

path = H5+'G2D1_narrow_UL_4core.h5'
THIN = 10
t0 = time.time()
with h5py.File(path,'r') as f:
    pn = [p.decode() if isinstance(p,(bytes,np.bytes_)) else str(p) for p in f['par_names'][:]]
    ds = f['samples_cold']
    N, P = ds.shape[1], ds.shape[2]
    log('N=%d P=%d chunks=%s' % (N, P, ds.chunks))
    im = {p: pn.index(p) for p in ('0_log10_mc','0_log10_h0','0_log10_fgw') if p in pn}
    log('mask cols: %s' % im)
    thin_rows = []
    full = {p: np.empty(N, np.float32) for p in im}
    B = 2_000_000
    for s in range(0, N, B):
        e = min(s+B, N)
        blk = ds[0, s:e, :]
        thin_rows.append(blk[::THIN].copy())
        for p,i in im.items():
            full[p][s:e] = blk[:, i]
        del blk
        if (s//B) % 5 == 0: log('read %d / %d rows  (%.0f s)' % (e, N, time.time()-t0))
X = np.vstack(thin_rows); del thin_rows
log('thinned matrix %s  (%.0f s)' % (X.shape, time.time()-t0))

taus = {}
for i,p in enumerate(pn):
    x = np.asarray(X[:,i], np.float64)
    if x.std() == 0: continue
    try:
        taus[p] = THIN*float(emcee.autocorr.integrated_time(x, quiet=True)[0])
    except Exception:
        taus[p] = float('nan')
log('--- per parameter tau, full chain, sorted worst first (%.0f s) ---' % (time.time()-t0))
cwgwb = [p for p in taus if p.startswith('0_') or p.startswith('gwb') or 'com' in p]
for p,t in sorted(taus.items(), key=lambda kv: -(kv[1] if kv[1]==kv[1] else -1))[:25]:
    tag = ' *CW/GWB*' if p in cwgwb else ''
    log('%-28s tau=%12.0f  ESS=%12.0f%s' % (p,t,N/t,tag) if t==t else p+' nan')
log('--- CW+GWB params only ---')
for p in sorted(cwgwb, key=lambda q: -(taus[q] if taus[q]==taus[q] else -1)):
    t = taus[p]
    log('%-28s tau=%12.0f  ESS=%12.0f' % (p,t,N/t) if t==t else p+' nan')

c = 299792458.0; Mpc = 3.086e22; T_sun = 1.327124400e20/c**3
dhat, eta = 75.4, 0.01
mc  = 10**np.asarray(full['0_log10_mc'], np.float64)
h0  = 10**np.asarray(full['0_log10_h0'], np.float64)
fgw = 10**np.asarray(full['0_log10_fgw'], np.float64) if '0_log10_fgw' in full else np.full_like(mc, 3.7e-9)
dL = 2*c*(mc*T_sun)**(5/3)*(np.pi*fgw)**(2/3)/h0/Mpc
idx = np.flatnonzero((dL >= dhat*(1-eta)) & (dL <= dhat*(1+eta)))
log('--- 1%% mask around %.1f Mpc ---' % dhat)
log('survivors: %d of %d' % (idx.size, N))
g = np.diff(idx)
log('gaps: mean=%.0f median=%.0f min=%d max=%d first=%d last=%d' % (g.mean(), np.median(g), g.min(), g.max(), idx[0], idx[-1]))
sub = np.log10(mc)[idx]
try:
    ts = float(emcee.autocorr.integrated_time(sub, quiet=True)[0])
    log('naive subset tau(log10_mc) = %.1f -> naive ESS = %.0f' % (ts, idx.size/ts))
except Exception as e:
    log('naive subset tau failed: %s' % e)
log('--- corrected masked ESS = min(Nsurv, N/tau), CW+GWB ---')
for p in sorted(cwgwb, key=lambda q: -(taus[q] if taus[q]==taus[q] else -1)):
    t = taus[p]
    if t==t: log('%-28s %12.0f' % (p, min(idx.size, N/t)))
log('DONE (%.0f s)' % (time.time()-t0))
