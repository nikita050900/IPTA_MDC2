#!/usr/bin/env python
# fig2_mc_hist.py: log10 Mc histograms from the Run D chain, unmasked and with the
# 1% dL mask, for the Figure 2 inset. Writes fig2_mc_hist.json.
import numpy as np, h5py, json, os
H5 = os.environ['PROJECT'] + '/projects/IPTA_MDC2/h5_files/'
PP = os.environ['PROJECT'] + '/projects/IPTA_MDC2/post_processing/'
mpc = 3.086e22; c = 299792458.0; Tsun = 1.327124400e20 / c**3
CONST = np.log10(2 * Tsun**(5/3) * np.pi**(2/3) * c / mpc)
TARGET = 75.4
with h5py.File(H5 + 'G2D1_narrow_UL_4core_UNMASKED_outfile.h5', 'r') as f:
    pn = [p.decode() if isinstance(p, bytes) else str(p) for p in f['par_names'][:]]
    ds = f['samples_cold']
    mc = ds[0, :, pn.index('0_log10_mc')].astype(np.float64)
    h = ds[0, :, pn.index('0_log10_h')].astype(np.float64)
dl = 10**(CONST + (5/3)*mc + (2/3)*np.log10(3.7e-9) - h)
edges = np.linspace(7.0, 11.0, 81)
out = {'edges': edges.tolist(),
       'unmasked': np.histogram(mc, bins=edges, density=True)[0].tolist(),
       'q95_unmasked': float(np.quantile(mc, 0.95))}
m = np.abs(dl - TARGET) < 0.01 * TARGET
out['masked_1pc'] = np.histogram(mc[m], bins=edges, density=True)[0].tolist()
out['q95_masked_1pc'] = float(np.quantile(mc[m], 0.95))
out['N_masked_1pc'] = int(m.sum())
json.dump(out, open(PP + 'fig2_mc_hist.json', 'w'))
print('unmasked q95 %.4f, masked 1%% q95 %.4f, N %d' % (out['q95_unmasked'], out['q95_masked_1pc'], out['N_masked_1pc']))
