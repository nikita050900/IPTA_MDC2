import sys, numpy as np, h5py, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, corner
from matplotlib.lines import Line2D
run = sys.argv[1]
base = '/anvil/projects/x-phy260442/projects/IPTA_MDC2/'
files = {'E': base+'h5_files/G2D2_broad_detect_tref_4core_UNMASKED_outfile.h5',
         'F': base+'h5_files/G2D2_narrow_detect_tref_4core_UNMASKED_outfile.h5'}
out = base+'post_processing/dl_corner_25sep/'
TSUN = 1.327124400e20/299792458.0**3; MPC_S = 3.086e22/299792458.0
DL_EM = 75.4; TOL = 0.01; THIN = 10
# column order: cos_gwtheta, cos_inc, gwphi, log10_fgw, log10_h, log10_mc, phase0, psi
def dl_mpc(lmc, lh, lf):
    return 2*(10**lmc*TSUN)**(5/3)*(np.pi*10**lf)**(2/3)/10**lh/MPC_S
un, ma = [], []
with h5py.File(files[run], 'r') as f:
    ds = f['samples_cold']; N = ds.shape[1]; B = 10_000_000
    for s in range(0, N, B):
        blk = ds[0, s:s+B, :]
        d = dl_mpc(blk[:,5], blk[:,4], blk[:,3])
        m = np.abs(d-DL_EM) <= TOL*DL_EM
        ma.append(blk[m]); un.append(blk[::THIN])
        print(run, s, m.sum(), flush=True)
un = np.concatenate(un); ma = np.concatenate(ma)
print('Nsurv', len(ma), 'Nthin', len(un))
cols = [5,4,3,1,6,7]
labels = [r'$\log_{10}(\mathcal{M}_c/M_\odot)$', r'$\log_{10} h_0$', r'$\log_{10}(f_{\rm GW}/{\rm Hz})$', r'$\cos\iota$', r'$\Phi_0$', r'$\psi$']
truths = [np.log10(4.3e9), -13.67, np.log10(3.7e-9), np.cos(0.841), None, None]
if run == 'F':
    keep = [0,1,3,4,5]
    cols = [cols[i] for i in keep]; labels = [labels[i] for i in keep]; truths = [truths[i] for i in keep]
U = un[:, cols]; M = ma[:, cols]
rng = []
for k in range(U.shape[1]):
    lo = min(np.percentile(U[:,k],0.5), np.percentile(M[:,k],0.5)); hi = max(np.percentile(U[:,k],99.5), np.percentile(M[:,k],99.5))
    pad = 0.03*(hi-lo); rng.append((lo-pad, hi+pad))
def kw(): return dict(labels=labels, range=rng, bins=40, smooth=1.0, levels=(0.393,0.865), plot_datapoints=False, plot_density=False, fill_contours=True, max_n_ticks=3, label_kwargs=dict(fontsize=22), hist_kwargs=dict(density=True, lw=1.8), contour_kwargs=dict(linewidths=1.2))
fig = corner.corner(U, color='C0', truths=truths, truth_color='k', **kw())
corner.corner(M, color='C3', fig=fig, **kw())
for ax in fig.get_axes(): ax.tick_params(labelsize=15)
lab = {'E': 'Run E, broad $f_{\\rm GW}$', 'F': 'Run F, fixed $f_{\\rm GW}$'}[run]
h = [Line2D([],[],color='C0',lw=3,label='before $d_L$ mask'), Line2D([],[],color='C3',lw=3,label='after $d_L$ mask ($\\eta_{\\rm tol}=1\\%%$, $N_{\\rm surv}=%d$)'%len(ma)), Line2D([],[],color='k',lw=2,label='injected')]
fig.legend(handles=h, loc='upper right', bbox_to_anchor=(0.98,0.98), fontsize=20, title=lab, title_fontsize=22, frameon=False)
fig.savefig(out+'dl_mask_corner_%s.png'%run, dpi=150, bbox_inches='tight'); fig.savefig(out+'dl_mask_corner_%s.pdf'%run, bbox_inches='tight')
json.dump({'Nsurv': int(len(ma)), 'Nthin': int(len(un)), 'range': rng}, open(out+'meta_%s.json'%run,'w'))
print('done', run)
