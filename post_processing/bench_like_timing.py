"""Benchmark for Section 6.1 XX/YY: per-update likelihood cost of QuickCW
on the IPTA MDC2 dataset. Mirrors runQuickMCMC_G2D2_narrow_with_tref.py
(Run F setup) with savefile=None. Times:
(a) projection update = FLIs[0].get_lnlikelihood(x0) using precomputed
    filter inner products (the cheap update, 9999 of 10000 per block);
(b) shape update = flm.recompute_FastLike(FLI_swap, x0, params), which
    rebuilds the inner products (1 of 10000 per block).
Medians over repeated batches after Numba JIT warm-up, like Becsy et al.
Appendix B Table III."""
import numpy as np, pickle, time
import QuickCW.QuickCW_G2D2_with_tref as QuickCW
from QuickCW.QuickMCMCUtils import ChainParams

data_pkl = '/scratch/na00078/projects/IPTA_MDC2/IPTA_MDC2_data/psr_objects/G2D2_IPTA_MDC2_all_pulsars.pkl'
noisefile = '/scratch/na00078/projects/IPTA_MDC2/IPTA_MDC2_data/noise_files/fit_psr_noise_dataset2.json'
with open(data_pkl,'rb') as f:
    psrs = pickle.load(f)
print('pulsars:', len(psrs), flush=True)

cos_gwtheta = np.cos(0.6387905062299246)
gwphi = 3.3335788713091694
TargFreq = 3.7e-09
n_int_block = 10000
chain_params = ChainParams(3., 4, 1,
    freq_bounds=np.array([TargFreq-1e-21, TargFreq+1e-21]),
    n_int_block=n_int_block, save_every_n=100000,
    fisher_eig_downsample=2000, rn_emp_dist_file=None,
    savefile=None, thin=10,
    prior_draw_prob=0.2, de_prob=0.6, fisher_prob=0.3,
    dist_jump_weight=0.2, rn_jump_weight=0.3, gwb_jump_weight=0.1,
    common_jump_weight=0.2, all_jump_weight=0.2,
    fix_rn=False, zero_rn=False, fix_gwb=False, zero_gwb=False,
    cos_gwtheta_bounds=[cos_gwtheta-1e-8, cos_gwtheta+1e-8],
    gwphi_bounds=[gwphi-1e-8, gwphi+1e-8])

t0 = time.time()
pta, mcc = QuickCW.QuickCW(chain_params, psrs, amplitude_prior='detection',
                           psr_distance_file=None, noise_json=noisefile)
print('setup done in %.1f s' % (time.time()-t0), flush=True)

x0 = mcc.x0s[0]
FLI = mcc.FLIs[0]
params = dict(zip(mcc.par_names, mcc.samples[0,0,:]))

# JIT warm-up
for _ in range(100):
    FLI.get_lnlikelihood(x0)
for _ in range(3):
    mcc.flm.recompute_FastLike(mcc.FLI_swap, x0, params)
print('warm-up done', flush=True)

# (a) projection update
NP = 100000
reps = []
for _ in range(7):
    t = time.perf_counter()
    for _ in range(NP):
        FLI.get_lnlikelihood(x0)
    reps.append((time.perf_counter()-t)/NP)
print('PROJECTION update: median %.4g ms (min %.4g, max %.4g; 7 batches of %d)'
      % (1e3*np.median(reps), 1e3*min(reps), 1e3*max(reps), NP), flush=True)

# (b) shape update
NS = 50
reps2 = []
for _ in range(7):
    t = time.perf_counter()
    for _ in range(NS):
        mcc.flm.recompute_FastLike(mcc.FLI_swap, x0, params)
    reps2.append((time.perf_counter()-t)/NS)
print('SHAPE update: median %.4g ms (min %.4g, max %.4g; 7 batches of %d)'
      % (1e3*np.median(reps2), 1e3*min(reps2), 1e3*max(reps2), NS), flush=True)

tp, ts = np.median(reps), np.median(reps2)
print('implied mean per-iteration cost in a %d-iter block: %.4g ms (compare t_eval in tab:like_cost)'
      % (n_int_block, 1e3*((ts+(n_int_block-1)*tp)/n_int_block)), flush=True)
