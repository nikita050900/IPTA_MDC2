%matplotlib inline
%config InlineBackend.figure_format = 'retina'
%load_ext autoreload
%autoreload 2

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import corner
import h5py

from scipy.stats import gaussian_kde
from enterprise_extensions import model_utils

#===CELL===
# Switch between targets here
TARGET = 'NGC3115'   # or 'NGC1316'

if TARGET == 'NGC3115':
    target_d_L = 9.7    # Mpc, Tonry et al. 2001 SBF
    h5_path = '/scratch/na00078/projects/15yr_broad_targeted/loki/h5_files/NGC3115_gauss_dL_UL_broad.h5'
elif TARGET == 'NGC1316':
    target_d_L = 20.8   # Mpc, Cantiello et al. 2013 SBF
    h5_path = '/scratch/na00078/projects/15yr_broad_targeted/loki/h5_files/NGC1316_gauss_dL_UL_broad.h5'
else:
    raise ValueError(f'Unknown target: {TARGET}')

dist_sigma_frac = 0.10

base = f'{TARGET}_broad_UL_gauss_dL_{target_d_L}_'
crnr_plt_title1 = base + 'corner_plot_log'
crnr_plt_title1_lin = base + 'corner_plot_linear'
crnr_plt_title2 = base + 'corner_plot_with_dL'
trace_plot_title = base + 'trace_plot'
UL_histogram_title = base + 'Mc_histogram'

print(f'Target: {TARGET}')
print(f'dL = {target_d_L} Mpc, sigma = {dist_sigma_frac*100:.0f}%')
print(f'h5 file: {h5_path}')

#===CELL===
first_n_param = 8

with h5py.File(h5_path, 'r') as f:
    Ts = f['T-ladder'][...]
    samples_cold = f['samples_cold'][:, :, :first_n_param]
    log_likelihood = f['log_likelihood'][:1, :]
    par_names = [x.decode('UTF-8') for x in list(f['par_names'])]
    acc_fraction = f['acc_fraction'][...]
    fisher_diag = f['fisher_diag'][...]

print(f'samples_cold shape: {samples_cold.shape}')
print(f'log_likelihood shape: {log_likelihood.shape}')
print('First 8 par_names:')
for i, n in enumerate(par_names[:first_n_param]):
    print(f'  [{i}] {n}')

#===CELL===
# Find parameter indices dynamically
idx = {name: par_names.index(name) for name in [
    '0_cos_gwtheta', '0_cos_inc', '0_gwphi',
    '0_log10_dist', '0_log10_fgw', '0_log10_mc',
    '0_phase0', '0_psi']}

print('Parameter indices:')
for k, v in idx.items():
    print(f'  {k:18s} -> {v}')

#===CELL===
burnin = 100_000 
thin = 1

megaparsec = 3.086e+22
speed_of_light = 299792458.0
T_sun = 1.327124400e20 / speed_of_light**3

cos_theta = samples_cold[0][burnin::thin, idx['0_cos_gwtheta']]
cos_inc   = samples_cold[0][burnin::thin, idx['0_cos_inc']]
gwphi     = samples_cold[0][burnin::thin, idx['0_gwphi']]
log10_d_L = samples_cold[0][burnin::thin, idx['0_log10_dist']]
log10_fgw = samples_cold[0][burnin::thin, idx['0_log10_fgw']]
log10_mc  = samples_cold[0][burnin::thin, idx['0_log10_mc']]
phase0    = samples_cold[0][burnin::thin, idx['0_phase0']]
psi       = samples_cold[0][burnin::thin, idx['0_psi']]

mmm = 10**log10_mc
fff = 10**log10_fgw
dLL = 10**log10_d_L

h_amp = 2.0 * (mmm * T_sun)**(5.0/3.0) * (np.pi * fff)**(2.0/3.0) * speed_of_light / (dLL * megaparsec)
log10_h = np.log10(h_amp)

print(f'Number of samples: {len(log10_h)}')
print(f'log10_h range:    [{log10_h.min():.3f}, {log10_h.max():.3f}]')
print(f'log10_mc range:   [{log10_mc.min():.3f}, {log10_mc.max():.3f}]')
print(f'log10_fgw range:  [{log10_fgw.min():.3f}, {log10_fgw.max():.3f}]')
print(f'log10_dL range:   [{log10_d_L.min():.3f}, {log10_d_L.max():.3f}]')
print(f'dL range (Mpc):   [{dLL.min():.3f}, {dLL.max():.3f}]')
print()
print('Gaussian dL prior diagnostics:')
print(f'  log10_dL median:    {np.median(log10_d_L):.4f}')
print(f'  log10_dL 16/50/84:  {np.percentile(log10_d_L, [16, 50, 84])}')
print(f'  log10_dL 1/99:      {np.percentile(log10_d_L, [1, 99])}')
print(f'  Expected mean:      {np.log10(target_d_L):.4f}')
print(f'  Expected sigma:     {dist_sigma_frac/np.log(10):.4f}')

#===CELL===
samples2plot = np.column_stack([
    cos_theta, cos_inc, gwphi,
    log10_fgw, log10_h, log10_mc,
    phase0, psi, log10_d_L
])

print(f'samples2plot shape: {samples2plot.shape}')

#===CELL===
labels = [r'$\cos \iota$', r'$\log_{10} f_{\rm GW}$', r'$\log_{10} h$',
          r'$\log_{10} \mathcal{M}$', r'$\Phi_0$', r'$\psi$']

samples2plot_detect = np.column_stack([
    cos_inc, log10_fgw, log10_h, log10_mc, phase0, psi
])

ranges = [(-1, 1),
          (-8.7, -7),
          (-18, -11),
          (6, 10),
          (0, 2*np.pi),
          (0, np.pi)]

fontsize_titles = 16

fig = corner.corner(
    samples2plot_detect,
    labels=labels,
    show_titles=True,
    quantiles=[0.16, 0.5, 0.84],
    range=ranges,
    hist_kwargs={'density': True},
    title_kwargs={'fontsize': fontsize_titles})

for ax in fig.get_axes():
    ax.tick_params(axis='both', labelsize=14)
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)

n = len(labels)
for i, ax in enumerate(fig.axes):
    if i == 0:
        Xs = np.linspace(-1, 1)
        ax.plot(Xs, Xs*0 + 0.5, color='xkcd:green')
    elif i == (n+1):
        Xs = np.linspace(-8.7, -7)
        ax.plot(Xs, Xs*0 + 1.0/1.7, color='xkcd:green')
    elif i == 2*(n+1):
        # LinearExp prior on log10_h: p(log10_h) propto 10^log10_h
        Xs = np.linspace(-18, -11)
        ax.plot(Xs, np.log(10)*10**Xs / (10**(-11) - 10**(-18)), color='xkcd:green')
    elif i == 3*(n+1):
        # LinearExp prior on log10_mc
        Xs = np.linspace(6, 10)
        ax.plot(Xs, np.log(10)*10**Xs / (10**10 - 10**6), color='xkcd:green')
    elif i == 4*(n+1):
        Xs = np.linspace(0, 2*np.pi)
        ax.plot(Xs, Xs*0 + 1.0/(2*np.pi), color='xkcd:green')
    elif i == 5*(n+1):
        Xs = np.linspace(0, np.pi)
        ax.plot(Xs, Xs*0 + 1.0/np.pi, color='xkcd:green')

fig.suptitle(crnr_plt_title1, fontsize=25, y=1.05)

#===CELL===
# Linear-scale variant: useful for UL visualization since the LinearExp
# prior on log10_mc maps to a uniform-in-Mc prior
labels_lin = [r'$\cos \iota$',
              r'$\log_{10} f_{\rm GW}$',
              r'$h$',
              r'$\mathcal{M}$',
              r'$\Phi_0$',
              r'$\psi$']

samples2plot_lin = np.column_stack([
    cos_inc, log10_fgw, h_amp, mmm, phase0, psi
])

ranges_lin = [(-1, 1),
              (-8.7, -7),
              (h_amp.min(), h_amp.max()),
              (mmm.min(), mmm.max()),
              (0, 2*np.pi),
              (0, np.pi)]

fig = corner.corner(
    samples2plot_lin,
    labels=labels_lin,
    show_titles=True,
    range=ranges_lin,
    hist_kwargs={'density': True})

for ax in fig.get_axes():
    ax.tick_params(axis='both', labelsize=14)
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)

fig.suptitle(crnr_plt_title1_lin, fontsize=25, y=1.05)

#===CELL===
# Build samples array with linear h and Mc instead of log
samples2plot_lin_hM = samples2plot.copy()
samples2plot_lin_hM[:, 4] = 10**samples2plot[:, 4]   # log10_h  -> h
samples2plot_lin_hM[:, 5] = 10**samples2plot[:, 5]   # log10_mc -> Mc

labels_full = [r'$\cos \theta$', r'$\cos \iota$', r'$\phi$',
               r'$\log_{10} f_{\rm GW}$', r'$h$',
               r'$\mathcal{M}$', r'$\Phi_0$', r'$\psi$',
               r'$\log_{10} d_L$']

# Use percentile-based ranges so extreme tails don't squash the plot
h_lo, h_hi = np.percentile(samples2plot_lin_hM[:, 4], [0.5, 99.5])
mc_lo, mc_hi = np.percentile(samples2plot_lin_hM[:, 5], [0.5, 99.5])

ranges_full = [(-1, 1), (-1, 1), (0, 2*np.pi),
               (-8.7, -7),
               (h_lo, h_hi),
               (mc_lo, mc_hi),
               (0, 2*np.pi), (0, np.pi),
               (np.log10(target_d_L*(1-3*dist_sigma_frac)),
                np.log10(target_d_L*(1+3*dist_sigma_frac)))]

fontsize_labels = 18
fontsize_titles = 16

fig = corner.corner(
    samples2plot_lin_hM,
    labels=labels_full,
    show_titles=True,
    quantiles=[0.16, 0.5, 0.84],
    range=ranges_full,
    hist_kwargs={'density': True},
    label_kwargs={'fontsize': fontsize_labels},
    title_kwargs={'fontsize': fontsize_titles})

n = len(labels_full)
for i, ax in enumerate(fig.axes):
    if i == 0 or i == (n+1):
        Xs = np.linspace(-1, 1)
        ax.plot(Xs, Xs*0 + 0.5, color='xkcd:green')
    elif i == 2*(n+1) or i == 6*(n+1):
        Xs = np.linspace(0, 2*np.pi)
        ax.plot(Xs, Xs*0 + 1.0/(2*np.pi), color='xkcd:green')
    elif i == 3*(n+1):
        Xs = np.linspace(-8.7, -7)
        ax.plot(Xs, Xs*0 + 1.0/1.7, color='xkcd:green')
    elif i == 4*(n+1):
        # Flat prior on h within the displayed range. The prior is uniform
        # in h across [10^-18, 10^-11] so the density is tiny on this scale;
        # we draw it spanning the displayed range so it shows as a faint line.
        Xs = np.linspace(h_lo, h_hi)
        ax.plot(Xs, Xs*0 + 1.0/(10**(-11) - 10**(-18)), color='xkcd:green')
    elif i == 5*(n+1):
        Xs = np.linspace(mc_lo, mc_hi)
        ax.plot(Xs, Xs*0 + 1.0/(10**10 - 10**6), color='xkcd:green')
    elif i == 7*(n+1):
        Xs = np.linspace(0, np.pi)
        ax.plot(Xs, Xs*0 + 1.0/np.pi, color='xkcd:green')
    elif i == 8*(n+1):
        mu = np.log10(target_d_L)
        sigma = dist_sigma_frac / np.log(10.0)
        Xs = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
        ax.plot(Xs, np.exp(-0.5*((Xs-mu)/sigma)**2) / (sigma*np.sqrt(2*np.pi)),
                color='xkcd:green')

fig.suptitle(crnr_plt_title2, fontsize=25)

#===CELL===
titles = ['gw cos_theta', 'cos_inc', 'gw phi', 'log10 fGW',
          'log10 h', 'log10 Mc', 'phase', 'psi', 'log10 dL']

row, column = 3, 3
fig, axs = plt.subplots(row, column, figsize=(30, 20))
fig.tight_layout(h_pad=3, w_pad=2)

j = -1
for i in range(len(titles)):
    if i % column == 0:
        j += 1
    axs[j, i % column].plot(np.arange(samples2plot.shape[0]), samples2plot[:, i])
    axs[j, i % column].set_title(titles[i], fontsize=30)
    axs[j, i % column].tick_params(axis='both', labelsize=25)
    axs[j, i % column].set_xlabel('sample index', fontsize=25)
    axs[j, i % column].set_ylabel('value', fontsize=25)

fig.suptitle(trace_plot_title, fontsize=28, y=1.08)
plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.show()

#===CELL===
mc_samples = 10**log10_mc
N = len(mc_samples)
q = 0.95

UL = np.quantile(mc_samples, q)
print(f'95% Chirp Mass Upper Limit = {UL:.5e}')
print(f'95% Chirp Mass Upper Limit log10 = {np.log10(UL):.5f}')

bins = 20
counts, bin_edges, _ = plt.hist(mc_samples, bins=bins, density=True, color='green')
plt.axvline(x=UL, color='red', label=f'95% UL = {UL:.3e}')
plt.xlabel(r'$\mathcal{M}_c$ [$M_\odot$]')
plt.ylabel('Posterior Probability Density')
plt.title(UL_histogram_title)
plt.legend()
plt.show()

bin_index = np.digitize(UL, bin_edges) - 1
y_value = counts[bin_index]
sigma_hist = np.sqrt(q*(1-q)/N) / y_value
print(f'95% UL error (histogram method) = {sigma_hist:.5e}')

#===CELL===
kde = gaussian_kde(mc_samples)
f_q = kde.evaluate([UL])[0]
UL_err = np.sqrt(q*(1-q)) / (f_q * np.sqrt(N))

print(f'95% UL (Mc) = {UL:.3e} +/- {UL_err:.1e}')

def linear_to_log10(UL, UL_err):
    logUL = np.log10(UL)
    logUL_err = UL_err / (UL * np.log(10))
    return logUL, logUL_err

logUL, logUL_err = linear_to_log10(UL, UL_err)
print(f'log10 UL = {logUL:.3f} +/- {logUL_err:.4f}')

#===CELL===
h_samples = 10**log10_h
UL_h = np.quantile(h_samples, q)
print(f'95% Strain Upper Limit = {UL_h:.5e}')
print(f'95% Strain Upper Limit log10 = {np.log10(UL_h):.5f}')

kde_h = gaussian_kde(h_samples)
f_q_h = kde_h.evaluate([UL_h])[0]
UL_h_err = np.sqrt(q*(1-q)) / (f_q_h * np.sqrt(N))

logUL_h, logUL_h_err = linear_to_log10(UL_h, UL_h_err)
print(f'log10 h UL = {logUL_h:.3f} +/- {logUL_h_err:.4f}')

plt.figure(figsize=(8, 5))
plt.hist(log10_h, bins=30, density=True, color='steelblue', alpha=0.7)
plt.axvline(x=logUL_h, color='red', label=f'95% UL: log10(h) = {logUL_h:.2f}')
plt.xlabel(r'$\log_{10} h$')
plt.ylabel('density')
plt.title(f'{TARGET} log10_h posterior')
plt.legend()
plt.show()

#===CELL===
print(f'='*60)
print(f'  {TARGET} broad freq targeted UL (Gaussian dL prior)')
print(f'='*60)
print(f'  Target distance:        {target_d_L} Mpc (+/- {dist_sigma_frac*100:.0f}%)')
print(f'  N samples:              {N}')
print(f'  log10_dL median:        {np.median(log10_d_L):.4f} (expected {np.log10(target_d_L):.4f})')
print()
print(f'  95% UL on Mc:           {UL:.3e} M_sun')
print(f'  95% UL on log10(Mc):    {logUL:.3f} +/- {logUL_err:.4f}')
print()
print(f'  95% UL on h:            {UL_h:.3e}')
print(f'  95% UL on log10(h):     {logUL_h:.3f} +/- {logUL_h_err:.4f}')
print(f'='*60)