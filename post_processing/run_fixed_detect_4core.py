import matplotlib; matplotlib.use('Agg')
def display(*a,**k):
    pass

import numpy as np
import h5py
import matplotlib.pyplot as plt
import corner

np.random.seed(42)

QCW_FILE = "/scratch/na00078/projects/IPTA_MDC2/h5_files/G2D2_narrow_detect_tref_4core_dLmasked_75.400Mpc.h5"
LOKI_FILE = "/scratch/na00078/projects/IPTA_MDC2/h5_files/G2D2_fixed_detect_loki_100M_lastTOA_4core.h5"
ENT_FILE = "/scratch/na00078/projects/IPTA_MDC2/h5_files/G2D2_core_fixed_new.h5"

ENT_BURN = 3000

# constants
megaparsec = 3.086e+22
speed_of_light = 299792458.0
T_sun = 1.327124400e20 / speed_of_light**3
D_MPC_FIXED = 75.400
F_GW_FIXED = np.log10(3.7e-9)  # injected fGW for fixed runs

def derive_log10_h0(log10_mc, log10_fgw, log10_dL_Mpc):
    return (np.log10(2.0)
            + (5.0/3.0) * (log10_mc + np.log10(T_sun))
            + (2.0/3.0) * (np.log10(np.pi) + log10_fgw)
            - (log10_dL_Mpc + np.log10(megaparsec) - np.log10(speed_of_light)))

# load
print("Loading...")
with h5py.File(QCW_FILE, "r") as f:
    qc = f["samples_masked"][...]
    qc_pn = [p.decode() if isinstance(p, bytes) else p for p in f["par_names"][:]]
with h5py.File(LOKI_FILE, "r") as f:
    lk = f["samples_cold"][0, :, :]
    lk_pn = [p.decode() if isinstance(p, bytes) else p for p in f["par_names"][:]]
with h5py.File(ENT_FILE, "r") as f:
    en_all = f["chain"][...]
    en_pn = [p.decode() if isinstance(p, bytes) else p for p in f["params"][:]]

# Enterprise: apply burn-in
en = en_all[ENT_BURN:, :]

# subsample QCW and Loki to match Enterprise post burn-in
N_TARGET = len(en)
qc_idx = np.random.choice(len(qc), min(N_TARGET, len(qc)), replace=False)
qc_sub = qc[qc_idx]
lk_idx = np.random.choice(len(lk), N_TARGET, replace=False)
lk_sub = lk[lk_idx]

print(f"All pipelines subsampled to match Enterprise (N = {N_TARGET})")
print(f"My QCW: {len(qc_sub):,}, Loki: {len(lk_sub):,}, Enterprise: {len(en):,}")

# extract (no fGW since it's fixed)
qcw_cosi = qc_sub[:, qc_pn.index("0_cos_inc")]
qcw_h    = qc_sub[:, qc_pn.index("0_log10_h")]
qcw_mc   = qc_sub[:, qc_pn.index("0_log10_mc")]
qcw_amp  = qc_sub[:, qc_pn.index("gwb_log10_A")]
qcw_gam  = qc_sub[:, qc_pn.index("gwb_gamma")]
qcw_phi  = qc_sub[:, qc_pn.index("0_phase0")]
qcw_psi  = qc_sub[:, qc_pn.index("0_psi")]

loki_cosi = lk_sub[:, lk_pn.index("0_cos_inc")]
loki_mc   = lk_sub[:, lk_pn.index("0_log10_mc")]
loki_dist = lk_sub[:, lk_pn.index("0_log10_dist")]
loki_h    = derive_log10_h0(loki_mc, np.full_like(loki_mc, F_GW_FIXED), loki_dist)
loki_amp  = lk_sub[:, lk_pn.index("gwb_log10_A")]
loki_gam  = lk_sub[:, lk_pn.index("gwb_gamma")]
loki_phi  = lk_sub[:, lk_pn.index("0_phase0")]
loki_psi  = lk_sub[:, lk_pn.index("0_psi")]

ent_cosi = en[:, en_pn.index("cos_inc")]
ent_mc   = en[:, en_pn.index("log10_mc")]
ent_h    = derive_log10_h0(ent_mc, np.full_like(ent_mc, F_GW_FIXED),
                           np.full_like(ent_mc, np.log10(D_MPC_FIXED)))
ent_amp  = en[:, en_pn.index("crn_log10_A")]
ent_gam  = en[:, en_pn.index("crn_gamma")]
ent_phi  = en[:, en_pn.index("phase0")]
ent_psi  = en[:, en_pn.index("psi")]

# injected truth values (from MDC2 JSON, no fGW since it's fixed)
t_cosi = 0.8412486994612669
t_h    = -13.668773493298787
t_mc   = np.log10(4.3e9)
t_amp  = -15.070581074285707
t_phi  = 0.24434609527920614
t_psi  = 1.1187560505283651
# gamma_GWB is not in the injection JSON

CB, CG, CO, CT = "#0072B2", "#009E73", "#D55E00", "#CC79A7"

# corner setup (7 params, no fGW)
labels_c = [r"$\cos\iota$", r"$\log_{10} h_0$",
            r"$\log_{10}\mathcal{M}_c$", r"$\log_{10} A_{\rm GWB}$",
            r"$\gamma_{\rm GWB}$", r"$\Phi_0$", r"$\psi$"]
ranges_c = [(-1, 1), (-15, -12), (8.5, 10.0), (-20, -11),
            (0, 7), (0, 2*np.pi), (0, np.pi)]
truths_c = [t_cosi, t_h, t_mc, t_amp, None, t_phi, t_psi]

qcw_c  = np.column_stack([qcw_cosi, qcw_h, qcw_mc, qcw_amp, qcw_gam, qcw_phi, qcw_psi])
loki_c = np.column_stack([loki_cosi, loki_h, loki_mc, loki_amp, loki_gam, loki_phi, loki_psi])
ent_c  = np.column_stack([ent_cosi, ent_h, ent_mc, ent_amp, ent_gam, ent_phi, ent_psi])

def make_corner(data_list, color_list, label_list, title):
    fig = corner.corner(data_list[0], labels=labels_c, truths=truths_c,
                        truth_color=CT, range=ranges_c, color=color_list[0])
    for d, c in zip(data_list[1:], color_list[1:]):
        corner.corner(d, fig=fig, labels=labels_c, truths=truths_c,
                      truth_color=CT, range=ranges_c, color=c)
    handles = [plt.Line2D([0],[0], color=c, lw=2.5, label=l)
               for c, l in zip(color_list, label_list)]
    handles.append(plt.Line2D([0],[0], color=CT, lw=1.5, label="Injected"))
    fig.legend(handles=handles, loc="upper right", fontsize=11,
               bbox_to_anchor=(0.98, 0.98))
    fig.suptitle(title, fontsize=14, y=1.0)
    return fig

# corner plots
make_corner([qcw_c, loki_c], [CB, CG],
            ["My QCW", "Loki"], "G2D2 fixed: My QCW vs Loki")
plt.show()

make_corner([loki_c, ent_c], [CG, CO],
            ["Loki", "Enterprise"], "G2D2 fixed: Loki vs Enterprise")
plt.show()

make_corner([qcw_c, ent_c], [CB, CO],
            ["My QCW", "Enterprise"], "G2D2 fixed: My QCW vs Enterprise")
plt.show()

make_corner([qcw_c, loki_c, ent_c], [CB, CG, CO],
            ["My QCW", "Loki", "Enterprise"], "G2D2 fixed: All three")
plt.show()

# trace plots (7 rows, no fGW)
trace_rows = [
    (r"$\cos\iota$",              t_cosi, [(qcw_cosi, CB), (loki_cosi, CG), (ent_cosi, CO)]),
    (r"$\log_{10}h_0$",           t_h,    [(qcw_h, CB),    (loki_h, CG),    (ent_h, CO)]),
    (r"$\log_{10}\mathcal{M}_c$", t_mc,   [(qcw_mc, CB),   (loki_mc, CG),   (ent_mc, CO)]),
    (r"$\log_{10}A_{\rm GWB}$",   t_amp,  [(qcw_amp, CB),  (loki_amp, CG),  (ent_amp, CO)]),
    (r"$\gamma_{\rm GWB}$",       None,   [(qcw_gam, CB),  (loki_gam, CG),  (ent_gam, CO)]),
    (r"$\Phi_0$",                 t_phi,  [(qcw_phi, CB),  (loki_phi, CG),  (ent_phi, CO)]),
    (r"$\psi$",                   t_psi,  [(qcw_psi, CB),  (loki_psi, CG),  (ent_psi, CO)]),
]

titles = [f"My QCW (n={len(qcw_cosi):,})",
          f"Loki (n={len(loki_cosi):,})",
          f"Enterprise (n={len(ent_cosi):,}, burn {ENT_BURN})"]

fig, axes = plt.subplots(len(trace_rows), 3, figsize=(18, 21), sharex="col")
for row_i, (ylabel, truth_val, data) in enumerate(trace_rows):
    for col_i, (arr, color) in enumerate(data):
        axes[row_i, col_i].plot(arr, marker=".", linestyle="None",
                                markersize=1, color=color, rasterized=True)
        if truth_val is not None:
            axes[row_i, col_i].axhline(truth_val, color=CT, lw=1.5)
    axes[row_i, 0].set_ylabel(ylabel)

for col_i, t in enumerate(titles):
    axes[0, col_i].set_title(t)
for col_i in range(3):
    axes[-1, col_i].set_xlabel("sample index")

plt.tight_layout()
plt.show()

# parameter ranges
print("\n" + "=" * 70)
print("Parameter ranges (fGW fixed at log10 = -8.4318)")
print("=" * 70)
all_labels = ["cos_inc", "log10_h0", "log10_Mc",
              "log10_A_GWB", "gamma_GWB", "phase0", "psi"]

for name, arrs in [
    ("My QCW", (qcw_cosi, qcw_h, qcw_mc, qcw_amp, qcw_gam, qcw_phi, qcw_psi)),
    ("Loki",   (loki_cosi, loki_h, loki_mc, loki_amp, loki_gam, loki_phi, loki_psi)),
    ("Ent",    (ent_cosi, ent_h, ent_mc, ent_amp, ent_gam, ent_phi, ent_psi)),
]:
    print(f"\n{name} (n={len(arrs[0]):,}):")
    for lab, a in zip(all_labels, arrs):
        print(f"  {lab}: [{a.min():.3f}, {a.max():.3f}]  median={np.median(a):.3f}")

import numpy as np
import h5py
import matplotlib.pyplot as plt
import corner

np.random.seed(42)

QCW_FILE = "/scratch/na00078/projects/IPTA_MDC2/h5_files/G2D2_narrow_detect_tref_4core_dLmasked_75.400Mpc.h5"
LOKI_FILE = "/scratch/na00078/projects/IPTA_MDC2/h5_files/G2D2_fixed_detect_loki_100M_lastTOA_4core.h5"
ENT_FILE = "/scratch/na00078/projects/IPTA_MDC2/h5_files/G2D2_core_fixed_new.h5"

ENT_BURN = 3000

# constants
megaparsec = 3.086e+22
speed_of_light = 299792458.0
T_sun = 1.327124400e20 / speed_of_light**3
D_MPC_FIXED = 75.400
F_GW_FIXED = np.log10(3.7e-9)

def derive_log10_h0(log10_mc, log10_fgw, log10_dL_Mpc):
    return (np.log10(2.0)
            + (5.0/3.0) * (log10_mc + np.log10(T_sun))
            + (2.0/3.0) * (np.log10(np.pi) + log10_fgw)
            - (log10_dL_Mpc + np.log10(megaparsec) - np.log10(speed_of_light)))

# load
print("Loading...")
with h5py.File(QCW_FILE, "r") as f:
    qc = f["samples_masked"][...]
    qc_pn = [p.decode() if isinstance(p, bytes) else p for p in f["par_names"][:]]
with h5py.File(LOKI_FILE, "r") as f:
    lk = f["samples_cold"][0, :, :]
    lk_pn = [p.decode() if isinstance(p, bytes) else p for p in f["par_names"][:]]
with h5py.File(ENT_FILE, "r") as f:
    en_all = f["chain"][...]
    en_pn = [p.decode() if isinstance(p, bytes) else p for p in f["params"][:]]

# Enterprise: apply burn-in
en = en_all[ENT_BURN:, :]

# subsample QCW and Loki to match Enterprise post burn-in (for CORNER plots)
N_TARGET = len(en)
qc_idx = np.random.choice(len(qc), min(N_TARGET, len(qc)), replace=False)
qc_sub = qc[qc_idx]
lk_idx = np.random.choice(len(lk), N_TARGET, replace=False)
lk_sub = lk[lk_idx]

print(f"Corner: subsampled to Enterprise N = {N_TARGET}")
print(f"  My QCW: {len(qc_sub):,}, Loki: {len(lk_sub):,}, Enterprise: {len(en):,}")
print(f"Trace: full chains (no thinning, no burn-in for QCW and Loki)")
print(f"  My QCW: {len(qc):,}, Loki: {len(lk):,}, Enterprise: {len(en):,}")

# ============================================================
# CORNER plot data (subsampled, no fGW since it's fixed)
# ============================================================
qcw_cosi = qc_sub[:, qc_pn.index("0_cos_inc")]
qcw_h    = qc_sub[:, qc_pn.index("0_log10_h")]
qcw_mc   = qc_sub[:, qc_pn.index("0_log10_mc")]
qcw_amp  = qc_sub[:, qc_pn.index("gwb_log10_A")]
qcw_phi  = qc_sub[:, qc_pn.index("0_phase0")]
qcw_psi  = qc_sub[:, qc_pn.index("0_psi")]

loki_cosi = lk_sub[:, lk_pn.index("0_cos_inc")]
loki_mc   = lk_sub[:, lk_pn.index("0_log10_mc")]
loki_dist = lk_sub[:, lk_pn.index("0_log10_dist")]
loki_h    = derive_log10_h0(loki_mc, np.full_like(loki_mc, F_GW_FIXED), loki_dist)
loki_amp  = lk_sub[:, lk_pn.index("gwb_log10_A")]
loki_phi  = lk_sub[:, lk_pn.index("0_phase0")]
loki_psi  = lk_sub[:, lk_pn.index("0_psi")]

ent_cosi = en[:, en_pn.index("cos_inc")]
ent_mc   = en[:, en_pn.index("log10_mc")]
ent_h    = derive_log10_h0(ent_mc, np.full_like(ent_mc, F_GW_FIXED),
                           np.full_like(ent_mc, np.log10(D_MPC_FIXED)))
ent_amp  = en[:, en_pn.index("crn_log10_A")]
ent_phi  = en[:, en_pn.index("phase0")]
ent_psi  = en[:, en_pn.index("psi")]

# truth (from MDC2 JSON, no fGW since it's fixed)
t_cosi = 0.8412486994612669
t_h    = -13.668773493298787
t_mc   = np.log10(4.3e9)
t_amp  = -15.070581074285707
t_phi  = 0.24434609527920614
t_psi  = 1.1187560505283651

CB, CG, CO, CT = "#0072B2", "#009E73", "#D55E00", "#CC79A7"

# corner setup (7 params, no fGW)
labels_c = [r"$\cos\iota$", r"$\log_{10} h_0$",
            r"$\log_{10}\mathcal{M}_c$", r"$\log_{10} A_{\rm GWB}$",
            r"$\gamma_{\rm GWB}$", r"$\Phi_0$", r"$\psi$"]
ranges_c = [(-1, 1), (-15, -12), (8.5, 10.0), (-20, -11),
            (0, 7), (0, 2*np.pi), (0, np.pi)]
truths_c = [t_cosi, t_h, t_mc, t_amp, None, t_phi, t_psi]

qcw_gam_c  = qc_sub[:, qc_pn.index("gwb_gamma")]
loki_gam_c = lk_sub[:, lk_pn.index("gwb_gamma")]
ent_gam_c  = en[:, en_pn.index("crn_gamma")]

qcw_c  = np.column_stack([qcw_cosi, qcw_h, qcw_mc, qcw_amp, qcw_gam_c, qcw_phi, qcw_psi])
loki_c = np.column_stack([loki_cosi, loki_h, loki_mc, loki_amp, loki_gam_c, loki_phi, loki_psi])
ent_c  = np.column_stack([ent_cosi, ent_h, ent_mc, ent_amp, ent_gam_c, ent_phi, ent_psi])

def make_corner(data_list, color_list, label_list, title):
    fig = corner.corner(data_list[0], labels=labels_c, truths=truths_c,
                        truth_color=CT, range=ranges_c, color=color_list[0])
    for d, c in zip(data_list[1:], color_list[1:]):
        corner.corner(d, fig=fig, labels=labels_c, truths=truths_c,
                      truth_color=CT, range=ranges_c, color=c)
    handles = [plt.Line2D([0],[0], color=c, lw=2.5, label=l)
               for c, l in zip(color_list, label_list)]
    handles.append(plt.Line2D([0],[0], color=CT, lw=1.5, label="Injected"))
    fig.legend(handles=handles, loc="upper right", fontsize=11,
               bbox_to_anchor=(0.98, 0.98))
    fig.suptitle(title, fontsize=14, y=1.0)
    return fig

# corner plots
make_corner([qcw_c, loki_c], [CB, CG],
            ["My QCW", "Loki"], "G2D2 fixed: My QCW vs Loki")
plt.show()

make_corner([loki_c, ent_c], [CG, CO],
            ["Loki", "Enterprise"], "G2D2 fixed: Loki vs Enterprise")
plt.show()

make_corner([qcw_c, ent_c], [CB, CO],
            ["My QCW", "Enterprise"], "G2D2 fixed: My QCW vs Enterprise")
plt.show()

make_corner([qcw_c, loki_c, ent_c], [CB, CG, CO],
            ["My QCW", "Loki", "Enterprise"], "G2D2 fixed: All three")
plt.show()

# ============================================================
# TRACE plot data (full chains for QCW and Loki, no thinning)
# Enterprise keeps burn-in cut as Bjorn specified
# ============================================================
# QCW full chain in order
qcw_t_cosi = qc[:, qc_pn.index("0_cos_inc")]
qcw_t_h    = qc[:, qc_pn.index("0_log10_h")]
qcw_t_mc   = qc[:, qc_pn.index("0_log10_mc")]
qcw_t_amp  = qc[:, qc_pn.index("gwb_log10_A")]
qcw_t_gam  = qc[:, qc_pn.index("gwb_gamma")]
qcw_t_phi  = qc[:, qc_pn.index("0_phase0")]
qcw_t_psi  = qc[:, qc_pn.index("0_psi")]

# Loki full chain in order
loki_t_cosi = lk[:, lk_pn.index("0_cos_inc")]
loki_t_mc   = lk[:, lk_pn.index("0_log10_mc")]
loki_t_dist = lk[:, lk_pn.index("0_log10_dist")]
loki_t_h    = derive_log10_h0(loki_t_mc, np.full_like(loki_t_mc, F_GW_FIXED), loki_t_dist)
loki_t_amp  = lk[:, lk_pn.index("gwb_log10_A")]
loki_t_gam  = lk[:, lk_pn.index("gwb_gamma")]
loki_t_phi  = lk[:, lk_pn.index("0_phase0")]
loki_t_psi  = lk[:, lk_pn.index("0_psi")]

# Enterprise full post burn-in in order
ent_t_cosi = en[:, en_pn.index("cos_inc")]
ent_t_mc   = en[:, en_pn.index("log10_mc")]
ent_t_h    = derive_log10_h0(ent_t_mc, np.full_like(ent_t_mc, F_GW_FIXED),
                             np.full_like(ent_t_mc, np.log10(D_MPC_FIXED)))
ent_t_amp  = en[:, en_pn.index("crn_log10_A")]
ent_t_gam  = en[:, en_pn.index("crn_gamma")]
ent_t_phi  = en[:, en_pn.index("phase0")]
ent_t_psi  = en[:, en_pn.index("psi")]

trace_rows = [
    (r"$\cos\iota$",              t_cosi, [(qcw_t_cosi, CB), (loki_t_cosi, CG), (ent_t_cosi, CO)]),
    (r"$\log_{10}h_0$",           t_h,    [(qcw_t_h, CB),    (loki_t_h, CG),    (ent_t_h, CO)]),
    (r"$\log_{10}\mathcal{M}_c$", t_mc,   [(qcw_t_mc, CB),   (loki_t_mc, CG),   (ent_t_mc, CO)]),
    (r"$\log_{10}A_{\rm GWB}$",   t_amp,  [(qcw_t_amp, CB),  (loki_t_amp, CG),  (ent_t_amp, CO)]),
    (r"$\gamma_{\rm GWB}$",       None,   [(qcw_t_gam, CB),  (loki_t_gam, CG),  (ent_t_gam, CO)]),
    (r"$\Phi_0$",                 t_phi,  [(qcw_t_phi, CB),  (loki_t_phi, CG),  (ent_t_phi, CO)]),
    (r"$\psi$",                   t_psi,  [(qcw_t_psi, CB),  (loki_t_psi, CG),  (ent_t_psi, CO)]),
]

titles = [f"My QCW (n={len(qcw_t_cosi):,})",
          f"Loki (n={len(loki_t_cosi):,})",
          f"Enterprise (n={len(ent_t_cosi):,}, burn {ENT_BURN})"]

# each column has its own x-axis (different chain lengths), so no sharex
fig, axes = plt.subplots(len(trace_rows), 3, figsize=(18, 21))
for row_i, (ylabel, truth_val, data) in enumerate(trace_rows):
    for col_i, (arr, color) in enumerate(data):
        axes[row_i, col_i].plot(arr, marker=".", linestyle="None",
                                markersize=1, color=color, rasterized=True)
        if truth_val is not None:
            axes[row_i, col_i].axhline(truth_val, color=CT, lw=1.5)
    axes[row_i, 0].set_ylabel(ylabel)

for col_i, t in enumerate(titles):
    axes[0, col_i].set_title(t)
for col_i in range(3):
    axes[-1, col_i].set_xlabel("sample index")

plt.tight_layout()
plt.show()

# parameter ranges (full chains)
print("\n" + "=" * 70)
print("Parameter ranges (full chains, fGW fixed)")
print("=" * 70)
all_labels = ["cos_inc", "log10_h0", "log10_Mc",
              "log10_A_GWB", "gamma_GWB", "phase0", "psi"]

for name, arrs in [
    ("My QCW", (qcw_t_cosi, qcw_t_h, qcw_t_mc, qcw_t_amp, qcw_t_gam, qcw_t_phi, qcw_t_psi)),
    ("Loki",   (loki_t_cosi, loki_t_h, loki_t_mc, loki_t_amp, loki_t_gam, loki_t_phi, loki_t_psi)),
    ("Ent",    (ent_t_cosi, ent_t_h, ent_t_mc, ent_t_amp, ent_t_gam, ent_t_phi, ent_t_psi)),
]:
    print(f"\n{name} (n={len(arrs[0]):,}):")
    for lab, a in zip(all_labels, arrs):
        print(f"  {lab}: [{a.min():.3f}, {a.max():.3f}]  median={np.median(a):.3f}")

print("\n" + "=" * 70)
print("Min / Max per parameter (full trace chains)")
print("=" * 70)

params = [
    ("cos_inc",     qcw_t_cosi, loki_t_cosi, ent_t_cosi),
    ("log10_h0",    qcw_t_h,    loki_t_h,    ent_t_h),
    ("log10_Mc",    qcw_t_mc,   loki_t_mc,   ent_t_mc),
    ("log10_A_GWB", qcw_t_amp,  loki_t_amp,  ent_t_amp),
    ("gamma_GWB",   qcw_t_gam,  loki_t_gam,  ent_t_gam),
    ("phase0",      qcw_t_phi,  loki_t_phi,  ent_t_phi),
    ("psi",         qcw_t_psi,  loki_t_psi,  ent_t_psi),
]

for name, q, l, e in params:
    print(f"\n{name}:")
    print(f"  QCW : {q.min():.4f}  {q.max():.4f}")
    print(f"  Loki: {l.min():.4f}  {l.max():.4f}")
    if e is None:
        print(f"  Ent : fixed (not sampled)")
    else:
        print(f"  Ent : {e.min():.4f}  {e.max():.4f}")

import h5py

files = {
    # G2D2 broad detection
    "QCW  G2D2 broad": ("/scratch/na00078/projects/IPTA_MDC2/h5_files/G2D2_broad_detect_tref_4core_dLmasked_75.400Mpc.h5", "samples_masked"),
    "Loki G2D2 broad": ("/scratch/na00078/projects/IPTA_MDC2/h5_files/G2D2_broad_detect_loki_100M_lastTOA_4core.h5", "samples_cold"),
    "Ent  G2D2 broad": ("/scratch/na00078/projects/IPTA_MDC2/h5_files/G2D2_varyfgw_core_new.h5", "chain"),
    # G2D2 fixed detection
    "QCW  G2D2 fixed": ("/scratch/na00078/projects/IPTA_MDC2/h5_files/G2D2_narrow_detect_tref_4core_dLmasked_75.400Mpc.h5", "samples_masked"),
    "Loki G2D2 fixed": ("/scratch/na00078/projects/IPTA_MDC2/h5_files/G2D2_fixed_detect_loki_100M_lastTOA_4core.h5", "samples_cold"),
    "Ent  G2D2 fixed": ("/scratch/na00078/projects/IPTA_MDC2/h5_files/G2D2_core_fixed_new.h5", "chain"),
}

for label, (path, key) in files.items():
    with h5py.File(path, "r") as f:
        d = f[key]
        # Loki stores [n_temp, n_samples, n_params]; take the cold chain
        n = d.shape[1] if key == "samples_cold" else d.shape[0]
        print(f"{label:16s} {key:14s} n = {n:,}")

# ===== SD BF via analytic induced prior on x = log10 h0 (Loki, Slack 17 Jun) =====
# 1) prior induced on x by priors Mc~U(7,10), dL (LokiCW: U in log10 dL over 75.4*(1+/-0.01);
#    ENT: fixed 75.4), fGW fixed -> analytic (Irwin-Hall convolution of scaled uniforms)
# 2) reweight derived x to flat (log-uniform h0) over the induced support
# 3) binned SD estimator: BF10 = prior/posterior density in lowest bin, several widths
from itertools import combinations
from math import factorial

LOG10_MC_MIN, LOG10_MC_MAX = 7.0, 10.0
QCW_H_MIN, QCW_H_MAX = -18.0, -11.0
DL0, ETA = 75.400, 0.01
BIN_COUNTS = (20, 30, 40, 50, 75, 100)

XCONST = (np.log10(2.0) + (5.0/3.0)*np.log10(T_sun) + (2.0/3.0)*np.log10(np.pi)
          - (np.log10(megaparsec) - np.log10(speed_of_light)))

def sum_uniform_pdf(y, widths):
    K = len(widths); y = np.asarray(y, dtype=float); out = np.zeros_like(y)
    for r in range(K + 1):
        for S in combinations(range(K), r):
            sh = sum(widths[k] for k in S)
            out += ((-1)**r) * np.where(y > sh, np.maximum(y - sh, 0.0)**(K - 1), 0.0)
    out = out / (factorial(K - 1) * np.prod(widths))
    return np.where((y >= 0.0) & (y <= sum(widths)), out, 0.0)

def induced_prior(terms, const):
    widths = [abs(s)*(hi - lo) for s, lo, hi in terms if abs(s)*(hi - lo) > 0]
    x_lo = const + sum(min(s*lo, s*hi) for s, lo, hi in terms)
    x_hi = x_lo + sum(widths)
    return (lambda x: sum_uniform_pdf(np.asarray(x) - x_lo, widths)), x_lo, x_hi

def sd_bf10(x, x_lo, x_hi, weights=None, label=""):
    if weights is None: weights = np.ones_like(x)
    pd = 1.0/(x_hi - x_lo); ws = weights.sum()
    ne = ws**2/(weights**2).sum()
    print(f"--- {label}")
    print(f"    support [{x_lo:.3f}, {x_hi:.3f}] width {x_hi-x_lo:.3f} N={len(x):,} Neff={ne:,.0f}")
    fin = []
    for nb in BIN_COUNTS:
        edges = np.linspace(x_lo, x_hi, nb + 1)
        wh, _ = np.histogram(x, bins=edges, weights=weights)
        rh, _ = np.histogram(x, bins=edges)
        d0 = wh[0]/(ws*(edges[1]-edges[0]))
        bf = pd/d0 if (rh[0] > 0 and d0 > 0) else np.inf
        print(f"    bins={nb:4d} raw n(lowest)={rh[0]:9d} BF10={bf:.6g}")
        if np.isfinite(bf): fin.append(bf)
    if fin:
        fin = np.array(fin)
        print(f"    BF10 = {fin.mean():.4g} +/- {fin.std():.2g} over finite variants")
    else:
        print("    BF10 = inf for all widths (no samples at null; quote lower limit)")
    # lower limit convention: BF10 > prior/(1 sample density) at finest binning
    nb = BIN_COUNTS[-1]; edges = np.linspace(x_lo, x_hi, nb+1)
    d1 = 1.0/(len(x)*(edges[1]-edges[0]))
    print(f"    (1-sample density floor at bins={nb}: BF10 limit ~ {pd/d1:.4g})")
    print()

qcw_x = qc[:, qc_pn.index("0_log10_h")]
lk_mc_f = lk[:, lk_pn.index("0_log10_mc")]
lk_dl_f = lk[:, lk_pn.index("0_log10_dist")]
en_mc_f = en[:, en_pn.index("log10_mc")]
loki_x = derive_log10_h0(lk_mc_f, np.full_like(lk_mc_f, F_GW_FIXED), lk_dl_f)
ent_x  = derive_log10_h0(en_mc_f, np.full_like(en_mc_f, F_GW_FIXED),
                         np.full_like(en_mc_f, np.log10(D_MPC_FIXED)))

print(f"loki dL range: [{lk_dl_f.min():.5f}, {lk_dl_f.max():.5f}] vs prior [{np.log10(DL0*0.99):.5f}, {np.log10(DL0*1.01):.5f}]")

mc_term = (5.0/3.0, LOG10_MC_MIN, LOG10_MC_MAX)
dl_term = (-1.0, np.log10(DL0*(1.0-ETA)), np.log10(DL0*(1.0+ETA)))
f_const = (2.0/3.0)*F_GW_FIXED
e_pdf, e_lo, e_hi = induced_prior([mc_term], XCONST + f_const - np.log10(DL0))
l_pdf, l_lo, l_hi = induced_prior([mc_term, dl_term], XCONST + f_const)

print(f"x ranges | QCW [{qcw_x.min():.3f}, {qcw_x.max():.3f}] prior [{QCW_H_MIN}, {QCW_H_MAX}]")
print(f"         | Loki [{loki_x.min():.3f}, {loki_x.max():.3f}] analytic [{l_lo:.3f}, {l_hi:.3f}]")
print(f"         | Ent  [{ent_x.min():.3f}, {ent_x.max():.3f}] analytic [{e_lo:.3f}, {e_hi:.3f}]")
print()

w_l = 1.0/np.clip(l_pdf(loki_x), 1e-300, None)
w_e = 1.0/np.clip(e_pdf(ent_x), 1e-300, None)

sd_bf10(qcw_x, QCW_H_MIN, QCW_H_MAX, label="QCW fixed: sampled log10 h0, flat prior, dL-masked")
sd_bf10(ent_x, e_lo, e_hi, weights=w_e, label="Enterprise fixed: derived log10 h0, reweighted")
sd_bf10(loki_x, l_lo, l_hi, weights=w_l, label="LokiCW fixed: derived log10 h0, reweighted")

for lbl, xx, ww, lo in (("QCW", qcw_x, np.ones_like(qcw_x), QCW_H_MIN),
                        ("Ent", ent_x, w_e, e_lo), ("Loki", loki_x, w_l, l_lo)):
    fr = [(t, float(ww[xx <= lo + t].sum()/ww.sum())) for t in (0.25, 0.5, 1.0, 2.0)]
    print(lbl, "posterior mass within lower bound + t:", fr)


import numpy as np
for nm, arr in [('QCW_h', qcw_x), ('Loki_h', loki_x), ('Ent_h', ent_x), ('QCW_mc', qc[:, qc_pn.index('0_log10_mc')]), ('Loki_mc', lk_mc_f), ('Ent_mc', en_mc_f), ('QCW_A', qc[:, qc_pn.index('gwb_log10_A')]), ('Loki_A', lk[:, lk_pn.index('gwb_log10_A')]), ('Ent_A', en[:, en_pn.index('crn_log10_A')])]:
    q = np.percentile(arr, [5, 50, 95])
    print(nm, 'p5=%.3f p50=%.3f p95=%.3f' % tuple(q))

# QCW post-mask SD BF under the MASKED (induced) prior, same convention as LokiCW
w_q = 1.0/np.clip(l_pdf(qcw_x), 1e-300, None)
m = (qcw_x >= l_lo) & (qcw_x <= l_hi)
print('QCW samples outside induced support:', int((~m).sum()))
sd_bf10(qcw_x[m], l_lo, l_hi, weights=w_q[m], label='QCW fixed: masked samples, reweighted to flat over induced support')

from enterprise_extensions import model_utils
import numpy as np
np.random.seed(7)

def bf_resampled(x, w, lo, hi, ntol):
    m = (x >= lo) & (x <= hi)
    xm, wm = x[m], w[m]
    idx = np.random.choice(len(xm), size=len(xm), replace=True, p=wm/wm.sum())
    return model_utils.bayes_fac(xm[idx], ntol=ntol, logAmin=lo, logAmax=hi)

for ntol in (200, 50, 10):
    print('ntol =', ntol)
    print('  QCW :', model_utils.bayes_fac(qcw_x, ntol=ntol, logAmin=-18.0, logAmax=-11.0))
    print('  Loki:', bf_resampled(loki_x, w_l, l_lo, l_hi, ntol))
    print('  Ent :', bf_resampled(ent_x, w_e, e_lo, e_hi, ntol))


# ===== publication corner, fixed, larger labels =====
import corner
import matplotlib.pyplot as plt
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
QCW_arr = np.column_stack([qcw_mc, qcw_h, qcw_cosi, qcw_phi, qcw_psi])
ENT_arr = np.column_stack([ent_mc, ent_h, ent_cosi, ent_phi, ent_psi])
DL_arr  = np.column_stack([loki_mc, loki_h, loki_cosi, loki_phi, loki_psi])
truths = [np.log10(4.3e9), -13.668773493298787, np.cos(0.8412486994612669), 0.24434609527920614, 1.1187560505283651]
labels = [r"$\log_{10}\mathcal{M}_c$", r"$\log_{10} h_0$", r"$\cos\iota$", r"$\Phi_0$", r"$\psi$"]
ranges = [(8.5, 10.0), (-15.0, -12.0), (-1, 1), (0, 2*np.pi), (0, np.pi)]
LW = 1.6
sig = np.array([0.5, 1.0, 1.5, 2.0]); levels = 1.0 - np.exp(-0.5*sig**2)
base_kw = dict(labels=labels, bins=30, smooth=1.0, levels=levels, range=ranges,
               plot_datapoints=False, plot_density=False, no_fill_contours=True,
               fill_contours=False, label_kwargs={"fontsize": 22})
C_QCW, C_ENT, C_DL = "#4477AA", "#228833", "#CC6677"
fig = corner.corner(QCW_arr, color=C_QCW, contour_kwargs={"linewidths": LW, "colors": C_QCW}, hist_kwargs={"density": True, "color": C_QCW, "lw": LW}, **base_kw)
corner.corner(ENT_arr, fig=fig, color=C_ENT, contour_kwargs={"linewidths": LW, "colors": C_ENT}, hist_kwargs={"density": True, "color": C_ENT, "lw": LW}, **base_kw)
corner.corner(DL_arr, fig=fig, color=C_DL, contour_kwargs={"linewidths": LW, "colors": C_DL}, hist_kwargs={"density": True, "color": C_DL, "lw": LW}, **base_kw)
corner.overplot_lines(fig, truths, color="k", lw=LW)
fig.savefig("g2d2_fixed_truth_4core.png", dpi=300, bbox_inches="tight")
fig.savefig("g2d2_fixed_truth_4core.pdf", bbox_inches="tight")
print("saved g2d2_fixed_truth_4core")


