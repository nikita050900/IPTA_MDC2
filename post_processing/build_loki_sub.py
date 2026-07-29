import h5py, numpy as np
H5="/scratch/na00078/projects/IPTA_MDC2/h5_files"
files=["G2D2_broad_detect_loki_100M_lastTOA_4core",
       "G2D2_fixed_detect_loki_100M_lastTOA_4core",
       "G2D2_broad_UL_loki_100M_lastTOA_4core",
       "G2D2_fixed_UL_loki_100M_lastTOA_ntol_10_4core"]
rng=np.random.default_rng(42)
for b in files:
    with h5py.File(f"{H5}/{b}.h5","r") as f:
        n=f["samples_cold"].shape[1]
        idx=np.sort(rng.choice(n, size=min(500000,n), replace=False))
        sub=f["samples_cold"][0][idx,:]
        pn=f["par_names"][...]
    with h5py.File(f"{H5}/{b}_sub.h5","w") as o:
        o.create_dataset("samples_cold", data=sub[np.newaxis])
        o.create_dataset("par_names", data=pn)
    print("built", b, sub.shape, flush=True)
print("BUILD_DONE", flush=True)
