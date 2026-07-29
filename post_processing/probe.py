import h5py,glob,os
H5="/scratch/na00078/projects/IPTA_MDC2/h5_files/"
for f in ["G2D2_broad_detect_tref_4core_outfile.h5","G2D2_narrow_detect_tref_4core_outfile.h5","G2D2_broad_detect_tref_4core_UNMASKED_outfile.h5","G2D2_broad_detect_tref_4core_UNMASKED_sub200k_outfile.h5","G2D2_broad_detect_tref_4core_dLmasked_75.400Mpc.h5","G2D2_narrow_detect_tref_4core_dLmasked_75.400Mpc.h5"]:
    p=H5+f
    if not os.path.exists(p):
        print("MISSING",f); continue
    with h5py.File(p,"r") as h:
        print(f, {k:(h[k].shape if hasattr(h[k],"shape") else "grp") for k in h.keys()})
