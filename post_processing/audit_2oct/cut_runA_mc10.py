import h5py,numpy as np
D='/scratch/na00078/projects/IPTA_MDC2/h5_files/dl_masked/'
src=D+'G2D1_broad_detect_4core_dLmasked_75.400Mpc.h5'; dst=D+'G2D1_broad_detect_4core_dLmasked_75.400Mpc_mcmax10.h5'
with h5py.File(src,'r') as f, h5py.File(dst,'w') as g:
    pn=[p.decode() if isinstance(p,bytes) else str(p) for p in f['par_names'][:]]
    S=f['samples_masked'][:]; N=S.shape[0]; keep=S[:,pn.index('0_log10_mc')]<=10.0
    for k in f.keys():
        d=f[k][()]
        if hasattr(d,'shape') and d.ndim>=1 and d.shape[0]==N and k!='par_names': d=d[keep]
        g.create_dataset(k,data=d,compression='gzip' if hasattr(d,'ndim') and d.ndim>=1 and d.size>1000 else None)
        print(k, f[k].shape, '->', g[k].shape)
    for a,v in f.attrs.items(): g.attrs[a]=v
    g.attrs['note']='Copy of the Run A masked file with the one sample above log10 Mc = 10 removed, to match the prior bound of Table 1.'
print('removed', int((~keep).sum()))
