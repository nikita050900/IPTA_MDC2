import h5py,numpy as np,glob,os
D='/scratch/na00078/projects/IPTA_MDC2/h5_files/dl_masked/'
for fn in sorted(glob.glob(D+'*.h5')):
    try:
        f=h5py.File(fn,'r'); pn=[p.decode() if isinstance(p,bytes) else str(p) for p in f['par_names'][:]]
        key='samples_masked' if 'samples_masked' in f else 'samples_cold'; ds=f[key]
        i=pn.index('0_log10_mc'); x=ds[:,i] if ds.ndim==2 else ds[0,:,i]
        print(os.path.basename(fn), key, ds.shape, 'max %.4f'%x.max(), 'n>10', int((x>10).sum()), list(f.keys()))
    except Exception as e: print(os.path.basename(fn),'ERR',e)
