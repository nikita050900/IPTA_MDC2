import h5py,numpy as np
D='/scratch/na00078/projects/IPTA_MDC2/h5_files/'
for fn in ['core_single_MDC2_DS1.h5','core_single_MDC2_DS1_varyfgw.h5','core_single_MDC2.h5','core_single_MDC2_varyfgw.h5','core_single_MDC2_DS1_new.h5']:
    try:
        f=h5py.File(D+fn,'r')
        def show(name,obj):
            if isinstance(obj,h5py.Dataset): print('   ',name,obj.shape,obj.dtype)
        print(fn); f.visititems(show)
        pn=[p.decode() if isinstance(p,bytes) else str(p) for p in f['params'][:]]
        print('   params', [p for p in pn if not p.startswith('J') and not p.startswith('B')])
        ch=f['chain']; print('   chain', ch.shape)
        if 'log10_fgw' in pn:
            x=ch[:,pn.index('log10_fgw')]; print('   fgw range',x.min(),x.max())
        mc=np.asarray(ch[3000:,pn.index('log10_mc')]); print('   q95 mc(lin)',np.log10(np.quantile(10**mc,0.95)))
    except Exception as e: print(fn,'ERR',repr(e))
