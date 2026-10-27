import numpy as np, h5py, os, json
H5=os.environ['PROJECT']+'/projects/IPTA_MDC2/h5_files/'
mpc=3.086e22; c=299792458.0; Tsun=1.327124400e20/c**3
CONST=np.log10(2*(Tsun)**(5/3)*np.pi**(2/3)*c/mpc)
with h5py.File(H5+'G2D1_narrow_UL_4core_UNMASKED_outfile.h5','r') as f:
    pn=[p.decode() if isinstance(p,bytes) else str(p) for p in f['par_names'][:]]
    print(pn)
    ds=f['samples_cold']; mc=ds[0,:,pn.index('0_log10_mc')].astype(np.float64); h=ds[0,:,pn.index('0_log10_h')].astype(np.float64)
print('mc min/max %.4f %.4f  h min/max %.3f %.3f'%(mc.min(),mc.max(),h.min(),h.max()))
print('mc quantiles 5,50,90,95,99:', np.round(np.quantile(mc,[.05,.5,.9,.95,.99]),4))
print('h  quantiles 50,95:', np.round(np.quantile(h,[.5,.95]),4))
dl=10**(CONST+(5/3)*mc+(2/3)*np.log10(3.7e-9)-h)
print('dl quantiles 5,50,95:', np.round(np.quantile(dl,[.05,.5,.95]),1))
for nt in (0.5,1.0,2.0,5.0):
    m=np.abs(dl-75.4)<nt*75.4
    print('ntol %.0f%%: n=%d UL=%.4f'%(nt*100,m.sum(),np.quantile(mc[m],0.95)))
hist,edges=np.histogram(mc,bins=60)
for a,b in zip(edges[:-1],hist): print('%.3f %d'%(a,b))
