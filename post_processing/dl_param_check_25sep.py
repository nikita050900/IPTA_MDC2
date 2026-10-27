#!/usr/bin/env python
# dl_param_check_25sep.py
# Sarah 3 Sep: does constraining dL change any CW parameter other than Mc?
# Runs E (broad fgw) and F (fixed fgw), CW+GWB detection chains.
# Produces corner plots of all CW params masked vs unmasked, plus a
# per parameter summary at several tolerances.
import numpy as np, h5py, json, time, os, traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H5='/anvil/projects/x-phy260442/projects/IPTA_MDC2/h5_files/'
PP='/anvil/projects/x-phy260442/projects/IPTA_MDC2/post_processing/'
OUTD=PP+'dl_check_25sep/'
os.makedirs(OUTD, exist_ok=True)
LOG=OUTD+'dl_param_check.log'
def log(s):
    with open(LOG,'a') as f: f.write(time.strftime('%H:%M:%S ')+str(s)+'\n')
    print(s, flush=True)

TARGET=75.4
mpc=3.086e22; c=299792458.0; Tsun=1.327124400e20/c**3
CONST=np.log10(2*(Tsun)**(5/3)*np.pi**(2/3)*c/mpc)
FGW_INJ=3.7e-9
INJ={'0_log10_mc':float(np.log10(4.3e9)), '0_log10_h':-13.67,
     '0_log10_fgw':float(np.log10(FGW_INJ)), '0_cos_inc':float(np.cos(0.841)),
     '0_phase0':0.244, '0_psi':1.119}
LBL={'0_log10_mc':r'$\log_{10}\mathcal{M}_c$','0_log10_h':r'$\log_{10}h_0$',
     '0_log10_fgw':r'$\log_{10}f_{\rm GW}$','0_cos_inc':r'$\cos\iota$',
     '0_phase0':r'$\Phi_0$','0_psi':r'$\psi$','gwb_log10_A':r'$\log_{10}A_{\rm GWB}$'}

with open(PP+'sec6_ess_4core.json') as f: ESSJ=json.load(f)

def dl_of(mc,h,logf):
    return 10**(CONST + (5/3)*mc + (2/3)*logf - h)

def summarize(v):
    q=np.percentile(v,[2.5,16,50,84,97.5])
    return {'median':float(q[2]),'q16':float(q[1]),'q84':float(q[3]),
            'w68':float(q[3]-q[1]),'w95':float(q[4]-q[0])}

try:
    import corner
    HAVE=True
except Exception as e:
    HAVE=False; log('corner import failed: %r'%e)

RES={}
try:
    log('=== START ===')
    for run,broad in (('E',True),('F',False)):
        fn=ESSJ[run]['file']
        want=['0_log10_mc','0_log10_h','0_cos_inc','0_phase0','0_psi']
        if broad: want.insert(2,'0_log10_fgw')
        want_all=want+['gwb_log10_A']
        log('=== run %s broad=%s file %s ==='%(run,broad,fn))
        C={}
        with h5py.File(H5+fn,'r') as f:
            pn=[p.decode() if isinstance(p,bytes) else str(p) for p in f['par_names'][:]]
            ds=f['samples_cold']
            log('samples_cold shape %s'%str(ds.shape))
            log('cw par_names present: %s'%[p for p in pn if p.startswith('0_') or p.startswith('gwb')])
            for w in want_all:
                if w in pn:
                    C[w]=np.asarray(ds[0,:,pn.index(w)],np.float64)
                else:
                    log('MISSING %s'%w)
        cols=[w for w in want if w in C]
        N=len(C['0_log10_mc']); log('N=%d cols=%s'%(N,cols))
        logf=C['0_log10_fgw'] if (broad and '0_log10_fgw' in C) else np.log10(FGW_INJ)
        dl=dl_of(C['0_log10_mc'],C['0_log10_h'],logf)
        log('implied dL pct [1,50,99] = %s'%np.percentile(dl,[1,50,99]))
        r={'file':fn,'N':int(N),'cols':cols,'injected':{k:INJ.get(k) for k in cols}}
        r['unmasked']={k:summarize(C[k]) for k in cols+['gwb_log10_A'] if k in C}
        for tol in (0.01,0.05,0.10,0.30):
            m=np.flatnonzero(np.abs(dl-TARGET)<tol*TARGET)
            key='tol_%g'%(tol*100)
            d={'Nsurv':int(m.size)}
            for k in cols+['gwb_log10_A']:
                if k in C: d[k]=summarize(C[k][m])
            r[key]=d
            log('%s %s Nsurv=%d'%(run,key,m.size))
        m1=np.flatnonzero(np.abs(dl-TARGET)<0.01*TARGET)
        if HAVE and m1.size>100:
            g=np.random.default_rng(42)
            nsub=int(min(m1.size, 150000))
            sub=g.choice(N,size=nsub,replace=False)
            A=np.column_stack([C[k][sub] for k in cols])
            B=np.column_stack([C[k][m1] for k in cols])
            labs=[LBL.get(k,k) for k in cols]
            tr=[INJ.get(k) for k in cols]
            rg=[(min(A[:,i].min(),B[:,i].min()),max(A[:,i].max(),B[:,i].max())) for i in range(len(cols))]
            fig=corner.corner(A,labels=labs,color='#4477AA',range=rg,plot_datapoints=False,
                              plot_density=False,levels=(0.39,0.68,0.86,0.95),
                              hist_kwargs={'density':True},max_n_ticks=3)
            corner.corner(B,fig=fig,color='#CC3311',range=rg,plot_datapoints=False,
                          plot_density=False,levels=(0.39,0.68,0.86,0.95),truths=tr,
                          truth_color='k',hist_kwargs={'density':True},max_n_ticks=3)
            for ax in fig.axes:
                ax.xaxis.label.set_size(26); ax.yaxis.label.set_size(26)
                ax.tick_params(labelsize=16)
            fig.savefig(OUTD+'dl_mask_allparams_%s.pdf'%run,bbox_inches='tight')
            fig.savefig(OUTD+'dl_mask_allparams_%s.png'%run,dpi=140,bbox_inches='tight')
            plt.close(fig); log('corner saved %s (unmasked n=%d, masked n=%d)'%(run,nsub,m1.size))
        RES[run]=r
        del C, dl
    with open(OUTD+'dl_param_check.json','w') as f: json.dump(RES,f,indent=1)
    log('=== DONE ===')
except Exception:
    log('FATAL: '+traceback.format_exc())
