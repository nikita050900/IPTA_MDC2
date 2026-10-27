# post_19oct.py : ESS choice per run, Eq. 26 order statistic intervals, pass counts, h0 slopes, sigma table
import numpy as np, json
D='/scratch/na00078/projects/IPTA_MDC2/post_processing/audit_19oct/'
J=json.load(open(D+'perbin_ess_19oct.json')); Q=0.95; EMIN=50
def lq(x,q): return float(np.log10(np.quantile(10**x,q)))
def fit(x,y):
    x=np.asarray(x); y=np.asarray(y); A=np.c_[np.ones_like(x),x]; co=np.linalg.lstsq(A,y,rcond=None)[0]; r=y-A@co; n=len(x)
    s2=r@r/(n-2); cov=s2*np.linalg.inv(A.T@A); return dict(n=n,slope=round(float(co[1]),3),se=round(float(np.sqrt(cov[1,1])),3))
def wfit(x,y,s):
    x=np.asarray(x); y=np.asarray(y); w=1/np.asarray(s)**2; A=np.c_[np.ones_like(x),x]; W=np.diag(w)
    cov=np.linalg.inv(A.T@W@A); co=cov@A.T@W@y; chi=float(((y-A@co)**2*w).sum()); return dict(slope=round(float(co[1]),3),se=round(float(np.sqrt(cov[1,1])),3),chi2=round(chi,1),dof=len(x)-2)
OUT={}
for tag in ['C','L']:
    S=np.load(D+'samples_%s.npz'%tag); lf,lm,lh=S['lf'],S['lm'],S['lh']; R=J[tag]; tau=R['tau_emcee'][0]
    f=10**lf; DF=J['DF_nHz']*1e-9; edges=np.arange(f.min(),f.max()+DF,DF); k=np.digitize(f,edges)-1
    rows=[]; bi=0
    for w in R['bins']:
        b=int(round((w['f_lo']*1e-9-edges[0])/DF)); m=k==b; x=lm[m]; xh=lh[m]
        ess=float(w['blocks']) if tag=='C' else float(w['ESS_Ntau'])
        d=np.sqrt(Q*(1-Q)/ess)
        lo,hi=lq(x,Q-d),lq(x,min(Q+d,1)); hlo,hhi=lq(xh,Q-d),lq(xh,min(Q+d,1))
        rows.append(dict(f_lo=round(w['f_lo'],3),f_c=w['f_c'],N=w['N'],ESS=round(ess,1),ESS_acor=round(w['ESS_Ntau_acor'],1),blocks=w['blocks'],sig_boot=round(w['sig_boot_q'],4),UL=round(w['UL'],4),lo=round(lo,4),hi=round(hi,4),hi_at_max=bool(Q+d>=1),ULh=round(w['ULh'],4),hlo=round(hlo,4),hhi=round(hhi,4),pas=bool(ess>=EMIN),old=bool(w['N']>=50)))
    P=[r for r in rows if r['pas']]; A=[r for r in rows if r['old']]
    lx=lambda RR:[np.log10(r['f_c']*1e-9) for r in RR]
    o=dict(tau=tau,npass=len(P),nbins=len(rows),nold=len(A),pass_f=[r['f_lo'] for r in P],
      ESS_range_all=[min(r['ESS'] for r in rows),max(r['ESS'] for r in rows)],ESS_range_old=[min(r['ESS'] for r in A),max(r['ESS'] for r in A)],
      h_pass=fit(lx(P),[r['ULh'] for r in P]) if len(P)>2 else None, mc_pass=fit(lx(P),[r['UL'] for r in P]) if len(P)>2 else None,
      h_old=fit(lx(A),[r['ULh'] for r in A]), mc_old=fit(lx(A),[r['UL'] for r in A]),
      h_pass_w=wfit(lx(P),[r['ULh'] for r in P],[max((r['hhi']-r['hlo'])/2,1e-3) for r in P]) if len(P)>2 else None,
      fmax_pass=max(r['f_lo'] for r in P)+J['DF_nHz'], rows=rows)
    OUT[tag]=o
    print(tag,'tau',round(tau,1),'npass',len(P),'/',len(rows),'old plotted',len(A),'pass bins',o['pass_f'])
    print(tag,'ESS all',o['ESS_range_all'],'ESS old plotted',o['ESS_range_old'])
    print(tag,'h0 slope pass',o['h_pass'],'weighted',o['h_pass_w'],'mc slope pass',o['mc_pass'])
    print(tag,'h0 slope oldplotted',o['h_old'],'mc slope old',o['mc_old'])
    for r in rows: print(tag,r['f_lo'],r['N'],r['ESS'],r['blocks'],r['UL'],r['lo'],r['hi'],'*' if r['hi_at_max'] else '',r['ULh'],r['hlo'],r['hhi'],'PASS' if r['pas'] else '')
# all bins h0 slope C and L combined info
# sigma table: broad marginalized limits
C=(9.751612663269043,0.010656919529871106); L=(9.771245956420898,0.0036385562900492327); H=tuple(J['H']['eq26_emcee']); Ha=tuple(J['H']['eq26_acor'])
def z(a,b): return round(abs(a[0]-b[0])/np.hypot(a[1],b[1]),2)
OUT['sigma']=dict(CH=z(C,H),LH=z(L,H),CL=z(C,L),CH_rounded=z((9.752,0.011),(9.799,0.014)),CH_acor=z(C,Ha),LH_acor=z(L,Ha),H=H,H_acor=Ha,Ltau_emcee_ratio_acor=J['L']['tau_emcee'][0]/J['L']['tau_acor'][0],H_ratio=J['H']['tau_emcee']/J['H']['tau_acor'])
print('sigma',OUT['sigma'])
json.dump(OUT,open(D+'post_19oct.json','w'),indent=1)
