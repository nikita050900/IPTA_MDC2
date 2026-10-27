import numpy as np, json
from scipy.stats import gaussian_kde
D="/scratch/na00078/projects/IPTA_MDC2/post_processing/audit_19oct/"
E=json.load(open("/scratch/na00078/projects/IPTA_MDC2/post_processing/audit_2oct/ess_bins_12oct.json"))
P=json.load(open(D+"post_19oct.json")); Q=0.95
for tag in ["C","L"]:
    S=np.load(D+"samples_%s.npz"%tag); lf,lm=S["lf"],S["lm"]
    f=10**lf; DF=1/(14.962*365.25*86400.0); edges=np.arange(f.min(),f.max()+DF,DF); k=np.digitize(f,edges)-1
    half={round(b["f_lo"],2):b for b in E[tag]["bins"]}
    zo=[];ratio=[]
    for r in P[tag]["rows"]:
        if r["N"]<50: continue
        b=int(round((r["f_lo"]*1e-9-edges[0])/DF)); x=10**lm[k==b]; ul=np.quantile(x,Q)
        sub=x if len(x)<500000 else x[::len(x)//500000]
        try: p=gaussian_kde(sub).evaluate([ul])[0]
        except Exception: continue
        sig=np.sqrt(Q*(1-Q)/len(x))/p/(ul*np.log(10))
        h=half.get(round(r["f_lo"],2))
        if h is None or h["UL_h1"] is None or h["UL_h2"] is None: continue
        dd=h["UL_h1"]-h["UL_h2"]; zo.append(dd/(2*sig)); ratio.append(r["sig_boot"]/sig)
    zo=np.array(zo); print(tag,"old bars (Eq7 raw n_k KDE): n",len(zo),"rms",round(float(np.sqrt(np.mean(zo**2))),2),"frac>2",round(float(np.mean(abs(zo)>2)),2),"median boot/old",round(float(np.median(ratio)),2),"range",round(float(np.min(ratio)),2),round(float(np.max(ratio)),2))
