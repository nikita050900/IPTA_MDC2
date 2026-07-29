import numpy as np, h5py, json
from enterprise_extensions import model_utils

H5  = "/scratch/na00078/projects/IPTA_MDC2/h5_files/"
OUT = "/scratch/na00078/projects/IPTA_MDC2/post_processing/"
RUNS = {"E": H5+"G2D2_broad_detect_tref_4core.h5",
        "F": H5+"G2D2_narrow_detect_tref_4core.h5"}

res = {}
for tag, f in RUNS.items():
    with h5py.File(f, "r") as h:
        par = [p.decode() if isinstance(p, bytes) else p for p in h["par_names"][:]]
        hits = [i for i, p in enumerate(par) if "log10_h" in p]
        print(tag, "h0 candidates:", [par[i] for i in hits], flush=True)
        c = h["samples_cold"][0, :, hits[0]]
    pre = model_utils.bayes_fac(c, ntol=200, logAmin=float(c.min()), logAmax=float(c.max()))
    res[tag] = {"N": int(c.size), "pre_mask_B10": [float(x) for x in np.atleast_1d(pre)]}
    print(tag, res[tag], flush=True)

json.dump(res, open(OUT+"bf_4core.json", "w"), indent=2)
print("wrote", OUT+"bf_4core.json")
