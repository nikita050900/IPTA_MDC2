#!/usr/bin/env python3
import numpy as np, h5py, sys

if len(sys.argv) < 2:
    print("Usage: python outfile_generator_4core.py <infile.h5>")
    sys.exit(1)

infile  = sys.argv[1]
outfile = infile.replace(".h5", "_outfile.h5")
n_keep  = 8
chunk   = 1_000_000

print(f"Input : {infile}")
print(f"Output: {outfile}")

with h5py.File(infile, 'r') as fin, h5py.File(outfile, 'w') as fout:
    sc = fin['samples_cold']
    nchain, nsamp, npar = sc.shape
    par_names = [x.decode('utf-8') for x in fin['par_names'][...]]
    print(f"  {nsamp} samples x {npar} params, keeping first {n_keep}")
    print(f"  kept params: {par_names[:n_keep]}")

    out_sc = fout.create_dataset('samples_cold', shape=(nchain, nsamp, n_keep), dtype=sc.dtype)
    for start in range(0, nsamp, chunk):
        end = min(start + chunk, nsamp)
        out_sc[0, start:end, :] = sc[0, start:end, :n_keep]

    for key in ['T-ladder', 'acc_fraction', 'fisher_diag', 'log_likelihood']:
        if key in fin:
            fout.create_dataset(key, data=fin[key][...])

    pn_bytes = np.array([s.encode('utf-8') for s in par_names[:n_keep]])
    fout.create_dataset('par_names', data=pn_bytes)

with h5py.File(outfile, 'r') as h:
    print("Saved contents:")
    for k, v in h.items():
        print(f"   {k}: {getattr(v,'shape',None)} {getattr(v,'dtype',None)}")
    print("   par_names:", [x.decode() for x in h['par_names'][...]])
