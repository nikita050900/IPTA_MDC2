#!/usr/bin/env python3
import numpy as np, h5py, sys

maskfile, rawfile, outfile = sys.argv[1], sys.argv[2], sys.argv[3]
n_keep = 8

print(f"Masked in : {maskfile}")
print(f"Raw meta  : {rawfile}")
print(f"Outfile   : {outfile}")

with h5py.File(maskfile, 'r') as fm:
    sm        = fm['samples_masked'][...]          # (Nkeep, 142)
    logL      = fm['logL_masked'][...]             # (Nkeep,)
    par_names = [x.decode('utf-8') for x in fm['par_names'][...]]
    Ts        = fm['T_ladder'][...]

print(f"  masked samples {sm.shape}, keeping first {n_keep} params")
print(f"  kept params: {par_names[:n_keep]}")

with h5py.File(rawfile, 'r') as fr:
    acc  = fr['acc_fraction'][...]
    fish = fr['fisher_diag'][...]

with h5py.File(outfile, 'w') as fo:
    fo.create_dataset('samples_cold',   data=sm[np.newaxis, :, :n_keep])
    fo.create_dataset('log_likelihood', data=logL[np.newaxis, :].astype('float32'))
    fo.create_dataset('T-ladder',       data=Ts)
    fo.create_dataset('acc_fraction',   data=acc)
    fo.create_dataset('fisher_diag',    data=fish)
    pn = np.array([s.encode('utf-8') for s in par_names[:n_keep]])
    fo.create_dataset('par_names', data=pn)

with h5py.File(outfile, 'r') as h:
    print("  saved:")
    for k, v in h.items():
        print("   ", k, v.shape, v.dtype)
    print("   par_names:", [x.decode() for x in h['par_names'][...]])
