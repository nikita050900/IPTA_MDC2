#!/usr/bin/env python3
import argparse
import numpy as np
import h5py
import sys
from pathlib import Path

BASE_DIR = Path("/scratch/na00078/projects/IPTA_MDC2/h5_files/")
FIRST_N = 8

def main():
    ap = argparse.ArgumentParser(description="Thin an HDF5 chain file without loading whole file.")
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--thin", type=int, default=1)
    args = ap.parse_args()

    infile_path = BASE_DIR / args.infile
    outfile_path = BASE_DIR / args.outfile

    if not infile_path.exists():
        print(f"ERROR: Infile does not exist: {infile_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Infile:  {infile_path}")
    print(f"Outfile: {outfile_path}")
    print(f"Keeping first {FIRST_N} parameters, thin={args.thin}")

    with h5py.File(infile_path, "r") as f_in:
        Ts = f_in["T-ladder"][...]
        acc_fraction = f_in["acc_fraction"][...]
        fisher_diag = f_in["fisher_diag"][...]
        par_names = [x.decode("UTF-8") for x in list(f_in["par_names"])]

        dset = f_in["samples_cold"]         # shape: (nchain, nsamp, npar)
        ll_dset = f_in["log_likelihood"]    # shape: (nchain, nsamp)

        nchain, nsamp, npar = dset.shape
        print("samples_cold shape:", dset.shape)

        k = min(FIRST_N, npar)
        out_nsamp = nsamp // args.thin
        print("Output samples:", out_nsamp)

        with h5py.File(outfile_path, "w") as f_out:

            # reduced samples: keep all chains, first k params
            d_out = f_out.create_dataset(
                "samples_cold",
                shape=(nchain, out_nsamp, k),
                dtype="float32",
                compression="gzip",
                chunks=True
            )

            # reduced log-likelihood: ONLY cold chain
            ll_out = f_out.create_dataset(
                "log_likelihood",
                shape=(1, out_nsamp),
                dtype="float32",
                compression="gzip",
                chunks=True
            )

            out_idx = 0
            chunk = 1_000_000

            for start in range(0, nsamp, chunk):
                end = min(start + chunk, nsamp)

                # load only needed portion + params
                block = dset[:, start:end, :k]
                block = block[:, ::args.thin, :]     # thin inside chunk

                bsize = block.shape[1]

                # write samples
                d_out[:, out_idx:out_idx + bsize, :] = block

                # write cold-chain likelihood
                ll_block = ll_dset[0, start:end:args.thin]  # shape (bsize,)
                ll_out[0, out_idx:out_idx + bsize] = ll_block

                out_idx += bsize

            f_out.create_dataset("par_names", data=np.array(par_names[:k], dtype="S"))
            f_out.create_dataset("acc_fraction", data=acc_fraction)
            f_out.create_dataset("fisher_diag", data=fisher_diag)
            f_out.create_dataset("T-ladder", data=Ts)

    print("Done.")

if __name__ == "__main__":
    main()
