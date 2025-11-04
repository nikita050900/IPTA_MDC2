#!/usr/bin/env python3
import argparse
import numpy as np
import h5py
import sys
from pathlib import Path

# Hardcoded base directory and number of parameters
BASE_DIR = Path("/scratch/na00078/projects/IPTA_MDC2/h5_files/")
FIRST_N = 8
THIN_FACTOR = 10   # adjust thinning here

def main():
    ap = argparse.ArgumentParser(description="Thin an HDF5 chain file.")
    ap.add_argument("--infile", required=True, help="Input HDF5 filename (just the name, not path)")
    ap.add_argument("--outfile", required=True, help="Output HDF5 filename (just the name, not path)")
    args = ap.parse_args()

    infile_path = BASE_DIR / args.infile
    outfile_path = BASE_DIR / args.outfile

    if not infile_path.exists():
        print(f"ERROR: Infile does not exist: {infile_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Infile:  {infile_path}")
    print(f"Outfile: {outfile_path}")
    print(f"Keeping first {FIRST_N} parameters with thinning factor {THIN_FACTOR}")

    with h5py.File(infile_path, "r") as f_in, h5py.File(outfile_path, "w") as f_out:
        # Copy small datasets directly
        Ts = f_in["T-ladder"][...]
        log_likelihood = f_in["log_likelihood"][:, ::THIN_FACTOR]
        par_names = [x.decode("UTF-8") for x in list(f_in["par_names"])]
        acc_fraction = f_in["acc_fraction"][...]
        fisher_diag = f_in["fisher_diag"][...]

        # Input dataset shape
        n_chain, n_iter, n_par = f_in["samples_cold"].shape
        k = min(FIRST_N, n_par)
        n_out_iter = n_iter // THIN_FACTOR

        print("Infile samples_cold shape:", (n_chain, n_iter, n_par))
        print("Outfile samples_cold shape:", (n_chain, n_out_iter, k))

        # Create output datasets
        dset_out = f_out.create_dataset(
            "samples_cold",
            shape=(n_chain, n_out_iter, k),
            dtype="f4",
            chunks=True,
            compression="gzip"
        )
        f_out.create_dataset("log_likelihood", data=log_likelihood, compression="gzip", chunks=True)
        f_out.create_dataset("par_names", data=np.array(par_names[:k], dtype="S"))
        f_out.create_dataset("acc_fraction", data=acc_fraction)
        f_out.create_dataset("fisher_diag", data=fisher_diag)
        f_out.create_dataset("T-ladder", data=Ts)

        # Process one chain at a time (avoid loading everything)
        for chain in range(n_chain):
            print(f"Processing chain {chain}...")
            data = f_in["samples_cold"][chain, ::THIN_FACTOR, :k]
            dset_out[chain, :, :] = data

    print("Done.")

if __name__ == "__main__":
    main()
