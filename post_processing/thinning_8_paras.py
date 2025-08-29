#!/usr/bin/env python3
import argparse
import numpy as np
import h5py
import sys
from pathlib import Path

# Hardcoded base directory and number of parameters
BASE_DIR = Path("/scratch/na00078/projects/IPTA_MDC2/h5_files/")
FIRST_N = 8

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
    print(f"Keeping first {FIRST_N} parameters")

    with h5py.File(infile_path, "r") as f:
        Ts = f["T-ladder"][...]
        samples_cold = f["samples_cold"][:, :, :]
        log_likelihood = f["log_likelihood"][:1, :]
        par_names = [x.decode("UTF-8") for x in list(f["par_names"])]
        acc_fraction = f["acc_fraction"][...]
        fisher_diag = f["fisher_diag"][...]

        print("Infile samples_cold shape:", samples_cold.shape)
        print("Infile log_likelihood shape:", log_likelihood.shape)

        k = min(FIRST_N, samples_cold.shape[-1])

    with h5py.File(outfile_path, "w") as f:
        f.create_dataset("samples_cold", data=samples_cold[:, :, :k], compression="gzip", chunks=True)
        f.create_dataset("log_likelihood", data=log_likelihood[:, :], compression="gzip", chunks=True)
        f.create_dataset("par_names", data=np.array(par_names, dtype="S"))
        f.create_dataset("acc_fraction", data=acc_fraction)
        f.create_dataset("fisher_diag", data=fisher_diag)
        f.create_dataset("T-ladder", data=Ts)

    print("Done.")

if __name__ == "__main__":
    main()
