#!/scratch/na00078/conda_envs/QuickCW/bin/python

import glob
from pathlib import Path
import libstempo as T2

# Directories
pardir = Path("/scratch/na00078/projects/IPTA_MDC2/mdc2/group2/dataset_2/par")
timdir = Path("/scratch/na00078/projects/IPTA_MDC2/mdc2/group2/dataset_2/tim")

parfiles = sorted(glob.glob(str(pardir / "*.par")))
timfiles = sorted(glob.glob(str(timdir / "*.tim")))

print(f"Found {len(parfiles)} parfiles and {len(timfiles)} timfiles\n")

for par in parfiles:
    # Get PSR name from the .par file
    psrname = None
    with open(par) as f:
        for line in f:
            if line.startswith("PSR"):
                psrname = line.split()[1].strip()
                break

    # Find corresponding .tim file by name match
    tim_match = str(timdir / (Path(par).stem + ".tim"))
    if not Path(tim_match).exists():
        print(f"{psrname}: No matching .tim file ({tim_match})")
        continue

    # Try to load with libstempo
    try:
        p = T2.tempopulsar(parfile=par, timfile=tim_match)
        ntoas = len(p.toas())
        print(f"{psrname}: Loaded with {ntoas} TOAs")
    except Exception as e:
        print(f"{psrname}: FAILED ({e})")

