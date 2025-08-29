#!/usr/bin/env python3
"""
G2D2_pint_pkl.py
----------------
Make PINT-compatible copies of dataset_2 .par files,
then build a single combined pickle of all pulsars.

Run:
    /scratch/na00078/conda_envs/QuickCW/bin/python G2D2_pint_pkl.py
"""

from pathlib import Path
import re
import pickle

# --- FIXED PATHS FOR dataset_2 ---
BASE     = Path("/scratch/na00078/projects/IPTA_MDC2/mdc2/group2/dataset_2")
PAR_IN   = BASE / "par_pint_clean"      # cleaned from NE_SW
PAR_OUT  = BASE / "par_pint_compat"     # new PINT-compatible versions
TIM_DIR  = BASE / "tim"
OUT_PKL  = BASE / "dataset2_all_pulsars_pint.pkl"

PAR_OUT.mkdir(parents=True, exist_ok=True)

def decide_binary_model(lines):
    """Infer replacement for BINARY T2/DDH."""
    has_eps  = any(re.match(r"\s*(EPS1|EPS2)\b", l) for l in lines)
    has_h3h4 = any(re.match(r"\s*(H3|H4|STIG)\b", l) for l in lines)
    if has_eps and has_h3h4:
        return "ELL1H"
    if has_eps:
        return "ELL1"
    return "DD"

def transform_par_text(text):
    lines = text.splitlines()
    out = []
    saw_units, saw_timeeph = False, False

    for line in lines:
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("C "):
            out.append(line); continue

        tok = s.split()[0]

        if tok == "UNITS":
            saw_units = True
            out.append("UNITS TDB")
            continue

        if tok == "TIMEEPH":
            saw_timeeph = True
            out.append("TIMEEPH FB90")
            continue

        if tok == "DILATEFREQ":
            continue

        if tok == "BINARY":
            parts = s.split()
            model = parts[1] if len(parts) > 1 else ""
            if model in ("T2", "DDH"):
                repl = decide_binary_model(lines)
                if model == "DDH" and repl not in ("ELL1H", "ELL1"):
                    repl = "DD"
                out.append(f"BINARY {repl}")
                continue

        out.append(line)

    if not saw_units:
        out.insert(0, "UNITS TDB")
    if not saw_timeeph:
        out.insert(1, "TIMEEPH FB90")

    return "\n".join(out) + "\n"

# --- Transform PARs ---
transformed = []
for par in sorted(PAR_IN.glob("*.par")):
    raw = par.read_text()
    new = transform_par_text(raw)
    dst = PAR_OUT / par.name
    dst.write_text(new)
    print(f"[compat] Wrote: {dst}")
    transformed.append(dst)

# --- Build Pulsars with PINT ---
from enterprise.pulsar import Pulsar

built, skipped = [], []
for par in transformed:
    tim = TIM_DIR / (par.stem + ".tim")
    if not tim.exists():
        skipped.append((par.name, "missing .tim")); continue
    try:
        psr = Pulsar(str(par), str(tim), timing_package="pint")
        built.append(psr)
        print(f"[ok]   {par.stem}: {len(psr.toas)} TOAs")
    except Exception as e:
        skipped.append((par.name, str(e)))
        print(f"[fail] {par.stem}: {e}")

with open(OUT_PKL, "wb") as f:
    pickle.dump(built, f, protocol=pickle.HIGHEST_PROTOCOL)

print("\\nSummary")
print("-------")
print(f"Saved {len(built)} pulsars (PINT) -> {OUT_PKL}")
if skipped:
    print("Remaining skips:", len(skipped))
    for name, reason in skipped:
        print(f"  - {name} -> {reason}")
print(f"Transformed PARs are in: {PAR_OUT}")
