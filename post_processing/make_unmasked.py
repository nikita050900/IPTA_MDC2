import os, sys
import numpy as np
import h5py

H5 = "/scratch/na00078/projects/IPTA_MDC2/h5_files/"
CHUNK = 2_000_000
FORCE = "--force" in sys.argv

JOBS = [
    ("broad",  H5 + "G2D2_broad_detect_tref_4core.h5",
               H5 + "G2D2_broad_detect_tref_4core_UNMASKED_outfile.h5"),
    ("narrow", H5 + "G2D2_narrow_detect_tref_4core.h5",
               H5 + "G2D2_narrow_detect_tref_4core_UNMASKED_outfile.h5"),
]

for tag, raw, out in JOBS:
    print("=" * 20, tag, flush=True)
    if os.path.exists(out) and not FORCE:
        with h5py.File(out, "r") as h:
            print("exists, shape", h["samples_cold"].shape, "skipping", flush=True)
        continue
    if not os.path.exists(raw):
        print("MISSING raw file", raw, flush=True)
        continue
    with h5py.File(raw, "r") as h:
        d = h["samples_cold"]
        N = d.shape[1]
        print("raw shape", d.shape, flush=True)
        with h5py.File(out + ".tmp", "w") as g:
            o = g.create_dataset("samples_cold", shape=(1, N, 8), dtype=np.float32)
            for a in range(0, N, CHUNK):
                b = min(a + CHUNK, N)
                o[0, a:b, :] = d[0, a:b, :8]
                print("  ", b, "/", N, flush=True)
    os.replace(out + ".tmp", out)
    with h5py.File(out, "r") as h:
        print("wrote", out, h["samples_cold"].shape, flush=True)

print("=== DONE ===", flush=True)
