#!/usr/bin/env python3
# Figure 1 (dL mask effect) regenerated on the 4 core chains.
# Replicates the exact convention of G2D2_detect__broad.ipynb In[20] / G2D2_detect__narrow.ipynb In[23].
import os, json
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner
from enterprise_extensions import model_utils

H5 = "/scratch/na00078/projects/IPTA_MDC2/h5_files/"
OUT = "/scratch/na00078/projects/IPTA_MDC2/post_processing/"

target_d_L = 75.4
megaparsec = 3.086e+22
speed_of_light = 299792458.0
T_sun = 1.327124400e20 / speed_of_light**3

RUNS = {
    "broad": dict(
        unmasked = H5 + "G2D2_broad_detect_tref_4core_UNMASKED_outfile.h5",
        raw      = H5 + "G2D2_broad_detect_tref_4core.h5",
        base     = "IPTA_MDC2_G2D2_broad_detection_dL_75.4_4core_",
        FIXED    = False,
    ),
    "fixed": dict(
        unmasked = H5 + "G2D2_narrow_detect_tref_4core_UNMASKED_outfile.h5",
        raw      = H5 + "G2D2_narrow_detect_tref_4core.h5",
        base     = "IPTA_MDC2_G2D2_narrow_detection_dL_75.4_4core_",
        FIXED    = True,
    ),
}

CHUNK = 2_000_000


def load_cols(cfg):
    """Return an (N,8) float32 array of the first 8 cold chain parameters."""
    if os.path.exists(cfg["unmasked"]):
        src = cfg["unmasked"]
        print("reading unmasked outfile", src, flush=True)
        with h5py.File(src, "r") as h:
            return h["samples_cold"][0, :, :8].astype(np.float32)
    src = cfg["raw"]
    print("reading raw run file in chunks", src, flush=True)
    with h5py.File(src, "r") as h:
        d = h["samples_cold"]
        N = d.shape[1]
        out = np.empty((N, 8), dtype=np.float32)
        for a in range(0, N, CHUNK):
            b = min(a + CHUNK, N)
            out[a:b] = d[0, a:b, :8]
            print("  ", b, "/", N, flush=True)
    # cache it so this never has to be repeated
    cache = cfg["unmasked"]
    print("writing cache", cache, flush=True)
    with h5py.File(cache, "w") as g:
        g.create_dataset("samples_cold", data=out[None, :, :])
    return out


res = {}
for tag in ["broad", "fixed"]:
    cfg = RUNS[tag]
    print("=" * 20, tag, flush=True)
    sc = load_cols(cfg)

    h_amp = 10.0 ** sc[:, 4].astype(np.float64)
    fff = 10.0 ** sc[:, 3].astype(np.float64)
    mmm = 10.0 ** sc[:, 5].astype(np.float64)
    dL = 2 * (mmm * T_sun) ** (5 / 3) * (np.pi * fff) ** (2 / 3) / h_amp * speed_of_light / megaparsec
    del h_amp, fff, mmm

    d_L_min = 