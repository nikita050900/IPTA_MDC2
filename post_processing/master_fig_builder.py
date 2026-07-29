
import matplotlib
matplotlib.use("Agg")
import numpy as np, h5py, json, os, subprocess, sys

PP = "/scratch/na00078/projects/IPTA_MDC2/post_processing"
HF = "/scratch/na00078/projects/IPTA_MDC2/h5_files"
os.chdir(PP)

# ---------- Part A: unmasked E outfile ----------
raw = HF + "/G2D2_broad_detect_tref_4core.h5"
unm = HF + "/G2D2_broad_detect_tref_4core_UNMASKED_outfile.h5"
if not os.path.exists(unm):
    print("A: generating unmasked E outfile", flush=True)
    with h5py.File(raw, "r") as f:
        ds = f["samples_cold"]
        print("   raw samples_cold shape", ds.shape, ds.dtype, flush=True)
        arr = ds[0, :, :8] if ds.ndim == 3 else ds[:, :8]
        pn = list(f["par_names"][:8])
    arr = np.asarray(arr)
    print("   unmasked array", arr.shape, flush=True)
    with h5py.File(unm, "w") as g:
        g.create_dataset("samples_cold", data=arr[None, :, :])
        g.create_dataset("par_names", data=pn)
    print("   wrote", unm, flush=True)
else:
    print("A: unmasked E outfile already exists", flush=True)

# ---------- Build Figure 1 script (pre-dL-mask corner) ----------
src = open("G2D2_pre_dl_masked_plot.py").read()
src = src.replace("/scratch/na00078/projects/IPTA_MDC2/h5_files/G2D2_broad_detect_outfile.h5", unm)
src = src.replace("corner_pre_dL_mask_detect_5params.png", "corner_pre_dL_mask_detect_5params_4core.png")
src = src.replace('fig.savefig(outfile, dpi=300, bbox_inches="tight")',
                  'fig.savefig(outfile, dpi=300, bbox_inches="tight")\nfig.savefig(outfile.replace(".png",".pdf"), bbox_inches="tight")\nprint("saved", outfile)')
open("G2D2_pre_dl_masked_plot_4core.py", "w").write(src)
print("built G2D2_pre_dl_masked_plot_4core.py", flush=True)

# ---------- Build Figure 8 script (all-sky map) ----------
nb = json.load(open("all_sky_healpy.ipynb"))
codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code" and "".join(c["source"]).strip()]
cell = max(codes, key=len)
cell = cell.replace("/scratch/na00078/projects/IPTA_MDC2/h5_files/G2D2_detect_allsky_outfile.h5",
                    HF + "/G2D2_detect_allsky_4core_outfile.h5")
hdr = "import matplotlib\nmatplotlib.use('Agg')\n"
foot = ('\ntry:\n    fig.savefig("all_sky_map_4core.png", dpi=300, bbox_inches="tight")\n'
        '    fig.savefig("all_sky_map_4core.pdf", bbox_inches="tight")\n    print("saved all_sky_map_4core")\n'
        'except Exception as e:\n    print("SAVEFIG ERR", repr(e))\n')
open("all_sky_healpy_4core.py", "w").write(hdr + cell + foot)
print("built all_sky_healpy_4core.py", flush=True)

# ---------- Run both ----------
for scr, log in [("G2D2_pre_dl_masked_plot_4core.py", "fig1_4core.log"),
                 ("all_sky_healpy_4core.py", "fig8_4core.log")]:
    print("RUN", scr, flush=True)
    rc = subprocess.run([sys.executable, scr], capture_output=True, text=True)
    open(log, "w").write(rc.stdout + "\n===STDERR===\n" + rc.stderr)
    print(scr, "rc=", rc.returncode, flush=True)
    print("  stdout_tail:", rc.stdout[-250:].replace("\n"," | "), flush=True)
    if rc.returncode != 0:
        print("  ERR_tail:", rc.stderr[-600:].replace("\n"," | "), flush=True)
print("ALL DONE", flush=True)
