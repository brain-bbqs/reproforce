import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pynwb import read_nwb

NWB_PATH = "Zenodo_compressed_groupSubject.nwb"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def save_json(data, path):
    def conv(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.floating, np.integer)): return o.item()
        return o
    with open(path, "w") as f:
        json.dump({k: conv(v) for k, v in data.items()}, f, indent=2)

nwbfile = read_nwb(NWB_PATH)
rrr = nwbfile.processing["analysis"]["reconstructed_route_rep"]

cond = np.asarray(rrr["condition"][:]).astype(str)
direction = np.asarray(rrr["direction"][:]).astype(str)
mtype = np.asarray(rrr["model_type"][:]).astype(str)

def idx(c, d, m):
    return np.where((cond == c) & (direction == d) & (mtype == m))[0][0]

def stack(c, d):
    a = np.asarray(rrr["route_reconstruction"][idx(c, d, "cosine")]).ravel()
    b = np.asarray(rrr["route_reconstruction"][idx(c, d, "-sine")]).ravel()
    return np.concatenate([a, b])

real = np.concatenate([stack("real", "left"), stack("real", "right")])
imag = np.concatenate([stack("imagined", "left"), stack("imagined", "right")])
R = float(np.corrcoef(real, imag)[0, 1])

save_json({
    "real": real, "imagined": imag,
    "R": R,
}, os.path.join(OUT_DIR, "fig_4d_real_vs_imag_scatter.json"))

plt.figure(figsize=(5.5, 4.5))
plt.scatter(real, imag, s=10, facecolors='none', edgecolors='k', alpha=0.4)
plt.xlabel('real walks'); plt.ylabel('imagined walks')
plt.tick_params(labelsize=18)
plt.title(f"R = {R:.3f}")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_4d_real_vs_imag_scatter.png"), dpi=150, bbox_inches="tight")
plt.close()
print("R =", R)