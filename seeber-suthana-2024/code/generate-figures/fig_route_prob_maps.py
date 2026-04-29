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
tbl = nwbfile.processing["analysis"]["route_probability_maps"]
lin_ax = np.array(tbl["lin_ax"][:]).squeeze()
H_walk = np.array(tbl["H2D_walk"][:]).squeeze()
H_ctrl = np.array(tbl["H2D_ctrl"][:]).squeeze()
H_imag = np.array(tbl["H2D_imag"][:]).squeeze()

def plot_map(data, tag, title):
    save_json({
        "lin_ax": lin_ax,
        "H": data,
        "vmin": 0, "vmax": 4.5, "cmap": "jet",
        "title": title,
    }, os.path.join(OUT_DIR, f"fig_route_prob_{tag}.json"))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    X, Y = np.meshgrid(lin_ax, lin_ax)
    pcm = ax.pcolormesh(X, Y, data.T, shading="gouraud", cmap="jet", vmin=0, vmax=4.5)
    ax.set_title(title)
    ax.set_xlabel("Actual positions [%]"); ax.set_ylabel("Estimated positions [%]")
    ax.tick_params(labelsize=18)
    cb = fig.colorbar(pcm, ax=ax); cb.set_label("recon. prob. [a.u.]")
    cb.set_ticks(np.arange(0, 5, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"fig_route_prob_{tag}.png"), dpi=150, bbox_inches="tight")
    plt.close()

plot_map(H_walk, "real", "real-world walks")
plot_map(H_imag, "imagined", "imagined walks")
plot_map(H_ctrl, "control", "control")