import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pynwb import read_nwb

NWB_PATH = "Zenodo_compressed_groupSubject.nwb"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

CLR = np.array([
    (0,     0.447, 0.741),
    (0.301, 0.745, 0.933),
    (0.466, 0.674, 0.188),
])

def save_json(data, path):
    def conv(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.floating, np.integer)): return o.item()
        return o
    with open(path, "w") as f:
        json.dump({k: conv(v) for k, v in data.items()}, f, indent=2)

nwbfile = read_nwb(NWB_PATH)
tbl = nwbfile.processing["analysis"]["circular_reconstruction_error"]
ph_ax = np.array(tbl["ph_ax"][:]).squeeze()

def get(name):
    return np.array(tbl[name][:]).squeeze()

err_walk = np.concatenate([get("err_walk_left"), get("err_walk_right")])
err_imag = np.concatenate([get("err_imag_left"), get("err_imag_right")])
err_ctrl = np.concatenate([get("err_ctrl_left"), get("err_ctrl_right")])

walk_angle = np.angle(err_walk[:, 0] + 1j * err_walk[:, 1])
imag_angle = np.angle(err_imag[:, 0] + 1j * err_imag[:, 1])
ctrl_angle = np.angle(err_ctrl[:, 0] + 1j * err_ctrl[:, 1])

def polar(angles, tag, title, color):
    save_json({
        "angles": angles, "bins": ph_ax,
        "color": list(color), "title": title,
    }, os.path.join(OUT_DIR, f"fig_circ_err_{tag}.json"))

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(5.5, 4.5))
    ax.hist(angles, bins=ph_ax, density=True, color=color,
            edgecolor="black", linewidth=0.6)
    ax.set_title(title); ax.set_ylim(0, 0.6)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 30)))
    ax.tick_params(labelsize=18); ax.set_rlabel_position(80)
    plt.savefig(os.path.join(OUT_DIR, f"fig_circ_err_{tag}.png"), dpi=150, bbox_inches="tight")
    plt.close()

polar(walk_angle, "real", "real-world", CLR[0])
polar(imag_angle, "imagined", "imagined", CLR[1])
polar(ctrl_angle, "control", "control", CLR[2])