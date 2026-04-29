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

def quant(arr, inc):
    return np.round(arr / inc) * inc

def track2img(track, vals, x, z, brush=0, img_2d=None):
    track = np.asarray(track); vals = np.asarray(vals)
    if vals.ndim == 1: vals = vals[:, None]
    n_chan = vals.shape[1]
    if img_2d is None:
        img_2d = np.full((len(x), len(z), n_chan), np.nan)
    for i in range(track.shape[0]):
        xi = np.argmin(np.abs(x - track[i, 0]))
        zi = np.argmin(np.abs(z - track[i, 1]))
        x0, x1 = max(0, xi - brush), min(len(x), xi + brush + 1)
        z0, z1 = max(0, zi - brush), min(len(z), zi + brush + 1)
        for c in range(n_chan):
            patch = img_2d[x0:x1, z0:z1, c]
            v = vals[i, c]
            img_2d[x0:x1, z0:z1, c] = np.where(np.isnan(patch), v, (patch + v) / 2)
    return img_2d

nwbfile = read_nwb(NWB_PATH)

avg_walk = nwbfile.processing["behavior"]["average_walks"]
dirs_w = np.array(avg_walk["direction"][:])
li = np.where(dirs_w == "left")[0]; ri = np.where(dirs_w == "right")[0]
lx = np.vstack([np.array(avg_walk["x_coordinates"][i]) for i in li]).mean(0)
ly = np.vstack([np.array(avg_walk["y_coordinates"][i]) for i in li]).mean(0)
rx = np.vstack([np.array(avg_walk["x_coordinates"][i]) for i in ri]).mean(0)
ry = np.vstack([np.array(avg_walk["y_coordinates"][i]) for i in ri]).mean(0)

dim, inc = 8, 0.01
z = np.arange(-dim, dim + inc, inc); x = z.copy()
left_track = quant(np.column_stack((lx, ly)), inc)
right_track = quant(np.column_stack((rx, ry)), inc)

tbl = nwbfile.processing["analysis"]["route_models"]

def get(condition, direction, model="-sine"):
    idx = np.where((tbl["condition"][:] == condition) &
                   (tbl["model_type"][:] == model) &
                   (tbl["direction"][:] == direction))[0][0]
    return np.array(tbl["theta_reconstruction"][idx]).squeeze()

def render(theta_l, theta_r, tag, title):
    img = track2img(right_track, theta_r, x, z, brush=17)
    img = track2img(left_track, theta_l, x, z, brush=17, img_2d=img)
    img = np.squeeze(img)

    save_json({
        "image": img,
        "x_axis": x, "z_axis": z,
        "extent": [x[0], x[-1], z[0], z[-1]],
        "vmin": -3, "vmax": 3, "cmap": "jet",
        "title": title,
    }, os.path.join(OUT_DIR, f"fig_4_{tag}.json"))

    fig, ax = plt.subplots(figsize=(6.55, 4.45))
    im = ax.imshow(np.ma.masked_invalid(img.T),
                   extent=[x[0], x[-1], z[0], z[-1]],
                   origin="lower", aspect="equal", cmap="jet", vmin=-3, vmax=3)
    ax.set_xlim([-8, 8]); ax.set_ylim([-7.5, 7.5])
    ax.set_xticks(np.arange(-8, 9, 2)); ax.set_yticks(np.arange(-6, 7, 2))
    ax.set_xlabel("space [m]"); ax.set_ylabel("space [m]")
    ax.tick_params(labelsize=18)
    cbar = fig.colorbar(im, ax=ax); cbar.set_label("theta [z]")
    plt.title(title); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"fig_4_{tag}.png"), dpi=150, bbox_inches="tight")
    plt.close()

render(get("baseline_model", "left"), get("baseline_model", "right"),
       "4a_baseline_model", "-sine route model")
render(get("real", "left"), get("real", "right"),
       "4b_real_reconstruction", "Real-world, reconstruction")
render(get("imagined", "left"), get("imagined", "right"),
       "4c_imagined_reconstruction", "Imagined, reconstruction")