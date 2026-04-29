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
])

def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj

def save_json(data, path):
    with open(path, "w") as f:
        json.dump({k: to_serializable(v) for k, v in data.items()}, f, indent=2)

# --- Load data ---
nwbfile = read_nwb(NWB_PATH)
route_table = nwbfile.processing["task"]["instructed_routes"]
route_names = route_table["direction"][:]

left_idx = np.where(route_names == "left")[0][0]
right_idx = np.where(route_names == "right")[0][0]

x_left = np.asarray(route_table["x_coordinates"][left_idx])
y_left = np.asarray(route_table["y_coordinates"][left_idx])
x_right = np.asarray(route_table["x_coordinates"][right_idx])
y_right = np.asarray(route_table["y_coordinates"][right_idx])

# --- Dump plotting data ---
save_json({
    "x_left": x_left, "y_left": y_left,
    "x_right": x_right, "y_right": y_right,
    "color_left": CLR[1].tolist(),
    "color_right": CLR[0].tolist(),
}, os.path.join(OUT_DIR, "fig_1a_instructed_routes.json"))

# --- Plot ---
plt.figure()
plt.plot(x_left, y_left, color=CLR[1], linewidth=4, label="left")
plt.plot(x_right, y_right, color=CLR[0], linewidth=4, label="right")
plt.xlim([-8, 8]); plt.ylim([-7.5, 7.5])
plt.xticks(np.arange(-8, 9, 2)); plt.yticks(np.arange(-6, 7, 2))
plt.xlabel("space [m]"); plt.ylabel("space [m]")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "fig_1a_instructed_routes.png"), dpi=150, bbox_inches="tight")
plt.close()