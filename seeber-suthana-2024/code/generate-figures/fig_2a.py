import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pynwb import read_nwb

NWB_PATH = "Zenodo_compressed_groupSubject.nwb"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

CLR_LEFT = (0.301, 0.745, 0.933)
CLR_RIGHT = (0, 0.447, 0.741)

def save_json(data, path):
    def conv(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, list): return [conv(x) for x in o]
        if isinstance(o, (np.floating, np.integer)): return o.item()
        return o
    with open(path, "w") as f:
        json.dump({k: conv(v) for k, v in data.items()}, f, indent=2)

nwbfile = read_nwb(NWB_PATH)
avg_walk = nwbfile.processing["behavior"]["average_walks"]
route_names = np.array(avg_walk["direction"][:])
left_idxs = np.where(route_names == "left")[0]
right_idxs = np.where(route_names == "right")[0]

left_walks = [(np.asarray(avg_walk["x_coordinates"][i]),
               np.asarray(avg_walk["y_coordinates"][i])) for i in left_idxs]
right_walks = [(np.asarray(avg_walk["x_coordinates"][i]),
                np.asarray(avg_walk["y_coordinates"][i])) for i in right_idxs]

save_json({
    "left_x": [w[0] for w in left_walks],
    "left_y": [w[1] for w in left_walks],
    "right_x": [w[0] for w in right_walks],
    "right_y": [w[1] for w in right_walks],
    "color_left": list(CLR_LEFT),
    "color_right": list(CLR_RIGHT),
}, os.path.join(OUT_DIR, "fig_2a_average_walks.json"))

plt.figure(figsize=(7, 7))
for i, (x, y) in enumerate(left_walks):
    plt.plot(x, y, color=CLR_LEFT, linewidth=2, alpha=0.5,
             label="left" if i == 0 else None)
for i, (x, y) in enumerate(right_walks):
    plt.plot(x, y, color=CLR_RIGHT, linewidth=2, alpha=0.5,
             label="right" if i == 0 else None)
plt.xlim([-8, 8]); plt.ylim([-7.5, 7.5])
plt.xticks(np.arange(-8, 9, 2)); plt.yticks(np.arange(-6, 7, 2))
plt.xlabel("space [m]"); plt.ylabel("space [m]")
plt.legend(); plt.axis("equal")
plt.savefig(os.path.join(OUT_DIR, "fig_2a_average_walks.png"), dpi=150, bbox_inches="tight")
plt.close()