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
corr_df = nwbfile.processing["analysis"]["corr_theta"].to_dataframe()

ctrl = np.asarray(corr_df.loc[corr_df["condition"] == "control", "correlation"].iloc[0])
imag = np.asarray(corr_df.loc[corr_df["condition"] == "imagined", "correlation"].iloc[0])
walks = np.asarray(corr_df.loc[corr_df["condition"] == "real", "correlation"].iloc[0])
data = np.column_stack([ctrl, imag, walks])
swarm_colors = CLR[[2, 1, 0], :]

# --- 3d swarm ---
xpos = np.array([1, 2, 3])
rng = np.random.default_rng(0)
jitter_width = 0.08
jittered_x = []
for i in range(3):
    jittered_x.append(np.full(data.shape[0], xpos[i], dtype=float)
                      + rng.uniform(-jitter_width, jitter_width, data.shape[0]))

save_json({
    "ctrl": ctrl, "imag": imag, "walks": walks,
    "data_columns": data,
    "x_positions": xpos,
    "jittered_x": np.array(jittered_x),
    "colors": swarm_colors,
    "labels": ["ctrl", "imag", "walks"],
}, os.path.join(OUT_DIR, "fig_3d_corr_swarm.json"))

fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=100)
for i in range(3):
    ax.scatter(jittered_x[i], data[:, i], s=50, facecolors='none',
               edgecolors=[swarm_colors[i]], linewidths=2, marker='o')
ax.grid(True); ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['ctrl', 'imag', 'walks'])
ax.set_yticks(np.arange(-0.4, 0.41, 0.2)); ax.set_ylim(-0.5, 0.5)
ax.set_ylabel('corr', fontsize=18); ax.tick_params(labelsize=18)
fig.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_3d_corr_swarm.png"), dpi=150, bbox_inches="tight")
plt.close()

# --- 3e scatter ---
save_json({
    "imag": data[:, 1], "real": data[:, 2],
    "color": CLR[0].tolist(),
}, os.path.join(OUT_DIR, "fig_3e_corr_scatter.json"))

fig, ax = plt.subplots(figsize=(5.5, 4.7), dpi=100)
ax.plot(data[:, 1], data[:, 2], 'o', linewidth=2,
        markerfacecolor='none', markeredgecolor=CLR[0], markeredgewidth=2)
ax.set_xlim(-0.1, 0.4); ax.set_ylim(-0.1, 0.5); ax.grid(True)
ax.set_xlabel('imagined walks', fontsize=18)
ax.set_ylabel('real-world walks', fontsize=18)
ax.tick_params(labelsize=18)
fig.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_3e_corr_scatter.png"), dpi=150, bbox_inches="tight")
plt.close()