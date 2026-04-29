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

avg_theta = nwbfile.processing["ecephys"]["avg_theta_turns"]
sem_obj = nwbfile.processing["ecephys"]["avg_theta_turns_sem"]
v_turn = nwbfile.processing["behavior"]["hip_speed"]["hip_speed_turns"]
dPhi = nwbfile.processing["behavior"]["hip_angular_velocity"]["hip_angular_velocity_turns"]

Fs = avg_theta.rate
theta = np.squeeze(np.asarray(avg_theta.data[:]))
sem = np.squeeze(np.asarray(sem_obj.data[:]))
v = np.squeeze(np.asarray(v_turn.data[:]))
w = np.squeeze(np.asarray(dPhi.data[:]))
t = np.arange(-3, 1 + 1/Fs, 1/Fs)[:theta.shape[0]]

save_json({
    "t": t, "theta": theta, "theta_sem": sem,
    "speed": v, "angular_velocity": w,
    "color_theta": CLR[0].tolist(),
    "color_speed": CLR[1].tolist(),
    "color_omega": CLR[2].tolist(),
}, os.path.join(OUT_DIR, "fig_2d_theta_turns.json"))

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.plot(t, theta, linewidth=2, color=CLR[0], label="theta [z]")
ax.fill_between(t, theta - sem, theta + sem, color=CLR[0], alpha=0.3)
ax.plot(t, v, linewidth=2, color=CLR[1], label="v [m/s]")
ax.plot(t, w, linewidth=2, color=CLR[2], label="\u03c9 [\N{DEGREE SIGN}/s]")
ax.grid(True); ax.set_xlim([-3, 1]); ax.axvline(0, color="k")
ax.set_ylim([-0.4, 1.2]); ax.set_yticks(np.arange(-0.4, 1.21, 0.4))
ax.set_xlabel("time [s]"); ax.tick_params(labelsize=18); ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_2d_theta_turns.png"), dpi=150, bbox_inches="tight")
plt.close()