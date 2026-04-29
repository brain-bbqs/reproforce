import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pynwb import read_nwb

NWB_PATH = "Zenodo_compressed_groupSubject.nwb"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

CLR_THETA = (0.494, 0.184, 0.556)
GREY = (0.7, 0.7, 0.7)

def save_json(data, path):
    def conv(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.floating, np.integer)): return o.item()
        return o
    with open(path, "w") as f:
        json.dump({k: conv(v) for k, v in data.items()}, f, indent=2)

nwbfile = read_nwb(NWB_PATH)

ts_obj = nwbfile.processing["ecephys"]["processed_broadband_iEEG"]
iEEG = np.asarray(ts_obj.data[:])
theta = np.asarray(nwbfile.processing["ecephys"]["processed_theta_iEEG"].data[:])
ts = ts_obj.starting_time + np.arange(iEEG.shape[0]) / ts_obj.rate

yoff = np.array([12, 4, -4, -12])
iEEG_off = iEEG + yoff
theta_off = theta + yoff

save_json({
    "ts": ts,
    "iEEG_offset": iEEG_off,
    "theta_offset": theta_off,
    "y_offsets": yoff,
    "color_iEEG": list(GREY),
    "color_theta": list(CLR_THETA),
}, os.path.join(OUT_DIR, "fig_1b_iEEG_example.json"))

plt.figure(figsize=(14, 4))
plt.plot(ts, iEEG_off, color=GREY, linewidth=4)
plt.plot(ts, theta_off, color=CLR_THETA, linewidth=4)
plt.ylim([-17, 17])
plt.xticks(np.arange(0, 6, 1))
plt.yticks([])
plt.xlabel("time [s]"); plt.ylabel("iEEG [z]")
plt.savefig(os.path.join(OUT_DIR, "fig_1b_iEEG_example.png"), dpi=150, bbox_inches="tight")
plt.close()