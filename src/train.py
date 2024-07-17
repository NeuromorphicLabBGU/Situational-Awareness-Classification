import os
import pickle
from pathlib import Path

import torch
import numpy as np

from src.utils import Topographies2SNN, Bands2TopographyLayer, Resonators2BandsLayer

full_resonator_array = torch.Tensor(
    [
        1.1,
        1.3,
        1.6,
        1.9,
        2.2,
        2.5,
        2.88,
        3.05,
        3.39,
        3.7,
        4.12,
        4.62,
        5.09,
        5.45,
        5.87,
        6.36,
        6.8,
        7.6,
        8.6,
        10.5,
        11.5,
        12.8,
        15.8,
        16.6,
        19.4,
        22.0,
        24.8,
        28.4,
        30.5,
        34.7,
        37.2,
        40.2,
        43.2,
        47.7,
        52.6,
        57.2,
    ]
)
bands = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 14),
    "Beta": (14, 32),
    "Gamma": (32, 62),
}


N = 11
xs = (
    np.array([-0.7, -0.66, 0, 0.2, 0.4, 0.5, 0.7, 0.7, 0.5, 0.4, 0.2, 0, -0.66, -0.7])
    + 1
) / 2
ys = (
    -np.array(
        [0.2, 0.6, 0.95, 0.6, 0.3, 0.6, 0.2, -0.2, -0.6, -0.3, -0.6, -0.95, -0.6, -0.2]
    )
    + 1
) / 2
channels = [
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
]

ch_pos = {ch: (x, y) for ch, (x, y) in zip(channels, zip(ys, xs))}

ch_pos_N = {k: (int(x * N), int(y * N)) for k, (x, y) in ch_pos.items()}
map_pos2ch = {(vx, vy): k for k, (vx, vy) in ch_pos_N.items()}

bands2topographies = {}
for band, (lf, hf) in bands.items():
    resonators2band = {}
    for ch in ch_pos.keys():
        freqs = full_resonator_array[
            (full_resonator_array >= lf) & (full_resonator_array < hf)
        ]
        clk_freqs = torch.where(freqs < 10, 15360, 153600)
        resonators2band[ch] = Resonators2BandsLayer(ch, band, clk_freqs)

    bands2topographies[band] = Bands2TopographyLayer(
        band, resonators2band, shape=(N, N), ch_pos=ch_pos_N
    )

snn = Topographies2SNN(bands2topographies, N * N, N * int(np.sqrt(N)), 3)

gpu = torch.cuda.is_available()
device = torch.device("cuda" if gpu else "cpu")
torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

snn.to(device)

trial = "0"
data = {
    "focus": [],
    "unfocus": [],
    "drowsed": [],
}

main_path = (
    Path("dataset")
    / "dataset"
    / "EEG_data_for_Mental_Attention_State_Detection"
    / "EEG_spikes"
)
for label in data.keys():
    for fname in os.listdir(main_path / trial / label):
        with open(main_path / trial / label / fname, "rb") as f:
            data[label].append(
                {
                    f"{k}-resonators": t.T.reshape(t.shape[1], 1, t.shape[0]).to(device)
                    for k, t in pickle.load(f).items()
                }
            )

input_layers = [l for l in snn.layers.keys() if l.endswith("resonators")]
time = 153600 // 4
inputs = data["focus"][0]
label_tensor = torch.tensor([0])
snn.run(inputs=inputs, time=time)

power_on_band = {label: {band: [] for band in bands.keys()} for label in data.keys()}
power_on_channel = {
    label: {channel: [] for channel in channels} for label in data.keys()
}


power = np.std
for label in data.keys():
    for channel_resonators, signals in data[label][0].items():
        channel, band, _ = channel_resonators.split("-")
        for j in range(signals.shape[2]):
            power_on_band[label][band].append(power(signals[:, 0, j]))
            power_on_channel[label][channel].append(power(signals[:, 0, j]))
