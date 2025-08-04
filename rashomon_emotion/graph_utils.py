# Graph construction utilities.
import numpy as np

def compute_plv(signal1, signal2):
    phase_diff = np.angle(signal1) - np.angle(signal2)
    return np.abs(np.mean(np.exp(1j * phase_diff)))

def build_connectivity_graph(eeg_data):
    n_channels = eeg_data.shape[0]
    graph = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(n_channels):
            graph[i, j] = compute_plv(eeg_data[i], eeg_data[j])
    return graph
