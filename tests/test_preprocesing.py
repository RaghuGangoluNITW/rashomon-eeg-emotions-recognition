import numpy as np
from rashomon_emotion import preprocessing

def test_normalize_trials():
    eeg = np.random.randn(2, 4, 256)  # 2 trials, 4 channels, 256 samples
    normed = preprocessing.normalize_trials(eeg)
    assert np.allclose(normed.mean(axis=-1), 0, atol=1e-6)
