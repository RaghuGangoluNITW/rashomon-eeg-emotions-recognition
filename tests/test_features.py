import numpy as np
from rashomon_emotion import features

def test_extract_wavelet_features():
    sig = np.random.randn(256)
    feats = features.extract_wavelet_features(sig)
    assert isinstance(feats, np.ndarray)
    assert feats.size > 0

def test_lorentzian_weighting():
    feats = np.array([1.0, 2.0, 3.0])
    weighted = features.lorentzian_weighting(feats, gamma=0.5)
    assert weighted.shape == feats.shape
