# Feature extraction from EEG data.
import numpy as np
import pywt

def extract_wavelet_features(signal, wavelet='db4', level=5):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return np.concatenate([np.mean(np.abs(c)) for c in coeffs])

def lorentzian_weighting(features, gamma=0.1):
    return features / (1 + (features / gamma)**2)
