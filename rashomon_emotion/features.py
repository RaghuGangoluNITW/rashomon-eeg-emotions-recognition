# Feature extraction from EEG data.
import numpy as np
import pywt
from scipy.stats import entropy


def extract_wavelet_features(signal, wavelet='db4', level=5):
    """Extract mean absolute values of wavelet coefficients"""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return np.array([np.mean(np.abs(c)) for c in coeffs], dtype=float)


def extract_dwt_subbands(signal, wavelet='db4', level=5):
    """
    Extract DWT subband features: energy of detail coefficients 
    at each level plus approximation energy.
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    energies = [np.mean(np.square(c)) for c in coeffs]
    return np.array(energies, dtype=float)


def lorentzian_weighting(features, gamma=0.1):
    """Apply Lorentzian weighting to features"""
    return features / (1 + (features / gamma)**2)


def extract_lorentzian_bandpower(signal, fs=128, bands=None, gamma=0.1):
    """
    Extract Lorentzian-weighted bandpower features.
    
    Args:
        signal: Time-domain EEG signal
        fs: Sampling frequency (Hz)
        bands: Dict of frequency bands, e.g., {'delta': (0.5, 4), 'theta': (4, 8), ...}
        gamma: Lorentzian parameter
    Returns:
        Lorentzian-weighted bandpower features
    """
    if bands is None:
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
    
    # Compute FFT
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    fft_vals = np.fft.rfft(signal)
    power = np.abs(fft_vals) ** 2
    
    # Extract bandpower for each band
    bandpowers = []
    for band_name, (low, high) in bands.items():
        idx = np.where((freqs >= low) & (freqs <= high))[0]
        if len(idx) > 0:
            bp = np.mean(power[idx])
        else:
            bp = 0.0
        bandpowers.append(bp)
    
    bandpowers = np.array(bandpowers)
    
    # Apply Lorentzian weighting
    return lorentzian_weighting(bandpowers, gamma)


def compute_hjorth_parameters(signal):
    """
    Compute Hjorth parameters: Activity, Mobility, and Complexity.
    
    Args:
        signal: Time-domain EEG signal
    Returns:
        Array of [activity, mobility, complexity]
    """
    # First derivative
    first_deriv = np.diff(signal)
    # Second derivative
    second_deriv = np.diff(first_deriv)
    
    # Activity (variance of signal)
    activity = np.var(signal)
    
    # Mobility (square root of variance of first derivative / variance of signal)
    if activity > 0:
        mobility = np.sqrt(np.var(first_deriv) / activity)
    else:
        mobility = 0.0
    
    # Complexity (ratio of mobility of first derivative to mobility of signal)
    if mobility > 0:
        mobility_deriv = np.sqrt(np.var(second_deriv) / np.var(first_deriv))
        complexity = mobility_deriv / mobility
    else:
        complexity = 0.0
    
    return np.array([activity, mobility, complexity])


def compute_spectral_entropy(signal, fs=128, method='shannon'):
    """
    Compute spectral entropy of a signal.
    
    Args:
        signal: Time-domain EEG signal
        fs: Sampling frequency
        method: 'shannon' or 'tsallis'
    Returns:
        Entropy value
    """
    # Compute power spectral density
    freqs = np.fft.rfftfreq(len(signal), d=1/fs)
    fft_vals = np.fft.rfft(signal)
    power = np.abs(fft_vals) ** 2
    
    # Normalize to get probability distribution
    power_norm = power / np.sum(power)
    
    if method == 'shannon':
        # Shannon entropy: -sum(p * log(p))
        return entropy(power_norm)
    elif method == 'tsallis':
        # Tsallis entropy with q=2: (1 - sum(p^2)) / (q-1)
        q = 2
        return (1 - np.sum(power_norm ** q)) / (q - 1)
    else:
        raise ValueError(f"Unknown entropy method: {method}")


def compute_sample_entropy(signal, m=2, r=0.2):
    """
    Compute Sample Entropy - measures complexity and irregularity.
    
    Args:
        signal: Time-domain EEG signal
        m: Embedding dimension
        r: Tolerance (as fraction of std dev)
    Returns:
        Sample entropy value
    """
    N = len(signal)
    
    # Normalize signal
    signal = (signal - np.mean(signal)) / np.std(signal)
    tolerance = r * np.std(signal)
    
    def _maxdist(xi, xj):
        return np.max(np.abs(xi - xj))
    
    def _phi(m):
        patterns = np.array([signal[i:i+m] for i in range(N - m)])
        count = 0
        for i in range(len(patterns)):
            for j in range(len(patterns)):
                if i != j and _maxdist(patterns[i], patterns[j]) < tolerance:
                    count += 1
        return count / (N - m)
    
    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    
    if phi_m > 0 and phi_m1 > 0:
        return -np.log(phi_m1 / phi_m)
    else:
        return 0.0


def extract_comprehensive_features(signal, fs=128, include_entropy=True):
    """
    Extract comprehensive feature set combining multiple methods.
    
    Args:
        signal: Time-domain EEG signal
        fs: Sampling frequency
        include_entropy: Whether to include entropy features (computationally expensive)
    Returns:
        Feature vector combining all methods
    """
    features = []
    
    # Wavelet features
    wavelet_feats = extract_wavelet_features(signal)
    features.append(wavelet_feats)
    
    # DWT subband features
    dwt_feats = extract_dwt_subbands(signal)
    features.append(dwt_feats)
    
    # Lorentzian bandpower
    lorentzian_feats = extract_lorentzian_bandpower(signal, fs)
    features.append(lorentzian_feats)
    
    # Hjorth parameters
    hjorth_feats = compute_hjorth_parameters(signal)
    features.append(hjorth_feats)
    
    # Entropy features (optional, can be slow)
    if include_entropy:
        spectral_ent = compute_spectral_entropy(signal, fs, method='shannon')
        sample_ent = compute_sample_entropy(signal)
        features.append(np.array([spectral_ent, sample_ent]))
    
    # Concatenate all features
    return np.concatenate(features)
