# Utility functions such as downsampling, pickling, etc.
import pickle
import scipy.signal

def downsample_signal(sig, target_len=256):
    return scipy.signal.resample(sig, target_len, axis=1)

def save_pickle(obj, fname):
    with open(fname, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)
