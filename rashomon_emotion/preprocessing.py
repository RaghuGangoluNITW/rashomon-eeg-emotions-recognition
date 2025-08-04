# EEG preprocessing logic (wavelets, normalization).
import numpy as np
import scipy.io

def load_deap_data(path):
    data = scipy.io.loadmat(path)
    # Extract EEG and labels here
    return eeg_data, labels

def normalize_trials(eeg_data):
    return (eeg_data - eeg_data.mean(axis=-1, keepdims=True)) / eeg_data.std(axis=-1, keepdims=True)
