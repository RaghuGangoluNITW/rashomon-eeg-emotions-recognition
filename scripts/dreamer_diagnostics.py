import os, sys
import scipy.io, numpy as np, json

# Ensure project root is on sys.path so `rashomon_emotion` imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rashomon_emotion.preprocessing import load_data
from rashomon_emotion.features import extract_wavelet_features
p='data/DREAMER/DREAMER.mat'
print('Inspecting file:', p)
# top-level keys
mat = scipy.io.loadmat(p)
keys = [k for k in mat.keys() if not k.startswith('__')]
print('Top-level keys in .mat:', keys)

# loader outputs
try:
    eeg, labels = load_data(p, dataset='DREAMER')
except Exception as e:
    print('Loader error:', e)
    raise

print('Loaded eeg type:', type(eeg))
try:
    eeg_arr = np.array(eeg)
    print('eeg_arr.shape:', eeg_arr.shape)
except Exception as e:
    print('Could not convert eeg to ndarray:', e)
    eeg_arr = eeg

print('Loaded labels type:', type(labels))
try:
    labels_arr = np.array(labels)
    print('labels_arr.shape:', labels_arr.shape)
except Exception as e:
    print('Could not convert labels to ndarray:', e)
    labels_arr = labels

# print label uniques for first few subjects
n_sub = labels_arr.shape[0] if hasattr(labels_arr,'shape') and len(labels_arr.shape)>0 else 1
print('n_subjects (inferred):', n_sub)
for i in range(min(5, n_sub)):
    try:
        vals = np.unique(np.array(labels_arr[i]))
        print(f'subject {i} label uniques:', vals.tolist())
    except Exception as e:
        print('could not inspect labels for subject', i, e)

# inspect subject0 eeg shape and dtype
try:
    s0 = np.array(eeg_arr[0])
    print('subject0 eeg shape:', s0.shape, 'dtype:', s0.dtype)
    if s0.ndim>=2:
        print('subject0 n_trials:', s0.shape[0])
except Exception as e:
    print('Could not inspect subject0 eeg:', e)

# quick feature leakage check for subject0
try:
    # build features for subject0 and rest
    X0 = np.array([extract_wavelet_features(t) for t in eeg_arr[0]])
    Xrest = np.vstack([np.array([extract_wavelet_features(t) for t in eeg_arr[i]]) for i in range(1, eeg_arr.shape[0])])
    print('X0.shape, Xrest.shape:', X0.shape, Xrest.shape)
    matches = [np.any(np.all(Xrest == row, axis=1)) for row in X0]
    print('Any exact match between subject0 trials and rest:', any(matches))
except Exception as e:
    print('Feature leakage check failed:', e)

# overall label sanity
try:
    flat_labels = labels_arr.flatten()
    unique_overall = np.unique(flat_labels)
    print('Overall label unique values:', unique_overall.tolist())
    print('If unique length==1 then labels are trivial')
except Exception as e:
    print('Could not compute overall label uniques:', e)

print('Diagnostic complete')
