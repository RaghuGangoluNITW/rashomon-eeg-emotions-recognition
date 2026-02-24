"""
EEG preprocessing utilities and dataset loaders.

This module provides flexible loaders for DEAP and DREAMER datasets.
The loaders attempt to read common variable names from .mat files but
also accept preprocessed numpy arrays. For DREAMER, place the .mat or
dataset folder under `data/` and pass the path to scripts.
"""
import os
import numpy as np
import scipy.io
from typing import Tuple
def _load_mat_flexible(path: str) -> dict:
    mat = scipy.io.loadmat(path)
    return {k: v for k, v in mat.items() if not k.startswith('__')}


def load_deap_data(path: str):
    data = _load_mat_flexible(path)
    eeg = None
    labels = None
    for key in ('data', 'eeg', 'EEG', 'X'):
        if key in data:
            eeg = data[key]
            break
    for key in ('labels', 'y', 'Y'):
        if key in data:
            labels = data[key].squeeze()
            break

    if eeg is None or labels is None:
        raise ValueError(f"Could not find EEG or labels in {path}. Available keys: {list(data.keys())}")

    return eeg, labels


def load_dreamer_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load DREAMER dataset from a .mat or directory.

    Supports .mat files where data is under 'Data' struct or top-level arrays.
    Returns (eeg, labels) where eeg is (n_subjects, n_trials, ...).
    """
    if os.path.isdir(path):
        for fname in os.listdir(path):
            if fname.endswith('.npz') or fname.endswith('.npy'):
                full = os.path.join(path, fname)
                if fname.endswith('.npz'):
                    arr = np.load(full)
                    if 'eeg' in arr and 'labels' in arr:
                        return arr['eeg'], arr['labels']
                else:
                    raise ValueError("Found .npy in DREAMER folder; please provide combined .npz with 'eeg' and 'labels' keys.")
        raise ValueError(f"No supported DREAMER archive found in {path}. Place a .npz with 'eeg' and 'labels' or a .mat file.")

    if path.endswith('.mat'):
        data = _load_mat_flexible(path)

        def _matobj_to_py(obj):
            if isinstance(obj, np.ndarray):
                if obj.dtype.names:
                    out = {}
                    for name in obj.dtype.names:
                        out[name] = _matobj_to_py(obj[name])
                    return out
                if obj.size == 1:
                    return _matobj_to_py(obj.flatten()[0])
                return [_matobj_to_py(o) for o in obj.flatten()]
            return obj

        # unwind single top-level wrapper like {'DREAMER': struct}
        if len(data) == 1:
            topk = next(iter(data))
            inner = data[topk]
            # Fast path: common MATLAB export with a 1x1 struct containing fields
            try:
                if isinstance(inner, np.ndarray) and inner.size == 1 and getattr(inner, 'dtype', None) is not None and getattr(inner, 'dtype').names:
                    struct = inner.flatten()[0]
                    fields = {}
                    for name in struct.dtype.names:
                        try:
                            fields[name] = struct[name]
                        except Exception:
                            pass
                    # if we've found numeric arrays under common keys, use them
                    if isinstance(fields, dict) and len(fields) > 0:
                        data = fields
                else:
                    # fallback to full recursive conversion
                    if topk != 'Data':
                        try:
                            inner = data[topk]
                            pyinner = _matobj_to_py(inner)
                            if isinstance(pyinner, dict):
                                data = pyinner
                        except Exception:
                            pass
            except Exception:
                # on any error, continue with original data and deeper parsing below
                pass

        if 'Data' in data:
            raw = data['Data']
            try:
                pydata = _matobj_to_py(raw)
            except Exception:
                pydata = raw

            # Helper: recursively find numeric numpy arrays inside nested objects
            def _find_numeric_arrays(obj, max_depth=6):
                found = []
                if max_depth < 0:
                    return found
                # handle numpy arrays (including object arrays)
                if isinstance(obj, np.ndarray):
                    # flatten object arrays
                    try:
                        if obj.dtype == object:
                            for o in obj.flatten():
                                found.extend(_find_numeric_arrays(o, max_depth-1))
                        else:
                            if np.issubdtype(obj.dtype, np.number):
                                found.append(obj)
                    except Exception:
                        pass
                    return found
                # structured numpy void (from MATLAB structs) -> iterate named fields
                if hasattr(obj, 'dtype') and getattr(obj, 'dtype') is not None and getattr(obj, 'dtype').names:
                    try:
                        for name in obj.dtype.names:
                            try:
                                v = obj[name]
                                found.extend(_find_numeric_arrays(v, max_depth-1))
                            except Exception:
                                pass
                    except Exception:
                        pass
                    return found
                # numpy scalar / generic numeric
                if isinstance(obj, np.generic):
                    try:
                        if np.issubdtype(type(obj), np.number):
                            found.append(np.array(obj))
                    except Exception:
                        pass
                    return found
                if isinstance(obj, (list, tuple)):
                    for o in obj:
                        found.extend(_find_numeric_arrays(o, max_depth-1))
                    return found
                if isinstance(obj, dict):
                    for v in obj.values():
                        found.extend(_find_numeric_arrays(v, max_depth-1))
                    return found
                return found

            # If pydata is a list (per-subject entries) try heuristic extraction
            if isinstance(pydata, list):
                eeg_list = []
                labels_list = []
                for idx, subj in enumerate(pydata):
                    # gather numeric arrays inside the subject entry
                    candidates = _find_numeric_arrays(subj)
                    if len(candidates) == 0:
                        raise ValueError(f"No numeric arrays found for subject index {idx} in DREAMER Data element; inspect the file structure.")

                    # pick EEG candidate: prefer arrays with large sample dimension (>1024), or 2D with second dim large
                    eeg_cand = None
                    for arr in candidates:
                        shape = getattr(arr, 'shape', ())
                        if len(shape) == 1 and shape[0] > 1024:
                            eeg_cand = arr
                            break
                        if len(shape) == 2 and (shape[0] > 4 and shape[1] > 1024 or shape[1] > 1024):
                            eeg_cand = arr
                            break

                    # fallback: choose the largest numeric array
                    if eeg_cand is None:
                        candidates_sorted = sorted(candidates, key=lambda a: np.prod(getattr(a, 'shape', (1,))), reverse=True)
                        eeg_cand = candidates_sorted[0]

                    # find label candidate: prefer short arrays (<200) which might be per-trial ratings
                    label_cand = None
                    for arr in candidates:
                        shape = getattr(arr, 'shape', ())
                        if len(shape) == 1 and 1 < shape[0] <= 200:
                            label_cand = arr
                            break
                    # if no label found, create zeros per trial
                    eeg_arr = np.array(eeg_cand)
                    # determine trial count heuristically
                    if eeg_arr.ndim == 1:
                        n_trials = 1
                    elif eeg_arr.ndim == 2:
                        # assume shape (channels, samples) or (trials, samples)
                        if eeg_arr.shape[0] <= 32 and eeg_arr.shape[1] > 32:
                            # channels x samples -> single trial
                            n_trials = 1
                        else:
                            n_trials = eeg_arr.shape[0]
                    else:
                        # safe fallback: if shape is empty (scalar/0-d), assume 1 trial
                        try:
                            if hasattr(eeg_arr, 'shape') and len(getattr(eeg_arr, 'shape', ())) > 0:
                                n_trials = eeg_arr.shape[0]
                            else:
                                n_trials = 1
                        except Exception:
                            n_trials = 1

                    if label_cand is not None:
                        labels_arr = np.array(label_cand).squeeze()
                    else:
                        labels_arr = np.zeros(n_trials, dtype=int)

                    eeg_list.append(eeg_arr)
                    labels_list.append(labels_arr)

                return np.array(eeg_list, dtype=object), np.array(labels_list, dtype=object)

            if isinstance(pydata, dict):
                # try to extract numeric arrays from dict
                candidates = _find_numeric_arrays(pydata)
                if len(candidates) >= 2:
                    # pick largest -> eeg, pick short -> labels
                    candidates_sorted = sorted(candidates, key=lambda a: np.prod(getattr(a, 'shape', (1,))), reverse=True)
                    eeg = np.array(candidates_sorted[0])
                    label = None
                    for arr in candidates_sorted[1:]:
                        if arr.ndim == 1 and arr.shape[0] <= 200:
                            label = np.array(arr).squeeze()
                            break
                    if label is None:
                        label = np.zeros(1, dtype=int)
                    return eeg, label

            raise ValueError(f"DREAMER .mat file 'Data' structure could not be parsed automatically. Top-level keys: {list(data.keys())}. Run the provided inspection scripts to view structure.")

        eeg = data.get('eeg') or data.get('EEG') or data.get('data') or data.get('EEGdata')
        labels = data.get('labels') or data.get('y') or data.get('Y') or data.get('labels_array')
        if eeg is None or labels is None:
            raise ValueError(f"DREAMER .mat file missing expected keys. Available: {list(data.keys())}")
        return eeg, np.array(labels).squeeze()

    if path.endswith('.npz'):
        arr = np.load(path)
        if 'eeg' in arr and 'labels' in arr:
            return arr['eeg'], arr['labels']
        raise ValueError(".npz must contain 'eeg' and 'labels' arrays")
    raise ValueError("Unsupported DREAMER path. Provide a .mat file, .npz archive, or a directory containing a .npz")


def load_data(path: str, dataset: str = 'DEAP') -> Tuple[np.ndarray, np.ndarray]:
    """Unified loader: choose dataset-specific loader by name.

    dataset: 'DEAP' or 'DREAMER' (case-insensitive)
    """
    ds = dataset.strip().lower()
    if ds == 'deap':
        return load_deap_data(path)
    elif ds == 'dreamer':
        return load_dreamer_data(path)
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'. Supported: DEAP, DREAMER")


def normalize_trials(eeg_data: np.ndarray) -> np.ndarray:
    return (eeg_data - eeg_data.mean(axis=-1, keepdims=True)) / (eeg_data.std(axis=-1, keepdims=True) + 1e-8)
