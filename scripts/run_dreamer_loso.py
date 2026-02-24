"""Run LOSO baseline on DREAMER with selectable preprocessing.

Usage:
    python scripts/run_dreamer_loso.py --data_path data/DREAMER/DREAMER.mat --preproc wavelet --epochs 30

This script assumes `load_data(..., dataset='DREAMER')` can load the file.
If the .mat contains subject-wise axis (n_subjects, n_trials, ...), LOSO is done over axis 0.
Otherwise, the script will try to infer subject grouping from a 'subject' key in the .mat file.
"""
import argparse
import time
import json
import os
import sys
import numpy as np

# Ensure project root is on sys.path so `rashomon_emotion` imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.metrics import accuracy_score, f1_score

from rashomon_emotion.preprocessing import load_data, normalize_trials
from rashomon_emotion.features import extract_wavelet_features, extract_dwt_subbands
from rashomon_emotion.model import EEGGNN
from rashomon_emotion.rashomon_runner import train_model


def extract_features_for_trials(trials, method='wavelet'):
    feats = []
    if method == 'wavelet':
        for t in trials:
            feats.append(extract_wavelet_features(t))
    else:
        for t in trials:
            feats.append(extract_dwt_subbands(t))
    return np.array(feats)


def loso_from_subject_axis(eeg, labels, preproc, epochs, hidden_dim, device):
    n_subjects = eeg.shape[0]
    fold_results = []
    def _ensure_numeric_array(arr):
        """Convert an array (possibly object-dtype with dicts) into numeric ndarray
        of shape (n_trials, ...). Raises RuntimeError with diagnostic if conversion fails."""
        a = np.asarray(arr)
        if a.dtype != object:
            return a

        elems = []
        for i, el in enumerate(a.flatten()):
            def _find_numeric(obj):
                # Recursively search obj for a numpy array or list that can be converted
                if obj is None:
                    return None
                if isinstance(obj, np.ndarray):
                    if obj.dtype != object:
                        return obj
                    if obj.size == 1:
                        return _find_numeric(obj.flatten()[0])
                if isinstance(obj, (list, tuple)):
                    # try to convert list of numbers or arrays
                    try:
                        cand = np.asarray(obj)
                        if cand.dtype != object:
                            return cand
                    except Exception:
                        pass
                    # else search elements
                    for sub in obj:
                        res = _find_numeric(sub)
                        if res is not None:
                            return res
                    return None
                if isinstance(obj, dict):
                    # prefer common EEG-like keys
                    for key in ('EEG', 'eeg', 'signal', 'data', 'trial', 'stimuli', 'baseline'):
                        if key in obj:
                            res = _find_numeric(obj[key])
                            if res is not None:
                                return res
                    # otherwise search all values
                    for v in obj.values():
                        res = _find_numeric(v)
                        if res is not None:
                            return res
                    return None
                # scalars
                try:
                    return np.asarray(obj)
                except Exception:
                    return None

            if isinstance(el, dict):
                val = _find_numeric(el)
                if val is None:
                    raise RuntimeError(f'Found dict element at index {i} with keys {list(el.keys())}; cannot map to numeric array')
            else:
                val = el
            elems.append(np.asarray(val))

        # ensure consistent shapes
        shapes = [x.shape for x in elems]
        if len(set(shapes)) != 1:
            raise RuntimeError(f'Inconsistent shapes within subject trials: sample shapes {shapes[:5]}')
        return np.stack(elems, axis=0)

    for s in range(n_subjects):
        # prepare train and test
        test_X = _ensure_numeric_array(eeg[s])
        test_y = np.array(labels[s])

        # build train arrays robustly (handles object arrays / 0-d entries)
        train_parts = []
        train_label_parts = []
        for i in range(n_subjects):
            if i == s:
                continue
            part = np.array(eeg[i])
            if part.ndim == 0:
                part = part.reshape(1)
            train_parts.append(part)

            lab_part = np.array(labels[i])
            if lab_part.ndim == 0:
                lab_part = lab_part.reshape(1)
            train_label_parts.append(lab_part)

        if len(train_parts) == 0:
            raise RuntimeError('No training data for LOSO fold {s}')

        train_X = np.concatenate(train_parts, axis=0)
        train_y = np.concatenate(train_label_parts, axis=0)

        # convert object arrays to numeric arrays if needed
        try:
            train_X = _ensure_numeric_array(train_X)
        except RuntimeError as e:
            raise RuntimeError(f'Failed to prepare training EEG arrays for LOSO fold {s}: {e}')
        test_X = _ensure_numeric_array(test_X)

        train_X = normalize_trials(train_X)
        test_X = normalize_trials(test_X)

        X_train = extract_features_for_trials(train_X, method=preproc)
        X_test = extract_features_for_trials(test_X, method=preproc)

        input_dim = X_train.shape[1]
        model = EEGGNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=len(np.unique(train_y)))

        start = time.time()
        trained = train_model(model, X_train, train_y, epochs=epochs, device=device)
        train_time = time.time() - start

        # evaluate
        import torch
        trained.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            outputs = trained(X_test_t)
            y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

        acc = accuracy_score(test_y, y_pred)
        f1 = f1_score(test_y, y_pred, average='weighted')
        fold_results.append({'subject': int(s), 'accuracy': float(acc), 'f1': float(f1), 'train_time_s': train_time})
        print(f"Fold {s}: acc={acc:.4f}, f1={f1:.4f}, train_time={train_time:.1f}s")

    return fold_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--preproc', default='wavelet', choices=['wavelet','dwt_bands'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--out', type=str, default='dreamer_loso_results.json')
    args = parser.parse_args()

    eeg, labels = load_data(args.data_path, dataset='DREAMER')

    # If data is subject-major (n_subjects, n_trials, ...)
    if eeg.ndim >= 3 and eeg.shape[0] == labels.shape[0]:
        print('Detected subject-major organization. Running LOSO on axis 0.')
        results = loso_from_subject_axis(eeg, labels, args.preproc, args.epochs, args.hidden_dim, args.device)
    else:
        # Try to handle flat list with subject ids per trial
        # Expect labels to be dict-like or array; here we expect labels to include subject ids if not subject-major
        try:
            # labels may be structured: (n_subjects, n_trials)
            if labels.ndim == 2 and labels.shape[0] == eeg.shape[0]:
                print('Detected labels shape (n_subjects, n_trials). Running LOSO on axis 0.')
                results = loso_from_subject_axis(eeg, labels, args.preproc, args.epochs, args.hidden_dim, args.device)
            else:
                raise ValueError('Unsupported layout for automated LOSO. Please reshape data to (n_subjects, n_trials, ...).')
        except Exception as e:
            raise RuntimeError(f'Could not infer LOSO grouping: {e}')

    # aggregate
    accs = [r['accuracy'] for r in results]
    f1s = [r['f1'] for r in results]
    summary = {'mean_accuracy': float(np.mean(accs)), 'std_accuracy': float(np.std(accs)), 'mean_f1': float(np.mean(f1s)), 'std_f1': float(np.std(f1s)), 'folds': results}

    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)

    print('LOSO finished. Summary:')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
