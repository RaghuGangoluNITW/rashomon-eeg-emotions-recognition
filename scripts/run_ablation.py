#!/usr/bin/env python
"""Run ablation experiments (limited subjects) for DEAP and DREAMER.

Selects 5 evenly spaced subjects and runs configurations varying feature
extraction and hidden dimensions. Saves per-run JSON and a summary CSV.
"""
import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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


def _ensure_numeric_array(arr):
    a = np.asarray(arr)
    if a.dtype != object:
        return a
    elems = []
    for el in a.flatten():
        if isinstance(el, dict):
            # try common keys
            for key in ('EEG','eeg','data','trial'):
                if key in el:
                    el = el[key]
                    break
        elems.append(np.asarray(el))
    shapes = [x.shape for x in elems]
    if len(set(shapes)) != 1:
        raise RuntimeError(f'Inconsistent shapes in trials: {shapes[:5]}')
    return np.stack(elems, axis=0)


def run_limited_loso(eeg, labels, subject_indices, preproc, hidden_dims, epochs, device, out_dir, seed=42):
    results = []
    n_subjects = eeg.shape[0]

    for s in subject_indices:
        print(f"Running subject {s} (of {n_subjects})")
        test_X = _ensure_numeric_array(eeg[s])
        test_y = np.array(labels[s])

        # build train
        train_parts = []
        train_label_parts = []
        for i in range(n_subjects):
            if i == s:
                continue
            part = np.array(eeg[i])
            train_parts.append(part)
            train_label_parts.append(np.array(labels[i]))

        train_X = np.concatenate(train_parts, axis=0)
        train_y = np.concatenate(train_label_parts, axis=0)

        train_X = normalize_trials(_ensure_numeric_array(train_X))
        test_X = normalize_trials(test_X)

        for pre in preproc:
            X_train = extract_features_for_trials(train_X, method=pre)
            X_test = extract_features_for_trials(test_X, method=pre)

            for hid in hidden_dims:
                model = EEGGNN(input_dim=X_train.shape[1], hidden_dim=hid, output_dim=len(np.unique(train_y)))
                run_name = f"sub{s}_{pre}_hid{hid}"
                hist_path = os.path.join(out_dir, f"history_{run_name}.json")
                start = time.time()
                trained = train_model(model, X_train, train_y, epochs=epochs, device=device, seed=seed, history_path=hist_path)
                train_time = time.time() - start

                # evaluate
                import torch
                trained.eval()
                with torch.no_grad():
                    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
                    outputs = trained(X_test_t)
                    y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

                acc = float((y_pred == test_y).mean())
                # simple F1
                from sklearn.metrics import f1_score
                f1 = float(f1_score(test_y, y_pred, average='weighted'))

                result = {
                    'subject': int(s), 'preproc': pre, 'hidden_dim': int(hid),
                    'accuracy': acc, 'f1': f1, 'train_time_s': train_time, 'history': hist_path
                }
                print(f"{run_name}: acc={acc:.4f} f1={f1:.4f} time={train_time:.1f}s")
                results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['DEAP','DREAMER'], required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out_dir', default='ablation_out')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    eeg, labels = load_data(args.data_path, dataset=args.dataset)
    # assume subject-major
    if eeg.ndim < 3 or eeg.shape[0] != labels.shape[0]:
        raise RuntimeError('Expected subject-major EEG array (n_subjects, n_trials, ...).')

    n_subjects = eeg.shape[0]
    # pick 5 evenly spaced subjects
    indices = np.linspace(0, n_subjects-1, 5, dtype=int)

    preproc = ['wavelet', 'dwt_bands']
    hidden_dims = [32, 64, 128]

    results = run_limited_loso(eeg, labels, indices, preproc, hidden_dims, args.epochs, args.device, args.out_dir)

    out_summary = os.path.join(args.out_dir, f"ablation_summary_{args.dataset}.json")
    with open(out_summary, 'w') as f:
        json.dump(results, f, indent=2)
    print('Ablation finished. Summary saved to', out_summary)


if __name__ == '__main__':
    main()
