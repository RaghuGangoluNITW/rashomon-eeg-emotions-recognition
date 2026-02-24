#!/usr/bin/env python
"""Run ablation experiments (limited subjects) for DEAP.

Selects 5 evenly spaced subject IDs and runs configurations varying feature
extraction and hidden dimensions. Saves per-run JSON and a summary CSV.
"""
import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rashomon_emotion.preprocessing import load_deap_data, normalize_trials
from rashomon_emotion.features import extract_wavelet_features, extract_dwt_subbands
from rashomon_emotion.model import EEGGNN
from rashomon_emotion.rashomon_runner import train_model
from sklearn.metrics import accuracy_score, f1_score


def extract_features_for_trials(trials, method='wavelet'):
    feats = []
    if method == 'wavelet':
        for t in trials:
            feats.append(extract_wavelet_features(t))
    else:
        for t in trials:
            feats.append(extract_dwt_subbands(t))
    return np.array(feats)


def load_deap_subject(subject_id, data_dir):
    """Load a single DEAP subject's .mat file."""
    fpath = os.path.join(data_dir, f's{subject_id:02d}.mat')
    eeg, labels = load_deap_data(fpath)
    # binarize labels: if labels is (n_trials, 4), use valence (index 0) > 5
    if labels.ndim == 2 and labels.shape[1] >= 1:
        labels = (labels[:, 0] > 5).astype(int)  # high/low valence binary
    elif labels.ndim == 1:
        labels = (labels > labels.median()).astype(int)
    return eeg, labels


def run_deap_loso_limited(subject_ids, data_dir, preproc, hidden_dims, epochs, device, out_dir, seed=42):
    """Run LOSO limited to specified subject IDs, ablating over preproc and hidden_dims."""
    results = []
    all_subjects = list(range(1, 33))  # DEAP has 32 subjects

    for test_subj in subject_ids:
        print(f"Running subject {test_subj}")
        test_X, test_y = load_deap_subject(test_subj, data_dir)
        test_X = normalize_trials(test_X)

        # build train set from all other selected subjects
        train_parts = []
        train_label_parts = []
        for subj in all_subjects:
            if subj == test_subj:
                continue
            try:
                train_eeg, train_labels = load_deap_subject(subj, data_dir)
                train_eeg = normalize_trials(train_eeg)
                train_parts.append(train_eeg)
                train_label_parts.append(train_labels)
            except Exception as e:
                print(f"Warning: could not load subject {subj}: {e}")
                continue

        if len(train_parts) == 0:
            raise RuntimeError(f"No training data for subject {test_subj}")

        train_X = np.concatenate(train_parts, axis=0)
        train_y = np.concatenate(train_label_parts, axis=0)

        for pre in preproc:
            X_train = extract_features_for_trials(train_X, method=pre)
            X_test = extract_features_for_trials(test_X, method=pre)

            for hid in hidden_dims:
                model = EEGGNN(input_dim=X_train.shape[1], hidden_dim=hid, output_dim=len(np.unique(train_y)))
                run_name = f"sub{test_subj}_{pre}_hid{hid}"
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

                acc = float(accuracy_score(test_y, y_pred))
                f1 = float(f1_score(test_y, y_pred, average='weighted'))

                result = {
                    'subject': int(test_subj), 'preproc': pre, 'hidden_dim': int(hid),
                    'accuracy': acc, 'f1': f1, 'train_time_s': train_time, 'history': hist_path
                }
                print(f"{run_name}: acc={acc:.4f} f1={f1:.4f} time={train_time:.1f}s")
                results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/DEAP/data_preprocessed_matlab')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out_dir', default='ablation_out_deap')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # pick 5 evenly spaced subjects (1..32)
    subject_ids = [1, 8, 16, 24, 32]

    preproc = ['wavelet', 'dwt_bands']
    hidden_dims = [32, 64, 128]

    results = run_deap_loso_limited(subject_ids, args.data_dir, preproc, hidden_dims, args.epochs, args.device, args.out_dir)

    out_summary = os.path.join(args.out_dir, "ablation_summary_DEAP.json")
    with open(out_summary, 'w') as f:
        json.dump(results, f, indent=2)
    print('Ablation finished. Summary saved to', out_summary)


if __name__ == '__main__':
    main()
