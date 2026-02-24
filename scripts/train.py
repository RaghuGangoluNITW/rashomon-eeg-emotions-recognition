# Script to train the model.
#!/usr/bin/env python
"""
Train script for Rashomon EEG Emotion Recognition.
"""

import argparse
from rashomon_emotion.preprocessing import load_data, normalize_trials
from rashomon_emotion.features import extract_wavelet_features, extract_dwt_subbands
from rashomon_emotion.model import EEGGNN
from rashomon_emotion.rashomon_runner import train_model

def main():
    parser = argparse.ArgumentParser(description="Train Rashomon EEG Model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset file or folder")
    parser.add_argument("--dataset", type=str, default='DEAP', help="Dataset to run: DEAP or DREAMER")
    parser.add_argument("--preproc", type=str, default='wavelet', choices=['wavelet', 'dwt_bands'], help="Preprocessing / feature extraction method")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--output_dim", type=int, default=2, help="Number of classes")
    args = parser.parse_args()

    # Load and preprocess data
    eeg, labels = load_data(args.data_path, dataset=args.dataset)
    eeg = normalize_trials(eeg)

    # Extract features (example: first trial only for demo)
    if args.preproc == 'wavelet':
        features = [extract_wavelet_features(trial) for trial in eeg]
    else:
        features = [extract_dwt_subbands(trial) for trial in eeg]

    # Define model
    model = EEGGNN(input_dim=len(features[0]), hidden_dim=args.hidden_dim, output_dim=args.output_dim)

    # Train model
    train_model(model, features, labels, epochs=args.epochs)

if __name__ == "__main__":
    main()
