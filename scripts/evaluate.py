# Script to evaluate the model.
#!/usr/bin/env python
"""
Evaluation script for Rashomon EEG Emotion Recognition.
"""

import argparse
from rashomon_emotion.utils import load_pickle
from rashomon_emotion.rashomon_runner import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Evaluate Rashomon EEG Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model file (.pt)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to DEAP .mat file")
    args = parser.parse_args()

    # Evaluate
    metrics = evaluate_model(args.model_path, args.data_path)
    print("Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
