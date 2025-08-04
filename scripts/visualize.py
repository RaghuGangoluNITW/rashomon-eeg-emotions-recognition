# Script for UMAP and SHAP visualizations.
#!/usr/bin/env python
"""
Visualization and interpretability script for Rashomon EEG Emotion Recognition.
"""

import argparse
from rashomon_emotion.utils import load_pickle
from rashomon_emotion.interpretability import explain_with_shap

def main():
    parser = argparse.ArgumentParser(description="Visualize and explain Rashomon EEG Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data file")
    args = parser.parse_args()

    # Load model + data (this assumes your interpretability function handles loading)
    explain_with_shap(args.model_path, args.data_path)

if __name__ == "__main__":
    main()
