#!/usr/bin/env python
"""Generate aggregated tables and plots from ablation summary JSON for paper inclusion.

Reads ablation_summary_DEAP.json and creates:
- CSV table of mean/std accuracy and F1 by (preproc, hidden_dim)
- Bar charts comparing ablation conditions
"""
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary_json', default='ablation_out_deap/ablation_summary_DEAP.json')
    parser.add_argument('--out_dir', default='ablation_out_deap')
    args = parser.parse_args()

    with open(args.summary_json, 'r') as f:
        results = json.load(f)

    df = pd.DataFrame(results)

    # aggregate over subjects
    agg = df.groupby(['preproc', 'hidden_dim']).agg({
        'accuracy': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'train_time_s': ['mean', 'std']
    }).reset_index()
    agg.columns = ['_'.join(col).strip('_') for col in agg.columns.values]

    csv_path = os.path.join(args.out_dir, 'ablation_aggregated.csv')
    agg.to_csv(csv_path, index=False)
    print('Saved', csv_path)

   # bar chart: accuracy by (preproc, hidden_dim)
    plt.figure(figsize=(8,5))
    sns.barplot(data=df, x='hidden_dim', y='accuracy', hue='preproc')
    plt.title('DEAP Ablation: Accuracy by Feature & Hidden Dim')
    plt.ylabel('Accuracy')
    plt.xlabel('Hidden Dimension')
    plt.tight_layout()
    bar_path = os.path.join(args.out_dir, 'ablation_accuracy_bar.png')
    plt.savefig(bar_path)
    plt.close()
    print('Saved', bar_path)

    # bar chart: F1 by (preproc, hidden_dim)
    plt.figure(figsize=(8,5))
    sns.barplot(data=df, x='hidden_dim', y='f1', hue='preproc')
    plt.title('DEAP Ablation: F1 by Feature & Hidden Dim')
    plt.ylabel('F1 Score')
    plt.xlabel('Hidden Dimension')
    plt.tight_layout()
    f1_bar_path = os.path.join(args.out_dir, 'ablation_f1_bar.png')
    plt.savefig(f1_bar_path)
    plt.close()
    print('Saved', f1_bar_path)

    # runtime comparison
    plt.figure(figsize=(8,5))
    sns.barplot(data=df, x='hidden_dim', y='train_time_s', hue='preproc')
    plt.title('DEAP Ablation: Training Time')
    plt.ylabel('Train Time (s)')
    plt.xlabel('Hidden Dimension')
    plt.tight_layout()
    time_bar_path = os.path.join(args.out_dir, 'ablation_time_bar.png')
    plt.savefig(time_bar_path)
    plt.close()
    print('Saved', time_bar_path)

if __name__ == '__main__':
    main()
