"""
Generate PDI visualizations for DREAMER dataset using predictions
Creates publication-quality figures for paper (no SHAP required)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr
import argparse

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def load_predictions(feature_dir):
    """Load all predictions from a feature directory"""
    feature_dir = Path(feature_dir)
    pred_files = sorted(feature_dir.glob('predictions_subject_*.npy'))
    
    predictions = []
    subject_ids = []
    
    for pred_file in pred_files:
        try:
            preds = np.load(pred_file)
            predictions.append(preds)
            subject_id = int(pred_file.stem.split('_')[-1])
            subject_ids.append(subject_id)
        except Exception as e:
            print(f"  Failed to load {pred_file}: {e}")
    
    print(f" Loaded {len(predictions)} prediction arrays")
    return predictions, subject_ids


def load_accuracies(feature_dir):
    """Load accuracies from JSON files"""
    feature_dir = Path(feature_dir)
    json_files = sorted(feature_dir.glob('loso_subject_*.json'))
    
    accuracies = []
    subject_ids = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                accuracies.append(data['test_accuracy'])
                subject_ids.append(data['test_subject'])
        except Exception as e:
            print(f"  Failed to load {json_file}: {e}")
    
    return np.array(accuracies), subject_ids


def compute_pdi_from_predictions(predictions):
    """
    Compute Predictive Diversity Index (PDI) from model predictions
    Uses cosine distance between prediction vectors
    """
    n_models = len(predictions)
    
    if n_models == 0:
        print("  No predictions to compute PDI")
        return None
    
    # Flatten predictions to 1D vectors for each model
    flattened = []
    for pred in predictions:
        if len(pred.shape) > 1:
            # If predictions are 2D (trials × classes), flatten
            flattened.append(pred.flatten())
        else:
            flattened.append(pred)
    
    # Compute PDI matrix (cosine distance between prediction vectors)
    pdi_matrix = np.zeros((n_models, n_models))
    
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                pdi_matrix[i, j] = 0.0
            else:
                # Cosine distance: 1 - cosine_similarity
                pdi_matrix[i, j] = cosine(flattened[i], flattened[j])
    
    # Compute mean PDI (excluding diagonal)
    mask = ~np.eye(n_models, dtype=bool)
    mean_pdi = pdi_matrix[mask].mean()
    
    print(f" Computed PDI matrix ({n_models}×{n_models})")
    print(f"  Mean PDI: {mean_pdi:.4f}")
    print(f"  Min PDI: {pdi_matrix[mask].min():.4f}")
    print(f"  Max PDI: {pdi_matrix[mask].max():.4f}")
    
    return pdi_matrix, mean_pdi


def plot_pdi_heatmap(pdi_matrix, subject_ids, feature_name, output_path):
    """Plot PDI heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(pdi_matrix, 
                xticklabels=subject_ids,
                yticklabels=subject_ids,
                cmap='RdYlBu_r',
                vmin=0, vmax=1,
                cbar_kws={'label': 'Predictive Diversity Index (PDI)'},
                ax=ax,
                square=True)
    
    ax.set_xlabel('Test Subject')
    ax.set_ylabel('Test Subject')
    ax.set_title(f'DREAMER PDI Matrix - {feature_name.capitalize()} Features\n'
                 f'(n={len(subject_ids)} subject-specific models)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved PDI heatmap: {output_path}")


def plot_accuracy_distribution(accuracies, feature_name, output_path):
    """Plot accuracy distribution across subjects"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.hist(accuracies * 100, bins=15, alpha=0.7, edgecolor='black')
    ax.axvline(accuracies.mean() * 100, color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {accuracies.mean()*100:.2f}%')
    
    ax.set_xlabel('Test Accuracy (%)')
    ax.set_ylabel('Number of Subjects')
    ax.set_title(f'DREAMER Accuracy Distribution - {feature_name.capitalize()} Features\n'
                 f'(n={len(accuracies)} subjects)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved accuracy distribution: {output_path}")


def plot_accuracy_vs_pdi(accuracies, pdi_matrix, feature_name, output_path):
    """Plot relationship between accuracy and mean PDI"""
    # Compute mean PDI for each model (excluding self)
    mask = ~np.eye(len(pdi_matrix), dtype=bool)
    mean_pdi_per_model = []
    
    for i in range(len(pdi_matrix)):
        pdi_values = pdi_matrix[i][mask[i]]
        mean_pdi_per_model.append(pdi_values.mean())
    
    mean_pdi_per_model = np.array(mean_pdi_per_model)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(mean_pdi_per_model, accuracies * 100, alpha=0.6, s=100)
    
    # Add trend line
    z = np.polyfit(mean_pdi_per_model, accuracies * 100, 1)
    p = np.poly1d(z)
    x_line = np.linspace(mean_pdi_per_model.min(), mean_pdi_per_model.max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    
    # Compute correlation
    corr, p_value = pearsonr(mean_pdi_per_model, accuracies)
    
    ax.set_xlabel('Mean PDI (Predictive Diversity)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title(f'DREAMER: Accuracy vs PDI - {feature_name.capitalize()}\n'
                 f'Correlation: r={corr:.3f}, p={p_value:.3f}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved accuracy vs PDI plot: {output_path}")


def plot_per_subject_accuracy(accuracies, subject_ids, feature_name, output_path):
    """Plot bar chart of per-subject accuracy"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    colors = ['green' if acc > 0.6 else 'orange' if acc > 0.5 else 'red' 
              for acc in accuracies]
    
    bars = ax.bar(range(len(subject_ids)), accuracies * 100, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(accuracies.mean() * 100, color='blue', linestyle='--', 
               linewidth=2, label=f'Mean: {accuracies.mean()*100:.2f}%')
    
    ax.set_xlabel('Test Subject')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title(f'DREAMER Per-Subject Accuracy - {feature_name.capitalize()} Features')
    ax.set_xticks(range(len(subject_ids)))
    ax.set_xticklabels(subject_ids, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved per-subject accuracy: {output_path}")


def plot_pdi_distribution(pdi_matrix, feature_name, output_path):
    """Plot distribution of PDI values"""
    mask = ~np.eye(len(pdi_matrix), dtype=bool)
    pdi_values = pdi_matrix[mask].flatten()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.hist(pdi_values, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(pdi_values.mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {pdi_values.mean():.4f}')
    ax.axvline(np.median(pdi_values), color='green', linestyle='--', 
               linewidth=2, label=f'Median: {np.median(pdi_values):.4f}')
    
    ax.set_xlabel('Predictive Diversity Index (PDI)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'DREAMER PDI Distribution - {feature_name.capitalize()} Features\n'
                 f'({len(pdi_values)} pairwise comparisons)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved PDI distribution: {output_path}")


def generate_summary_stats(accuracies, pdi_matrix, feature_name, output_path):
    """Generate text summary of statistics"""
    mask = ~np.eye(len(pdi_matrix), dtype=bool)
    pdi_values = pdi_matrix[mask].flatten()
    
    summary = f"""
DREAMER Dataset Analysis Summary - {feature_name.capitalize()} Features
{'='*70}

Sample Size:
  - Number of subjects: {len(accuracies)}
  - Number of LOSO folds: {len(accuracies)}
  - Total pairwise PDI comparisons: {len(pdi_values)}

Accuracy Statistics:
  - Mean accuracy: {accuracies.mean()*100:.2f}% ± {accuracies.std()*100:.2f}%
  - Min accuracy: {accuracies.min()*100:.2f}%
  - Max accuracy: {accuracies.max()*100:.2f}%
  - Median accuracy: {np.median(accuracies)*100:.2f}%
  - Subjects with >60% accuracy: {(accuracies > 0.6).sum()} ({(accuracies > 0.6).sum()/len(accuracies)*100:.1f}%)

PDI Statistics:
  - Mean PDI: {pdi_values.mean():.4f} ± {pdi_values.std():.4f}
  - Min PDI: {pdi_values.min():.4f}
  - Max PDI: {pdi_values.max():.4f}
  - Median PDI: {np.median(pdi_values):.4f}
  - Q1 (25th percentile): {np.percentile(pdi_values, 25):.4f}
  - Q3 (75th percentile): {np.percentile(pdi_values, 75):.4f}

Correlation Analysis:
"""
    
    # Compute mean PDI per model
    mean_pdi_per_model = []
    for i in range(len(pdi_matrix)):
        pdi_vals = pdi_matrix[i][mask[i]]
        mean_pdi_per_model.append(pdi_vals.mean())
    mean_pdi_per_model = np.array(mean_pdi_per_model)
    
    corr, p_value = pearsonr(mean_pdi_per_model, accuracies)
    summary += f"  - Accuracy vs Mean PDI: r={corr:.3f}, p={p_value:.3f}\n"
    
    summary += f"\n{'='*70}\n"
    
    with open(output_path, 'w') as f:
        f.write(summary)
    
    print(f" Saved summary statistics: {output_path}")
    print(summary)


def main():
    parser = argparse.ArgumentParser(description='Generate PDI visualizations for DREAMER')
    parser.add_argument('--dreamer_dir', type=str, default='dreamer_with_shap',
                        help='Directory containing DREAMER results')
    parser.add_argument('--output_dir', type=str, default='figures/dreamer',
                        help='Output directory for figures')
    args = parser.parse_args()
    
    dreamer_dir = Path(args.dreamer_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
     
    print("DREAMER PDI VISUALIZATION PIPELINE")
    print("="*70 + "\n")
    
    # Process both feature types
    for feature_name in ['wavelet', 'lorentzian']:
        print(f"\n{'─'*70}")
        print(f"Processing {feature_name.upper()} features")
        print('─'*70 + "\n")
        
        # Find feature directory
        feature_dirs = list(dreamer_dir.glob(f'{feature_name}_*'))
        if not feature_dirs:
            print(f"  No directory found for {feature_name}, skipping...")
            continue
        
        feature_dir = feature_dirs[0]
        print(f" Using directory: {feature_dir}")
        
        # Load predictions and accuracies
        predictions, pred_subject_ids = load_predictions(feature_dir)
        accuracies, acc_subject_ids = load_accuracies(feature_dir)
        
        if len(predictions) == 0 or len(accuracies) == 0:
            print(f"  No data found for {feature_name}, skipping...")
            continue
        
        # Compute PDI
        pdi_result = compute_pdi_from_predictions(predictions)
        if pdi_result is None:
            continue
        
        pdi_matrix, mean_pdi = pdi_result
        
        # Generate all visualizations
        print(f"\n Generating visualizations...")
        
        # 1. PDI heatmap
        plot_pdi_heatmap(pdi_matrix, pred_subject_ids, feature_name,
                         output_dir / f'pdi_heatmap_{feature_name}.png')
        
        # 2. Accuracy distribution
        plot_accuracy_distribution(accuracies, feature_name,
                                    output_dir / f'accuracy_distribution_{feature_name}.png')
        
        # 3. Accuracy vs PDI
        plot_accuracy_vs_pdi(accuracies, pdi_matrix, feature_name,
                             output_dir / f'accuracy_vs_pdi_{feature_name}.png')
        
        # 4. Per-subject accuracy
        plot_per_subject_accuracy(accuracies, acc_subject_ids, feature_name,
                                   output_dir / f'per_subject_accuracy_{feature_name}.png')
        
        # 5. PDI distribution
        plot_pdi_distribution(pdi_matrix, feature_name,
                              output_dir / f'pdi_distribution_{feature_name}.png')
        
        # 6. Summary statistics
        generate_summary_stats(accuracies, pdi_matrix, feature_name,
                               output_dir / f'summary_stats_{feature_name}.txt')
        
        print(f"\n Completed {feature_name} visualizations\n")
    
     
    print(" DREAMER VISUALIZATION COMPLETE")
    print(f" All figures saved to: {output_dir}")



if __name__ == '__main__':
    main()
