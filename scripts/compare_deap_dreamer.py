"""
Cross-Dataset Comparison: DEAP vs DREAMER
Analyzes SHAP patterns, PDI distributions, and model performance across both datasets
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats
import argparse

def load_pdi_matrix(pdi_path):
    """Load PDI matrix from numpy file"""
    if Path(pdi_path).exists():
        return np.load(pdi_path)
    return None

def load_accuracies(results_dir, dataset_name):
    """Load accuracy values from LOSO results"""
    accuracies = []
    
    summary_path = Path(results_dir) / "loso_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            data = json.load(f)
            accuracies = data.get('subject_accuracies', [])
    
    return np.array(accuracies)

def load_shap_values(shap_dir):
    """Load all SHAP values from a directory"""
    shap_dir = Path(shap_dir)
    shap_values = []
    
    for shap_file in sorted(shap_dir.glob("shap_subject_*.npy")):
        shap_vals = np.load(shap_file)
        shap_values.append(shap_vals)
    
    return shap_values

def plot_pdi_comparison(deap_pdi, dreamer_pdi, output_path):
    """Compare PDI distributions between DEAP and DREAMER"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Flatten upper triangles (exclude diagonal)
    deap_upper = deap_pdi[np.triu_indices_from(deap_pdi, k=1)]
    dreamer_upper = dreamer_pdi[np.triu_indices_from(dreamer_pdi, k=1)]
    
    # Plot distributions
    axes[0].hist(deap_upper, bins=30, alpha=0.6, label='DEAP', color='steelblue', density=True)
    axes[0].hist(dreamer_upper, bins=30, alpha=0.6, label='DREAMER', color='coral', density=True)
    axes[0].axvline(np.mean(deap_upper), color='steelblue', linestyle='--', linewidth=2, label=f'DEAP mean: {np.mean(deap_upper):.3f}')
    axes[0].axvline(np.mean(dreamer_upper), color='coral', linestyle='--', linewidth=2, label=f'DREAMER mean: {np.mean(dreamer_upper):.3f}')
    axes[0].set_xlabel('PDI Value')
    axes[0].set_ylabel('Density')
    axes[0].set_title('PDI Distribution Comparison')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Box plot comparison
    data_to_plot = [deap_upper, dreamer_upper]
    bp = axes[1].boxplot(data_to_plot, labels=['DEAP', 'DREAMER'], patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('coral')
    axes[1].set_ylabel('PDI Value')
    axes[1].set_title('PDI Value Distribution')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add statistics
    t_stat, p_value = stats.ttest_ind(deap_upper, dreamer_upper)
    axes[1].text(0.5, 0.95, f't-test p={p_value:.4f}', transform=axes[1].transAxes,
                ha='center', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_accuracy_comparison(deap_acc, dreamer_acc, output_path):
    """Compare model accuracies between datasets"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Distribution comparison
    axes[0].hist(deap_acc, bins=20, alpha=0.6, label='DEAP', color='steelblue', density=True)
    axes[0].hist(dreamer_acc, bins=20, alpha=0.6, label='DREAMER', color='coral', density=True)
    axes[0].axvline(np.mean(deap_acc), color='steelblue', linestyle='--', linewidth=2)
    axes[0].axvline(np.mean(dreamer_acc), color='coral', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Accuracy')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Accuracy Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Statistics
    deap_mean, deap_std = np.mean(deap_acc), np.std(deap_acc)
    dreamer_mean, dreamer_std = np.mean(dreamer_acc), np.std(dreamer_acc)
    
    datasets = ['DEAP', 'DREAMER']
    means = [deap_mean, dreamer_mean]
    stds = [deap_std, dreamer_std]
    
    x_pos = np.arange(len(datasets))
    axes[1].bar(x_pos, means, yerr=stds, capsize=5, color=['steelblue', 'coral'], alpha=0.7)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(datasets)
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Mean Accuracy Comparison')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        axes[1].text(i, mean + std + 1, f'{mean:.2f}±{std:.2f}%', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_rashomon_set_size_comparison(deap_pdi, dreamer_pdi, threshold, output_path):
    """
    Compare Rashomon set sizes at different PDI thresholds
    Rashomon set = models within threshold of each other
    """
    thresholds = np.linspace(0, 1, 50)
    deap_sizes = []
    dreamer_sizes = []
    
    for thresh in thresholds:
        # Count pairs within threshold
        deap_count = np.sum(deap_pdi <= thresh)
        dreamer_count = np.sum(dreamer_pdi <= thresh)
        deap_sizes.append(deap_count)
        dreamer_sizes.append(dreamer_count)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, deap_sizes, label='DEAP', color='steelblue', linewidth=2)
    ax.plot(thresholds, dreamer_sizes, label='DREAMER', color='coral', linewidth=2)
    ax.axvline(threshold, color='gray', linestyle='--', alpha=0.5, label=f'Threshold={threshold}')
    ax.set_xlabel('PDI Threshold')
    ax.set_ylabel('Number of Model Pairs within Threshold')
    ax.set_title('Rashomon Set Size Comparison')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def compute_summary_statistics(deap_pdi, dreamer_pdi, deap_acc, dreamer_acc):
    """Compute and print summary statistics"""
 
    
    # PDI statistics
    deap_pdi_upper = deap_pdi[np.triu_indices_from(deap_pdi, k=1)]
    dreamer_pdi_upper = dreamer_pdi[np.triu_indices_from(dreamer_pdi, k=1)]
    
    print("\nPDI Statistics:")
    print(f"  DEAP:    mean={np.mean(deap_pdi_upper):.4f}, std={np.std(deap_pdi_upper):.4f}, "
          f"median={np.median(deap_pdi_upper):.4f}")
    print(f"  DREAMER: mean={np.mean(dreamer_pdi_upper):.4f}, std={np.std(dreamer_pdi_upper):.4f}, "
          f"median={np.median(dreamer_pdi_upper):.4f}")
    
    t_stat, p_value = stats.ttest_ind(deap_pdi_upper, dreamer_pdi_upper)
    print(f"  t-test: t={t_stat:.4f}, p={p_value:.4f}")
    
    # Accuracy statistics
    print("\nAccuracy Statistics:")
    print(f"  DEAP:    mean={np.mean(deap_acc):.2f}%, std={np.std(deap_acc):.2f}%")
    print(f"  DREAMER: mean={np.mean(dreamer_acc):.2f}%, std={np.std(dreamer_acc):.2f}%")
    
    t_stat, p_value = stats.ttest_ind(deap_acc, dreamer_acc)
    print(f"  t-test: t={t_stat:.4f}, p={p_value:.4f}")
    
    # Rashomon set size (at PDI=0.1 threshold)
    threshold = 0.1
    deap_rashomon = np.sum(deap_pdi_upper <= threshold)
    dreamer_rashomon = np.sum(dreamer_pdi_upper <= threshold)
    
    print(f"\nRashomon Set Size (PDI ≤ {threshold}):")
    print(f"  DEAP:    {deap_rashomon} model pairs")
    print(f"  DREAMER: {dreamer_rashomon} model pairs")
    
    # Interpretability diversity
    print("\nInterpretability Diversity:")
    print(f"  DEAP:    {np.sum(deap_pdi_upper > 0.5)} highly diverse pairs (PDI > 0.5)")
    print(f"  DREAMER: {np.sum(dreamer_pdi_upper > 0.5)} highly diverse pairs (PDI > 0.5)")

def main():
    parser = argparse.ArgumentParser(description="Compare DEAP and DREAMER results")
    parser.add_argument("--deap_dir", type=str, default="deap_results",
                       help="Directory with DEAP results")
    parser.add_argument("--dreamer_dir", type=str, default="dreamer_with_shap",
                       help="Directory with DREAMER results")
    parser.add_argument("--output_dir", type=str, default="figures/comparison",
                       help="Output directory for comparison figures")
    parser.add_argument("--feature", type=str, default="wavelet",
                       help="Feature method to compare")
    parser.add_argument("--graphs", type=str, nargs='+', 
                       default=['plv', 'coherence', 'correlation', 'mi', 'aec'],
                       help="Graph types used")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct graph string
    graph_str = "_".join(args.graphs)
    
    # Load DEAP data
    print("Loading DEAP data...")
    deap_pdi_path = Path(args.deap_dir) / f"{args.feature}_{graph_str}" / f"pdi_matrix_{args.feature}_{graph_str}.npy"
    deap_pdi = load_pdi_matrix(deap_pdi_path)
    
    if deap_pdi is None:
        print(f" Could not find DEAP PDI matrix at: {deap_pdi_path}")
        return
    
    deap_acc = load_accuracies(Path(args.deap_dir) / f"{args.feature}_{graph_str}", "DEAP")
    
    # Load DREAMER data
    print("Loading DREAMER data...")
    dreamer_pdi_path = Path(args.dreamer_dir) / f"{args.feature}_{graph_str}" / f"pdi_matrix_{args.feature}_{graph_str}.npy"
    dreamer_pdi = load_pdi_matrix(dreamer_pdi_path)
    
    if dreamer_pdi is None:
        print(f" Could not find DREAMER PDI matrix at: {dreamer_pdi_path}")
        return
    
    dreamer_acc = load_accuracies(Path(args.dreamer_dir) / f"{args.feature}_{graph_str}", "DREAMER")
    
    # Generate comparisons
    print("\nGenerating comparison visualizations...")
    
    # PDI comparison
    plot_pdi_comparison(deap_pdi, dreamer_pdi, 
                       output_dir / f"pdi_comparison_{args.feature}.png")
    
    # Accuracy comparison
    if len(deap_acc) > 0 and len(dreamer_acc) > 0:
        plot_accuracy_comparison(deap_acc, dreamer_acc,
                                output_dir / f"accuracy_comparison_{args.feature}.png")
    
    # Rashomon set size comparison
    plot_rashomon_set_size_comparison(deap_pdi, dreamer_pdi, threshold=0.1,
                                     output_path=output_dir / f"rashomon_size_comparison_{args.feature}.png")
    
    # Compute statistics
    compute_summary_statistics(deap_pdi, dreamer_pdi, deap_acc, dreamer_acc)
    
 
    print(f"\nFigures saved to: {output_dir}/")
    print("\nKey insights to check:")
    print("1. Are PDI distributions similar? (suggests consistent Rashomon set properties)")
    print("2. Which dataset has larger Rashomon sets? (more interpretability diversity)")
    print("3. Is accuracy vs PDI trade-off similar? (suggests method generalizes)")
    print("\nUse these figures and statistics in the paper's cross-dataset comparison section.")

if __name__ == '__main__':
    main()
