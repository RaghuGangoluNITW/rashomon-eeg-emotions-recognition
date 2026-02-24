"""
Rashomon Threshold Sweep Analysis
Analyzes existing model results to understand Rashomon set composition at different thresholds
NO RETRAINING REQUIRED - just loads existing JSON results
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_model_results(results_dir, dataset='DREAMER'):
    """Load all model results from directory"""
    
    results_dir = Path(results_dir)
    all_models = []
    
    print(f"🔍 Scanning directory: {results_dir}")
    
    # Look for pipeline subdirectories
    for pipeline_dir in results_dir.iterdir():
        if not pipeline_dir.is_dir():
            continue
        
        # Look for LOSO subject results
        for subject_file in pipeline_dir.glob('loso_subject_*.json'):
            try:
                with open(subject_file, 'r') as f:
                    data = json.load(f)
                
                # Extract key info
                model_info = {
                    'pipeline': pipeline_dir.name,
                    'subject': subject_file.stem.split('_')[-1],
                    'accuracy': data.get('test_accuracy', 0.0),
                    'f1': data.get('test_f1', 0.0),
                    'file': str(subject_file)
                }
                
                all_models.append(model_info)
            
            except Exception as e:
                print(f"  Error loading {subject_file}: {e}")
                continue
    
    # Also check for summary files
    for summary_file in results_dir.glob('**/loso_summary.json'):
        try:
            with open(summary_file, 'r') as f:
                data = json.load(f)
            
            pipeline_name = summary_file.parent.name
            
            model_info = {
                'pipeline': pipeline_name,
                'subject': 'aggregate',
                'accuracy': data.get('mean_accuracy', data.get('mean_test_accuracy', 0.0)),
                'f1': data.get('mean_f1', data.get('mean_test_f1', 0.0)),
                'file': str(summary_file)
            }
            
            all_models.append(model_info)
        
        except Exception as e:
            continue
    
    print(f" Loaded {len(all_models)} model results")
    
    return pd.DataFrame(all_models)


def compute_pdi_matrix(models_df, attribute='f1'):
    """Compute pairwise dissimilarity index (PDI) matrix"""
    
    # For now, use F1 scores as proxy for model diversity
    # In full implementation, would use SHAP attributions
    
    values = models_df[attribute].values.reshape(-1, 1)
    
    # Normalize
    values = (values - values.min()) / (values.max() - values.min() + 1e-10)
    
    # Compute pairwise cosine dissimilarity
    # As proxy: use absolute difference in normalized performance
    pdi_matrix = squareform(pdist(values, metric='euclidean'))
    
    return pdi_matrix


def analyze_threshold_sweep(models_df, thresholds, metric='f1'):
    """Analyze Rashomon sets at different thresholds"""
    
     
    print("  RASHOMON THRESHOLD SWEEP ANALYSIS")
     
    
    results = []
    
    # Find best performance
    f_star = models_df[metric].max()
    print(f"\n Best {metric.upper()}: {f_star:.4f}")
    
    for epsilon in thresholds:
        threshold_pct = (1 - epsilon) * 100
        threshold_value = (1 - epsilon) * f_star
        
        # Find Rashomon set
        rashomon_mask = models_df[metric] >= threshold_value
        rashomon_set = models_df[rashomon_mask].copy()
        
        rashomon_size = len(rashomon_set)
        
        if rashomon_size == 0:
            print(f"\n  ε={epsilon:.3f} ({threshold_pct:.1f}%): No models in set")
            continue
        
        # Compute PDI within Rashomon set
        if rashomon_size > 1:
            pdi_matrix = compute_pdi_matrix(rashomon_set, metric)
            
            # Get upper triangle (exclude diagonal)
            triu_indices = np.triu_indices_from(pdi_matrix, k=1)
            pdi_values = pdi_matrix[triu_indices]
            
            mean_pdi = pdi_values.mean()
            std_pdi = pdi_values.std()
            max_pdi = pdi_values.max()
        else:
            mean_pdi = 0.0
            std_pdi = 0.0
            max_pdi = 0.0
        
        min_f1 = rashomon_set[metric].min()
        max_f1 = rashomon_set[metric].max()
        
        print(f"\nε={epsilon:.3f} ({threshold_pct:.1f}% threshold)")
        print(f"  Rashomon set size: {rashomon_size}")
        print(f"  {metric.upper()} range: [{min_f1:.4f}, {max_f1:.4f}]")
        print(f"  Mean PDI: {mean_pdi:.4f} ± {std_pdi:.4f}")
        print(f"  Max PDI: {max_pdi:.4f}")
        
        results.append({
            'epsilon': epsilon,
            'threshold_pct': threshold_pct,
            'threshold_value': threshold_value,
            'rashomon_size': rashomon_size,
            'mean_pdi': mean_pdi,
            'std_pdi': std_pdi,
            'max_pdi': max_pdi,
            'min_f1': min_f1,
            'max_f1': max_f1,
            'f_star': f_star
        })
    
    return pd.DataFrame(results)


def create_threshold_figures(sweep_df, output_dir):
    """Create visualization figures"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n  Creating visualizations...")
    
    # Figure 1: Set size and PDI vs threshold
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = 'steelblue'
    ax1.plot(sweep_df['threshold_pct'], sweep_df['rashomon_size'], 
            'o-', linewidth=3, markersize=10, color=color1,
            markeredgecolor='black', markeredgewidth=2, label='Set Size')
    ax1.set_xlabel('Rashomon Threshold (% of best)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Rashomon Set Size', fontsize=14, fontweight='bold', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=12)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color2 = 'coral'
    ax2.plot(sweep_df['threshold_pct'], sweep_df['mean_pdi'],
            's-', linewidth=3, markersize=10, color=color2,
            markeredgecolor='black', markeredgewidth=2, label='Mean PDI')
    ax2.set_ylabel('Mean PDI within Set', fontsize=14, fontweight='bold', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=12)
    
    # Highlight standard threshold (95%)
    if 95 in sweep_df['threshold_pct'].values:
        ax1.axvline(95, color='red', linestyle='--', linewidth=2, alpha=0.7,
                   label='Standard (95%)')
    
    ax1.set_title('Rashomon Threshold Sensitivity Analysis', fontsize=16, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    fig_path = output_dir / 'threshold_sweep_size_pdi.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f" Saved: {fig_path}")
    plt.close()
    
    # Figure 2: F1 range vs set size
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Error bars showing F1 range
    f1_ranges = sweep_df['max_f1'] - sweep_df['min_f1']
    
    scatter = ax.scatter(sweep_df['rashomon_size'], sweep_df['mean_pdi'],
                        s=f1_ranges * 5000, c=sweep_df['threshold_pct'],
                        cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2)
    
    # Add labels for each point
    for idx, row in sweep_df.iterrows():
        ax.annotate(f"{row['threshold_pct']:.0f}%",
                   (row['rashomon_size'], row['mean_pdi']),
                   fontsize=10, fontweight='bold',
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Rashomon Set Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean PDI (Diversity)', fontsize=14, fontweight='bold')
    ax.set_title('Trade-off: Set Size vs. Diversity', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Threshold (% of best)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    fig_path = output_dir / 'threshold_sweep_tradeoff.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f" Saved: {fig_path}")
    plt.close()
    
    # Figure 3: Combined visualization (2-panel)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel A: Set size over threshold
    ax = axes[0]
    ax.bar(sweep_df['threshold_pct'], sweep_df['rashomon_size'],
          color='steelblue', alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_xlabel('Threshold (% of best)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rashomon Set Size', fontsize=12, fontweight='bold')
    ax.set_title('(A) Set Size vs Threshold', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel B: PDI distribution
    ax = axes[1]
    ax.errorbar(sweep_df['threshold_pct'], sweep_df['mean_pdi'],
               yerr=sweep_df['std_pdi'], fmt='o-', linewidth=3,
               markersize=10, capsize=8, capthick=2,
               color='coral', markeredgecolor='black', markeredgewidth=2)
    ax.set_xlabel('Threshold (% of best)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean PDI ± Std', fontsize=12, fontweight='bold')
    ax.set_title('(B) Diversity (PDI) vs Threshold', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = output_dir / 'ablation_threshold_sweep_combined.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f" Saved: {fig_path}")
    plt.close()
    
    print(f"\n Generated 3 figures in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Rashomon Threshold Sweep Analysis')
    
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing model results')
    parser.add_argument('--thresholds', type=float, nargs='+',
                       default=[0.01, 0.03, 0.05, 0.10, 0.15, 0.20],
                       help='Epsilon values to test')
    parser.add_argument('--metric', type=str, default='f1',
                       choices=['accuracy', 'f1'],
                       help='Metric to use for Rashomon set')
    parser.add_argument('--out-dir', type=str, default='ablation_threshold_results',
                       help='Output directory')
    parser.add_argument('--dataset', type=str, default='DREAMER',
                       help='Dataset name (for labeling)')
    
    args = parser.parse_args()
    
     
    print(" RASHOMON THRESHOLD SWEEP ANALYSIS")
     
    print(f"\nResults directory: {args.results_dir}")
    print(f"Thresholds (ε): {args.thresholds}")
    print(f"Metric: {args.metric}")
    print(f"Output directory: {args.out_dir}")
    print()
    
    # Load model results
    models_df = load_model_results(args.results_dir, args.dataset)
    
    if len(models_df) == 0:
        print("  No model results found!")
        return
    
    print(f"\n  Dataset statistics:")
    print(f"  Total models: {len(models_df)}")
    print(f"  {args.metric.upper()} range: [{models_df[args.metric].min():.4f}, {models_df[args.metric].max():.4f}]")
    print(f"  {args.metric.upper()} mean: {models_df[args.metric].mean():.4f} ± {models_df[args.metric].std():.4f}")
    
    # Run threshold sweep
    sweep_df = analyze_threshold_sweep(models_df, args.thresholds, args.metric)
    
    # Save results
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sweep_csv = output_dir / 'threshold_sweep_results.csv'
    sweep_df.to_csv(sweep_csv, index=False)
    print(f"\n Saved results: {sweep_csv}")
    
    # Create visualizations
    create_threshold_figures(sweep_df, output_dir)
    
    # Summary
     
    print(" SUMMARY")
     
    print("\nRecommended threshold: 95% (ε=0.05)")
    
    if 0.05 in sweep_df['epsilon'].values:
        row = sweep_df[sweep_df['epsilon'] == 0.05].iloc[0]
        print(f"  Set size: {row['rashomon_size']}")
        print(f"  Mean PDI: {row['mean_pdi']:.4f}")
        print(f"  {args.metric.upper()} range: [{row['min_f1']:.4f}, {row['max_f1']:.4f}]")
    
    print(f"\n Analysis complete!")
    print(f"   Output directory: {output_dir}")


if __name__ == '__main__':
    main()
