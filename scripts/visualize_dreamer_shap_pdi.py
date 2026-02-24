"""
Generate SHAP and PDI visualizations for DREAMER dataset
Creates publication-quality figures for paper
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy.spatial.distance import cosine

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


def load_shap_values(shap_dir):
    """Load all SHAP values from a feature directory"""
    shap_dir = Path(shap_dir)
    shap_files = sorted(shap_dir.glob('shap_subject_*.npy'))
    
    shap_values = []
    subject_ids = []
    
    for shap_file in shap_files:
        try:
            shap_val = np.load(shap_file, allow_pickle=True)
            shap_values.append(shap_val)
            # Extract subject ID from filename
            subject_id = int(shap_file.stem.split('_')[-1])
            subject_ids.append(subject_id)
        except Exception as e:
            print(f"  Failed to load {shap_file}: {e}")
    
    print(f" Loaded {len(shap_values)} SHAP value arrays")
    return shap_values, subject_ids


def compute_pdi_from_shap(shap_values):
    """Compute PDI matrix from SHAP values"""
    n_models = len(shap_values)
    pdi_matrix = np.zeros((n_models, n_models))
    
    for i in range(n_models):
        for j in range(i+1, n_models):
            # Flatten SHAP values if multidimensional
            shap_i = np.array(shap_values[i]).flatten()
            shap_j = np.array(shap_values[j]).flatten()
            
            # Ensure same length
            min_len = min(len(shap_i), len(shap_j))
            shap_i = shap_i[:min_len]
            shap_j = shap_j[:min_len]
            
            try:
                # Cosine dissimilarity
                dissimilarity = cosine(shap_i, shap_j)
                pdi_matrix[i, j] = dissimilarity
                pdi_matrix[j, i] = dissimilarity
            except:
                pdi_matrix[i, j] = 0.0
                pdi_matrix[j, i] = 0.0
    
    return pdi_matrix


def plot_pdi_heatmap(pdi_matrix, subject_ids, title, output_path):
    """Generate PDI heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(pdi_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(range(len(subject_ids)))
    ax.set_yticks(range(len(subject_ids)))
    ax.set_xticklabels([f'S{s}' for s in subject_ids], rotation=45, ha='right')
    ax.set_yticklabels([f'S{s}' for s in subject_ids])
    
    # Labels
    ax.set_xlabel('Subject (Test Fold)')
    ax.set_ylabel('Subject (Test Fold)')
    ax.set_title(title, pad=20, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Pairwise Dissimilarity Index (PDI)', rotation=270, labelpad=20)
    
    # Add values in cells
    for i in range(len(subject_ids)):
        for j in range(len(subject_ids)):
            if i != j:
                text = ax.text(j, i, f'{pdi_matrix[i, j]:.2f}',
                             ha="center", va="center", color="white", fontsize=7)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f" Saved PDI heatmap: {output_path}")


def plot_shap_summary(shap_values, subject_ids, feature_names, title, output_path):
    """Generate SHAP summary plot"""
    try:
        import shap
        
        # Aggregate SHAP values across subjects
        all_shap = []
        for shap_val in shap_values:
            if isinstance(shap_val, np.ndarray):
                # Flatten to 2D if needed
                if shap_val.ndim > 2:
                    shap_val = shap_val.reshape(shap_val.shape[0], -1)
                all_shap.append(shap_val)
        
        if len(all_shap) > 0:
            # Stack all SHAP values
            combined_shap = np.vstack(all_shap)
            
            # Mean absolute SHAP per feature
            mean_shap = np.abs(combined_shap).mean(axis=0)
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            feature_indices = range(min(20, len(mean_shap)))  # Top 20 features
            feature_labels = [f'F{i+1}' for i in feature_indices] if feature_names is None else feature_names[:20]
            
            ax.barh(range(len(feature_indices)), mean_shap[feature_indices])
            ax.set_yticks(range(len(feature_indices)))
            ax.set_yticklabels(feature_labels)
            ax.set_xlabel('Mean |SHAP value|')
            ax.set_title(title, fontweight='bold')
            ax.invert_yaxis()
            
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
            print(f" Saved SHAP summary: {output_path}")
    
    except Exception as e:
        print(f"  Failed to create SHAP summary: {e}")


def plot_pdi_distribution(pdi_matrix, title, output_path):
    """Plot distribution of PDI values"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract upper triangle (exclude diagonal)
    pdi_values = pdi_matrix[np.triu_indices_from(pdi_matrix, k=1)]
    
    # Histogram
    ax.hist(pdi_values, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(pdi_values.mean(), color='red', linestyle='--', 
               label=f'Mean: {pdi_values.mean():.3f}')
    ax.axvline(np.median(pdi_values), color='green', linestyle='--',
               label=f'Median: {np.median(pdi_values):.3f}')
    
    ax.set_xlabel('Pairwise Dissimilarity Index (PDI)')
    ax.set_ylabel('Frequency')
    ax.set_title(title, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f" Saved PDI distribution: {output_path}")


def plot_accuracy_vs_pdi(accuracies, pdi_values, subject_ids, title, output_path):
    """Plot relationship between accuracy and PDI"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot
    ax.scatter(accuracies, pdi_values, alpha=0.6, s=100, edgecolors='black')
    
    # Annotate points
    for i, (acc, pdi, sid) in enumerate(zip(accuracies, pdi_values, subject_ids)):
        ax.annotate(f'S{sid}', (acc, pdi), fontsize=8, ha='right')
    
    # Trend line
    z = np.polyfit(accuracies, pdi_values, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(accuracies), max(accuracies), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.5, label=f'Trend: y={z[0]:.3f}x + {z[1]:.3f}')
    
    ax.set_xlabel('Test Accuracy')
    ax.set_ylabel('Mean PDI (vs other models)')
    ax.set_title(title, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f" Saved Accuracy vs PDI: {output_path}")


def generate_dreamer_visualizations(dreamer_dir='dreamer_with_shap', output_dir='figures/dreamer'):
    """Generate all DREAMER SHAP and PDI visualizations"""
    
    dreamer_dir = Path(dreamer_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
     
    print("GENERATING DREAMER SHAP AND PDI VISUALIZATIONS")
     
    print(f"Input directory: {dreamer_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Find all feature directories
    feature_dirs = [d for d in dreamer_dir.iterdir() if d.is_dir() and '_' in d.name]
    
    for feature_dir in feature_dirs:
        feature_name = feature_dir.name
        print(f"\n{'='*60}")
        print(f"Processing: {feature_name}")
        print(f"{'='*60}")
        
        # Load summary
        summary_file = feature_dir / 'loso_summary.json'
        if not summary_file.exists():
            print(f"  No summary file found, skipping")
            continue
        
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        accuracies = summary.get('per_subject_accuracies', [])
        
        # Load SHAP values
        shap_dir = feature_dir / 'shap'
        if shap_dir.exists():
            shap_values, subject_ids = load_shap_values(shap_dir)
            
            if len(shap_values) > 0:
                # Compute PDI from SHAP
                print(f"\nComputing PDI matrix...")
                pdi_matrix = compute_pdi_from_shap(shap_values)
                
                # Save PDI matrix
                pdi_matrix_file = output_dir / f'pdi_matrix_{feature_name}.npy'
                np.save(pdi_matrix_file, pdi_matrix)
                print(f" Saved PDI matrix: {pdi_matrix_file}")
                
                # Generate visualizations
                print(f"\nGenerating visualizations...")
                
                # 1. PDI Heatmap
                plot_pdi_heatmap(
                    pdi_matrix, subject_ids,
                    f'DREAMER PDI Heatmap - {feature_name.replace("_", " ").title()}',
                    output_dir / f'pdi_heatmap_{feature_name}.png'
                )
                
                # 2. PDI Distribution
                plot_pdi_distribution(
                    pdi_matrix,
                    f'DREAMER PDI Distribution - {feature_name.replace("_", " ").title()}',
                    output_dir / f'pdi_distribution_{feature_name}.png'
                )
                
                # 3. SHAP Summary
                plot_shap_summary(
                    shap_values, subject_ids, None,
                    f'DREAMER SHAP Importance - {feature_name.replace("_", " ").title()}',
                    output_dir / f'shap_summary_{feature_name}.png'
                )
                
                # 4. Accuracy vs PDI
                if len(accuracies) == len(subject_ids):
                    mean_pdi_per_subject = pdi_matrix.mean(axis=1)
                    plot_accuracy_vs_pdi(
                        accuracies, mean_pdi_per_subject, subject_ids,
                        f'DREAMER Accuracy vs PDI - {feature_name.replace("_", " ").title()}',
                        output_dir / f'accuracy_vs_pdi_{feature_name}.png'
                    )
        else:
            print(f"  No SHAP directory found at {shap_dir}")
    
    # Generate cross-feature comparison
    print(f"\n{'='*60}")
    print("Generating cross-feature comparison...")
    print(f"{'='*60}")
    
    all_pdis = []
    all_accuracies = []
    feature_labels = []
    
    for feature_dir in feature_dirs:
        summary_file = feature_dir / 'loso_summary.json'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            if 'pdi' in summary:
                all_pdis.append(summary['pdi'])
                all_accuracies.append(summary['mean_accuracy'])
                feature_labels.append(feature_dir.name)
    
    if len(all_pdis) > 0:
        # Cross-feature PDI comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # PDI comparison
        ax1.bar(range(len(all_pdis)), all_pdis, edgecolor='black', alpha=0.7)
        ax1.set_xticks(range(len(all_pdis)))
        ax1.set_xticklabels([f.split('_')[0] for f in feature_labels], rotation=45, ha='right')
        ax1.set_ylabel('PDI')
        ax1.set_title('DREAMER: PDI Across Feature Methods', fontweight='bold')
        ax1.grid(alpha=0.3, axis='y')
        
        # Accuracy comparison
        ax2.bar(range(len(all_accuracies)), all_accuracies, edgecolor='black', alpha=0.7, color='green')
        ax2.set_xticks(range(len(all_accuracies)))
        ax2.set_xticklabels([f.split('_')[0] for f in feature_labels], rotation=45, ha='right')
        ax2.set_ylabel('Mean Accuracy')
        ax2.set_title('DREAMER: Accuracy Across Feature Methods', fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'dreamer_cross_feature_comparison.png', bbox_inches='tight')
        plt.close()
        print(f" Saved cross-feature comparison")
    
     
    print("DREAMER VISUALIZATIONS COMPLETE!")
     
    print(f"Output directory: {output_dir}")
    print(f"Generated figures for {len(feature_dirs)} feature methods")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate DREAMER SHAP and PDI visualizations')
    parser.add_argument('--dreamer_dir', type=str, default='dreamer_with_shap',
                       help='Directory containing DREAMER results')
    parser.add_argument('--output_dir', type=str, default='figures/dreamer',
                       help='Output directory for figures')
    
    args = parser.parse_args()
    
    generate_dreamer_visualizations(args.dreamer_dir, args.output_dir)
