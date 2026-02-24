"""
DEAP SHAP PDI Analysis and Beautiful 3D Visualizations
Generate impactful visualizations from complete DEAP predictions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import pandas as pd
from scipy.spatial.distance import cosine, pdist, squareform
from sklearn.manifold import MDS
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_deap_results_full():
    """Load all DEAP results with predictions"""
    deap_dir = Path('final_paper_results')
    results = []
    
    for json_file in sorted(deap_dir.glob('loso_subject_*.json')):
        try:
            with open(json_file) as f:
                data = json.load(f)
                
                # Load predictions if saved
                pred_file = json_file.parent / f"predictions_{json_file.stem}.npy"
                if pred_file.exists():
                    data['predictions'] = np.load(pred_file)
                
                # Load ground truth if saved
                gt_file = json_file.parent / f"groundtruth_{json_file.stem}.npy"
                if gt_file.exists():
                    data['groundtruth'] = np.load(gt_file)
                
                results.append(data)
                print(f" Loaded: {json_file.name}")
        except Exception as e:
            print(f" Error loading {json_file.name}: {e}")
    
    print(f"\n Loaded {len(results)} DEAP results")
    return results


def compute_pdi_matrix(predictions_list):
    """Compute PDI matrix from predictions using cosine distance"""
    n_models = len(predictions_list)
    pdi_matrix = np.zeros((n_models, n_models))
    
    for i in range(n_models):
        for j in range(n_models):
            if i != j:
                pred_i = predictions_list[i].flatten()
                pred_j = predictions_list[j].flatten()
                # Cosine distance (1 - cosine similarity)
                pdi_matrix[i, j] = cosine(pred_i, pred_j)
            else:
                pdi_matrix[i, j] = 0.0
    
    return pdi_matrix


def create_deap_pdi_heatmap(results, output_dir):
    """Beautiful PDI heatmap for DEAP"""

    
    predictions_list = []
    subject_labels = []
    accuracies = []
    
    for r in results:
        if 'predictions' in r:
            predictions_list.append(r['predictions'])
            subject_labels.append(f"S{r['test_subject']}")
            accuracies.append(r['test_accuracy'] * 100)
    
    if len(predictions_list) < 2:
        print(" Not enough predictions for PDI analysis")
        return
    
    # Compute PDI
    pdi_matrix = compute_pdi_matrix(predictions_list)
    
    # Create beautiful heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(pdi_matrix), k=1)
    
    # Plot heatmap
    im = ax.imshow(pdi_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('PDI (Cosine Distance)', fontsize=12, fontweight='bold')
    
    # Set ticks and labels
    ax.set_xticks(range(len(subject_labels)))
    ax.set_yticks(range(len(subject_labels)))
    ax.set_xticklabels(subject_labels, rotation=45, ha='right')
    ax.set_yticklabels(subject_labels)
    
    # Add accuracy annotations
    for i in range(len(subject_labels)):
        for j in range(len(subject_labels)):
            if i != j:
                text = ax.text(j, i, f'{pdi_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=8,
                             fontweight='bold' if pdi_matrix[i, j] > 0.5 else 'normal')
    
    # Add accuracy bar on the side
    acc_colors = plt.cm.viridis(np.array(accuracies) / 100)
    for i, (acc, color) in enumerate(zip(accuracies, acc_colors)):
        ax.add_patch(plt.Rectangle((-0.5, i-0.4), -0.3, 0.8, 
                                   facecolor=color, edgecolor='black', linewidth=1))
        ax.text(-1.2, i, f'{acc:.1f}%', ha='right', va='center', fontsize=9, fontweight='bold')
    
    ax.set_xlim(-1.5, len(subject_labels)-0.5)
    
    ax.set_title('DEAP: Pairwise Dissimilarity Index (PDI) Matrix\nRashomon Set Analysis', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Model (Subject Left Out)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model (Subject Left Out)', fontsize=12, fontweight='bold')
    
    # Add grid
    ax.set_xticks(np.arange(len(subject_labels))-0.5, minor=True)
    ax.set_yticks(np.arange(len(subject_labels))-0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add statistics box
    mean_pdi = np.mean(pdi_matrix[pdi_matrix > 0])
    std_pdi = np.std(pdi_matrix[pdi_matrix > 0])
    stats_text = f'Mean PDI: {mean_pdi:.3f}\nStd PDI: {std_pdi:.3f}\n'
    stats_text += f'Mean Acc: {np.mean(accuracies):.2f}%\nModels: {len(results)}'
    
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=2))
    
    plt.tight_layout()
    output_path = output_dir / 'deap_pdi_heatmap_beautiful.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {output_path}")
    
    return pdi_matrix, accuracies


def create_3d_rashomon_space_deap(results, pdi_matrix, accuracies, output_dir):
    """Beautiful 3D Rashomon space for DEAP"""

    
    # Prepare data
    subjects = [r['test_subject'] for r in results if 'predictions' in r]
    
    # Use MDS to embed PDI in 3D
    mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
    pdi_3d = mds.fit_transform(pdi_matrix)
    
    # Create 3D scatter
    fig = go.Figure()
    
    # Color by accuracy
    colors = accuracies
    
    fig.add_trace(go.Scatter3d(
        x=pdi_3d[:, 0],
        y=pdi_3d[:, 1],
        z=pdi_3d[:, 2],
        mode='markers+text',
        marker=dict(
            size=20,
            color=colors,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Accuracy (%)", thickness=20),
            line=dict(color='black', width=2),
            opacity=0.9
        ),
        text=[f'S{s}<br>{acc:.1f}%' for s, acc in zip(subjects, accuracies)],
        textposition="top center",
        textfont=dict(size=12, color='black', family='Arial Black'),
        name='DEAP Models',
        hovertemplate='<b>Subject %{text}</b><br>' +
                     'Accuracy: %{marker.color:.2f}%<br>' +
                     'PDI-X: %{x:.3f}<br>' +
                     'PDI-Y: %{y:.3f}<br>' +
                     'PDI-Z: %{z:.3f}<br>' +
                     '<extra></extra>'
    ))
    
    # Add lines connecting models (show relationships)
    for i in range(len(pdi_3d)):
        for j in range(i+1, len(pdi_3d)):
            if pdi_matrix[i, j] < 0.3:  # Only connect similar models
                fig.add_trace(go.Scatter3d(
                    x=[pdi_3d[i, 0], pdi_3d[j, 0]],
                    y=[pdi_3d[i, 1], pdi_3d[j, 1]],
                    z=[pdi_3d[i, 2], pdi_3d[j, 2]],
                    mode='lines',
                    line=dict(color='lightblue', width=2),
                    opacity=0.3,
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text='<b>DEAP Rashomon Set in 3D Space</b><br>' +
                 '<sub>MDS Embedding of Prediction Dissimilarity (PDI)</sub>',
            font=dict(size=20, color='darkblue', family='Arial Black')
        ),
        scene=dict(
            xaxis=dict(title='PDI Dimension 1', backgroundcolor="rgb(230, 230,230)",
                      gridcolor="white", showbackground=True),
            yaxis=dict(title='PDI Dimension 2', backgroundcolor="rgb(230, 230,230)",
                      gridcolor="white", showbackground=True),
            zaxis=dict(title='PDI Dimension 3', backgroundcolor="rgb(230, 230,230)",
                      gridcolor="white", showbackground=True),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        width=1200,
        height=900,
        font=dict(family="Arial", size=14),
        hovermode='closest',
        paper_bgcolor='white',
        plot_bgcolor='rgba(240,240,240,0.9)'
    )
    
    # Add annotation
    mean_acc = np.mean(accuracies)
    annotation_text = f'Mean Accuracy: {mean_acc:.2f}%<br>'
    annotation_text += f'Models in Rashomon Set: {len(results)}<br>'
    annotation_text += f'Mean PDI: {np.mean(pdi_matrix[pdi_matrix>0]):.3f}'
    
    fig.add_annotation(
        text=annotation_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=14, color="black", family="Arial Black"),
        align="left",
        bgcolor="rgba(255, 255, 200, 0.9)",
        bordercolor="black",
        borderwidth=2,
        borderpad=10
    )
    
    output_path = output_dir / '3d_deap_rashomon_space.html'
    fig.write_html(str(output_path))
    print(f" Saved: {output_path}")


def create_3d_accuracy_pdi_relationship(results, pdi_matrix, accuracies, output_dir):
    """3D visualization of accuracy vs PDI relationship"""

    
    subjects = [r['test_subject'] for r in results if 'predictions' in r]
    
    # Compute mean PDI for each model
    mean_pdis = [np.mean(pdi_matrix[i][pdi_matrix[i] > 0]) for i in range(len(pdi_matrix))]
    
    # Compute diversity (std of predictions)
    diversities = []
    for r in results:
        if 'predictions' in r:
            pred_probs = r['predictions']
            # Diversity = entropy or std of predictions
            diversity = np.std(pred_probs)
            diversities.append(diversity)
    
    # Create 3D scatter
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=accuracies,
        y=mean_pdis,
        z=diversities,
        mode='markers+text',
        marker=dict(
            size=18,
            color=accuracies,
            colorscale='Turbo',
            showscale=True,
            colorbar=dict(title="Accuracy (%)", thickness=20),
            line=dict(color='black', width=2),
            opacity=0.9
        ),
        text=[f'S{s}' for s in subjects],
        textposition="top center",
        textfont=dict(size=14, color='black', family='Arial Black'),
        name='DEAP Models',
        hovertemplate='<b>Subject %{text}</b><br>' +
                     'Accuracy: %{x:.2f}%<br>' +
                     'Mean PDI: %{y:.3f}<br>' +
                     'Diversity: %{z:.3f}<br>' +
                     '<extra></extra>'
    ))
    
    # Add trend surface (if enough points)
    if len(accuracies) >= 4:
        from scipy.interpolate import griddata
        
        # Create grid
        acc_range = np.linspace(min(accuracies), max(accuracies), 20)
        pdi_range = np.linspace(min(mean_pdis), max(mean_pdis), 20)
        acc_grid, pdi_grid = np.meshgrid(acc_range, pdi_range)
        
        # Interpolate diversity
        points = np.array([accuracies, mean_pdis]).T
        div_grid = griddata(points, diversities, (acc_grid, pdi_grid), method='linear')
        
        fig.add_trace(go.Surface(
            x=acc_grid,
            y=pdi_grid,
            z=div_grid,
            colorscale='Blues',
            opacity=0.3,
            showscale=False,
            name='Trend Surface',
            hoverinfo='skip'
        ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text='<b>DEAP: Accuracy × PDI × Diversity Landscape</b><br>' +
                 '<sub>Exploring the Rashomon Set Tradeoffs</sub>',
            font=dict(size=20, color='darkgreen', family='Arial Black')
        ),
        scene=dict(
            xaxis=dict(title='<b>Test Accuracy (%)</b>', backgroundcolor="rgb(230, 230,250)",
                      gridcolor="white", showbackground=True),
            yaxis=dict(title='<b>Mean PDI</b>', backgroundcolor="rgb(250, 230,230)",
                      gridcolor="white", showbackground=True),
            zaxis=dict(title='<b>Prediction Diversity</b>', backgroundcolor="rgb(230, 250,230)",
                      gridcolor="white", showbackground=True),
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.4))
        ),
        width=1200,
        height=900,
        font=dict(family="Arial", size=14),
        hovermode='closest',
        paper_bgcolor='white'
    )
    
    output_path = output_dir / '3d_deap_accuracy_pdi_diversity.html'
    fig.write_html(str(output_path))
    print(f" Saved: {output_path}")


def create_deap_comprehensive_panel(results, pdi_matrix, accuracies, output_dir):
    """Comprehensive 6-panel figure for DEAP"""

    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    subjects = [r['test_subject'] for r in results if 'predictions' in r]
    mean_pdis = [np.mean(pdi_matrix[i][pdi_matrix[i] > 0]) for i in range(len(pdi_matrix))]
    
    # 1. Accuracy distribution
    axes[0, 0].hist(accuracies, bins=8, alpha=0.7, color='steelblue', edgecolor='black', linewidth=2)
    axes[0, 0].axvline(np.mean(accuracies), color='red', linestyle='--', linewidth=3,
                      label=f'Mean: {np.mean(accuracies):.2f}%')
    axes[0, 0].set_xlabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Accuracy Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. PDI distribution
    pdi_values = pdi_matrix[pdi_matrix > 0]
    axes[0, 1].hist(pdi_values, bins=10, alpha=0.7, color='coral', edgecolor='black', linewidth=2)
    axes[0, 1].axvline(np.mean(pdi_values), color='blue', linestyle='--', linewidth=3,
                      label=f'Mean: {np.mean(pdi_values):.3f}')
    axes[0, 1].set_xlabel('PDI (Cosine Distance)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('PDI Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Accuracy vs PDI scatter
    axes[0, 2].scatter(mean_pdis, accuracies, s=200, c=accuracies, cmap='viridis',
                      alpha=0.8, edgecolor='black', linewidth=2)
    for i, subj in enumerate(subjects):
        axes[0, 2].annotate(f'S{subj}', (mean_pdis[i], accuracies[i]),
                          fontsize=10, fontweight='bold', ha='center')
    axes[0, 2].set_xlabel('Mean PDI', fontsize=12, fontweight='bold')
    axes[0, 2].set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0, 2].set_title('Accuracy vs PDI Tradeoff', fontsize=14, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Add correlation
    corr = np.corrcoef(mean_pdis, accuracies)[0, 1]
    axes[0, 2].text(0.05, 0.95, f'Correlation: {corr:.3f}',
                   transform=axes[0, 2].transAxes, fontsize=11, fontweight='bold',
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # 4. Per-subject accuracy bars
    colors = plt.cm.viridis(np.array(accuracies) / max(accuracies))
    bars = axes[1, 0].bar(range(len(subjects)), accuracies, color=colors,
                          edgecolor='black', linewidth=2, alpha=0.8)
    axes[1, 0].set_xticks(range(len(subjects)))
    axes[1, 0].set_xticklabels([f'S{s}' for s in subjects], fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Per-Subject Performance', fontsize=14, fontweight='bold')
    axes[1, 0].axhline(np.mean(accuracies), color='red', linestyle='--', linewidth=2,
                      label='Mean', alpha=0.8)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 5. Confusion matrix (aggregate)
    if 'predictions' in results[0] and 'groundtruth' in results[0]:
        all_preds = []
        all_gts = []
        for r in results:
            if 'predictions' in r and 'groundtruth' in r:
                preds = (r['predictions'] > 0.5).astype(int).flatten()
                gts = r['groundtruth'].flatten()
                all_preds.extend(preds)
                all_gts.extend(gts)
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(all_gts, all_preds)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                   cbar_kws={'label': 'Count'}, linewidths=2, linecolor='black',
                   annot_kws={'fontsize': 14, 'fontweight': 'bold'})
        axes[1, 1].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('True Label', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Aggregate Confusion Matrix', fontsize=14, fontweight='bold')
        axes[1, 1].set_xticklabels(['Negative', 'Positive'], fontsize=11)
        axes[1, 1].set_yticklabels(['Negative', 'Positive'], fontsize=11)
    
    # 6. Rashomon set size vs threshold
    thresholds = np.linspace(90, 100, 20)
    best_acc = max(accuracies)
    rashomon_sizes = []
    for thresh in thresholds:
        cutoff = best_acc * (thresh / 100)
        size = sum([acc >= cutoff for acc in accuracies])
        rashomon_sizes.append(size)
    
    axes[1, 2].plot(thresholds, rashomon_sizes, 'o-', linewidth=3, markersize=10,
                   color='purple', markeredgecolor='black', markeredgewidth=2)
    axes[1, 2].axvline(95, color='red', linestyle='--', linewidth=2, alpha=0.7,
                      label='Typical threshold (95%)')
    axes[1, 2].set_xlabel('Performance Threshold (% of best)', fontsize=12, fontweight='bold')
    axes[1, 2].set_ylabel('Rashomon Set Size', fontsize=12, fontweight='bold')
    axes[1, 2].set_title('Rashomon Set Size vs Threshold', fontsize=14, fontweight='bold')
    axes[1, 2].legend(fontsize=11)
    axes[1, 2].grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle('DEAP Dataset: Comprehensive Rashomon Set Analysis',
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    output_path = output_dir / 'deap_comprehensive_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {output_path}")


def main():

    
    # Create output directories
    output_2d = Path('figures/paper_ready')
    output_3d = Path('figures/3d_interactive')
    output_2d.mkdir(parents=True, exist_ok=True)
    output_3d.mkdir(parents=True, exist_ok=True)
    
    # Load DEAP results
    results = load_deap_results_full()
    
    if len(results) < 2:
        print("\n Need at least 2 complete DEAP results for PDI analysis")
        return

    
    pdi_matrix, accuracies = create_deap_pdi_heatmap(results, output_2d)
    create_deap_comprehensive_panel(results, pdi_matrix, accuracies, output_2d)
    

    
    create_3d_rashomon_space_deap(results, pdi_matrix, accuracies, output_3d)
    create_3d_accuracy_pdi_relationship(results, pdi_matrix, accuracies, output_3d)
    

    print("\nGenerated:")
    print("  • Beautiful PDI heatmap with accuracy bars")
    print("  • Comprehensive 6-panel analysis")
    print("  • 3D Rashomon space (MDS embedding)")
    print("  • 3D Accuracy-PDI-Diversity landscape")


if __name__ == '__main__':
    main()
