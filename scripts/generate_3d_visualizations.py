"""
Generate rich interactive 3D visualizations for EEG Emotion Recognition paper
Creates HTML files with interactive Plotly 3D graphs
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
import pandas as pd

# EEG electrode positions (10-20 system for DREAMER - 14 channels)
ELECTRODE_POSITIONS_3D = {
    'AF3': [-0.3, 0.7, 0.4],
    'F7': [-0.7, 0.3, 0.2],
    'F3': [-0.4, 0.5, 0.5],
    'FC5': [-0.6, 0.2, 0.4],
    'T7': [-0.8, 0.0, 0.1],
    'P7': [-0.7, -0.4, 0.1],
    'O1': [-0.3, -0.8, 0.2],
    'O2': [0.3, -0.8, 0.2],
    'P8': [0.7, -0.4, 0.1],
    'T8': [0.8, 0.0, 0.1],
    'FC6': [0.6, 0.2, 0.4],
    'F4': [0.4, 0.5, 0.5],
    'F8': [0.7, 0.3, 0.2],
    'AF4': [0.3, 0.7, 0.4],
}


def create_3d_rashomon_space(output_dir):
    """
    3D scatter plot of Rashomon set: Accuracy vs PDI vs Complexity
    """
    print(" Creating 3D Rashomon Space...")
    
    # Load DREAMER results
    dreamer_dir = Path('dreamer_with_shap')
    
    models_data = []
    for feature in ['wavelet', 'lorentzian']:
        feature_dirs = list(dreamer_dir.glob(f'{feature}_*'))
        if not feature_dirs:
            continue
        
        feature_dir = feature_dirs[0]
        json_files = sorted(feature_dir.glob('loso_subject_*.json'))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    models_data.append({
                        'feature': feature.capitalize(),
                        'subject': data['test_subject'],
                        'accuracy': data['test_accuracy'] * 100,
                        'epochs': data.get('num_epochs', 100),
                        'hidden_dim': data.get('hidden_dim', 64),
                    })
            except:
                continue
    
    df = pd.DataFrame(models_data)
    
    # Compute PDI-like metric (variance across subjects per feature)
    df['pdi'] = df.groupby('feature')['accuracy'].transform(lambda x: np.abs(x - x.mean()))
    df['complexity'] = df['hidden_dim'] * df['epochs'] / 1000  # Normalized complexity
    
    # Create 3D scatter
    fig = go.Figure()
    
    for feature in df['feature'].unique():
        feature_df = df[df['feature'] == feature]
        
        fig.add_trace(go.Scatter3d(
            x=feature_df['accuracy'],
            y=feature_df['pdi'],
            z=feature_df['complexity'],
            mode='markers',
            name=feature,
            marker=dict(
                size=8,
                color=feature_df['accuracy'],
                colorscale='Viridis',
                showscale=True,
                opacity=0.8,
                line=dict(color='white', width=0.5)
            ),
            text=[f"Subject {s}<br>Acc: {a:.2f}%<br>PDI: {p:.2f}<br>Feature: {f}" 
                  for s, a, p, f in zip(feature_df['subject'], feature_df['accuracy'], 
                                       feature_df['pdi'], feature_df['feature'])],
            hoverinfo='text'
        ))
    
    fig.update_layout(
        title=dict(
            text='<b>3D Rashomon Set Visualization: DREAMER Dataset</b><br>' +
                 '<i>Interactive: Rotate, Zoom, Pan to Explore Model Space</i>',
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        scene=dict(
            xaxis=dict(title='<b>Test Accuracy (%)</b>', backgroundcolor="rgb(230, 230,230)"),
            yaxis=dict(title='<b>PDI (Prediction Diversity)</b>', backgroundcolor="rgb(230, 230,230)"),
            zaxis=dict(title='<b>Model Complexity (×10³)</b>', backgroundcolor="rgb(230, 230,230)"),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        width=1200,
        height=800,
        font=dict(size=12),
        hoverlabel=dict(font_size=14),
        legend=dict(x=0.02, y=0.98, font=dict(size=14))
    )
    
    output_path = output_dir / '3d_rashomon_space.html'
    fig.write_html(output_path)
    print(f" Saved: {output_path}")
    return output_path


def create_3d_brain_network(output_dir):
    """
    3D Brain connectivity network showing EEG electrodes and connections
    """
    print(" Creating 3D Brain Network...")
    
    # Create electrode nodes
    electrodes = list(ELECTRODE_POSITIONS_3D.keys())
    positions = np.array(list(ELECTRODE_POSITIONS_3D.values()))
    
    # Create example connectivity (simulate PLV connections)
    n_electrodes = len(electrodes)
    np.random.seed(42)
    connectivity = np.random.rand(n_electrodes, n_electrodes)
    connectivity = (connectivity + connectivity.T) / 2  # Symmetric
    np.fill_diagonal(connectivity, 0)
    
    # Threshold for strong connections
    threshold = 0.7
    
    # Create network graph
    fig = go.Figure()
    
    # Add edges (connections)
    edge_traces = []
    for i in range(n_electrodes):
        for j in range(i+1, n_electrodes):
            if connectivity[i, j] > threshold:
                x_coords = [positions[i, 0], positions[j, 0], None]
                y_coords = [positions[i, 1], positions[j, 1], None]
                z_coords = [positions[i, 2], positions[j, 2], None]
                
                fig.add_trace(go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode='lines',
                    line=dict(
                        color=px.colors.sequential.Plasma[int(connectivity[i,j]*10)],
                        width=connectivity[i, j] * 8
                    ),
                    hoverinfo='skip',
                    showlegend=False,
                    opacity=0.6
                ))
    
    # Add nodes (electrodes)
    node_importance = connectivity.sum(axis=1)  # Sum of connections
    
    fig.add_trace(go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode='markers+text',
        marker=dict(
            size=node_importance * 3,
            color=node_importance,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Node<br>Importance"),
            line=dict(color='white', width=2),
            opacity=0.9
        ),
        text=electrodes,
        textposition='top center',
        textfont=dict(size=10, color='black', family='Arial Black'),
        hovertext=[f"<b>{e}</b><br>Connections: {ni:.2f}<br>Position: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})"
                   for e, ni, p in zip(electrodes, node_importance, positions)],
        hoverinfo='text',
        name='EEG Electrodes'
    ))
    
    # Add brain surface (hemisphere wireframe)
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_brain = 0.9 * np.outer(np.cos(u), np.sin(v))
    y_brain = 0.9 * np.outer(np.sin(u), np.sin(v))
    z_brain = 0.6 * np.outer(np.ones(np.size(u)), np.cos(v))
    
    fig.add_trace(go.Surface(
        x=x_brain,
        y=y_brain,
        z=z_brain,
        colorscale=[[0, 'lightblue'], [1, 'lightblue']],
        opacity=0.1,
        showscale=False,
        hoverinfo='skip',
        name='Brain Surface'
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>3D EEG Functional Connectivity Network</b><br>' +
                 '<i>Interactive Brain Network: 14 DREAMER Electrodes</i>',
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        scene=dict(
            xaxis=dict(title='', showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(title='', showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(title='', showgrid=False, zeroline=False, showticklabels=False),
            bgcolor='rgb(240, 240, 245)',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.2)
            )
        ),
        width=1200,
        height=900,
        font=dict(size=12),
        showlegend=True
    )
    
    output_path = output_dir / '3d_brain_network.html'
    fig.write_html(output_path)
    print(f" Saved: {output_path}")
    return output_path


def create_3d_performance_landscape(output_dir):
    """
    3D surface plot showing performance landscape across features and subjects
    """
    print(" Creating 3D Performance Landscape...")
    
    # Load DREAMER results
    dreamer_dir = Path('dreamer_with_shap')
    
    # Create accuracy matrix: subjects × features
    subjects = list(range(1, 24))
    features = ['wavelet', 'lorentzian']
    
    accuracy_matrix = np.zeros((len(subjects), len(features)))
    
    for f_idx, feature in enumerate(features):
        feature_dirs = list(dreamer_dir.glob(f'{feature}_*'))
        if not feature_dirs:
            continue
        
        feature_dir = feature_dirs[0]
        for s_idx, subject in enumerate(subjects):
            json_file = feature_dir / f'loso_subject_{subject:02d}.json'
            if json_file.exists():
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        accuracy_matrix[s_idx, f_idx] = data['test_accuracy'] * 100
                except:
                    pass
    
    # Create meshgrid for surface plot
    X, Y = np.meshgrid(range(len(features)), subjects)
    Z = accuracy_matrix
    
    fig = go.Figure(data=[go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale='Turbo',
        colorbar=dict(title='Accuracy<br>(%)', tickfont=dict(size=12)),
        hovertemplate='<b>Feature:</b> %{x}<br><b>Subject:</b> %{y}<br><b>Accuracy:</b> %{z:.2f}%<extra></extra>',
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project=dict(z=True))
        )
    )])
    
    fig.update_layout(
        title=dict(
            text='<b>3D Performance Landscape: DREAMER Dataset</b><br>' +
                 '<i>Accuracy Surface Across Subjects and Features</i>',
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        scene=dict(
            xaxis=dict(
                title='<b>Feature Type</b>',
                tickmode='array',
                tickvals=[0, 1],
                ticktext=['Wavelet', 'Lorentzian']
            ),
            yaxis=dict(title='<b>Test Subject ID</b>'),
            zaxis=dict(title='<b>Test Accuracy (%)</b>'),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.3)
            )
        ),
        width=1200,
        height=800,
        font=dict(size=12)
    )
    
    output_path = output_dir / '3d_performance_landscape.html'
    fig.write_html(output_path)
    print(f" Saved: {output_path}")
    return output_path


def create_3d_pdi_cube(output_dir):
    """
    3D visualization of PDI relationships between models
    """
    print(" Creating 3D PDI Cube...")
    
    # Load predictions to compute PDI
    dreamer_dir = Path('dreamer_with_shap/wavelet_plv_coherence_correlation_mi_aec')
    
    pred_files = sorted(dreamer_dir.glob('predictions_subject_*.npy'))[:15]  # First 15 for visualization
    
    predictions = []
    subject_ids = []
    for pred_file in pred_files:
        preds = np.load(pred_file)
        predictions.append(preds.flatten())
        subject_ids.append(int(pred_file.stem.split('_')[-1]))
    
    predictions = np.array(predictions)
    
    # Compute pairwise cosine distances (PDI)
    from scipy.spatial.distance import pdist, squareform
    from scipy.spatial.distance import cosine
    
    n_models = len(predictions)
    pdi_matrix = np.zeros((n_models, n_models))
    
    for i in range(n_models):
        for j in range(n_models):
            if i != j:
                pdi_matrix[i, j] = cosine(predictions[i], predictions[j])
    
    # Create 3D scatter with connections based on PDI
    fig = go.Figure()
    
    # Use MDS to embed models in 3D based on PDI
    from sklearn.manifold import MDS
    mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
    positions_3d = mds.fit_transform(pdi_matrix)
    
    # Add edges between close models
    threshold = np.percentile(pdi_matrix[pdi_matrix > 0], 30)
    
    for i in range(n_models):
        for j in range(i+1, n_models):
            if pdi_matrix[i, j] < threshold:
                fig.add_trace(go.Scatter3d(
                    x=[positions_3d[i, 0], positions_3d[j, 0], None],
                    y=[positions_3d[i, 1], positions_3d[j, 1], None],
                    z=[positions_3d[i, 2], positions_3d[j, 2], None],
                    mode='lines',
                    line=dict(color='lightgray', width=1),
                    hoverinfo='skip',
                    showlegend=False,
                    opacity=0.3
                ))
    
    # Add nodes (models)
    mean_pdi = pdi_matrix.sum(axis=1) / (n_models - 1)
    
    fig.add_trace(go.Scatter3d(
        x=positions_3d[:, 0],
        y=positions_3d[:, 1],
        z=positions_3d[:, 2],
        mode='markers+text',
        marker=dict(
            size=15,
            color=mean_pdi,
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title="Mean<br>PDI"),
            line=dict(color='black', width=1),
            opacity=0.9
        ),
        text=[f"S{s}" for s in subject_ids],
        textposition='top center',
        textfont=dict(size=10, color='black'),
        hovertext=[f"<b>Subject {s}</b><br>Mean PDI: {mpdi:.4f}" 
                   for s, mpdi in zip(subject_ids, mean_pdi)],
        hoverinfo='text',
        name='Subject Models'
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>3D PDI Model Space (MDS Embedding)</b><br>' +
                 '<i>Models Positioned by Prediction Diversity</i>',
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        scene=dict(
            xaxis=dict(title='<b>MDS Dimension 1</b>'),
            yaxis=dict(title='<b>MDS Dimension 2</b>'),
            zaxis=dict(title='<b>MDS Dimension 3</b>'),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1200,
        height=900,
        font=dict(size=12)
    )
    
    output_path = output_dir / '3d_pdi_model_space.html'
    fig.write_html(output_path)
    print(f" Saved: {output_path}")
    return output_path


def create_3d_training_trajectory(output_dir):
    """
    3D trajectory of training dynamics across subjects
    """
    print(" Creating 3D Training Trajectory...")
    
    # Simulate/load training history
    dreamer_dir = Path('dreamer_with_shap/wavelet_plv_coherence_correlation_mi_aec')
    
    fig = go.Figure()
    
    # Load training histories from JSON files
    json_files = sorted(dreamer_dir.glob('loso_subject_*.json'))[:10]
    
    colors = px.colors.qualitative.Set3
    
    for idx, json_file in enumerate(json_files):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                subject_id = data['test_subject']
                
                if 'training_history' in data and 'loss' in data['training_history']:
                    losses = data['training_history']['loss'][:50]  # First 50 epochs
                    epochs = list(range(len(losses)))
                    
                    # Create 3D spiral trajectory
                    theta = np.linspace(0, 4*np.pi, len(losses))
                    x = np.array(epochs)
                    y = np.cos(theta) * np.array(losses)
                    z = np.sin(theta) * np.array(losses)
                    
                    fig.add_trace(go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        mode='lines+markers',
                        name=f'Subject {subject_id}',
                        line=dict(color=colors[idx % len(colors)], width=3),
                        marker=dict(size=3),
                        hovertext=[f"Subject {subject_id}<br>Epoch {e}<br>Loss: {l:.3f}" 
                                   for e, l in zip(epochs, losses)],
                        hoverinfo='text'
                    ))
        except:
            continue
    
    fig.update_layout(
        title=dict(
            text='<b>3D Training Dynamics Trajectories</b><br>' +
                 '<i>Loss Evolution Across Subjects (Spiral Projection)</i>',
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        scene=dict(
            xaxis=dict(title='<b>Training Epoch</b>'),
            yaxis=dict(title='<b>Loss × cos(φ)</b>'),
            zaxis=dict(title='<b>Loss × sin(φ)</b>'),
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.2)
            )
        ),
        width=1200,
        height=800,
        font=dict(size=12),
        legend=dict(x=0.02, y=0.98)
    )
    
    output_path = output_dir / '3d_training_trajectory.html'
    fig.write_html(output_path)
    print(f" Saved: {output_path}")
    return output_path


def create_3d_feature_comparison(output_dir):
    """
    3D comparison of Wavelet vs Lorentzian features across subjects
    """
    print(" Creating 3D Feature Comparison...")
    
    # Load both feature results
    dreamer_dir = Path('dreamer_with_shap')
    
    data_points = []
    for feature in ['wavelet', 'lorentzian']:
        feature_dirs = list(dreamer_dir.glob(f'{feature}_*'))
        if not feature_dirs:
            continue
        
        feature_dir = feature_dirs[0]
        json_files = sorted(feature_dir.glob('loso_subject_*.json'))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                    # Load predictions
                    pred_file = feature_dir / f"predictions_subject_{data['test_subject']:02d}.npy"
                    if pred_file.exists():
                        preds = np.load(pred_file)
                        
                        data_points.append({
                            'feature': feature.capitalize(),
                            'subject': data['test_subject'],
                            'accuracy': data['test_accuracy'] * 100,
                            'confidence': preds.max(),
                            'entropy': -(preds * np.log(preds + 1e-10)).sum()
                        })
            except:
                continue
    
    df = pd.DataFrame(data_points)
    
    # Create 3D scatter
    fig = go.Figure()
    
    for feature in df['feature'].unique():
        feature_df = df[df['feature'] == feature]
        
        fig.add_trace(go.Scatter3d(
            x=feature_df['accuracy'],
            y=feature_df['confidence'],
            z=feature_df['entropy'],
            mode='markers',
            name=feature,
            marker=dict(
                size=10,
                symbol='diamond' if feature == 'Wavelet' else 'circle',
                color=feature_df['accuracy'],
                colorscale='Viridis',
                showscale=True,
                line=dict(color='white', width=1),
                opacity=0.8
            ),
            text=[f"<b>{f} - Subject {s}</b><br>Accuracy: {a:.2f}%<br>Confidence: {c:.3f}<br>Entropy: {e:.3f}"
                  for f, s, a, c, e in zip(feature_df['feature'], feature_df['subject'],
                                          feature_df['accuracy'], feature_df['confidence'],
                                          feature_df['entropy'])],
            hoverinfo='text'
        ))
    
    fig.update_layout(
        title=dict(
            text='<b>3D Feature Space Comparison: Wavelet vs Lorentzian</b><br>' +
                 '<i>Accuracy, Confidence, and Entropy Distribution</i>',
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        scene=dict(
            xaxis=dict(title='<b>Test Accuracy (%)</b>'),
            yaxis=dict(title='<b>Prediction Confidence</b>'),
            zaxis=dict(title='<b>Prediction Entropy</b>'),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.3)
            )
        ),
        width=1200,
        height=800,
        font=dict(size=12),
        legend=dict(x=0.02, y=0.98, font=dict(size=14))
    )
    
    output_path = output_dir / '3d_feature_comparison.html'
    fig.write_html(output_path)
    print(f" Saved: {output_path}")
    return output_path


def main():
    output_dir = Path('figures/3d_interactive')
    output_dir.mkdir(parents=True, exist_ok=True)
    
     
    print(" GENERATING INTERACTIVE 3D VISUALIZATIONS")
    
    
    generated_files = []
    
    try:
        generated_files.append(create_3d_rashomon_space(output_dir))
    except Exception as e:
        print(f" Failed to create Rashomon space: {e}")
    
    try:
        generated_files.append(create_3d_brain_network(output_dir))
    except Exception as e:
        print(f" Failed to create brain network: {e}")
    
    try:
        generated_files.append(create_3d_performance_landscape(output_dir))
    except Exception as e:
        print(f" Failed to create performance landscape: {e}")
    
    try:
        generated_files.append(create_3d_pdi_cube(output_dir))
    except Exception as e:
        print(f" Failed to create PDI cube: {e}")
    
    try:
        generated_files.append(create_3d_training_trajectory(output_dir))
    except Exception as e:
        print(f" Failed to create training trajectory: {e}")
    
    try:
        generated_files.append(create_3d_feature_comparison(output_dir))
    except Exception as e:
        print(f" Failed to create feature comparison: {e}")
    
     
    print(f" GENERATED {len(generated_files)} INTERACTIVE 3D VISUALIZATIONS")
     
    print("\n Files saved to: figures/3d_interactive/")
    print("\n Open in browser to interact:")
    for file in generated_files:
        print(f"   • {file.name}")
    print("\n  Features:")
    print("   • Rotate: Click and drag")
    print("   • Zoom: Scroll wheel")
    print("   • Pan: Right-click and drag")
    print("   • Reset: Double-click")
    print("   • Hover: See detailed information")
    


if __name__ == '__main__':
    main()
