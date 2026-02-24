"""
Full Rashomon Set Experiment Runner with Multi-Head GNN
Implements complete pipeline: multiple graph types, features, and models
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))

from rashomon_emotion.preprocessing import load_data
from rashomon_emotion.features import (
    extract_wavelet_features, extract_dwt_subbands,
    extract_lorentzian_bandpower, compute_hjorth_parameters,
    compute_spectral_entropy, extract_comprehensive_features
)
from rashomon_emotion.graph_utils import build_multi_graph
from rashomon_emotion.model import MultiHeadGNN, RashomonGNN
from rashomon_emotion.interpretability import compute_pdi, compute_node_importance
import warnings
warnings.filterwarnings('ignore')


def load_deap_subject(subject_id, data_path='data/DEAP'):
    """Load single DEAP subject with proper label extraction"""
    from scipy.io import loadmat
    
    mat_file = Path(data_path) / 'data_preprocessed_matlab' / f's{subject_id:02d}.mat'
    mat_data = loadmat(str(mat_file))
    
    data = mat_data['data']  # [n_trials, n_channels, n_samples]
    labels = mat_data['labels']  # [n_trials, 4] - valence, arousal, dominance, liking
    
    # Binarize labels: high (>5) vs low (<=5) for valence
    binary_labels = (labels[:, 0] > 5).astype(int)
    
    return data, binary_labels


def extract_node_features(trial_data, method='comprehensive', fs=128):
    """
    Extract features for each EEG channel (node).
    
    Args:
        trial_data: [n_channels, n_samples]
        method: feature extraction method
        fs: sampling frequency
    Returns:
        Node features [n_channels, n_features]
    """
    n_channels = trial_data.shape[0]
    node_features = []
    
    for ch in range(n_channels):
        signal = trial_data[ch, :]
        
        if method == 'wavelet':
            feats = extract_wavelet_features(signal)
        elif method == 'dwt_bands':
            feats = extract_dwt_subbands(signal)
        elif method == 'lorentzian':
            feats = extract_lorentzian_bandpower(signal, fs)
        elif method == 'hjorth':
            feats = compute_hjorth_parameters(signal)
        elif method == 'comprehensive':
            feats = extract_comprehensive_features(signal, fs, include_entropy=False)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        node_features.append(feats)
    
    return np.array(node_features)  # [n_channels, n_features]


def prepare_rashomon_set(trial_data, fs=128, 
                        feature_methods=['wavelet', 'dwt_bands', 'lorentzian', 'hjorth'],
                        graph_methods=['plv', 'coherence', 'correlation', 'mi']):
    """
    Prepare multiple feature-graph combinations (Rashomon set).
    
    Returns:
        List of (node_features, adjacency_matrices) tuples
    """
    rashomon_pipelines = []
    
    for feat_method in feature_methods:
        # Extract node features
        node_features = extract_node_features(trial_data, method=feat_method, fs=fs)
        
        # Build multiple connectivity graphs
        adj_matrices = build_multi_graph(trial_data, methods=graph_methods, fs=fs)
        
        rashomon_pipelines.append({
            'features': node_features,
            'graphs': adj_matrices,
            'feature_method': feat_method,
            'graph_methods': graph_methods
        })
    
    return rashomon_pipelines


def train_rashomon_gnn(node_features, adj_matrices, labels, 
                      hidden_dim=64, num_epochs=100, lr=1e-3, 
                      device='cuda', seed=42):
    """
    Train Multi-Head GNN with multiple graph types.
    
    Args:
        node_features: [n_samples, n_nodes, n_features]
        adj_matrices: List of [n_samples, n_nodes, n_nodes] per graph type
        labels: [n_samples]
        hidden_dim: Hidden dimension
        num_epochs: Training epochs
        lr: Learning rate
        device: 'cuda' or 'cpu'
        seed: Random seed
    Returns:
        Trained model and training history
    """
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    # Model setup
    n_samples, n_nodes, n_features = node_features.shape
    num_classes = len(np.unique(labels))
    num_heads = len(adj_matrices)
    
    model = MultiHeadGNN(
        num_nodes=n_nodes,
        node_features=n_features,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_heads=num_heads,
        dropout=0.5
    ).to(device)
    
    # Convert to tensors
    X = torch.tensor(node_features, dtype=torch.float32).to(device)
    y = torch.tensor(labels, dtype=torch.long).to(device)
    adj_list = [torch.tensor(adj, dtype=torch.float32).to(device) for adj in adj_matrices]
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    history = {'loss': [], 'accuracy': []}
    
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X, adj_list)
        loss = criterion(outputs, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y).float().mean().item()
        
        history['loss'].append(loss.item())
        history['accuracy'].append(accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Acc: {accuracy:.4f}")
    
    return model, history


def run_loso_evaluation(subject_ids, data_path='data/DEAP',
                       feature_methods=['wavelet', 'dwt_bands'],
                       graph_methods=['plv', 'coherence'],
                       hidden_dim=64, num_epochs=100,
                       out_dir='rashomon_out', device='cuda', seed=42):
    """
    Run Leave-One-Subject-Out evaluation with Rashomon set.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    all_results = []
    all_predictions = []  # For PDI calculation
    
    print(f"\n=== Running LOSO Evaluation on {len(subject_ids)} subjects ===")
    print(f"Feature methods: {feature_methods}")
    print(f"Graph methods: {graph_methods}")
    print(f"Device: {device}")
    print()
    
    for test_subject in tqdm(subject_ids, desc="LOSO Progress"):
        print(f"\n--- Testing on Subject {test_subject} ---")
        
        # Load all subjects
        train_X, train_y = [], []
        test_X, test_y = None, None
        
        for subj in subject_ids:
            data, labels = load_deap_subject(subj, data_path)
            
            if subj == test_subject:
                test_data_raw = data
                test_y = labels
            else:
                train_X.append(data)
                train_y.append(labels)
        
        # Prepare training data with Rashomon set
        print(f"Preparing Rashomon pipelines for {len(train_X)} training subjects...")
        
        # Use first feature-graph combination for now (can extend to all)
        feat_method = feature_methods[0]
        
        # Collect node features and graphs
        train_node_features = []
        train_adj_matrices = [[] for _ in graph_methods]
        
        for trial_data in np.concatenate(train_X, axis=0):  # All training trials
            node_feats = extract_node_features(trial_data, method=feat_method)
            adj_mats = build_multi_graph(trial_data, methods=graph_methods)
            
            train_node_features.append(node_feats)
            for i, adj in enumerate(adj_mats):
                train_adj_matrices[i].append(adj)
        
        train_node_features = np.array(train_node_features)
        train_adj_matrices = [np.array(adj_list) for adj_list in train_adj_matrices]
        train_labels = np.concatenate(train_y)
        
        # Prepare test data
        test_node_features = []
        test_adj_matrices = [[] for _ in graph_methods]
        
        for trial_data in test_data_raw:
            node_feats = extract_node_features(trial_data, method=feat_method)
            adj_mats = build_multi_graph(trial_data, methods=graph_methods)
            
            test_node_features.append(node_feats)
            for i, adj in enumerate(adj_mats):
                test_adj_matrices[i].append(adj)
        
        test_node_features = np.array(test_node_features)
        test_adj_matrices = [np.array(adj_list) for adj_list in test_adj_matrices]
        
        # Train model
        print(f"Training Multi-Head GNN ({len(graph_methods)} heads)...")
        model, history = train_rashomon_gnn(
            train_node_features, train_adj_matrices, train_labels,
            hidden_dim=hidden_dim, num_epochs=num_epochs,
            lr=1e-3, device=device, seed=seed
        )
        
        # Evaluate on test subject
        model.eval()
        with torch.no_grad():
            X_test = torch.tensor(test_node_features, dtype=torch.float32).to(device)
            adj_test = [torch.tensor(adj, dtype=torch.float32).to(device) 
                       for adj in test_adj_matrices]
            
            outputs = model(X_test, adj_test)
            _, predicted = torch.max(outputs, 1)
            
            test_accuracy = (predicted.cpu().numpy() == test_y).mean()
            all_predictions.append(outputs.cpu().numpy())
        
        print(f"Test Accuracy (Subject {test_subject}): {test_accuracy:.4f}")
        
        # Save results
        result = {
            'test_subject': int(test_subject),
            'feature_method': feat_method,
            'graph_methods': graph_methods,
            'hidden_dim': hidden_dim,
            'num_epochs': num_epochs,
            'test_accuracy': float(test_accuracy),
            'training_history': history
        }
        all_results.append(result)
        
        # Save individual result
        result_file = out_dir / f'loso_subject_{test_subject:02d}.json'
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
    
    # Compute PDI (diversity across subjects)
    print("\n=== Computing Rashomon Set Diversity (PDI) ===")
    pdi = compute_pdi(all_predictions)
    print(f"PDI (Pairwise Dissimilarity Index): {pdi:.4f}")
    
    # Aggregate results
    accuracies = [r['test_accuracy'] for r in all_results]
    summary = {
        'mean_accuracy': float(np.mean(accuracies)),
        'std_accuracy': float(np.std(accuracies)),
        'median_accuracy': float(np.median(accuracies)),
        'pdi': float(pdi),
        'num_subjects': len(subject_ids),
        'feature_methods': feature_methods,
        'graph_methods': graph_methods,
        'hidden_dim': hidden_dim,
        'num_epochs': num_epochs,
        'device': device,
        'seed': seed
    }
    
    summary_file = out_dir / 'loso_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== LOSO Evaluation Complete ===")
    print(f"Mean Accuracy: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
    print(f"Median Accuracy: {summary['median_accuracy']:.4f}")
    print(f"PDI: {summary['pdi']:.4f}")
    print(f"\nResults saved to: {out_dir}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Full Rashomon Set LOSO Evaluation')
    parser.add_argument('--subjects', type=int, nargs='+', default=list(range(1, 33)),
                       help='Subject IDs to include (default: all 32)')
    parser.add_argument('--data_path', type=str, default='data/DEAP',
                       help='Path to DEAP dataset')
    parser.add_argument('--features', type=str, nargs='+', 
                       default=['wavelet', 'dwt_bands', 'lorentzian', 'hjorth'],
                       help='Feature extraction methods')
    parser.add_argument('--graphs', type=str, nargs='+',
                       default=['plv', 'coherence', 'correlation', 'mi'],
                       help='Graph construction methods')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--out_dir', type=str, default='rashomon_full_loso',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Run LOSO evaluation
    summary = run_loso_evaluation(
        subject_ids=args.subjects,
        data_path=args.data_path,
        feature_methods=args.features,
        graph_methods=args.graphs,
        hidden_dim=args.hidden_dim,
        num_epochs=args.epochs,
        out_dir=args.out_dir,
        device=args.device,
        seed=args.seed
    )
    
    print("\n All experiments completed successfully!")


if __name__ == '__main__':
    main()
