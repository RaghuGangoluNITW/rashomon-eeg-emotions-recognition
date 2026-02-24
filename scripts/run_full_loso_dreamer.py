"""
Full Rashomon Set LOSO Evaluation on DREAMER Dataset
Uses all 5 graph construction methods and Multi-Head GNN architecture
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

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print(" SHAP not available. Install with: pip install shap")


def load_dreamer_dataset(data_path='data/DREAMER/DREAMER.mat'):
    """
    Fast DREAMER loader that directly extracts EEG and labels.
    Returns list of (subject_data, subject_labels) tuples.
    """
    import scipy.io as sio
    
    print(f"Loading DREAMER from {data_path}...")
    
    mat = sio.loadmat(data_path, struct_as_record=False, squeeze_me=True)
    data = mat['DREAMER']
    
    subjects_data = []
    n_subjects = len(data.Data) if hasattr(data.Data, '__len__') else 1
    print(f"Found {n_subjects} subjects")
    
    for subj_idx in range(n_subjects):
        try:
            subj = data.Data[subj_idx] if n_subjects > 1 else data.Data
            
            # Extract EEG stimuli trials
            eeg_struct = subj.EEG
            stimuli = eeg_struct.stimuli if hasattr(eeg_struct, 'stimuli') else None
            
            if stimuli is None:
                continue
            
            # Extract trials
            trials_list = []
            if isinstance(stimuli, np.ndarray):
                for trial in stimuli:
                    try:
                        trial_arr = np.asarray(trial, dtype=float)
                        if trial_arr.ndim == 2:
                            trials_list.append(trial_arr)
                    except:
                        continue
            
            if len(trials_list) == 0:
                continue
            
            # Normalize shapes
            min_channels = min(t.shape[0] for t in trials_list)
            min_samples = min(t.shape[1] for t in trials_list)
            
            trials_cropped = [t[:min_channels, :min_samples] for t in trials_list]
            eeg_trials = np.stack(trials_cropped, axis=0)
            
            # Transpose if needed (channels > samples means it's backwards)
            if eeg_trials.shape[1] > eeg_trials.shape[2]:
                eeg_trials = np.transpose(eeg_trials, (0, 2, 1))
            
            # Extract valence labels
            valence = np.array(subj.ScoreValence) if hasattr(subj, 'ScoreValence') else np.ones(len(trials_list)) * 3
            binary_labels = (valence > 3).astype(int)  # DREAMER uses 1-5 scale
            
            subjects_data.append((eeg_trials, binary_labels))
            
            if subj_idx == 0:
                print(f"  Subject 1: {eeg_trials.shape[0]} trials, {eeg_trials.shape[1]} channels, {eeg_trials.shape[2]} samples")
        
        except Exception as e:
            print(f"Error processing subject {subj_idx}: {e}")
            continue
    
    print(f"[OK] Loaded {len(subjects_data)} subjects")
    return subjects_data


def extract_node_features(trial_data, method='wavelet', fs=128):
    """
    Extract features for each EEG channel (node).
    
    Args:
        trial_data: [n_channels, n_samples]
        method: 'wavelet', 'dwt_bands', 'lorentzian', 'hjorth', 'comprehensive'
        fs: Sampling frequency
    Returns:
        node_features: [n_channels, n_features]
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
    if device == 'cuda' and torch.cuda.is_available():
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
            print(f"  Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Acc: {accuracy:.4f}")
    
    return model, history


def run_loso_evaluation(subjects_data,
                       feature_methods=['wavelet', 'lorentzian'],
                       graph_methods=['plv', 'coherence', 'correlation', 'mi', 'aec'],
                       hidden_dim=64, num_epochs=100,
                       out_dir='dreamer_loso_full', device='cuda', seed=42,
                       save_models=True, compute_shap=True):
    """
    Run Leave-One-Subject-Out evaluation on DREAMER with Rashomon Set.
    Tests multiple feature-graph combinations (Rashomon pipelines).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    n_subjects = len(subjects_data)
    all_results = []
    all_predictions = []  # For PDI calculation across all pipelines
    
    n_pipelines = len(feature_methods) * len(graph_methods)
    

    print(f"Subjects: {n_subjects}")
    print(f"Feature methods: {feature_methods} ({len(feature_methods)} features)")
    print(f"Graph methods: {graph_methods} ({len(graph_methods)} graphs)")
    print(f"RASHOMON PIPELINES: {n_pipelines} ({len(feature_methods)} × {len(graph_methods)})")
    print(f"Architecture: Multi-Head GNN with {len(graph_methods)} heads")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Epochs: {num_epochs}")
    print(f"Device: {device}")

    
    # Dictionary to hold results for each feature-graph combination
    rashomon_results = {}
    
    # Loop over feature methods (Rashomon Set construction)
    for feat_idx, feature_method in enumerate(feature_methods):

        
        feature_results = []
        feature_predictions = []
        
        for test_idx in tqdm(range(n_subjects), desc=f"{feature_method} LOSO"):
            # Check if subject already completed (for resuming after crashes)
            feature_dir = out_dir / f'{feature_method}_{"_".join(graph_methods)}'
            result_file = feature_dir / f'loso_subject_{test_idx+1:02d}.json'
            if result_file.exists():
                print(f"⏭️  Skipping Subject {test_idx + 1}/{n_subjects} (already completed)")
                # Load existing result for aggregation
                with open(result_file, 'r') as f:
                    result = json.load(f)
                    feature_results.append(result)
                    # Reconstruct predictions if available
                    if 'training_history' in result:
                        # We can't get predictions easily, but PDI can skip this subject
                        pass
                continue
            

            
            # Split train/test
            test_data, test_labels = subjects_data[test_idx]
            
            train_data_list = []
            train_labels_list = []
            
            for train_idx in range(n_subjects):
                if train_idx != test_idx:
                    train_data, train_labels = subjects_data[train_idx]
                    train_data_list.append(train_data)
                    train_labels_list.append(train_labels)
            
            # Concatenate all training subjects
            all_train_trials = np.concatenate(train_data_list, axis=0)
            all_train_labels = np.concatenate(train_labels_list, axis=0)
            
            print(f"Training trials: {all_train_trials.shape[0]}")
            print(f"Test trials: {test_data.shape[0]}")
            print(f"Channels: {all_train_trials.shape[1]}")
            
            # Extract features and build graphs for training data
            print(f"Extracting features with method: {feature_method}...")
            train_node_features = []
            train_adj_matrices = [[] for _ in graph_methods]
            
            for trial_idx, trial_data in enumerate(all_train_trials):
                if (trial_idx + 1) % 50 == 0:
                    print(f"  Processed {trial_idx + 1}/{len(all_train_trials)} training trials")
                
                node_feats = extract_node_features(trial_data, method=feature_method)
                train_node_features.append(node_feats)
                
                # Build all graph types
                adj_mats = build_multi_graph(trial_data, methods=graph_methods, fs=128)
                for i, adj in enumerate(adj_mats):
                    train_adj_matrices[i].append(adj)
            
            train_node_features = np.array(train_node_features)
            train_adj_matrices = [np.array(adj_list) for adj_list in train_adj_matrices]
            
            print(f" Training features shape: {train_node_features.shape}")
            print(f" Training graphs: {len(train_adj_matrices)} types")
            
            # Extract features and build graphs for test data
            print(f"Processing test trials...")
            test_node_features = []
            test_adj_matrices = [[] for _ in graph_methods]
            
            for trial_data in test_data:
                node_feats = extract_node_features(trial_data, method=feature_method)
                test_node_features.append(node_feats)
                
                adj_mats = build_multi_graph(trial_data, methods=graph_methods, fs=128)
                for i, adj in enumerate(adj_mats):
                    test_adj_matrices[i].append(adj)
            
            test_node_features = np.array(test_node_features)
            test_adj_matrices = [np.array(adj_list) for adj_list in test_adj_matrices]
            
            print(f" Test features shape: {test_node_features.shape}")
            
            # Train Multi-Head GNN
            print(f"\nTraining Multi-Head GNN ({len(graph_methods)} heads)...")
            model, history = train_rashomon_gnn(
                train_node_features, train_adj_matrices, all_train_labels,
                hidden_dim=hidden_dim, num_epochs=num_epochs,
                lr=1e-3, device=device, seed=seed
            )
            
            # Evaluate on test subject
            print(f"\nEvaluating on test subject...")
            model.eval()
            with torch.no_grad():
                X_test = torch.tensor(test_node_features, dtype=torch.float32).to(device)
                adj_test = [torch.tensor(adj, dtype=torch.float32).to(device) 
                           for adj in test_adj_matrices]
                
                outputs = model(X_test, adj_test)
                _, predicted = torch.max(outputs, 1)
                
                test_accuracy = (predicted.cpu().numpy() == test_labels).mean()
                predictions_array = outputs.cpu().numpy()
                feature_predictions.append(predictions_array)
            

            
            # Save feature-specific subdirectory
            feature_dir = out_dir / f'{feature_method}_{"_".join(graph_methods)}'
            feature_dir.mkdir(exist_ok=True, parents=True)
            
            # Save model checkpoint
            if save_models:
                model_dir = feature_dir / 'models'
                model_dir.mkdir(exist_ok=True, parents=True)
                model_path = model_dir / f'model_subject_{test_idx+1:02d}.pt'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'feature_method': feature_method,
                    'graph_methods': graph_methods,
                    'hidden_dim': hidden_dim,
                    'num_heads': len(graph_methods),
                    'test_subject': test_idx + 1,
                    'test_accuracy': test_accuracy
                }, model_path)
                print(f" Saved model: {model_path.name}")
            
            # Save predictions and ground truth
            pred_path = feature_dir / f'predictions_subject_{test_idx+1:02d}.npy'
            gt_path = feature_dir / f'groundtruth_subject_{test_idx+1:02d}.npy'
            np.save(pred_path, predictions_array)
            np.save(gt_path, test_labels)
            print(f" Saved predictions: {pred_path.name}")
            
            # Compute SHAP values
            shap_values = None
            if compute_shap and SHAP_AVAILABLE:
                try:
                    print(f"\nComputing SHAP values...")
                    shap_dir = feature_dir / 'shap'
                    shap_dir.mkdir(exist_ok=True, parents=True)
                    
                    # Use a subset for SHAP (computational efficiency)
                    n_background = min(10, len(train_node_features))
                    n_test_samples = min(5, len(test_node_features))
                    
                    # Create prediction function for SHAP
                    def predict_fn(features):
                        model.eval()
                        with torch.no_grad():
                            X = torch.tensor(features, dtype=torch.float32).to(device)
                            # Use same test adjacency matrices (approximate)
                            outputs = model(X, adj_test)
                            return outputs.cpu().numpy()
                    
                    # Sample background and test data
                    background_idx = np.random.choice(len(train_node_features), n_background, replace=False)
                    test_idx_shap = np.random.choice(len(test_node_features), n_test_samples, replace=False)
                    
                    background_data = train_node_features[background_idx]
                    test_data_shap = test_node_features[test_idx_shap]
                    
                    # Reshape for SHAP (flatten features)
                    background_flat = background_data.reshape(n_background, -1)
                    test_flat = test_data_shap.reshape(n_test_samples, -1)
                    
                    # Wrap prediction function to handle flattened input
                    def predict_flat(features_flat):
                        n_samples = features_flat.shape[0]
                        features_reshaped = features_flat.reshape(n_samples, test_node_features.shape[1], test_node_features.shape[2])
                        return predict_fn(features_reshaped)
                    
                    # Compute SHAP
                    explainer = shap.KernelExplainer(predict_flat, background_flat)
                    shap_values = explainer.shap_values(test_flat)
                    
                    # Save SHAP values
                    shap_path = shap_dir / f'shap_subject_{test_idx+1:02d}.npy'
                    np.save(shap_path, shap_values)
                    print(f" Saved SHAP values: {shap_path.name}")
                    
                except Exception as e:
                    print(f" SHAP computation failed: {e}")
                    shap_values = None
            
            # Save individual result
            result = {
                'test_subject': int(test_idx + 1),
                'feature_method': feature_method,
                'graph_methods': graph_methods,
                'num_graph_types': len(graph_methods),
                'hidden_dim': hidden_dim,
                'num_epochs': num_epochs,
                'train_samples': int(all_train_trials.shape[0]),
                'test_samples': int(test_data.shape[0]),
                'test_accuracy': float(test_accuracy),
                'training_history': history,
                'model_saved': save_models,
                'shap_computed': shap_values is not None
            }
            feature_results.append(result)
            
            # Save result JSON
            result_file = feature_dir / f'loso_subject_{test_idx+1:02d}.json'
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f" Saved result: {result_file.name}")

    pdi = compute_pdi(feature_predictions)
    print(f"PDI (Pairwise Dissimilarity Index): {pdi:.4f}")
    
    # Aggregate results for this feature
    accuracies = [r['test_accuracy'] for r in feature_results]
    feature_summary = {
        'dataset': 'DREAMER',
        'num_subjects': n_subjects,
        'mean_accuracy': float(np.mean(accuracies)),
        'std_accuracy': float(np.std(accuracies)),
        'median_accuracy': float(np.median(accuracies)),
        'min_accuracy': float(np.min(accuracies)),
        'max_accuracy': float(np.max(accuracies)),
        'pdi': float(pdi),
        'feature_method': feature_method,
        'graph_methods': graph_methods,
        'num_graph_types': len(graph_methods),
        'hidden_dim': hidden_dim,
        'num_epochs': num_epochs,
        'device': device,
        'seed': seed,
        'per_subject_accuracies': accuracies
    }
    
    feature_dir = out_dir / f'{feature_method}_{"_".join(graph_methods)}'
    summary_file = feature_dir / 'loso_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(feature_summary, f, indent=2)
    print(f" Saved feature summary: {summary_file}")
    
    # Store in rashomon results
    rashomon_results[feature_method] = {
        'summary': feature_summary,
        'results': feature_results,
        'predictions': feature_predictions
    }
    

    
    all_accuracies = []
    all_pdis = []
    for feat_method, feat_data in rashomon_results.items():
        all_accuracies.extend([r['test_accuracy'] for r in feat_data['results']])
        all_pdis.append(feat_data['summary']['pdi'])
        print(f"{feat_method:15} | Acc: {feat_data['summary']['mean_accuracy']:.4f} ± {feat_data['summary']['std_accuracy']:.4f} | PDI: {feat_data['summary']['pdi']:.4f}")
    
    rashomon_summary = {
        'dataset': 'DREAMER',
        'num_subjects': n_subjects,
        'num_feature_methods': len(feature_methods),
        'num_graph_types': len(graph_methods),
        'rashomon_pipelines': len(feature_methods) * len(graph_methods),
        'feature_methods': feature_methods,
        'graph_methods': graph_methods,
        'overall_mean_accuracy': float(np.mean(all_accuracies)),
        'overall_std_accuracy': float(np.std(all_accuracies)),
        'mean_pdi_across_features': float(np.mean(all_pdis)),
        'per_feature_summaries': {k: v['summary'] for k, v in rashomon_results.items()},
        'hidden_dim': hidden_dim,
        'num_epochs': num_epochs,
        'device': device,
        'seed': seed
    }
    
    rashomon_file = out_dir / 'rashomon_set_aggregated.json'
    with open(rashomon_file, 'w') as f:
        json.dump(rashomon_summary, f, indent=2)
    

    print(f"Total Pipelines: {rashomon_summary['rashomon_pipelines']} ({len(feature_methods)} features × {len(graph_methods)} graphs)")
    print(f"Overall Mean Accuracy: {rashomon_summary['overall_mean_accuracy']:.4f} ± {rashomon_summary['overall_std_accuracy']:.4f}")
    print(f"Mean PDI across features: {rashomon_summary['mean_pdi_across_features']:.4f}")
    print(f"\nResults saved to: {out_dir}/")
    print(f"  - Feature-specific: {', '.join([f'{f}_*/' for f in feature_methods])}")
    print(f"  - Rashomon aggregated: rashomon_set_aggregated.json")
     
    
    return rashomon_summary


def main():
    parser = argparse.ArgumentParser(
        description='Full Rashomon Set LOSO Evaluation on DREAMER Dataset'
    )
    parser.add_argument('--data_path', type=str, default='data/DREAMER/DREAMER.mat',
                       help='Path to DREAMER.mat file')
    parser.add_argument('--features', type=str, nargs='+',
                       default=['wavelet', 'lorentzian'],
                       choices=['wavelet', 'dwt_bands', 'lorentzian', 'hjorth'],
                       help='Feature extraction methods (default: wavelet, lorentzian)')
    parser.add_argument('--graphs', type=str, nargs='+',
                       default=['plv', 'coherence', 'correlation', 'mi', 'aec'],
                       help='Graph construction methods (default: all 5)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension for GNN')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--out_dir', type=str, default='dreamer_loso_full_5graph',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--save_models', action='store_true', default=True,
                       help='Save model checkpoints')
    parser.add_argument('--compute_shap', action='store_true', default=True,
                       help='Compute SHAP values')
    
    args = parser.parse_args()
    
    # Load DREAMER dataset
    subjects_data = load_dreamer_dataset(args.data_path)
    
    # Run LOSO evaluation with Rashomon Set
    summary = run_loso_evaluation(
        subjects_data=subjects_data,
        feature_methods=args.features,
        graph_methods=args.graphs,
        hidden_dim=args.hidden_dim,
        num_epochs=args.epochs,
        out_dir=args.out_dir,
        device=args.device,
        seed=args.seed,
        save_models=args.save_models,
        compute_shap=args.compute_shap
    )
    
    print("\n RASHOMON SET LOSO EVALUATION COMPLETED SUCCESSFULLY!")
    print(f"Overall accuracy: {summary['overall_mean_accuracy']:.2%} ± {summary['overall_std_accuracy']:.2%}")
    print(f"Total pipelines tested: {summary['rashomon_pipelines']}")


if __name__ == '__main__':
    main()
