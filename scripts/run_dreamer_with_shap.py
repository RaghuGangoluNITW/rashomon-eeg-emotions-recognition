"""
DREAMER LOSO with SHAP and PDI Analysis
Saves models, predictions, and computes interpretability metrics
"""

import numpy as np
import torch
import json
from pathlib import Path
import sys
sys.path.append('.')

from rashomon_emotion.preprocessing import extract_node_features
from rashomon_emotion.graph_utils import construct_graphs
from rashomon_emotion.model import train_rashomon_gnn
from rashomon_emotion.interpretability import compute_pdi
from scripts.load_dreamer_fast import load_dreamer_data

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print(" SHAP not available. Install with: pip install shap")


def compute_shap_for_model(model, test_features, test_adj, train_features, train_adj, 
                           device='cuda', n_background=50):
    """Compute SHAP values for a trained model"""
    if not SHAP_AVAILABLE:
        print(" Skipping SHAP (not installed)")
        return None
    
    model.eval()
    
    # Select background samples
    n_bg = min(n_background, len(train_features))
    bg_indices = np.random.choice(len(train_features), n_bg, replace=False)
    background_features = train_features[bg_indices]
    background_adj = [adj[bg_indices] for adj in train_adj]
    
    # Create wrapper function for SHAP
    def model_predict(X):
        """Wrapper that takes numpy array and returns predictions"""
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            # Use the same adjacency matrices for all samples
            adj_tensors = [torch.tensor(adj, dtype=torch.float32).to(device).mean(dim=0, keepdim=True).expand(len(X), -1, -1)
                          for adj in background_adj]
            outputs = model(X_tensor, adj_tensors)
            return outputs.cpu().numpy()
    
    try:
        # Use KernelExplainer (works better for complex models)
        explainer = shap.KernelExplainer(model_predict, background_features[:10])
        shap_values = explainer.shap_values(test_features[:5])  # Explain first 5 test samples
        return shap_values
    except Exception as e:
        print(f" SHAP computation failed: {e}")
        return None


def run_dreamer_loso_with_interpretability(
    data_path='data/DREAMER/DREAMER.mat',
    feature_methods=['wavelet', 'lorentzian'],
    graph_methods=['plv', 'coherence', 'correlation', 'mi', 'aec'],
    hidden_dim=64,
    num_epochs=100,
    device='cuda',
    out_dir='dreamer_with_shap',
    seed=42
):
    """Run complete DREAMER LOSO with SHAP and PDI analysis"""
    
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
     
    print("DREAMER LOSO WITH SHAP AND PDI ANALYSIS")
     
    print(f"Output directory: {out_dir}")
    print(f"Feature methods: {feature_methods}")
    print(f"Graph methods: {graph_methods}")
    print(f"Device: {device}")
    print()
    
    # Load DREAMER data
    print("Loading DREAMER data...")
    all_subjects_data, all_subjects_labels = load_dreamer_data(data_path)
    n_subjects = len(all_subjects_data)
    print(f" Loaded {n_subjects} subjects")
    
    # Storage for Rashomon set
    all_models = []
    all_predictions = []
    all_shap_values = []
    
    for feature_method in feature_methods:
   
        
        feature_dir = out_dir / f'{feature_method}_{"_".join(graph_methods)}'
        feature_dir.mkdir(exist_ok=True, parents=True)
        
        models_dir = feature_dir / 'models'
        models_dir.mkdir(exist_ok=True)
        
        shap_dir = feature_dir / 'shap'
        shap_dir.mkdir(exist_ok=True)
        
        feature_results = []
        feature_predictions = []
        feature_shap_values = []
        
        # LOSO Protocol
        for test_idx in range(n_subjects):
            print(f"\n{'='*60}")
            print(f"FOLD {test_idx + 1}/{n_subjects}: TEST SUBJECT {test_idx + 1}")
            print(f"{'='*60}")
            
            # Split data
            test_data = all_subjects_data[test_idx]
            test_labels = all_subjects_labels[test_idx]
            
            train_data = [all_subjects_data[i] for i in range(n_subjects) if i != test_idx]
            train_labels = [all_subjects_labels[i] for i in range(n_subjects) if i != test_idx]
            
            all_train_trials = np.concatenate(train_data, axis=0)
            all_train_labels = np.concatenate(train_labels, axis=0)
            
            print(f"Training trials: {all_train_trials.shape[0]}")
            print(f"Test trials: {test_data.shape[0]}")
            
            # Extract features
            print(f"\nExtracting {feature_method} features...")
            train_node_features = extract_node_features(all_train_trials, method=feature_method)
            test_node_features = extract_node_features(test_data, method=feature_method)
            
            # Construct graphs
            print(f"Constructing {len(graph_methods)} graph types...")
            train_adj_matrices = []
            test_adj_matrices = []
            
            for graph_method in graph_methods:
                print(f"  - {graph_method}")
                train_adj = construct_graphs(all_train_trials, method=graph_method)
                test_adj = construct_graphs(test_data, method=graph_method)
                train_adj_matrices.append(train_adj)
                test_adj_matrices.append(test_adj)
            
            print(f" Training features shape: {train_node_features.shape}")
            print(f" Test features shape: {test_node_features.shape}")
            
            # Train model
            print(f"\nTraining Multi-Head GNN ({len(graph_methods)} heads)...")
            model, history = train_rashomon_gnn(
                train_node_features, train_adj_matrices, all_train_labels,
                hidden_dim=hidden_dim, num_epochs=num_epochs,
                lr=1e-3, device=device, seed=seed
            )
            
            # Save model checkpoint
            model_path = models_dir / f'model_subject_{test_idx+1:02d}.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'test_subject': test_idx + 1,
                'feature_method': feature_method,
                'graph_methods': graph_methods,
                'hidden_dim': hidden_dim,
                'num_epochs': num_epochs,
                'train_accuracy': history['accuracy'][-1] if 'accuracy' in history else None
            }, model_path)
            print(f" Saved model: {model_path}")
            
            # Evaluate and save predictions
            print(f"\nEvaluating on test subject...")
            model.eval()
            with torch.no_grad():
                X_test = torch.tensor(test_node_features, dtype=torch.float32).to(device)
                adj_test = [torch.tensor(adj, dtype=torch.float32).to(device) 
                           for adj in test_adj_matrices]
                
                outputs = model(X_test, adj_test)
                _, predicted = torch.max(outputs, 1)
                
                test_accuracy = (predicted.cpu().numpy() == test_labels).mean()
                predictions_np = outputs.cpu().numpy()
                feature_predictions.append(predictions_np)
            
            print(f" Test Accuracy: {test_accuracy:.4f}")
            
            # Compute SHAP values
            if SHAP_AVAILABLE:
                print(f"\nComputing SHAP values...")
                shap_values = compute_shap_for_model(
                    model, test_node_features, test_adj_matrices,
                    train_node_features, train_adj_matrices,
                    device=device, n_background=50
                )
                
                if shap_values is not None:
                    # Save SHAP values
                    shap_path = shap_dir / f'shap_subject_{test_idx+1:02d}.npy'
                    np.save(shap_path, shap_values)
                    print(f" Saved SHAP: {shap_path}")
                    feature_shap_values.append(shap_values)
            
            # Save predictions
            pred_path = feature_dir / f'predictions_subject_{test_idx+1:02d}.npy'
            np.save(pred_path, predictions_np)
            
            # Save ground truth
            gt_path = feature_dir / f'groundtruth_subject_{test_idx+1:02d}.npy'
            np.save(gt_path, test_labels)
            
            # Save result JSON
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
                'model_saved': True,
                'predictions_saved': True,
                'shap_saved': shap_values is not None
            }
            feature_results.append(result)
            
            result_file = feature_dir / f'loso_subject_{test_idx+1:02d}.json'
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f" Saved result: {result_file}")
        
        # Compute PDI for this feature
        print(f"\n{'='*60}")
        print(f"Computing PDI for {feature_method}...")
        pdi = compute_pdi(feature_predictions)
        print(f" PDI (Pairwise Dissimilarity Index): {pdi:.4f}")
        
        # Aggregate results
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
            'per_subject_accuracies': [float(a) for a in accuracies],
            'models_saved': True,
            'shap_computed': len(feature_shap_values) > 0
        }
        
        summary_file = feature_dir / 'loso_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(feature_summary, f, indent=2)
        print(f" Saved summary: {summary_file}")
        
        all_models.append(feature_results)
        all_predictions.append(feature_predictions)
        all_shap_values.append(feature_shap_values)
    
    # Compute cross-feature PDI

    all_preds_flat = [pred for feature_preds in all_predictions for pred in feature_preds]
    overall_pdi = compute_pdi(all_preds_flat)
    print(f" Overall PDI across all models: {overall_pdi:.4f}")
    
    # Save Rashomon set summary
    rashomon_summary = {
        'dataset': 'DREAMER',
        'num_feature_methods': len(feature_methods),
        'num_graph_types': len(graph_methods),
        'total_models': len(feature_methods) * n_subjects,
        'overall_pdi': float(overall_pdi),
        'feature_methods': feature_methods,
        'graph_methods': graph_methods,
        'models_saved': True,
        'shap_computed': SHAP_AVAILABLE and len(all_shap_values) > 0
    }
    
    rashomon_file = out_dir / 'rashomon_set_summary.json'
    with open(rashomon_file, 'w') as f:
        json.dump(rashomon_summary, f, indent=2)
    print(f" Saved Rashomon summary: {rashomon_file}")
    

    print(f"Output directory: {out_dir}")
    print(f"Total models trained: {len(feature_methods) * n_subjects}")
    print(f"Overall PDI: {overall_pdi:.4f}")
    print(f"SHAP available: {SHAP_AVAILABLE}")
    
    return rashomon_summary


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run DREAMER LOSO with SHAP and PDI')
    parser.add_argument('--data_path', type=str, default='data/DREAMER/DREAMER.mat')
    parser.add_argument('--features', nargs='+', default=['wavelet', 'lorentzian'])
    parser.add_argument('--graphs', nargs='+', default=['plv', 'coherence', 'correlation', 'mi', 'aec'])
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out_dir', type=str, default='dreamer_with_shap')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    run_dreamer_loso_with_interpretability(
        data_path=args.data_path,
        feature_methods=args.features,
        graph_methods=args.graphs,
        hidden_dim=args.hidden_dim,
        num_epochs=args.epochs,
        device=args.device,
        out_dir=args.out_dir,
        seed=args.seed
    )
