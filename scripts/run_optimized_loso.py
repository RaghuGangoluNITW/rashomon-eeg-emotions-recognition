"""
Optimized Full LOSO Experiment - Uses simpler feature approach for speed
This version completes in ~2-4 hours instead of days
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import argparse
from tqdm import tqdm
from scipy.io import loadmat
import sys
sys.path.append(str(Path(__file__).parent.parent))

from rashomon_emotion.features import extract_wavelet_features, extract_dwt_subbands
from rashomon_emotion.model import EEGGNN  # Use simpler model for speed
import warnings
warnings.filterwarnings('ignore')


def _set_seed(seed):
    """Set all random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_deap_subject(subject_id, data_path='data/DEAP'):
    """Load single DEAP subject"""
    mat_file = Path(data_path) / 'data_preprocessed_matlab' / f's{subject_id:02d}.mat'
    mat_data = loadmat(str(mat_file))
    
    data = mat_data['data']  # [n_trials, n_channels, n_samples]
    labels = mat_data['labels']  # [n_trials, 4]
    
    # Binarize valence labels: high (>5) vs low (<=5)
    binary_labels = (labels[:, 0] > 5).astype(int)
    
    return data, binary_labels


def extract_trial_features(trial_data, method='wavelet', n_channels=32):
    """
    Extract features from EEG trial and flatten for simple classifier.
    
    Args:
        trial_data: [n_channels, n_samples]
        method: 'wavelet' or 'dwt_bands'
        n_channels: Number of EEG channels
    Returns:
        Feature vector (flattened across all channels)
    """
    features = []
    
    for ch in range(n_channels):
        signal = trial_data[ch, :]
        
        if method == 'wavelet':
            ch_feats = extract_wavelet_features(signal)
        elif method == 'dwt_bands':
            ch_feats = extract_dwt_subbands(signal)
        else:
            # Simple statistical features as fallback
            ch_feats = np.array([
                np.mean(signal),
                np.std(signal),
                np.max(signal),
                np.min(signal)
            ])
        
        features.append(ch_feats)
    
    return np.concatenate(features)  # Flatten all channel features


def train_model(X_train, y_train, X_test, y_test, 
               hidden_dim=64, num_epochs=100, lr=1e-3,
               device='cuda', seed=42):
    """
    Train simple neural network classifier.
    
    Args:
        X_train: [n_train, n_features]
        y_train: [n_train]
        X_test: [n_test, n_features]
        y_test: [n_test]
    Returns:
        Test accuracy, F1 score, training history
    """
    _set_seed(seed)
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)
    
    # Model
    input_dim = X_train.shape[1]
    model = EEGGNN(input_dim, hidden_dim, output_dim=2).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'epoch_time': []
    }
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        import time
        start_time = time.time()
        
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
        # Metrics
        _, predicted = torch.max(outputs, 1)
        train_acc = (predicted == y_train_t).float().mean().item()
        
        epoch_time = time.time() - start_time
        
        history['train_loss'].append(loss.item())
        history['train_accuracy'].append(train_acc)
        history['epoch_time'].append(epoch_time)
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}")
    
    # Test evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        _, test_predicted = torch.max(test_outputs, 1)
        
        test_acc = (test_predicted == y_test_t).float().mean().item()
        
        # F1 score
        from sklearn.metrics import f1_score
        test_f1 = f1_score(y_test_t.cpu().numpy(), test_predicted.cpu().numpy(), average='weighted')
    
    return test_acc, test_f1, history


def run_loso_experiment(subject_ids, data_path='data/DEAP',
                       feature_method='wavelet', hidden_dim=64,
                       num_epochs=100, lr=1e-3,
                       out_dir='final_paper_results',
                       device='cuda', seed=42):
    """
    Run Leave-One-Subject-Out cross-validation.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    all_results = []
    
    print(f"Subjects: {len(subject_ids)}")
    print(f"Feature method: {feature_method}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Epochs: {num_epochs}")
    print(f"Device: {device}")
    print(f"Output: {out_dir}")

    
    for test_idx, test_subject in enumerate(tqdm(subject_ids, desc="LOSO Progress")):
        print(f"\n--- LOSO Fold {test_idx+1}/{len(subject_ids)}: Testing on Subject {test_subject} ---")
        
        # Load all data
        train_X, train_y = [], []
        test_X, test_y = None, None
        
        print(f"Loading data for {len(subject_ids)} subjects...")
        for subj in subject_ids:
            data, labels = load_deap_subject(subj, data_path)
            
            # Extract features for all trials
            trial_features = []
            for trial in data:
                feats = extract_trial_features(trial, method=feature_method)
                trial_features.append(feats)
            trial_features = np.array(trial_features)
            
            if subj == test_subject:
                test_X = trial_features
                test_y = labels
            else:
                train_X.append(trial_features)
                train_y.append(labels)
        
        # Concatenate training data
        train_X = np.vstack(train_X)
        train_y = np.concatenate(train_y)
        
        print(f"Training samples: {len(train_X)}, Test samples: {len(test_X)}")
        print(f"Feature dimension: {train_X.shape[1]}")
        
        # Train model
        print(f"Training model...")
        test_acc, test_f1, history = train_model(
            train_X, train_y, test_X, test_y,
            hidden_dim=hidden_dim,
            num_epochs=num_epochs,
            lr=lr,
            device=device,
            seed=seed
        )
        
        print(f" Test Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
        
        # Save result
        result = {
            'test_subject': int(test_subject),
            'test_accuracy': float(test_acc),
            'test_f1': float(test_f1),
            'train_samples': int(len(train_X)),
            'test_samples': int(len(test_X)),
            'feature_method': feature_method,
            'hidden_dim': hidden_dim,
            'num_epochs': num_epochs,
            'lr': lr,
            'seed': seed,
            'device': device,
            'training_history': history
        }
        all_results.append(result)
        
        # Save individual fold result
        fold_file = out_dir / f'loso_subject_{test_subject:02d}.json'
        with open(fold_file, 'w') as f:
            json.dump(result, f, indent=2)
    
    # Compute summary statistics
    accuracies = [r['test_accuracy'] for r in all_results]
    f1_scores = [r['test_f1'] for r in all_results]
    
    summary = {
        'mean_accuracy': float(np.mean(accuracies)),
        'std_accuracy': float(np.std(accuracies)),
        'median_accuracy': float(np.median(accuracies)),
        'min_accuracy': float(np.min(accuracies)),
        'max_accuracy': float(np.max(accuracies)),
        'mean_f1': float(np.mean(f1_scores)),
        'std_f1': float(np.std(f1_scores)),
        'median_f1': float(np.median(f1_scores)),
        'num_subjects': len(subject_ids),
        'feature_method': feature_method,
        'hidden_dim': hidden_dim,
        'num_epochs': num_epochs,
        'device': device,
        'seed': seed
    }
    
    # Save summary
    summary_file = out_dir / 'loso_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary

    print(f"Mean Accuracy: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
    print(f"Median Accuracy: {summary['median_accuracy']:.4f}")
    print(f"Range: [{summary['min_accuracy']:.4f}, {summary['max_accuracy']:.4f}]")
    print(f"Mean F1: {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")
    print(f"\nResults saved to: {out_dir}")

    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Optimized DEAP LOSO Experiment')
    parser.add_argument('--subjects', type=int, nargs='+', default=list(range(1, 33)),
                       help='Subject IDs (default: all 32)')
    parser.add_argument('--data_path', type=str, default='data/DEAP',
                       help='Path to DEAP dataset')
    parser.add_argument('--feature', type=str, default='wavelet',
                       choices=['wavelet', 'dwt_bands'],
                       help='Feature extraction method')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--out_dir', type=str, default='final_paper_results',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Run experiment
    summary = run_loso_experiment(
        subject_ids=args.subjects,
        data_path=args.data_path,
        feature_method=args.feature,
        hidden_dim=args.hidden_dim,
        num_epochs=args.epochs,
        lr=args.lr,
        out_dir=args.out_dir,
        device=args.device,
        seed=args.seed
    )
    
    print("\n Experiment completed successfully!")


if __name__ == '__main__':
    main()
