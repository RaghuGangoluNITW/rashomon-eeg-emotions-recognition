
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rashomon_emotion.preprocessing import load_deap_data
from rashomon_emotion.features import (
    extract_wavelet_features,
    extract_lorentzian_bandpower,
    compute_hjorth_parameters,
    extract_dwt_subbands
)
from rashomon_emotion.graph_utils import build_connectivity_graph



def build_plv_fast(eeg_data):
    """Vectorized PLV matrix.
    Precomputes all 32 analytic signals via a single hilbert call (axis=1),
    then forms the PLV matrix with a single matrix multiply — ~32x faster
    than calling scipy.signal.hilbert per pair.
    """
    from scipy.signal import hilbert as sp_hilbert
    analytic = sp_hilbert(eeg_data, axis=1)          # [n_ch, n_samples]
    exp_phases = np.exp(1j * np.angle(analytic))     # [n_ch, n_samples]
    plv = np.abs(exp_phases @ exp_phases.conj().T) / eeg_data.shape[1]
    np.fill_diagonal(plv, 1.0)
    return plv.real.astype(np.float32)


def build_correlation_fast(eeg_data):
    """Full correlation matrix via np.corrcoef — single call, O(n^2 * T)."""
    return np.abs(np.corrcoef(eeg_data)).astype(np.float32)


def build_coherence_fast(eeg_data, fs=128, nperseg=256):
    """Coherence matrix.
    Precomputes the PSD (Welch) for each channel once (32 calls), then
    only calls scipy.signal.csd per pair (496 calls) — avoids ~992 redundant
    Welch computations that the original pairwise coherence approach incurred.
    """
    from scipy.signal import welch, csd
    n_ch = eeg_data.shape[0]
    psds = []
    for i in range(n_ch):
        _, Pxx = welch(eeg_data[i], fs=fs, nperseg=nperseg)
        psds.append(Pxx + 1e-12)          # avoid div-by-zero
    coh_mat = np.eye(n_ch, dtype=np.float32)
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            _, Pxy = csd(eeg_data[i], eeg_data[j], fs=fs, nperseg=nperseg)
            c = float(np.mean(np.abs(Pxy) ** 2 / (psds[i] * psds[j])))
            coh_mat[i, j] = c
            coh_mat[j, i] = c
    return coh_mat


def build_mi_fast(eeg_data, n_bins=50):
    """MI matrix using fast histogram-based estimator.
    Quantizes each channel into n_bins bins once, then computes the 2-D joint
    histogram per pair with np.add.at (O(n) per pair in pure NumPy).
    ~100x faster than the sklearn k-NN estimator for n=7680 samples.
    """
    n_ch, n_samples = eeg_data.shape

    # Quantize each channel to [0, n_bins) once
    quantized = np.empty((n_ch, n_samples), dtype=np.int32)
    for i in range(n_ch):
        sig = eeg_data[i]
        lo, hi = sig.min(), sig.max()
        if hi > lo:
            quantized[i] = ((sig - lo) / (hi - lo) * (n_bins - 1)).astype(np.int32)
        else:
            quantized[i] = 0

    # Marginal distributions (computed once per channel)
    marginals = np.array([
        np.bincount(quantized[i], minlength=n_bins) / n_samples
        for i in range(n_ch)
    ])   # shape [n_ch, n_bins]

    mi_mat = np.zeros((n_ch, n_ch), dtype=np.float32)

    for i in range(n_ch):
        px = marginals[i]
        for j in range(i + 1, n_ch):
            py = marginals[j]
            # Joint histogram via flat-index trick
            idx = quantized[i] * n_bins + quantized[j]
            joint_flat = np.bincount(idx, minlength=n_bins * n_bins) / n_samples
            joint = joint_flat.reshape(n_bins, n_bins)
            # I(X;Y) = sum p(x,y) log[ p(x,y) / (p(x)*p(y)) ]
            outer = np.outer(px, py)
            mask = (joint > 0) & (outer > 0)
            mi_val = float(np.sum(joint[mask] * np.log(joint[mask] / outer[mask])))
            mi_mat[i, j] = mi_val
            mi_mat[j, i] = mi_val

    # Normalize off-diagonal values to [0, 1]
    off = mi_mat.copy()
    np.fill_diagonal(off, 0.0)
    mx = off.max()
    if mx > 0:
        mi_mat = mi_mat / mx
    np.fill_diagonal(mi_mat, 1.0)
    return mi_mat


_FAST_GRAPH_FNS = {
    'plv':         build_plv_fast,
    'correlation': build_correlation_fast,
    'coherence':   lambda e: build_coherence_fast(e, fs=128),
    'mi':          build_mi_fast,
}

def build_all_graphs_fast(trial_eeg, graph_methods):
    """Build all adjacency matrices for one trial using vectorized routines."""
    adjs = []
    for method in graph_methods:
        if method in _FAST_GRAPH_FNS:
            adjs.append(torch.FloatTensor(_FAST_GRAPH_FNS[method](trial_eeg)))
        else:
            # Fallback to original pairwise implementation
            adjs.append(torch.FloatTensor(
                build_connectivity_graph(trial_eeg, method=method, fs=128)))
    return adjs



class GCNLayer(nn.Module):
    """GCN layer - matches production rashomon_emotion/model.py exactly"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        """
        Args:
            x: [batch, num_nodes, in_features]
            adj: [batch, num_nodes, num_nodes]
        Returns:
            [batch, num_nodes, out_features]
        """
        # Normalize adjacency: D^(-1/2) * A * D^(-1/2)  (matches production model)
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1)  # [batch, num_nodes, 1]
        degree_inv_sqrt = torch.pow(degree, -0.5)
        adj_normalized = adj * degree_inv_sqrt * degree_inv_sqrt.transpose(-1, -2)

        # Graph convolution: A_norm * X * W
        support = torch.matmul(adj_normalized, x)   # [batch, num_nodes, in_features]
        output = self.linear(support)                # [batch, num_nodes, out_features]
        return output


class MultiHeadGNN_FusionAblation(nn.Module):
    
    def __init__(self, num_nodes, node_features, hidden_dim, num_classes=2,
                 num_heads=4, dropout=0.5, fusion_type='learnable'):
        """
        Args:
            fusion_type: 'none' (final layer only), 'late' (averaging), or 'learnable' (proposed)
        """
        super().__init__()
        self.num_heads = num_heads
        self.num_nodes = num_nodes
        self.fusion_type = fusion_type
        
        # Multiple GCN heads (3 layers each)
        self.gcn_heads = nn.ModuleList([
            nn.ModuleList([
                GCNLayer(node_features if l == 0 else hidden_dim, hidden_dim)
                for l in range(3)  # 3 GCN layers
            ]) for _ in range(num_heads)
        ])
        
        # Learnable fusion weights (if using learnable fusion)
        if fusion_type == 'learnable':
            self.fusion_weights = nn.Parameter(torch.ones(3))  # 3 layers
        
        # Attention weights for head fusion (across graph types)
        self.attention = nn.Linear(hidden_dim, 1)
        
        # Classification head
        self.fc1 = nn.Linear(hidden_dim * num_nodes, hidden_dim * 2)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Multi-head outputs (valence, arousal, dominance)
        self.fc_valence = nn.Linear(hidden_dim * 2, num_classes)
        self.fc_arousal = nn.Linear(hidden_dim * 2, num_classes)
        self.fc_dominance = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x, adj_list):
        """
        Args:
            x: [batch, num_nodes, node_features]
            adj_list: List of [batch, num_nodes, num_nodes] adjacencies
        Returns:
            dict with 'valence', 'arousal', 'dominance' logits
        """
        batch_size = x.size(0)
        
        # Process each graph type (head)
        head_outputs = []
        for head_idx, gcn_layers in enumerate(self.gcn_heads):
            adj = adj_list[head_idx]
            
            # Forward through 3 GCN layers, collecting intermediate embeddings
            layer_embeddings = []
            h = x
            for layer in gcn_layers:
                h = F.relu(layer(h, adj))
                layer_embeddings.append(h)  # [batch, num_nodes, hidden_dim]
            
            # Apply fusion strategy
            if self.fusion_type == 'none':
                # Use only final layer
                h_fused = layer_embeddings[-1]
            
            elif self.fusion_type == 'late':
                # Fixed averaging across layers
                h_fused = torch.stack(layer_embeddings, dim=0).mean(dim=0)
            
            elif self.fusion_type == 'learnable':
               
                weights = F.softmax(self.fusion_weights, dim=0)
                h_stack = torch.stack(layer_embeddings, dim=0)  # [3, batch, nodes, hidden]
                h_fused = (h_stack * weights.view(3, 1, 1, 1)).sum(dim=0)
            
            else:
                raise ValueError(f"Unknown fusion type: {self.fusion_type}")
            
            head_outputs.append(h_fused)
        
        # Attention-based fusion across graph types
        head_stack = torch.stack(head_outputs, dim=1)  # [batch, num_heads, nodes, hidden]
        attn_scores = self.attention(head_stack)  # [batch, num_heads, nodes, 1]
        attn_weights = F.softmax(attn_scores, dim=1)
        fused = (head_stack * attn_weights).sum(dim=1)  # [batch, nodes, hidden]
        
        # Global pooling
        pooled = fused.view(batch_size, -1)  # [batch, nodes * hidden]
        
        # Shared feature transformation
        out = F.relu(self.fc1(pooled))
        out = self.dropout_layer(out)
        
        # Multi-head outputs
        return {
            'valence': self.fc_valence(out),
            'arousal': self.fc_arousal(out),
            'dominance': self.fc_dominance(out)
        }



def train_one_epoch(model, train_data, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for x, adj_list, labels in train_data:
        x = x.to(device)
        adj_list = [adj.to(device) for adj in adj_list]
        
        optimizer.zero_grad()
        outputs = model(x, adj_list)
        
        # Multi-task loss 
        loss = 0
        for i, target in enumerate(['valence', 'arousal', 'dominance']):
            if target in labels:
                y = labels[target].to(device)
                loss += F.cross_entropy(outputs[target], y)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_data)


def evaluate_model(model, test_data, device):
    """Evaluate model"""
    model.eval()
    
    all_preds = {'valence': [], 'arousal': [], 'dominance': []}
    all_labels = {'valence': [], 'arousal': [], 'dominance': []}
    
    with torch.no_grad():
        for x, adj_list, labels in test_data:
            x = x.to(device)
            adj_list = [adj.to(device) for adj in adj_list]
            
            outputs = model(x, adj_list)
            
            for target in ['valence', 'arousal', 'dominance']:
                if target in labels:
                    preds = outputs[target].argmax(dim=1).cpu().numpy()
                    y = labels[target].cpu().numpy()
                    all_preds[target].extend(preds)
                    all_labels[target].extend(y)
    
    # Calculate metrics
    results = {}
    for target in ['valence', 'arousal', 'dominance']:
        if all_preds[target]:
            preds = np.array(all_preds[target])
            labels = np.array(all_labels[target])
            accuracy = (preds == labels).mean()
            
            # F1 score
            tp = ((preds == 1) & (labels == 1)).sum()
            fp = ((preds == 1) & (labels == 0)).sum()
            fn = ((preds == 0) & (labels == 1)).sum()
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            
            results[target] = {'accuracy': accuracy, 'f1': f1}
        else:
            results[target] = {'accuracy': 0.0, 'f1': 0.0}
    
    avg_acc = np.mean([r['accuracy'] for r in results.values()])
    avg_f1 = np.mean([r['f1'] for r in results.values()])
    
    return avg_acc, avg_f1, results


def prepare_data_deap(subject_id, feature_method, graph_methods, data_dir='data/DEAP/data_preprocessed_matlab'):

    
    # Load DEAP subject
    mat_path = Path(data_dir) / f's{subject_id:02d}.mat'
    eeg_data, labels_raw = load_deap_data(str(mat_path))  # Returns tuple (eeg, labels)
    
    eeg_data = eeg_data[:, :32, :]  # [trials, 32 channels, samples]
    
    # Binarize labels (valence, arousal, dominance)
    valence_labels = (labels_raw[:, 0] > 5).astype(int)
    arousal_labels = (labels_raw[:, 1] > 5).astype(int)
    dominance_labels = (labels_raw[:, 2] > 5).astype(int)
    
    samples = []
    n_trials = len(eeg_data)

    for trial_idx in range(n_trials):
        if trial_idx % 10 == 0:
            print(f"    subject {subject_id:02d}: trial {trial_idx+1}/{n_trials}", flush=True)
        trial_eeg = eeg_data[trial_idx]  # [32 channels, samples]
        
        # Extract node features 
       
        node_features = []
        for ch in range(32):
            if feature_method == 'wavelet':
                feat = extract_wavelet_features(trial_eeg[ch])
            elif feature_method == 'lorentzian':
                feat = extract_lorentzian_bandpower(trial_eeg[ch], fs=128)
            elif feature_method == 'hjorth':
                feat = compute_hjorth_parameters(trial_eeg[ch])
            elif feature_method == 'dwt_bands':
                feat = extract_dwt_subbands(trial_eeg[ch])
            else:
                raise ValueError(f"Unknown feature method: {feature_method}")
            node_features.append(feat)
        
        x = torch.FloatTensor(np.array(node_features))  # [32, feat_dim]
        
        # Construct graphs using fast vectorized routines
        adj_matrices = build_all_graphs_fast(trial_eeg, graph_methods)
        
        labels = {
            'valence': torch.LongTensor([valence_labels[trial_idx]]),
            'arousal': torch.LongTensor([arousal_labels[trial_idx]]),
            'dominance': torch.LongTensor([dominance_labels[trial_idx]])
        }
        
        samples.append((x.unsqueeze(0), adj_matrices, labels))
    
    return samples


def run_loso_fold(subject_ids, test_subject_id, feature_method, graph_methods,
                 fusion_type, hidden_dim, epochs, device, seed=42):
    """
    Run one LOSO fold with specified fusion strategy.
    Returns: test accuracy, test F1, train time
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"  Loading data for {len(subject_ids)} subjects...")
    
    # Load all subjects
    all_data = []
    for subj_id in subject_ids:
        try:
            samples = prepare_data_deap(subj_id, feature_method, graph_methods)
            all_data.extend(samples)
        except Exception as e:
            print(f"    Warning: Failed to load subject {subj_id}: {e}")
            continue
    
    if len(all_data) == 0:
        raise ValueError("No data loaded!")

    # subject_ids already excludes test_subject_id, so all_data IS the training set
    train_data = all_data

    # Load test subject separately
    test_data = prepare_data_deap(test_subject_id, feature_method, graph_methods)

    print(f"  Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    
    # Get dimensions from first sample
    sample_x, sample_adj, _ = train_data[0]
    num_nodes = sample_x.shape[1]
    node_features = sample_x.shape[2]
    num_heads = len(graph_methods)
    
    # Create model with specified fusion type
    model = MultiHeadGNN_FusionAblation(
        num_nodes=num_nodes,
        node_features=node_features,
        hidden_dim=hidden_dim,
        num_classes=2,
        num_heads=num_heads,
        dropout=0.5,
        fusion_type=fusion_type
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    print(f"  Training {fusion_type} fusion model for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_data, optimizer, device)
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")
    
    # Evaluate
    test_acc, test_f1, detailed = evaluate_model(model, test_data, device)
    train_time = time.time() - start_time
    
    return test_acc, test_f1, train_time


def run_ablation_study(args):
   
    

    print(f"\nFeatures: {args.features}")
    print(f"Graphs: {args.graphs} ")
    print(f"Fusion types: {args.fusion_types}")
    print(f"Subjects: {args.subjects}")
    print(f"Test subject (LOSO): {args.test_subject}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}\n")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # All subjects except test subject
    train_subjects = [s for s in args.subjects if s != args.test_subject]
    
    # Results storage
    all_results = []
    
    total_runs = len(args.features) * len(args.fusion_types)
    current_run = 0
    
    for feature_method in args.features:
        for fusion_type in args.fusion_types:
            current_run += 1
            
            print(f"\n{'='*60}")
            print(f"Run {current_run}/{total_runs}")
            print(f"Feature: {feature_method}, Fusion: {fusion_type}")
            print(f"Test Subject: {args.test_subject} (LOSO)")
            print(f"{'='*60}")
            
            try:
                test_acc, test_f1, train_time = run_loso_fold(
                    subject_ids=train_subjects,
                    test_subject_id=args.test_subject,
                    feature_method=feature_method,
                    graph_methods=args.graphs,
                    fusion_type=fusion_type,
                    hidden_dim=args.hidden_dim,
                    epochs=args.epochs,
                    device=device,
                    seed=args.seed
                )
                
                print(f"\nCompleted in {train_time:.1f}s")
                print(f"  Test Accuracy: {test_acc:.4f}")
                print(f"  Test F1: {test_f1:.4f}")
                
                result = {
                    'feature': feature_method,
                    'fusion_type': fusion_type,
                    'test_subject': args.test_subject,
                    'test_accuracy': test_acc,
                    'test_f1': test_f1,
                    'train_time_s': train_time,
                    'hidden_dim': args.hidden_dim,
                    'epochs': args.epochs
                }
                
                all_results.append(result)
            
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save results

    
    df = pd.DataFrame(all_results)
    
    # Detailed results
    df.to_csv(out_dir / 'fusion_ablation_detailed.csv', index=False)
    print(f"Saved: {out_dir / 'fusion_ablation_detailed.csv'}")
    
    # Summary by fusion type
    if len(df) > 0:
        summary = df.groupby(['feature', 'fusion_type']).agg({
            'test_accuracy': ['mean', 'std'],
            'test_f1': ['mean', 'std'],
            'train_time_s': 'mean'
        }).round(4)
        
        summary.to_csv(out_dir / 'fusion_ablation_summary.csv')
        print(f"Saved: {out_dir / 'fusion_ablation_summary.csv'}")
        
        print("\nSUMMARY:")
        print(summary)
    
    # Save config
    with open(out_dir / 'fusion_ablation_config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"\nAblation study complete! Results in: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description='Fusion Layer Ablation ')
    
    parser.add_argument('--features', nargs='+', default=['wavelet', 'lorentzian'],
                       help='Feature methods (wavelet, lorentzian, hjorth, dwt_bands)')
    parser.add_argument('--graphs', nargs='+', default=['plv', 'coherence'],
                       help='Graph construction methods (plv, coherence, correlation, mi)')
    parser.add_argument('--fusion-types', nargs='+', default=['none', 'late', 'learnable'],
                       help='Fusion strategies: none (final layer), late (avg), learnable (proposed)')
    parser.add_argument('--subjects', type=int, nargs='+', default=[1, 5, 10, 15, 20],
                       help='All subject IDs for training')
    parser.add_argument('--test-subject', type=int, default=1,
                       help='Subject ID to use as test (LOSO)')
    parser.add_argument('--hidden-dim', type=int, default=64,
                       help='Hidden dimension')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--out-dir', type=str, default='ablation_fusion',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    run_ablation_study(args)



if __name__ == '__main__':
    main()
