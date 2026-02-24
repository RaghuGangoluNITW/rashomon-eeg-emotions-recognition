# SHAP and other interpretability methods.
import numpy as np
import torch
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")


def explain_with_shap(model, data, background_data=None, max_samples=100):
    """
    Generate SHAP explanations for a PyTorch model.
    
    Args:
        model: PyTorch model
        data: Input data to explain (numpy array or torch tensor)
        background_data: Background dataset for SHAP (if None, uses data subsample)
        max_samples: Maximum samples for background data
    Returns:
        SHAP values array
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed")
    
    model.eval()
    
    # Convert to numpy if tensor
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    
    # Use subset as background if not provided
    if background_data is None:
        n_samples = min(max_samples, len(data))
        indices = np.random.choice(len(data), n_samples, replace=False)
        background_data = data[indices]
    
    # Create SHAP explainer (DeepExplainer for neural networks)
    explainer = shap.DeepExplainer(model, torch.tensor(background_data, dtype=torch.float32))
    
    # Compute SHAP values
    shap_values = explainer.shap_values(torch.tensor(data, dtype=torch.float32))
    
    return shap_values


def compute_node_importance(model, data, adj_list, node_names=None):
    """
    Compute node (electrode) importance using integrated gradients.
    
    Args:
        model: PyTorch GNN model
        data: Node features [batch_size, num_nodes, node_features]
        adj_list: List of adjacency matrices
        node_names: Optional list of node names (e.g., electrode names)
    Returns:
        Dictionary with node importance scores
    """
    model.eval()
    
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)
    
    data.requires_grad = True
    
    # Forward pass
    output = model(data, adj_list)
    
    # Get prediction class
    pred_class = output.argmax(dim=1)
    
    # Compute gradients
    model.zero_grad()
    output[0, pred_class[0]].backward()
    
    # Node importance = gradient magnitude for each node
    node_importance = data.grad.abs().mean(dim=-1).squeeze()  # [num_nodes]
    
    if node_names is not None:
        return {name: float(imp) for name, imp in zip(node_names, node_importance)}
    else:
        return node_importance.cpu().detach().numpy()


def compute_lobe_importance(node_importance, lobe_mapping):
    """
    Aggregate node importance to brain lobe level.
    
    Args:
        node_importance: Dict or array of node-level importance
        lobe_mapping: Dict mapping node indices/names to lobe names
    Returns:
        Dict of lobe-level importance scores
    """
    lobe_scores = {}
    
    for node, importance in enumerate(node_importance):
        lobe = lobe_mapping.get(node, 'unknown')
        if lobe not in lobe_scores:
            lobe_scores[lobe] = []
        lobe_scores[lobe].append(importance)
    
    # Average importance per lobe
    return {lobe: np.mean(scores) for lobe, scores in lobe_scores.items()}


def compute_pdi(predictions_list):
    """
    Compute Pairwise Dissimilarity Index (PDI) across multiple models.
    Measures diversity in the Rashomon set.
    
    Args:
        predictions_list: List of prediction arrays from different models
                         Each array: [n_samples, n_classes] (probabilities or logits)
    Returns:
        PDI value (higher = more diverse)
    """
    n_models = len(predictions_list)
    
    if n_models < 2:
        return 0.0
    
    # Convert to numpy and normalize to probabilities
    predictions = []
    for pred in predictions_list:
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().detach().numpy()
        # Softmax if not already probabilities
        if np.max(pred) > 1.0 or np.min(pred) < 0.0:
            pred = np.exp(pred) / np.exp(pred).sum(axis=1, keepdims=True)
        predictions.append(pred)
    
    predictions = np.array(predictions)  # [n_models, n_samples, n_classes]
    
    # Compute pairwise KL divergences
    total_divergence = 0.0
    n_pairs = 0
    
    for i in range(n_models):
        for j in range(i + 1, n_models):
            # KL divergence: sum over samples
            pred_i = predictions[i] + 1e-10  # Avoid log(0)
            pred_j = predictions[j] + 1e-10
            
            kl_ij = np.sum(pred_i * np.log(pred_i / pred_j))
            kl_ji = np.sum(pred_j * np.log(pred_j / pred_i))
            
            # Symmetric KL divergence
            total_divergence += (kl_ij + kl_ji) / 2
            n_pairs += 1
    
    # Average divergence across all pairs
    pdi = total_divergence / n_pairs if n_pairs > 0 else 0.0
    
    return pdi


def compute_rashomon_diversity(model_outputs, metric='pdi'):
    """
    Compute diversity metrics for Rashomon set.
    
    Args:
        model_outputs: List of model outputs (predictions, features, etc.)
        metric: 'pdi' (Pairwise Dissimilarity Index) or 'variance'
    Returns:
        Diversity score
    """
    if metric == 'pdi':
        return compute_pdi(model_outputs)
    elif metric == 'variance':
        # Prediction variance across models
        predictions = np.array([out.cpu().detach().numpy() if isinstance(out, torch.Tensor) 
                               else out for out in model_outputs])
        return np.mean(np.var(predictions, axis=0))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def generate_shap_summary(model, data, adj_list, save_path=None):
    """
    Generate comprehensive SHAP summary for model explainability.
    
    Args:
        model: PyTorch model
        data: Input data
        adj_list: Adjacency matrices
        save_path: Optional path to save summary plots
    Returns:
        Dict with SHAP values and summary statistics
    """
    if not SHAP_AVAILABLE:
        print("SHAP not available, skipping explainability analysis")
        return None
    
    model.eval()
    
    # Flatten data for SHAP (if needed)
    if len(data.shape) == 3:  # [batch, nodes, features]
        batch_size, num_nodes, num_features = data.shape
        data_flat = data.reshape(batch_size, -1)
    else:
        data_flat = data
    
    # Wrapper for model that handles graph structure
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model, adj_list):
            super().__init__()
            self.model = model
            self.adj_list = adj_list
            
        def forward(self, x):
            # Reshape if needed
            if len(x.shape) == 2:
                batch_size = x.shape[0]
                x = x.reshape(batch_size, num_nodes, num_features)
            return self.model(x, self.adj_list)
    
    wrapped_model = ModelWrapper(model, adj_list)
    
    try:
        # Compute SHAP values
        shap_values = explain_with_shap(wrapped_model, data_flat)
        
        summary = {
            'shap_values': shap_values,
            'feature_importance': np.mean(np.abs(shap_values), axis=0),
            'top_features': np.argsort(np.mean(np.abs(shap_values), axis=0))[::-1][:10]
        }
        
        # Save plots if requested
        if save_path:
            import matplotlib.pyplot as plt
            shap.summary_plot(shap_values, data_flat, show=False)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        
        return summary
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        return None
