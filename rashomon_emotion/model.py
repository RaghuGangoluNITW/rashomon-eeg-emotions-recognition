# PyTorch model architecture.
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGGNN(nn.Module):
    """Simple baseline GNN for compatibility"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EEGGNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class GCNLayer(nn.Module):
    """Graph Convolutional Network Layer"""
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        """
        Args:
            x: Node features [batch_size, num_nodes, in_features]
            adj: Adjacency matrix [batch_size, num_nodes, num_nodes]
        Returns:
            Updated node features [batch_size, num_nodes, out_features]
        """
        # Normalize adjacency matrix: D^(-1/2) * A * D^(-1/2)
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1)  # [batch, num_nodes, 1]
        degree_inv_sqrt = torch.pow(degree, -0.5)
        adj_normalized = adj * degree_inv_sqrt * degree_inv_sqrt.transpose(-1, -2)
        
        # Graph convolution: A_norm * X * W
        support = torch.matmul(adj_normalized, x)  # [batch, num_nodes, in_features]
        output = self.linear(support)  # [batch, num_nodes, out_features]
        return output


class MultiHeadGNN(nn.Module):
    """
    Multi-head Graph Neural Network with attention-based fusion.
    Implements the Rashomon-GNN architecture from the paper.
    """
    def __init__(self, num_nodes, node_features, hidden_dim, num_classes, 
                 num_heads=4, dropout=0.5):
        super(MultiHeadGNN, self).__init__()
        self.num_heads = num_heads
        self.num_nodes = num_nodes
        
        # Multiple GCN heads
        self.gcn_heads = nn.ModuleList([
            nn.Sequential(
                GCNLayer(node_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                GCNLayer(hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(num_heads)
        ])
        
        # Attention weights for head fusion
        self.attention = nn.Linear(hidden_dim, 1)
        
        # Global pooling and classification
        self.fc1 = nn.Linear(hidden_dim * num_nodes, hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x, adj_list):
        """
        Args:
            x: Node features [batch_size, num_nodes, node_features]
            adj_list: List of adjacency matrices (one per head) 
                     [batch_size, num_nodes, num_nodes] each
        Returns:
            Class logits [batch_size, num_classes]
        """
        batch_size = x.size(0)
        
        # Process each head
        head_outputs = []
        for i, gcn in enumerate(self.gcn_heads):
            adj = adj_list[i] if isinstance(adj_list, list) else adj_list
            h = x
            for layer in gcn:
                if isinstance(layer, GCNLayer):
                    h = layer(h, adj)
                else:
                    h = layer(h)
            head_outputs.append(h)  # [batch, num_nodes, hidden_dim]
        
        # Attention-based fusion
        head_stack = torch.stack(head_outputs, dim=1)  # [batch, num_heads, num_nodes, hidden]
        attn_scores = self.attention(head_stack)  # [batch, num_heads, num_nodes, 1]
        attn_weights = F.softmax(attn_scores, dim=1)  # Normalize over heads
        
        # Weighted sum of heads
        fused = (head_stack * attn_weights).sum(dim=1)  # [batch, num_nodes, hidden]
        
        # Global pooling (flatten)
        pooled = fused.view(batch_size, -1)  # [batch, num_nodes * hidden]
        
        # Classification
        out = F.relu(self.fc1(pooled))
        out = self.dropout(out)
        logits = self.fc2(out)
        
        return logits


class RashomonGNN(nn.Module):
    """
    Complete Rashomon-GNN implementation with multiple graph construction
    methods and feature extraction strategies.
    """
    def __init__(self, num_nodes, node_features, hidden_dim, num_classes,
                 graph_types=['plv', 'coherence', 'correlation', 'mi'],
                 dropout=0.5):
        super(RashomonGNN, self).__init__()
        self.graph_types = graph_types
        self.num_heads = len(graph_types)
        
        # Multi-head GNN (one head per graph type)
        self.gnn = MultiHeadGNN(
            num_nodes=num_nodes,
            node_features=node_features,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_heads=self.num_heads,
            dropout=dropout
        )
        
    def forward(self, x, adj_list):
        """
        Args:
            x: Node features [batch_size, num_nodes, node_features]
            adj_list: List of adjacency matrices (one per graph type)
        Returns:
            Class logits [batch_size, num_classes]
        """
        return self.gnn(x, adj_list)
