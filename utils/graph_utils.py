"""
Graph Utilities Module

This module provides functions for constructing spatial and feature neighbor graphs.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, to_undirected
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from typing import Tuple, Optional, Dict, Union
import warnings


def build_spatial_graph(
    coords: np.ndarray,
    k: int = 6,
    include_self_loops: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build spatial neighbor graph based on spatial coordinates.
    
    Args:
        coords: Spatial coordinates (N x 2 or N x 3)
        k: Number of nearest neighbors
        include_self_loops: Whether to include self-loops
        
    Returns:
        edge_index: Edge index tensor (2 x E)
        edge_weight: Edge weight tensor (E,) based on distance
    """
    n_samples = coords.shape[0]
    k = min(k, n_samples - 1)  # Ensure k is valid
    
    if k < 1:
        # Return empty graph for single node
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros(0)
        return edge_index, edge_weight
    
    # Find k nearest neighbors
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree')
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)
    
    # Build edge list (exclude self-connections from kNN results)
    source_nodes = []
    target_nodes = []
    weights = []
    
    for i in range(n_samples):
        for j_idx in range(1, k + 1):  # Start from 1 to skip self
            j = indices[i, j_idx]
            source_nodes.append(i)
            target_nodes.append(j)
            # Use inverse distance as weight
            dist = distances[i, j_idx]
            weight = 1.0 / (dist + 1e-6)
            weights.append(weight)
    
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float)
    
    # Make undirected
    edge_index, edge_weight = to_undirected(edge_index, edge_weight)
    
    # Add self-loops if requested
    if include_self_loops:
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=n_samples)
    
    return edge_index, edge_weight


def build_feature_graph(
    features: np.ndarray,
    k: int = 20,
    include_self_loops: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build feature neighbor graph based on feature similarity.
    
    Args:
        features: Feature matrix (N x D)
        k: Number of nearest neighbors
        include_self_loops: Whether to include self-loops
        
    Returns:
        edge_index: Edge index tensor (2 x E)
        edge_weight: Edge weight tensor (E,) based on cosine similarity
    """
    n_samples = features.shape[0]
    k = min(k, n_samples - 1)
    
    if k < 1:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros(0)
        return edge_index, edge_weight
    
    # Normalize features for cosine similarity
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    
    # Find k nearest neighbors in feature space
    nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine')
    nn.fit(features_norm)
    distances, indices = nn.kneighbors(features_norm)
    
    # Build edge list
    source_nodes = []
    target_nodes = []
    weights = []
    
    for i in range(n_samples):
        for j_idx in range(1, k + 1):
            j = indices[i, j_idx]
            source_nodes.append(i)
            target_nodes.append(j)
            # Convert cosine distance to similarity
            similarity = 1.0 - distances[i, j_idx]
            weights.append(max(similarity, 0.0))
    
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float)
    
    # Make undirected
    edge_index, edge_weight = to_undirected(edge_index, edge_weight)
    
    if include_self_loops:
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=n_samples)
    
    return edge_index, edge_weight


def build_heterogeneous_graph(
    coords: np.ndarray,
    features: np.ndarray,
    spatial_k: int = 6,
    feature_k: int = 20,
    include_self_loops: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build spatial-feature heterogeneous graph using union of spatial and feature graphs.
    
    Args:
        coords: Spatial coordinates (N x 2)
        features: Feature matrix (N x D)
        spatial_k: K for spatial neighbor graph
        feature_k: K for feature neighbor graph
        include_self_loops: Whether to include self-loops
        
    Returns:
        edge_index: Combined edge index tensor (2 x E)
        edge_weight: Combined edge weight tensor (E,)
    """
    # Build spatial graph
    spatial_edge_index, spatial_edge_weight = build_spatial_graph(
        coords, k=spatial_k, include_self_loops=False
    )
    
    # Build feature graph
    feature_edge_index, feature_edge_weight = build_feature_graph(
        features, k=feature_k, include_self_loops=False
    )
    
    # Combine edges (union)
    combined_edge_index = torch.cat([spatial_edge_index, feature_edge_index], dim=1)
    combined_edge_weight = torch.cat([spatial_edge_weight, feature_edge_weight], dim=0)
    
    # Remove duplicate edges (keep edge with maximum weight)
    n_nodes = coords.shape[0]
    edge_dict = {}
    
    for i in range(combined_edge_index.shape[1]):
        src, dst = combined_edge_index[0, i].item(), combined_edge_index[1, i].item()
        weight = combined_edge_weight[i].item()
        edge_key = (min(src, dst), max(src, dst))
        
        if edge_key in edge_dict:
            edge_dict[edge_key] = max(edge_dict[edge_key], weight)
        else:
            edge_dict[edge_key] = weight
    
    # Rebuild edge index and weights
    source_nodes = []
    target_nodes = []
    weights = []
    
    for (src, dst), weight in edge_dict.items():
        # Add both directions for undirected graph
        source_nodes.extend([src, dst])
        target_nodes.extend([dst, src])
        weights.extend([weight, weight])
    
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float)
    
    # Add self-loops
    if include_self_loops:
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=n_nodes)
    
    return edge_index, edge_weight


def build_lr_hr_bipartite_graph(
    hr_coords: np.ndarray,
    lr_to_hr_mapping: Dict[int, list],
    k_neighbors: int = 6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build bipartite graph between LR and HR nodes for diffusion.
    
    Each LR node is connected to its corresponding HR nodes based on
    the downsampling mapping, plus additional spatial neighbors.
    
    Args:
        hr_coords: HR spatial coordinates (N_hr x 2)
        lr_to_hr_mapping: Mapping from LR index to list of HR indices
        k_neighbors: Additional spatial neighbors to connect
        
    Returns:
        hr_edge_index: Edge index for HR spatial graph
        hr_edge_weight: Edge weights for HR spatial graph
        lr_hr_mapping_tensor: Tensor of (lr_idx, hr_idx) pairs
    """
    n_hr = hr_coords.shape[0]
    n_lr = len(lr_to_hr_mapping)
    
    # Build HR spatial graph
    hr_edge_index, hr_edge_weight = build_spatial_graph(
        hr_coords, k=k_neighbors, include_self_loops=True
    )
    
    # Build LR-HR mapping tensor
    lr_indices = []
    hr_indices = []
    
    for lr_idx, hr_idx_list in lr_to_hr_mapping.items():
        for hr_idx in hr_idx_list:
            lr_indices.append(lr_idx)
            hr_indices.append(hr_idx)
    
    lr_hr_mapping_tensor = torch.tensor([lr_indices, hr_indices], dtype=torch.long)
    
    return hr_edge_index, hr_edge_weight, lr_hr_mapping_tensor


def create_pyg_data(
    features: np.ndarray,
    coords: np.ndarray,
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor] = None,
    labels: Optional[np.ndarray] = None
) -> Data:
    """
    Create PyTorch Geometric Data object.
    
    Args:
        features: Node features (N x D)
        coords: Spatial coordinates (N x 2)
        edge_index: Edge index tensor
        edge_weight: Optional edge weights
        labels: Optional node labels
        
    Returns:
        data: PyG Data object
    """
    data = Data(
        x=torch.tensor(features, dtype=torch.float),
        pos=torch.tensor(coords, dtype=torch.float),
        edge_index=edge_index
    )
    
    if edge_weight is not None:
        data.edge_attr = edge_weight.unsqueeze(-1)
    
    if labels is not None:
        data.y = torch.tensor(labels, dtype=torch.float)
    
    return data


def compute_graph_statistics(edge_index: torch.Tensor, n_nodes: int) -> Dict:
    """
    Compute graph statistics.
    
    Args:
        edge_index: Edge index tensor
        n_nodes: Number of nodes
        
    Returns:
        stats: Dictionary of graph statistics
    """
    n_edges = edge_index.shape[1] // 2  # Divide by 2 for undirected
    
    # Compute degree
    degree = torch.zeros(n_nodes)
    for i in range(edge_index.shape[1]):
        degree[edge_index[0, i]] += 1
    
    stats = {
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'avg_degree': degree.mean().item(),
        'max_degree': degree.max().item(),
        'min_degree': degree.min().item(),
        'density': n_edges / (n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else 0
    }
    
    return stats
