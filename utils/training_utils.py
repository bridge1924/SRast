"""
Training Utilities Module

Provides common utility functions for training pipelines.
Consolidates duplicated code from train_stage1_single.py, train_stage2.py, etc.
"""

import gc
import numpy as np
import torch
from typing import Dict, Optional
from collections import defaultdict


def clear_memory():
    """Clear GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str = 'auto') -> str:
    """
    Get the appropriate device string.
    
    Args:
        device_str: Device specification ('auto', 'cuda', 'cpu', 'cuda:0', etc.)
        
    Returns:
        Resolved device string
    """
    if device_str == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_str


def estimate_memory_usage(
    n_hr: int, 
    n_lr: int, 
    n_genes: int, 
    latent_dim: int
) -> Dict[str, float]:
    """
    Estimate memory usage for a sample in GB.
    
    Args:
        n_hr: Number of HR spots
        n_lr: Number of LR spots  
        n_genes: Number of genes (HVG)
        latent_dim: Latent dimension
        
    Returns:
        Dict with memory estimates in GB
    """
    bytes_per_float32 = 4
    
    # x_hr: (n_hr, n_genes) float32
    x_hr_bytes = n_hr * n_genes * bytes_per_float32
    
    # x_lr: (n_lr, n_genes) float32
    x_lr_bytes = n_lr * n_genes * bytes_per_float32
    
    # z_lr: (n_lr, latent_dim) float32
    z_lr_bytes = n_lr * latent_dim * bytes_per_float32
    
    # hr_coords, lr_coords: (n, 2) float32
    coords_bytes = (n_hr + n_lr) * 2 * bytes_per_float32
    
    # indices: int64
    indices_bytes = n_hr * 3 * 8  # group_indices, local_hr_indices, sample_indices
    
    total_bytes = x_hr_bytes + x_lr_bytes + z_lr_bytes + coords_bytes + indices_bytes
    
    return {
        'x_hr_gb': x_hr_bytes / (1024**3),
        'x_lr_gb': x_lr_bytes / (1024**3),
        'z_lr_gb': z_lr_bytes / (1024**3),
        'coords_gb': coords_bytes / (1024**3),
        'indices_gb': indices_bytes / (1024**3),
        'total_gb': total_bytes / (1024**3)
    }


def spatial_uniform_subsample(
    coords: np.ndarray,
    n_samples: int,
    random_state: int = 42
) -> np.ndarray:
    """
    Perform spatially uniform subsampling using grid-based method.
    
    This ensures the subsampled points are spatially distributed uniformly,
    rather than being biased towards dense regions. Uses a fast grid-based
    approach instead of slow K-Means clustering.
    
    Args:
        coords: Spot coordinates (n, 2)
        n_samples: Number of samples to select
        random_state: Random seed
        
    Returns:
        indices: Selected indices
    """
    n_total = coords.shape[0]
    if n_samples >= n_total:
        return np.arange(n_total)
    
    np.random.seed(random_state)
    
    # Method: Grid-based spatial sampling (much faster than K-Means)
    # 1. Divide space into grid cells
    # 2. Sample proportionally from each cell
    
    # Normalize coordinates to [0, 1]
    coords_min = coords.min(axis=0)
    coords_max = coords.max(axis=0)
    coords_range = coords_max - coords_min
    coords_range[coords_range == 0] = 1  # Avoid division by zero
    normalized_coords = (coords - coords_min) / coords_range
    
    # Determine grid size to get roughly n_samples cells
    # Use sqrt(n_samples) as grid dimension for 2D
    grid_dim = max(10, int(np.sqrt(n_samples / 2)))
    
    # Assign each point to a grid cell
    cell_x = np.clip((normalized_coords[:, 0] * grid_dim).astype(int), 0, grid_dim - 1)
    cell_y = np.clip((normalized_coords[:, 1] * grid_dim).astype(int), 0, grid_dim - 1)
    cell_ids = cell_x * grid_dim + cell_y
    
    # Group points by cell
    cell_to_points = defaultdict(list)
    for idx, cell_id in enumerate(cell_ids):
        cell_to_points[cell_id].append(idx)
    
    # Calculate how many points to sample from each non-empty cell
    n_cells = len(cell_to_points)
    points_per_cell = max(1, n_samples // n_cells)
    
    selected_indices = []
    cells = list(cell_to_points.keys())
    np.random.shuffle(cells)
    
    # First pass: sample points_per_cell from each cell
    for cell_id in cells:
        points = cell_to_points[cell_id]
        n_to_sample = min(points_per_cell, len(points))
        sampled = np.random.choice(points, size=n_to_sample, replace=False)
        selected_indices.extend(sampled.tolist())
        
        if len(selected_indices) >= n_samples:
            break
    
    # If we still need more, randomly sample from remaining points
    if len(selected_indices) < n_samples:
        selected_set = set(selected_indices)
        remaining = [i for i in range(n_total) if i not in selected_set]
        additional_needed = n_samples - len(selected_indices)
        if remaining and additional_needed > 0:
            additional = np.random.choice(
                remaining, 
                size=min(additional_needed, len(remaining)),
                replace=False
            )
            selected_indices.extend(additional.tolist())
    
    # Return sorted indices (truncate to exactly n_samples)
    return np.array(sorted(selected_indices[:n_samples]))


class EarlyStopping:
    """
    Early stopping handler for training loops.
    
    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        verbose: Whether to print messages
        mode: 'min' for loss, 'max' for metrics like PCC
    """
    
    def __init__(
        self, 
        patience: int = 50, 
        min_delta: float = 0.0001,
        verbose: bool = True,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.mode = mode
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.epochs_without_improvement = 0
        self.should_stop = False
        
    def step(self, value: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            value: Current metric value
            
        Returns:
            True if training should stop
        """
        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = value
            self.epochs_without_improvement = 0
            if self.verbose:
                print(f"  [EarlyStop] New best value: {self.best_value:.6f}")
        else:
            self.epochs_without_improvement += 1
            if self.verbose:
                print(f"  [EarlyStop] No improvement for {self.epochs_without_improvement}/{self.patience} epochs (best: {self.best_value:.6f})")
        
        if self.epochs_without_improvement >= self.patience:
            self.should_stop = True
            if self.verbose:
                print(f"  [EarlyStop] Stopping: no improvement for {self.patience} epochs")
        
        return self.should_stop
    
    def reset(self):
        """Reset the early stopping state."""
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.epochs_without_improvement = 0
        self.should_stop = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking metrics during training.
    """
    
    def __init__(self, name: str = ''):
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history = []
    
    def update(self, val: float, n: int = 1):
        """
        Update statistics with a new value.
        
        Args:
            val: New value
            n: Number of samples (for weighted average)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def record(self):
        """Record current average to history."""
        self.history.append(self.avg)
    
    def __str__(self):
        return f"{self.name}: {self.val:.4f} (avg: {self.avg:.4f})"
