"""
I/O Utilities Module

This module provides functions for saving and loading models and data.
"""

import os
import json
import pickle
import numpy as np
import torch
import anndata as ad
import scanpy as sc
from typing import Dict, Any, Optional, Union
from datetime import datetime


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    loss: float,
    path: str,
    additional_info: Optional[Dict] = None
):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optional optimizer
        epoch: Current epoch
        loss: Current loss
        path: Path to save checkpoint
        additional_info: Additional information to save
    """
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if additional_info is not None:
        checkpoint['additional_info'] = additional_info
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        path: Path to checkpoint
        model: Optional model to load weights into
        optimizer: Optional optimizer to load state into
        device: Device to load tensors to
        
    Returns:
        checkpoint: Loaded checkpoint dictionary
    """
    checkpoint = torch.load(path, map_location=device)
    
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {path} (epoch {checkpoint['epoch']})")
    
    return checkpoint


def save_config(config: Dict, path: str):
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        path: Path to save
    """
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    
    with open(path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    print(f"Config saved to {path}")


def load_config(path: str) -> Dict:
    """
    Load configuration from JSON file.
    
    Args:
        path: Path to config file
        
    Returns:
        config: Configuration dictionary
    """
    with open(path, 'r') as f:
        config = json.load(f)
    
    return config


def save_results_h5ad(
    original_adata: ad.AnnData,
    reconstructed_expression: np.ndarray,
    output_path: str,
    metrics: Optional[Dict] = None,
    latent_features: Optional[np.ndarray] = None
):
    """
    Save super-resolution results to h5ad file.
    
    Args:
        original_adata: Original AnnData object (for reference)
        reconstructed_expression: Reconstructed gene expression matrix
        output_path: Path to save output h5ad
        metrics: Optional evaluation metrics
        latent_features: Optional latent representations
    """
    # Create new AnnData with reconstructed expression
    result_adata = ad.AnnData(X=reconstructed_expression)
    
    # Copy gene names if available
    if original_adata.var_names is not None:
        result_adata.var_names = original_adata.var_names
    
    # Copy observation names if dimensions match
    if reconstructed_expression.shape[0] == original_adata.n_obs:
        result_adata.obs_names = original_adata.obs_names
        
        # Copy spatial coordinates
        if 'spatial' in original_adata.obsm:
            result_adata.obsm['spatial'] = original_adata.obsm['spatial'].copy()
        
        # Copy other obs columns
        for col in original_adata.obs.columns:
            result_adata.obs[col] = original_adata.obs[col].values
    
    # Add reconstructed flag
    result_adata.uns['srast_reconstructed'] = True
    result_adata.uns['reconstruction_timestamp'] = datetime.now().isoformat()
    
    # Add metrics if provided
    if metrics is not None:
        result_adata.uns['srast_metrics'] = metrics
    
    # Add latent features if provided
    if latent_features is not None:
        result_adata.obsm['X_latent'] = latent_features
    
    # Save original expression for comparison
    result_adata.layers['original'] = original_adata.X.copy() if hasattr(original_adata, 'X') else None
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    
    # Save
    result_adata.write_h5ad(output_path)
    print(f"Results saved to {output_path}")
    
    return result_adata


def save_training_history(
    history: Dict[str, list],
    path: str
):
    """
    Save training history.
    
    Args:
        history: Dictionary of training metrics over epochs
        path: Path to save
    """
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Training history saved to {path}")


def load_training_history(path: str) -> Dict[str, list]:
    """
    Load training history.
    
    Args:
        path: Path to history file
        
    Returns:
        history: Training history dictionary
    """
    with open(path, 'r') as f:
        history = json.load(f)
    
    return history


def create_output_directory(base_dir: str, experiment_name: Optional[str] = None) -> str:
    """
    Create output directory with timestamp.
    
    Args:
        base_dir: Base output directory
        experiment_name: Optional experiment name
        
    Returns:
        output_dir: Created output directory path
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_name:
        output_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    else:
        output_dir = os.path.join(base_dir, timestamp)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
    
    print(f"Output directory created: {output_dir}")
    
    return output_dir


def save_numpy_arrays(arrays: Dict[str, np.ndarray], path: str):
    """
    Save multiple numpy arrays to a single file.
    
    Args:
        arrays: Dictionary of arrays
        path: Path to save
    """
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    np.savez(path, **arrays)
    print(f"Arrays saved to {path}")


def load_numpy_arrays(path: str) -> Dict[str, np.ndarray]:
    """
    Load numpy arrays from file.
    
    Args:
        path: Path to npz file
        
    Returns:
        arrays: Dictionary of loaded arrays
    """
    data = np.load(path)
    arrays = {key: data[key] for key in data.files}
    return arrays
