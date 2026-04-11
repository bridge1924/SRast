"""
Data Loading Utilities Module

Provides unified data loading functions for training and inference.
Consolidates duplicated code from train_stage2.py, inference.py, and recalculate_metrics.py.
"""

import os
import gc
import json
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import issparse

from data import load_h5ad
from data.preprocessing import DataPreprocessor


def build_lr_hr_mapping(
    lr_coords: np.ndarray,
    hr_coords: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Build LR-HR mapping where each HR spot is assigned to its nearest LR spot.
    
    Args:
        lr_coords: LR spot coordinates (n_lr, 2)
        hr_coords: HR spot coordinates (n_hr, 2)
        
    Returns:
        lr_hr_mapping: (2, N_HR) tensor with LR indices and HR indices
        group_indices: (N_HR,) tensor with LR parent index for each HR spot
        local_hr_indices: (N_HR,) tensor with local index within each LR group
        lr_hr_mapping_np: numpy array for metric computation
    """
    n_hr = hr_coords.shape[0]
    
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(lr_coords)
    distances, indices = nbrs.kneighbors(hr_coords)
    
    parent_lr = indices.flatten()
    group_indices = torch.tensor(parent_lr, dtype=torch.long)
    
    local_hr_indices = torch.zeros(n_hr, dtype=torch.long)
    group_counts = {}
    for hr_idx in range(n_hr):
        lr_idx = parent_lr[hr_idx]
        if lr_idx not in group_counts:
            group_counts[lr_idx] = 0
        local_hr_indices[hr_idx] = group_counts[lr_idx]
        group_counts[lr_idx] += 1
    
    lr_indices = torch.tensor(parent_lr, dtype=torch.long)
    hr_indices = torch.arange(n_hr, dtype=torch.long)
    lr_hr_mapping = torch.stack([lr_indices, hr_indices])
    
    # Also return numpy version for metric computation
    lr_hr_mapping_np = np.stack([parent_lr, np.arange(n_hr)])
    
    return lr_hr_mapping, group_indices, local_hr_indices, lr_hr_mapping_np


def load_stage1_latent_norm(sample_dir: str, latent_dim: int):
    """
    Load trained LatentNorm from Stage 1 decoder_weights.pt.
    
    Args:
        sample_dir: Stage 1 sample directory
        latent_dim: Latent dimension
        
    Returns:
        LatentNorm module with loaded parameters, or None if not available
    """
    from models.graph_vae import LatentNorm
    
    decoder_path = os.path.join(sample_dir, 'decoder_weights.pt')
    if not os.path.exists(decoder_path):
        return None
    
    try:
        decoder_weights = torch.load(decoder_path, map_location='cpu')
        
        if not decoder_weights.get('use_latent_norm', False):
            return None
        
        latent_norm_state = decoder_weights.get('latent_norm_state_dict')
        if latent_norm_state is None:
            return None
        
        latent_norm = LatentNorm(latent_dim=latent_dim, affine=True)
        latent_norm.load_state_dict(latent_norm_state)
        latent_norm.eval()
        
        return latent_norm
        
    except Exception as e:
        print(f"  [WARNING] Failed to load LatentNorm from {decoder_path}: {e}")
        return None


def apply_latent_norm(z_lr: torch.Tensor, latent_norm) -> torch.Tensor:
    """
    Apply LatentNorm normalization to z_lr.
    
    Args:
        z_lr: Original encoder output (N, latent_dim)
        latent_norm: LatentNorm module (can be None)
        
    Returns:
        Normalized z_lr (or original if latent_norm is None)
    """
    if latent_norm is None:
        return z_lr
    
    with torch.no_grad():
        return latent_norm.normalize_external(z_lr)


def load_test_sample(
    sample_id: str,
    stage1_dir: str,
    data_config: Dict,
    device: str,
    include_raw: bool = True
) -> Optional[Dict]:
    """
    Load a test sample for inference.
    
    Args:
        sample_id: Sample identifier
        stage1_dir: Stage 1 output directory
        data_config: Data configuration dictionary
        device: Device string
        include_raw: Whether to include raw expression data
        
    Returns:
        Sample data dictionary or None if failed
    """
    # Get sample paths from data config
    datasets = data_config.get('datasets', {})
    if sample_id not in datasets:
        print(f"  [ERROR] Sample {sample_id} not found in data config")
        return None
    
    sample_config = datasets[sample_id]
    lr_path = sample_config.get('lr_path')
    hr_path = sample_config.get('hr_path')
    
    if not lr_path or not hr_path:
        print(f"  [ERROR] {sample_id}: Missing lr_path or hr_path")
        return None
    
    if not os.path.exists(hr_path):
        print(f"  [ERROR] {sample_id}: HR data not found at {hr_path}")
        return None
    
    # Load Stage 1 outputs
    sample_dir = os.path.join(stage1_dir, sample_id)
    
    required_files = ['latent_representations.npz', 'preprocessor.pkl']
    for f in required_files:
        if not os.path.exists(os.path.join(sample_dir, f)):
            print(f"  [ERROR] {sample_id}: Missing Stage 1 file {f}")
            return None
    
    try:
        # Load latent representations
        latent_data = np.load(os.path.join(sample_dir, 'latent_representations.npz'))
        z_lr = torch.tensor(latent_data['z_lr'], dtype=torch.float32)
        lr_coords = latent_data['lr_coords']
        lr_hvg_expression = latent_data['lr_hvg_expression']
        
        # Load preprocessor
        preprocessor = DataPreprocessor.load(os.path.join(sample_dir, 'preprocessor.pkl'))
        
        # Load HR data
        hr_adata = load_h5ad(hr_path)
        hr_coords = hr_adata.obsm['spatial'].copy()
        hr_hvg_expression = preprocessor.get_hvg_expression(hr_adata)
        
        result = {
            'sample_id': sample_id,
            'z_lr': z_lr,
            'lr_coords': lr_coords,
            'preprocessor': preprocessor,
            'hr_adata': hr_adata,
            'x_lr_np': lr_hvg_expression,
        }
        
        # Include raw expression if requested
        if include_raw:
            hr_X_raw = hr_adata.X
            if issparse(hr_X_raw):
                hr_X_raw = hr_X_raw.toarray()
            hr_raw_hvg = hr_X_raw[:, preprocessor.hvg_indices_]
            
            lr_adata = load_h5ad(lr_path)
            lr_X_raw = lr_adata.X
            if issparse(lr_X_raw):
                lr_X_raw = lr_X_raw.toarray()
            lr_raw_hvg = lr_X_raw[:, preprocessor.hvg_indices_]
            
            result['hr_raw_hvg'] = hr_raw_hvg
            result['lr_raw_hvg'] = lr_raw_hvg
        
        # Build mappings
        lr_hr_mapping, group_indices, local_hr_indices, lr_hr_mapping_np = build_lr_hr_mapping(
            lr_coords, hr_coords
        )
        
        # Convert to tensors
        x_hr = torch.tensor(hr_hvg_expression, dtype=torch.float32)
        x_lr = torch.tensor(lr_hvg_expression, dtype=torch.float32)
        lr_coords_t = torch.tensor(lr_coords, dtype=torch.float32)
        hr_coords_t = torch.tensor(hr_coords, dtype=torch.float32)
        
        result.update({
            'x_hr': x_hr,
            'x_lr': x_lr,
            'lr_coords': lr_coords_t,
            'hr_coords': hr_coords_t,
            'lr_hr_mapping': lr_hr_mapping,
            'lr_hr_mapping_np': lr_hr_mapping_np,
            'group_indices': group_indices,
            'local_hr_indices': local_hr_indices,
            'n_lr': z_lr.shape[0],
            'n_hr': x_hr.shape[0],
            'n_genes': x_hr.shape[1],
        })
        
        return result
        
    except Exception as e:
        print(f"  [ERROR] {sample_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def load_training_sample(
    sample_id: str,
    stage1_dir: str,
    hr_path: str,
    preprocessor_or_config: Union[DataPreprocessor, Dict],
    max_hr_spots: Optional[int] = None,
    spatial_uniform_sampling: bool = True,
    use_latent_norm: bool = True
) -> Optional[Dict]:
    """
    Load a training sample for Stage 2 training.
    
    Args:
        sample_id: Sample identifier
        stage1_dir: Stage 1 output directory
        hr_path: Path to HR h5ad file
        preprocessor_or_config: Either a preprocessor or config dict
        max_hr_spots: Maximum HR spots (None = no limit)
        spatial_uniform_sampling: Use spatial uniform sampling if subsampling
        use_latent_norm: Whether to apply LatentNorm to z_lr
        
    Returns:
        Sample data dictionary or None if failed
    """
    from .training_utils import spatial_uniform_subsample
    
    sample_dir = os.path.join(stage1_dir, sample_id)
    
    required_files = ['latent_representations.npz', 'preprocessor.pkl', 'config.json']
    for f in required_files:
        if not os.path.exists(os.path.join(sample_dir, f)):
            print(f"  [SKIP] {sample_id}: Missing {f}")
            return None
    
    try:
        # Load latent representations
        latent_data = np.load(os.path.join(sample_dir, 'latent_representations.npz'))
        z_lr = torch.tensor(latent_data['z_lr'], dtype=torch.float32)
        lr_coords = latent_data['lr_coords']
        lr_hvg_expression = latent_data['lr_hvg_expression']
        
        # Load preprocessor
        preprocessor = DataPreprocessor.load(os.path.join(sample_dir, 'preprocessor.pkl'))
        
        # Load sample info
        with open(os.path.join(sample_dir, 'sample_info.json'), 'r') as f:
            sample_info = json.load(f)
        
        n_hvg = sample_info['n_hvg']
        latent_dim = sample_info['latent_dim']
        
        # Apply LatentNorm if requested
        if use_latent_norm:
            latent_norm = load_stage1_latent_norm(sample_dir, latent_dim)
            if latent_norm is not None:
                z_lr = apply_latent_norm(z_lr, latent_norm)
                print(f"  [LatentNorm] {sample_id}: Applied LatentNorm to z_lr")
            else:
                print(f"  [WARNING] {sample_id}: LatentNorm requested but not found in Stage1")
        
        if not os.path.exists(hr_path):
            print(f"  [SKIP] {sample_id}: HR data not found at {hr_path}")
            return None
        
        # Load HR data
        hr_adata = load_h5ad(hr_path)
        hr_coords_full = hr_adata.obsm['spatial'].copy()
        n_hr_original = hr_adata.n_obs
        
        # Check if subsampling is needed
        subsample_indices = None
        if max_hr_spots is not None and max_hr_spots > 0 and n_hr_original > max_hr_spots:
            print(f"  [SUBSAMPLE] {sample_id}: {n_hr_original:,} HR spots > max {max_hr_spots:,}")
            
            if spatial_uniform_sampling:
                print(f"    Using spatial-uniform subsampling...")
                subsample_indices = spatial_uniform_subsample(
                    hr_coords_full, max_hr_spots, random_state=42
                )
            else:
                print(f"    Using random subsampling...")
                np.random.seed(42)
                subsample_indices = np.random.choice(
                    n_hr_original, size=max_hr_spots, replace=False
                )
                subsample_indices = np.sort(subsample_indices)
            
            hr_coords = hr_coords_full[subsample_indices]
            print(f"    Subsampled to {len(subsample_indices):,} HR spots")
        else:
            hr_coords = hr_coords_full
        
        # Get HVG expression
        if subsample_indices is not None:
            hr_adata_sub = hr_adata[subsample_indices].copy()
            hr_hvg_expression = preprocessor.get_hvg_expression(hr_adata_sub)
            del hr_adata_sub
        else:
            hr_hvg_expression = preprocessor.get_hvg_expression(hr_adata)
        
        del hr_adata
        gc.collect()
        
        # Build mapping
        lr_hr_mapping, group_indices, local_hr_indices, _ = build_lr_hr_mapping(
            lr_coords, hr_coords
        )
        
        # Convert to tensors
        x_hr = torch.tensor(np.asarray(hr_hvg_expression, dtype=np.float32), dtype=torch.float32)
        x_lr = torch.tensor(np.asarray(lr_hvg_expression, dtype=np.float32), dtype=torch.float32)
        lr_coords_t = torch.tensor(np.asarray(lr_coords, dtype=np.float32), dtype=torch.float32)
        hr_coords_t = torch.tensor(np.asarray(hr_coords, dtype=np.float32), dtype=torch.float32)
        
        return {
            'sample_id': sample_id,
            'x_hr': x_hr,
            'x_lr': x_lr,
            'z_lr': z_lr,
            'lr_coords': lr_coords_t,
            'hr_coords': hr_coords_t,
            'lr_hr_mapping': lr_hr_mapping,
            'group_indices': group_indices,
            'local_hr_indices': local_hr_indices,
            'n_lr': z_lr.shape[0],
            'n_hr': x_hr.shape[0],
            'n_genes': n_hvg,
            'latent_dim': latent_dim,
            'subsampled': subsample_indices is not None,
            'n_hr_original': n_hr_original
        }
        
    except Exception as e:
        print(f"  [ERROR] {sample_id}: {e}")
        import traceback
        traceback.print_exc()
        return None
