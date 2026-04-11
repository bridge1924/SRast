"""
Stage 2 Flow Matching Model Training

 Flow Matching 

:
1. OT-Flow Matching: 
2. HR Spatial Prior:  HR spots 
3. 

:
- 
- 
- 

Usage:
    python train_stage2_flow.py --config configs/stage2_config.yaml
    python train_stage2_flow.py --epochs 100 --batch_size 1024
"""

import os
import sys
import argparse
import yaml
import json
import torch
import torch.nn as nn
import numpy as np
import gc
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors
from functools import partial

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from data import load_h5ad
from data.preprocessing import DataPreprocessor
from models.flow_matching import FlowMatchingRatio, FlowMatchingTrainer
from models.graph_vae import LatentNorm
from utils.config_loader import (
    load_stage2_config,
    load_unified_data_config,
    get_training_samples_dict
)
from utils import (
    clear_memory,
    estimate_memory_usage,
    spatial_uniform_subsample,
    compute_metrics,
    build_lr_hr_mapping,
    load_stage1_latent_norm,
    apply_latent_norm
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Stage 2: Flow Matching Model Training',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--config', type=str, default='configs/stage2_config.yaml',
                        help='Path to config file')
    parser.add_argument('--stage1_dir', type=str, default=None,
                        help='Override stage 1 output directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override output directory')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device')
    parser.add_argument('--samples', type=str, nargs='+', default=None,
                        help='Specific samples to use')
    parser.add_argument('--skip_test', action='store_true',
                        help='Skip testing after training')
    parser.add_argument('--num_steps', type=int, default=None,
                        help='Override sampling steps for testing')
    parser.add_argument('--max_hr_spots', type=int, default=None,
                        help='Override max HR spots per sample (for memory optimization)')
    parser.add_argument('--spot_mask_npz', type=str, default=None,
                        help='Path to NPZ file storing per-sample LR spot permutations')
    parser.add_argument('--spot_mask_percentage', type=int, default=None,
                        help='Percentage for per-sample LR spot subset (required with --spot_mask_npz)')
    
    # LatentNorm 
    parser.add_argument('--use_latent_norm', action='store_true',
                        help='Apply LatentNorm to z_lr (override config)')
    parser.add_argument('--no_latent_norm', action='store_true',
                        help='Disable LatentNorm on z_lr (override config)')
    
    return parser.parse_args()


#  HVG 
REQUIRED_N_HVG = 3000


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ============================================================================
# Note: The following functions have been moved to utils module:
# - clear_memory -> utils.training_utils.clear_memory
# - estimate_memory_usage -> utils.training_utils.estimate_memory_usage
# - spatial_uniform_subsample -> utils.training_utils.spatial_uniform_subsample
# - compute_metrics, compute_snr, compute_ssim -> utils.metrics
# - load_latent_norm_from_stage1, apply_latent_norm -> utils.data_loading
# - build_lr_hr_mapping -> utils.data_loading
# ============================================================================


class LargeSampleInfo:
    """
     epoch-wise 
    
    LR
    - LR spots
    - LR spotsHR spots
    - LR-HR
    """
    def __init__(
        self,
        sample_id: str,
        hr_path: str,
        preprocessor: 'DataPreprocessor',
        z_lr: torch.Tensor,
        x_lr: torch.Tensor,
        lr_coords: torch.Tensor,
        hr_to_lr_mapping: np.ndarray,  # HR spotLR spot
        hr_global_indices: Optional[np.ndarray],
        n_hr_total: int,
        n_genes: int,
        latent_dim: int,
        max_lr_per_epoch: int = 20000  # LR
    ):
        self.sample_id = sample_id
        self.hr_path = hr_path
        self.preprocessor = preprocessor
        self.z_lr_full = z_lr  # LR latent
        self.x_lr_full = x_lr  # LR expression
        self.lr_coords_full = lr_coords  # LR
        self.hr_to_lr_mapping = hr_to_lr_mapping  # HR->LR
        if hr_global_indices is None:
            hr_global_indices = np.arange(n_hr_total, dtype=np.int64)
        self.hr_global_indices = np.asarray(hr_global_indices, dtype=np.int64)
        if self.hr_global_indices.shape[0] != n_hr_total:
            raise ValueError(
                f"hr_global_indices length mismatch: {self.hr_global_indices.shape[0]} != {n_hr_total}"
            )
        self.n_hr_total = n_hr_total
        self.n_lr_total = z_lr.shape[0]
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.max_lr_per_epoch = min(max_lr_per_epoch, self.n_lr_total)
        
        # LRHR
        self.lr_to_hr_indices = {}
        for hr_idx, lr_idx in enumerate(hr_to_lr_mapping):
            if lr_idx not in self.lr_to_hr_indices:
                self.lr_to_hr_indices[lr_idx] = []
            self.lr_to_hr_indices[lr_idx].append(hr_idx)
        
        # LRHR
        hr_counts = [len(v) for v in self.lr_to_hr_indices.values()]
        self.avg_hr_per_lr = np.mean(hr_counts) if hr_counts else 0
        
        # Cached data for current epoch
        self._cached_data = None
        self._current_epoch = -1
        
    def load_epoch_data(self, epoch: int, verbose: bool = False):
        """
        LRLR spotsHR spots
        LR-HR
        """
        if self._current_epoch == epoch and self._cached_data is not None:
            return  # Already loaded for this epoch
        
        if verbose:
            print(f"    Loading epoch {epoch} data for {self.sample_id}...")
        
        # Random seed based on epoch
        np.random.seed(42 + epoch)
        
        # 1. LR spots
        all_lr_indices = np.arange(self.n_lr_total)
        if self.max_lr_per_epoch < self.n_lr_total:
            # LR spots
            lr_coords_np = self.lr_coords_full.numpy()
            selected_lr_indices = spatial_uniform_subsample(
                lr_coords_np, self.max_lr_per_epoch, random_state=42 + epoch
            )
        else:
            selected_lr_indices = all_lr_indices
        selected_lr_indices = np.asarray(selected_lr_indices, dtype=np.int64)
        
        # 2. LR spotsHR spots
        selected_hr_indices = []
        for lr_idx in selected_lr_indices:
            if lr_idx in self.lr_to_hr_indices:
                selected_hr_indices.extend(self.lr_to_hr_indices[lr_idx])
        selected_hr_local_indices = np.asarray(sorted(selected_hr_indices), dtype=np.int64)
        selected_hr_global_indices = self.hr_global_indices[selected_hr_local_indices]
        
        if verbose:
            print(f"    Selected {len(selected_lr_indices):,} LR spots -> {len(selected_hr_local_indices):,} HR spots")
        
        # 3. HR
        hr_adata = load_h5ad(self.hr_path)
        hr_coords_full = hr_adata.obsm['spatial'].copy()
        
        # HR
        hr_coords = hr_coords_full[selected_hr_global_indices]
        hr_adata_sub = hr_adata[selected_hr_global_indices].copy()
        hr_hvg_expression = self.preprocessor.get_hvg_expression(hr_adata_sub)
        
        del hr_adata, hr_adata_sub
        gc.collect()
        
        # 4. LR
        x_lr = self.x_lr_full[selected_lr_indices]
        z_lr = self.z_lr_full[selected_lr_indices]
        lr_coords = self.lr_coords_full[selected_lr_indices]
        
        # 5. LR-HR
        # LR
        old_to_new_lr = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_lr_indices)}
        
        # HR spotsgroup_indices
        group_indices = []
        local_hr_indices_list = []
        lr_hr_count = {}  # LRHR
        
        for hr_idx in selected_hr_local_indices:
            old_lr_idx = self.hr_to_lr_mapping[hr_idx]
            new_lr_idx = old_to_new_lr[old_lr_idx]
            group_indices.append(new_lr_idx)
            
            # local_hr_index
            if new_lr_idx not in lr_hr_count:
                lr_hr_count[new_lr_idx] = 0
            local_hr_indices_list.append(lr_hr_count[new_lr_idx])
            lr_hr_count[new_lr_idx] += 1
        
        group_indices = torch.tensor(group_indices, dtype=torch.long)
        local_hr_indices = torch.tensor(local_hr_indices_list, dtype=torch.long)
        
        # 6. tensor
        x_hr = torch.tensor(np.asarray(hr_hvg_expression, dtype=np.float32))
        hr_coords_t = torch.tensor(np.asarray(hr_coords, dtype=np.float32))
        
        self._cached_data = {
            'x_hr': x_hr,
            'x_lr': x_lr,
            'z_lr': z_lr,
            'lr_coords': lr_coords,
            'hr_coords': hr_coords_t,
            'group_indices': group_indices,
            'local_hr_indices': local_hr_indices,
            'n_hr': x_hr.shape[0],
            'n_lr': x_lr.shape[0],
            # Global indices in full sample space for cross-epoch coverage stats
            'selected_lr_indices': selected_lr_indices,
            'selected_hr_indices': selected_hr_local_indices,
            'selected_hr_global_indices': selected_hr_global_indices,
        }
        self._current_epoch = epoch
        
        if verbose:
            print(f"    Epoch {epoch}: {self._cached_data['n_lr']:,} LR, {self._cached_data['n_hr']:,} HR")
    
    def get_data(self):
        """Get cached data for current epoch."""
        return self._cached_data
    
    def clear_cache(self):
        """Clear cached data to free memory."""
        self._cached_data = None
        self._current_epoch = -1
        gc.collect()
        self._cached_local_hr_indices = None
        self._current_epoch = -1
        gc.collect()


class UnifiedFlowDataset(Dataset):
    """
    Unified Dataset for Flow Matching training.
     ratio_diffusion  dataset 
    """
    
    def __init__(
        self,
        all_samples_data: List[Dict],
        verbose: bool = True
    ):
        self.verbose = verbose
        
        if verbose:
            print("\n" + "=" * 70)
            print("Building Unified Flow Matching Dataset")
            print("=" * 70)
        
        self._aggregate_data(all_samples_data)
        
        if verbose:
            self._print_statistics()
    
    def _aggregate_data(self, all_samples_data: List[Dict]):
        """"""
        all_x_hr = []
        all_x_lr = []
        all_z_lr = []
        all_lr_coords = []
        all_hr_coords = []
        all_group_indices = []
        all_local_hr_indices = []
        all_sample_indices = []
        
        hr_offset = 0
        lr_offset = 0
        
        self.sample_info = []
        self.sample_id_list = []
        
        for sample_idx, sample_data in enumerate(all_samples_data):
            n_hr = sample_data['n_hr']
            n_lr = sample_data['n_lr']
            sample_id = sample_data['sample_id']
            
            self.sample_id_list.append(sample_id)
            
            all_x_hr.append(sample_data['x_hr'])
            all_x_lr.append(sample_data['x_lr'])
            all_z_lr.append(sample_data['z_lr'])
            all_lr_coords.append(sample_data['lr_coords'])
            all_hr_coords.append(sample_data['hr_coords'])
            
            all_group_indices.append(sample_data['group_indices'] + lr_offset)
            all_local_hr_indices.append(sample_data['local_hr_indices'])
            
            all_sample_indices.extend([sample_idx] * n_hr)
            
            self.sample_info.append({
                'sample_id': sample_id,
                'sample_idx': sample_idx,
                'hr_range': (hr_offset, hr_offset + n_hr),
                'lr_range': (lr_offset, lr_offset + n_lr),
                'n_hr': n_hr,
                'n_lr': n_lr,
                'n_genes': sample_data['n_genes']
            })
            
            hr_offset += n_hr
            lr_offset += n_lr
        
        self.x_hr = torch.cat(all_x_hr, dim=0)
        self.x_lr = torch.cat(all_x_lr, dim=0)
        self.z_lr = torch.cat(all_z_lr, dim=0)
        self.lr_coords = torch.cat(all_lr_coords, dim=0)
        self.hr_coords = torch.cat(all_hr_coords, dim=0)
        self.group_indices = torch.cat(all_group_indices, dim=0)
        self.local_hr_indices = torch.cat(all_local_hr_indices, dim=0)
        self.sample_indices = torch.tensor(all_sample_indices, dtype=torch.long)
        
        self.n_samples = len(all_samples_data)
        self.total_hr = self.x_hr.shape[0]
        self.total_lr = self.x_lr.shape[0]
        self.n_genes = self.x_hr.shape[1]
        self.latent_dim = self.z_lr.shape[1]
        
        lr_indices = self.group_indices
        hr_indices = torch.arange(self.total_hr, dtype=torch.long)
        self.lr_hr_mapping = torch.stack([lr_indices, hr_indices])
    
    def _print_statistics(self):
        print(f"\nDataset Statistics:")
        print(f"  Total samples: {self.n_samples}")
        print(f"  Total HR spots: {self.total_hr:,}")
        print(f"  Total LR spots: {self.total_lr:,}")
        print(f"  Genes: {self.n_genes}")
        print(f"  Latent dim: {self.latent_dim}")
        print(f"  HR/LR ratio: {self.total_hr / self.total_lr:.2f}")
        
        print(f"\nPer-sample breakdown:")
        for info in self.sample_info:
            print(f"  {info['sample_id']}: {info['n_hr']:,} HR / {info['n_lr']:,} LR")
    
    def __len__(self):
        return self.total_hr
    
    def __getitem__(self, idx):
        return {
            'hr_idx': idx,
            'sample_idx': self.sample_indices[idx].item(),
            'group_idx': self.group_indices[idx].item(),
            'local_hr_idx': self.local_hr_indices[idx].item()
        }


def collate_flow_batch(batch_list: List[Dict], dataset: UnifiedFlowDataset) -> Dict[str, torch.Tensor]:
    """Custom collate function for Flow Matching training."""
    hr_indices = torch.tensor([item['hr_idx'] for item in batch_list], dtype=torch.long)
    batch_size = len(hr_indices)
    
    x_hr_batch = dataset.x_hr[hr_indices]
    hr_coords_batch = dataset.hr_coords[hr_indices]
    local_hr_indices_batch = dataset.local_hr_indices[hr_indices]
    
    group_indices_global = dataset.group_indices[hr_indices]
    
    unique_lr_indices, inverse_indices = torch.unique(group_indices_global, return_inverse=True)
    n_lr_batch = len(unique_lr_indices)
    
    x_lr_batch = dataset.x_lr[unique_lr_indices]
    z_lr_batch = dataset.z_lr[unique_lr_indices]
    lr_coords_batch = dataset.lr_coords[unique_lr_indices]
    
    group_indices_batch = inverse_indices
    
    lr_hr_mapping_batch = torch.stack([
        group_indices_batch,
        torch.arange(batch_size, dtype=torch.long)
    ])
    
    return {
        'x_hr': x_hr_batch,
        'x_lr': x_lr_batch,
        'z_lr': z_lr_batch,
        'lr_coords': lr_coords_batch,
        'hr_coords': hr_coords_batch,
        'group_indices': group_indices_batch,
        'local_hr_indices': local_hr_indices_batch,
        'lr_hr_mapping': lr_hr_mapping_batch,
        'batch_size': batch_size,
        'n_lr': n_lr_batch
    }


def build_lr_hr_group_mapping(
    lr_coords: np.ndarray,
    hr_coords: np.ndarray,
    k_neighbors: int = 1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build LR-HR mapping where each HR spot is assigned to its nearest LR spot.
    NOTE: This is a wrapper around utils.data_loading.build_lr_hr_mapping for backwards compatibility.
    """
    lr_hr_mapping, group_indices, local_hr_indices, _ = build_lr_hr_mapping(lr_coords, hr_coords)
    return lr_hr_mapping, group_indices, local_hr_indices


def load_sample_data(
    sample_id: str,
    stage1_dir: str,
    hr_path: str,
    config: Dict,
    device: str,
    max_hr_spots: Optional[int] = None,
    spatial_uniform_sampling: bool = True,
    lazy_load_threshold: Optional[int] = None,
    use_latent_norm: bool = True,  #  LatentNorm
    selected_lr_indices: Optional[np.ndarray] = None,
) -> Optional[Dict]:
    """
    Load sample data for Flow Matching training.
    
    Args:
        sample_id: Sample identifier
        stage1_dir: Stage 1 output directory
        hr_path: Path to HR h5ad file
        config: Configuration dict
        device: Device string
        max_hr_spots: Maximum HR spots to load per epoch (None = no limit)
        spatial_uniform_sampling: Use spatial uniform sampling if subsampling
        lazy_load_threshold: If n_hr > this, use lazy loading (return LargeSampleInfo)
        use_latent_norm: Whether to apply LatentNorm to z_lr ()
        
    Returns:
        Sample data dict, LargeSampleInfo object, or None if failed
    """
    sample_dir = os.path.join(stage1_dir, sample_id)
    
    required_files = ['latent_representations.npz', 'preprocessor.pkl', 'config.json']
    for f in required_files:
        if not os.path.exists(os.path.join(sample_dir, f)):
            print(f"  [SKIP] {sample_id}: Missing {f}")
            return None
    
    try:
        latent_data = np.load(os.path.join(sample_dir, 'latent_representations.npz'))
        z_lr_full_np = latent_data['z_lr']
        lr_coords_full_np = latent_data['lr_coords']
        lr_hvg_expression_full_np = latent_data['lr_hvg_expression']

        z_lr_np = z_lr_full_np
        lr_coords = lr_coords_full_np
        lr_hvg_expression = lr_hvg_expression_full_np

        if selected_lr_indices is not None:
            selected_lr_indices = np.asarray(selected_lr_indices, dtype=np.int64)
            selected_lr_indices = selected_lr_indices[
                (selected_lr_indices >= 0) & (selected_lr_indices < z_lr_full_np.shape[0])
            ]
            if selected_lr_indices.size == 0:
                print(f"  [SKIP] {sample_id}: Spot mask selected 0 valid LR indices")
                return None

            z_lr_np = z_lr_full_np[selected_lr_indices]
            lr_coords = lr_coords_full_np[selected_lr_indices]
            lr_hvg_expression = lr_hvg_expression_full_np[selected_lr_indices]

            ratio = 100.0 * selected_lr_indices.size / max(1, z_lr_full_np.shape[0])
            print(
                f"  [SPOT-MASK] {sample_id}: using {selected_lr_indices.size:,}/"
                f"{z_lr_full_np.shape[0]:,} LR spots ({ratio:.2f}%)"
            )

        z_lr = torch.tensor(z_lr_np, dtype=torch.float32)
        
        preprocessor = DataPreprocessor.load(os.path.join(sample_dir, 'preprocessor.pkl'))
        
        with open(os.path.join(sample_dir, 'sample_info.json'), 'r') as f:
            sample_info = json.load(f)
        
        n_hvg = sample_info['n_hvg']
        latent_dim = sample_info['latent_dim']
        
        #  LatentNorm ()
        if use_latent_norm:
            latent_norm = load_stage1_latent_norm(sample_dir, latent_dim)
            if latent_norm is not None:
                z_lr = apply_latent_norm(z_lr, latent_norm)
                print(f"  [LatentNorm] {sample_id}: Applied LatentNorm to z_lr")
            else:
                print(f"  [WARNING] {sample_id}: LatentNorm requested but not found in Stage1")
        
        if n_hvg != REQUIRED_N_HVG:
            print(f"  [WARNING] {sample_id}: n_hvg={n_hvg}, expected={REQUIRED_N_HVG}")
        
        if not os.path.exists(hr_path):
            print(f"  [SKIP] {sample_id}: HR data not found at {hr_path}")
            return None
        
        # First, just get HR spot count and coordinates
        hr_adata = load_h5ad(hr_path)
        hr_coords_full_all = hr_adata.obsm['spatial'].copy()
        n_hr_full = int(hr_adata.n_obs)
        hr_global_indices = np.arange(n_hr_full, dtype=np.int64)

        # Keep only HR spots linked to selected LR spots for all sample sizes.
        if selected_lr_indices is not None:
            lr_coords_for_hr_filter = np.asarray(lr_coords_full_np, dtype=np.float32)
            hr_coords_for_filter = np.asarray(hr_coords_full_all, dtype=np.float32)

            nbrs_full = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(lr_coords_for_hr_filter)
            _, nearest_full_lr_indices = nbrs_full.kneighbors(hr_coords_for_filter)
            nearest_full_lr_indices = nearest_full_lr_indices.flatten()

            selected_lr_mask = np.zeros(z_lr_full_np.shape[0], dtype=np.bool_)
            selected_lr_mask[selected_lr_indices] = True
            keep_hr_mask = selected_lr_mask[nearest_full_lr_indices]

            hr_global_indices = hr_global_indices[keep_hr_mask]
            if hr_global_indices.size == 0:
                print(f"  [SKIP] {sample_id}: Spot mask produced 0 linked HR spots")
                return None

            hr_coords_full = hr_coords_full_all[keep_hr_mask]
            ratio_hr = 100.0 * hr_global_indices.size / max(1, n_hr_full)
            print(
                f"  [SPOT-MASK-HR] {sample_id}: using {hr_global_indices.size:,}/"
                f"{n_hr_full:,} HR spots ({ratio_hr:.2f}%)"
            )
        else:
            hr_coords_full = hr_coords_full_all

        n_hr_original = int(hr_coords_full.shape[0])
        
        # Check if lazy loading should be used
        use_lazy = (lazy_load_threshold is not None and 
                    lazy_load_threshold > 0 and 
                    n_hr_original > lazy_load_threshold)
        
        if use_lazy:
            # LRHR
            avg_hr_per_lr = n_hr_original / z_lr.shape[0]
            # max_hr_spotsLR
            max_lr_per_epoch = int(max_hr_spots / avg_hr_per_lr) if max_hr_spots else z_lr.shape[0]
            max_lr_per_epoch = min(max_lr_per_epoch, z_lr.shape[0])
            expected_hr = int(max_lr_per_epoch * avg_hr_per_lr)
            
            print(f"  [LAZY] {sample_id}: {n_hr_original:,} HR / {z_lr.shape[0]:,} LR (avg {avg_hr_per_lr:.1f} HR/LR)")
            print(f"    Strategy: Sample {max_lr_per_epoch:,} LR spots -> ~{expected_hr:,} HR spots per epoch")
            
            #  HR->LR 
            print(f"    Building HR->LR mapping...")
            lr_coords_np = np.asarray(lr_coords, dtype=np.float32)
            hr_coords_np = np.asarray(hr_coords_full, dtype=np.float32)
            
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(lr_coords_np)
            _, indices = nbrs.kneighbors(hr_coords_np)
            hr_to_lr_mapping = indices.flatten()  # HRLR
            
            # Close HR adata - will reload per epoch
            del hr_adata
            gc.collect()
            
            # Prepare LR data
            x_lr = torch.tensor(np.asarray(lr_hvg_expression, dtype=np.float32), dtype=torch.float32)
            lr_coords_t = torch.tensor(lr_coords_np, dtype=torch.float32)
            
            return LargeSampleInfo(
                sample_id=sample_id,
                hr_path=hr_path,
                preprocessor=preprocessor,
                z_lr=z_lr,
                x_lr=x_lr,
                lr_coords=lr_coords_t,
                hr_to_lr_mapping=hr_to_lr_mapping,
                hr_global_indices=hr_global_indices,
                n_hr_total=n_hr_original,
                n_genes=n_hvg,
                latent_dim=latent_dim,
                max_lr_per_epoch=max_lr_per_epoch
            )
        
        # Regular loading (small dataset or no lazy loading)
        # Check if subsampling is needed
        subsample_indices = None
        if max_hr_spots is not None and max_hr_spots > 0 and n_hr_original > max_hr_spots:
            print(f"  [SUBSAMPLE] {sample_id}: {n_hr_original:,} HR spots > max {max_hr_spots:,}")
            
            # Estimate memory savings
            mem_full = estimate_memory_usage(n_hr_original, z_lr.shape[0], n_hvg, latent_dim)
            mem_sub = estimate_memory_usage(max_hr_spots, z_lr.shape[0], n_hvg, latent_dim)
            print(f"    Memory: {mem_full['total_gb']:.2f} GB -> {mem_sub['total_gb']:.2f} GB "
                  f"(saving {mem_full['total_gb'] - mem_sub['total_gb']:.2f} GB)")
            
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
            
            # Apply subsampling to coordinates
            hr_coords = hr_coords_full[subsample_indices]
            selected_hr_global_indices = hr_global_indices[subsample_indices]
            print(f"    Subsampled to {len(subsample_indices):,} HR spots")
        else:
            hr_coords = hr_coords_full
            selected_hr_global_indices = hr_global_indices
        
        # Get HVG expression (apply subsampling if needed)
        # Memory-efficient: only extract needed spots
        if selected_hr_global_indices.shape[0] != hr_adata.n_obs:
            # Subset by global HR indices so spot masking and subsampling are both respected.
            hr_adata_sub = hr_adata[selected_hr_global_indices].copy()
            hr_hvg_expression = preprocessor.get_hvg_expression(hr_adata_sub)
            del hr_adata_sub
        else:
            hr_hvg_expression = preprocessor.get_hvg_expression(hr_adata)
        
        # Clear original adata to free memory
        del hr_adata
        gc.collect()
        
        lr_hr_mapping, group_indices, local_hr_indices = build_lr_hr_group_mapping(
            lr_coords, hr_coords
        )
        
        # Ensure proper dtype for torch conversion (some datasets have uint32 coords)
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


# Note: compute_snr, compute_ssim, compute_metrics have been moved to utils.metrics


def main():
    args = parse_args()
    
    print("\n" + "=" * 70)
    print("SRast Stage 2: Flow Matching Model Training")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ==========================================================================
    # Load configuration
    # ==========================================================================
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"[ERROR] Config file not found: {args.config}")
        return 1
    
    # Try to load unified data config
    try:
        _, data_config, training_dataset_ids, test_dataset_ids = load_stage2_config(
            config_path=args.config,
            experiment_config_path=None
        )
    except Exception as e:
        print(f"Warning: Could not load unified config: {e}")
        data_config = None
        training_dataset_ids = None
        test_dataset_ids = None
    
    # Get flow_matching config
    flow_config = config.get('flow_matching', {})
    
    # Override from command line
    stage1_dir = args.stage1_dir or config.get('stage1_dir', 'checkpoints/stage1')
    output_dir = args.output_dir or config.get('paths', {}).get('output_dir', 'checkpoints/stage2/flow_matching')
    epochs = args.epochs or flow_config.get('epochs', 100)
    batch_size = args.batch_size or flow_config.get('batch_size', 1024)
    lr = args.lr or flow_config.get('learning_rate', 0.0001)
    
    # Device
    if args.device:
        device = args.device
    elif config.get('training', {}).get('device') == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = config.get('training', {}).get('device', 'cuda')
    
    # LatentNorm 
    # :  >  > (true)
    if args.no_latent_norm:
        use_latent_norm = False
    elif args.use_latent_norm:
        use_latent_norm = True
    else:
        use_latent_norm = config.get('use_latent_norm', True)
    
    print(f"\nConfiguration:")
    print(f"  Stage 1 dir: {stage1_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Use HR Spatial: {flow_config.get('use_hr_spatial', True)}")
    print(f"  HR Neighbors: {flow_config.get('num_hr_neighbors', 6)}")
    print(f"  Use LatentNorm: {use_latent_norm}  {'(: z_lr)' if use_latent_norm else '(: z_lr)'}")
    
    # Memory optimization config
    mem_opt_config = config.get('memory_optimization', {})
    max_hr_spots = args.max_hr_spots or mem_opt_config.get('max_hr_spots_per_sample', None)
    lazy_load_threshold = mem_opt_config.get('lazy_load_threshold', None)
    spatial_uniform_sampling = mem_opt_config.get('spatial_uniform_sampling', True)
    print_memory_estimate = mem_opt_config.get('print_memory_estimate', True)
    
    if max_hr_spots:
        print(f"\nMemory Optimization:")
        print(f"  Max HR spots per sample: {max_hr_spots:,}")
        print(f"  Lazy load threshold: {lazy_load_threshold:,}" if lazy_load_threshold else "  Lazy load: disabled")
        print(f"  Spatial uniform sampling: {spatial_uniform_sampling}")

    # Spot-level subset mask config
    spot_mask_permutations = None
    spot_mask_percentage = None
    if args.spot_mask_npz:
        if args.spot_mask_percentage is None:
            print("[ERROR] --spot_mask_percentage is required when --spot_mask_npz is provided")
            return 1
        if args.spot_mask_percentage <= 0 or args.spot_mask_percentage >= 100:
            print(f"[ERROR] Invalid --spot_mask_percentage: {args.spot_mask_percentage}")
            return 1
        if not os.path.exists(args.spot_mask_npz):
            print(f"[ERROR] Spot mask NPZ not found: {args.spot_mask_npz}")
            return 1

        spot_mask_percentage = int(args.spot_mask_percentage)
        try:
            mask_npz = np.load(args.spot_mask_npz)
            spot_mask_permutations = {key: np.asarray(mask_npz[key], dtype=np.int64) for key in mask_npz.files}
            mask_npz.close()
            print(f"\nSpot Mask Configuration:")
            print(f"  Mask file: {args.spot_mask_npz}")
            print(f"  Percentage: {spot_mask_percentage}%")
            print(f"  Samples in mask file: {len(spot_mask_permutations)}")
        except Exception as e:
            print(f"[ERROR] Failed to load spot mask NPZ: {e}")
            return 1
    
    # ==========================================================================
    # Get training samples
    # ==========================================================================
    if data_config is not None and training_dataset_ids is not None:
        training_samples = get_training_samples_dict(
            data_config, training_dataset_ids,
            use_lr_path=True, use_hr_path=True
        )
        test_samples = get_training_samples_dict(
            data_config, test_dataset_ids,
            use_lr_path=True, use_hr_path=True
        ) if test_dataset_ids else {}
    else:
        training_samples = config.get('training_samples', {})
        test_samples = config.get('test_samples', {})
    
    if args.samples:
        training_samples = {k: v for k, v in training_samples.items() if k in args.samples}
    
    print(f"\nTraining samples: {len(training_samples)}")

    selected_lr_indices_by_sample = {}
    if spot_mask_permutations is not None:
        for sample_id in training_samples.keys():
            if sample_id not in spot_mask_permutations:
                print(f"[ERROR] Spot mask missing sample: {sample_id}")
                return 1
            perm = spot_mask_permutations[sample_id]
            n_total = int(perm.shape[0])
            n_select = max(1, int(np.ceil(n_total * (spot_mask_percentage / 100.0))))
            selected_lr_indices_by_sample[sample_id] = perm[:n_select]
    
    # ==========================================================================
    # Phase 1: Load all samples
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Phase 1: Loading Sample Data")
    print("-" * 70)
    
    all_samples_data = []  # Regular samples (dict)
    large_samples = []      # Large samples (LargeSampleInfo)
    n_genes = None
    latent_dim = None
    
    for sample_id, sample_cfg in training_samples.items():
        print(f"\nLoading: {sample_id}")
        
        hr_path = sample_cfg.get('hr_path')
        if not hr_path:
            print(f"  [SKIP] No HR path configured")
            continue
        
        sample_data = load_sample_data(
            sample_id=sample_id,
            stage1_dir=stage1_dir,
            hr_path=hr_path,
            config=config,
            device=device,
            max_hr_spots=max_hr_spots,
            spatial_uniform_sampling=spatial_uniform_sampling,
            lazy_load_threshold=lazy_load_threshold,
            use_latent_norm=use_latent_norm,  # 
            selected_lr_indices=selected_lr_indices_by_sample.get(sample_id)
        )
        
        if sample_data is not None:
            # Check if this is a LargeSampleInfo (lazy loading)
            if isinstance(sample_data, LargeSampleInfo):
                large_samples.append(sample_data)
                if n_genes is None:
                    n_genes = sample_data.n_genes
                    latent_dim = sample_data.latent_dim
                expected_hr = int(sample_data.max_lr_per_epoch * sample_data.avg_hr_per_lr)
                print(f"  [OK] LAZY: {sample_data.n_lr_total:,} LR, {sample_data.n_hr_total:,} HR total")
                print(f"       Per epoch: ~{sample_data.max_lr_per_epoch:,} LR -> ~{expected_hr:,} HR")
            else:
                all_samples_data.append(sample_data)
                
                if n_genes is None:
                    n_genes = sample_data['n_genes']
                    latent_dim = sample_data['latent_dim']
                
                if sample_data['n_genes'] != n_genes:
                    print(f"  [WARNING] Gene count mismatch: {sample_data['n_genes']} vs {n_genes}")
                
                # Show subsampling info if applicable
                if sample_data.get('subsampled', False):
                    print(f"  [OK] n_lr={sample_data['n_lr']}, n_hr={sample_data['n_hr']} "
                          f"(subsampled from {sample_data['n_hr_original']:,}), n_genes={sample_data['n_genes']}")
                else:
                    print(f"  [OK] n_lr={sample_data['n_lr']}, n_hr={sample_data['n_hr']}, n_genes={sample_data['n_genes']}")
    
    if not all_samples_data and not large_samples:
        print("\n[ERROR] No valid samples loaded!")
        return 1
    
    # Summarize regular samples
    total_lr_regular = sum(d['n_lr'] for d in all_samples_data) if all_samples_data else 0
    total_hr_regular = sum(d['n_hr'] for d in all_samples_data) if all_samples_data else 0
    n_subsampled = sum(1 for d in all_samples_data if d.get('subsampled', False))
    
    # Add large samples info (LR per epoch, not total)
    lazy_lr_per_epoch = sum(s.max_lr_per_epoch for s in large_samples)
    lazy_hr_total = sum(s.n_hr_total for s in large_samples)
    lazy_hr_per_epoch = sum(int(s.max_lr_per_epoch * s.avg_hr_per_lr) for s in large_samples)
    
    total_lr = total_lr_regular + lazy_lr_per_epoch
    total_hr = total_hr_regular + lazy_hr_per_epoch
    
    print(f"\nLoaded Summary:")
    print(f"  Regular samples: {len(all_samples_data)}, {total_hr_regular:,} HR spots")
    if large_samples:
        print(f"  Lazy samples: {len(large_samples)}, {lazy_hr_total:,} total HR spots")
        print(f"    Per epoch: ~{lazy_lr_per_epoch:,} LR -> ~{lazy_hr_per_epoch:,} HR (with correct LR-HR pairing)")
    print(f"  Total per epoch: {total_lr:,} LR, {total_hr:,} HR")
    if n_subsampled > 0:
        print(f"  ({n_subsampled} regular samples were subsampled)")
    print(f"Genes: {n_genes}, Latent dim: {latent_dim}")
    
    # Estimate and print total memory usage (per epoch)
    if print_memory_estimate:
        total_mem = estimate_memory_usage(total_hr, total_lr, n_genes, latent_dim)
        print(f"\nEstimated memory usage (per epoch):")
        print(f"  x_hr (HR expression): {total_mem['x_hr_gb']:.2f} GB")
        print(f"  x_lr (LR expression): {total_mem['x_lr_gb']:.2f} GB")
        print(f"  z_lr (LR latent):     {total_mem['z_lr_gb']:.2f} GB")
        print(f"  Coordinates:          {total_mem['coords_gb']:.2f} GB")
        print(f"  Indices:              {total_mem['indices_gb']:.2f} GB")
        print(f"  Total:                {total_mem['total_gb']:.2f} GB")

    # ======================================================================
    # Spot participation tracking (run-level true usage statistics)
    # ======================================================================
    participation_tracker = {
        'regular_samples': {},
        'lazy_samples': {},
        'epoch_stats': []
    }

    for sample_data in all_samples_data:
        sample_id = sample_data['sample_id']
        participation_tracker['regular_samples'][sample_id] = {
            'sample_id': sample_id,
            'mode': 'regular',
            'n_lr_total': int(sample_data['n_lr']),
            'n_hr_total': int(sample_data['n_hr']),
            'n_lr_per_epoch': int(sample_data['n_lr']),
            'n_hr_per_epoch': int(sample_data['n_hr']),
            'n_lr_unique_seen': int(sample_data['n_lr']),
            'n_hr_unique_seen': int(sample_data['n_hr']),
        }

    for large_sample in large_samples:
        participation_tracker['lazy_samples'][large_sample.sample_id] = {
            'sample_id': large_sample.sample_id,
            'mode': 'lazy',
            'n_lr_total': int(large_sample.n_lr_total),
            'n_hr_total': int(large_sample.n_hr_total),
            'n_lr_per_epoch_target': int(large_sample.max_lr_per_epoch),
            'lr_seen_mask': np.zeros(large_sample.n_lr_total, dtype=np.bool_),
            'hr_seen_mask': np.zeros(large_sample.n_hr_total, dtype=np.bool_),
            'epochs': {}
        }

    def record_lazy_epoch_participation(sample_dict: Dict[str, Any], epoch_index: int):
        """Track which global LR/HR spots were actually used for a lazy sample in this epoch."""
        sample_id = sample_dict.get('sample_id')
        if sample_id not in participation_tracker['lazy_samples']:
            return

        tracker = participation_tracker['lazy_samples'][sample_id]
        selected_lr = sample_dict.get('selected_lr_indices')
        selected_hr = sample_dict.get('selected_hr_indices')

        if isinstance(selected_lr, np.ndarray) and selected_lr.size > 0:
            tracker['lr_seen_mask'][selected_lr.astype(np.int64)] = True
            n_lr_selected = int(selected_lr.size)
        else:
            n_lr_selected = int(sample_dict.get('n_lr', 0))

        if isinstance(selected_hr, np.ndarray) and selected_hr.size > 0:
            tracker['hr_seen_mask'][selected_hr.astype(np.int64)] = True
            n_hr_selected = int(selected_hr.size)
        else:
            n_hr_selected = int(sample_dict.get('n_hr', 0))

        tracker['epochs'][str(epoch_index + 1)] = {
            'n_lr_selected': n_lr_selected,
            'n_hr_selected': n_hr_selected,
        }
    
    # ==========================================================================
    # Phase 2: Create Dataset and DataLoader
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Phase 2: Creating Unified Dataset")
    print("-" * 70)
    
    # Clear intermediate data to free memory before creating dataset
    clear_memory()
    
    # For lazy-loaded samples, we need to load their data for epoch 0
    if large_samples:
        print(f"\nLoading lazy samples data for epoch 0...")
        for large_sample in large_samples:
            large_sample.load_epoch_data(epoch=0, verbose=True)
            # Convert to dict format for UnifiedFlowDataset
            sample_dict = large_sample.get_data()
            sample_dict['sample_id'] = large_sample.sample_id
            sample_dict['n_genes'] = large_sample.n_genes
            sample_dict['latent_dim'] = large_sample.latent_dim
            sample_dict['subsampled'] = True
            sample_dict['n_hr_original'] = large_sample.n_hr_total
            sample_dict['is_lazy'] = True
            record_lazy_epoch_participation(sample_dict, epoch_index=0)
            all_samples_data.append(sample_dict)
    
    dataset = UnifiedFlowDataset(all_samples_data, verbose=True)
    
    collate_fn = partial(collate_flow_batch, dataset=dataset)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"\nDataLoader created:")
    print(f"  Total HR spots: {len(dataset):,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Batches per epoch: {len(dataloader):,}")
    
    # ==========================================================================
    # Phase 3: Initialize and Train Model
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Phase 3: Training Flow Matching Model")
    print("-" * 70)
    
    # v5.2 : DiT 
    use_dit = flow_config.get('use_dit', False)
    mlp_ratio = flow_config.get('mlp_ratio', 4.0)
    
    # v5.4 : 
    kd_config = flow_config.get('kd_loss', {})
    kd_enabled = kd_config.get('enabled', True)
    kd_weight = kd_config.get('weight', 1.0)
    kd_temperature = kd_config.get('temperature', 3.0)
    
    # v5.5 : velocity loss  ()
    velocity_config = flow_config.get('velocity_loss', {})
    velocity_loss_enabled = velocity_config.get('enabled', True)
    velocity_loss_weight = velocity_config.get('weight', 1.0)
    
    model = FlowMatchingRatio(
        n_genes=n_genes,
        latent_dim=latent_dim,
        hidden_dim=flow_config.get('hidden_dim', 256),
        num_heads=flow_config.get('num_heads', 8),
        num_layers=flow_config.get('num_layers', 4),
        dropout=flow_config.get('dropout', 0.1),
        max_hr_per_lr=flow_config.get('max_hr_per_lr', 16),
        use_hr_spatial=flow_config.get('use_hr_spatial', True),
        num_hr_neighbors=flow_config.get('num_hr_neighbors', 6),
        sigma_min=flow_config.get('sigma_min', 0.001),
        # v5.2 
        use_dit=use_dit,
        mlp_ratio=mlp_ratio,
        # v5.4 : 
        kd_enabled=kd_enabled,
        kd_weight=kd_weight,
        kd_temperature=kd_temperature,
        # v5.5 : velocity loss 
        velocity_loss_enabled=velocity_loss_enabled,
        velocity_loss_weight=velocity_loss_weight
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    arch_name = "DiT (AdaLN)" if use_dit else "Original"
    print(f"Model architecture: {arch_name}")
    print(f"Model parameters: {n_params:,}")
    
    # 
    print(f"\nLoss Configuration:")
    if velocity_loss_enabled:
        print(f"  Velocity Loss: enabled (weight={velocity_loss_weight})")
    else:
        print(f"  Velocity Loss: DISABLED ()")
    if kd_enabled:
        print(f"  KD Loss: enabled (weight={kd_weight}, temperature={kd_temperature})")
    else:
        print(f"  KD Loss: disabled")
    
    # v5.3: 
    # : total_steps  epochs scheduler.step()  epoch 
    scheduler_config = flow_config.get('scheduler', {})
    estimated_steps_per_epoch = len(dataloader)
    # total_steps  CosineAnnealingLR  T_max epoch  batch 
    total_steps = epochs  # :  epochs * batches
    
    # v5.3: 
    time_sampling_config = flow_config.get('time_sampling', {})
    
    trainer = FlowMatchingTrainer(
        model=model,
        learning_rate=lr,
        weight_decay=flow_config.get('weight_decay', 0.0001),
        device=device,
        validation_steps=flow_config.get('validation_steps', 50),
        scheduler_config=scheduler_config,
        total_steps=total_steps,
        time_sampling_config=time_sampling_config
    )
    
    # 
    scheduler_type = scheduler_config.get('type', 'cosine')
    print(f"\nTraining configuration:")
    print(f"  Learning rate: {lr}")
    print(f"  Scheduler type: {scheduler_type}")
    print(f"  Batches per epoch: {estimated_steps_per_epoch}")
    if scheduler_type == 'cosine':
        t_max = scheduler_config.get('t_max') or total_steps
        eta_min = scheduler_config.get('eta_min', 1e-6)
        print(f"  Scheduler T_max: {t_max} (epochs), eta_min: {eta_min}")
    warmup_epochs = scheduler_config.get('warmup_epochs', 0)
    if warmup_epochs > 0:
        print(f"  Warmup epochs: {warmup_epochs}")
    
    # v5.3: 
    time_strategy = time_sampling_config.get('strategy', 'uniform')
    print(f"  Time sampling: {time_strategy}")
    if time_strategy == 'importance':
        print(f"    importance_beta: {time_sampling_config.get('importance_beta', 2.0)}")
    
    log_interval = config.get('training', {}).get('log_every', 10)
    validate_interval = flow_config.get('validate_interval', 10)  # Nepoch
    validate_batches = flow_config.get('validate_batches', 5)  # batch
    
    # Early stopping 
    early_stop_config = flow_config.get('early_stopping', {})
    early_stop_enabled = early_stop_config.get('enabled', False)
    early_stop_patience = early_stop_config.get('patience', 50)
    early_stop_min_delta = early_stop_config.get('min_delta', 0.0001)
    early_stop_verbose = early_stop_config.get('verbose', True)
    
    # Early stopping 
    best_train_loss = float('inf')
    epochs_without_improvement = 0
    early_stopped = False
    
    # Training loop
    print("\nStarting training (v5.4 with KD Loss)...")
    if large_samples:
        print(f"  [INFO] {len(large_samples)} large sample(s) will be re-sampled each epoch")
    if early_stop_enabled:
        print(f"  [INFO] Early stopping enabled: patience={early_stop_patience}, min_delta={early_stop_min_delta}")
    print(f"{'='*80}")
    # 
    header_parts = [f"{'Epoch':>6}", f"{'Batch':>6}", f"{'Loss':>8}"]
    if velocity_loss_enabled:
        header_parts.append(f"{'Vel':>8}")
    if kd_enabled:
        header_parts.append(f"{'KD':>8}")
    header_parts.extend([f"{'t':>5}", f"{'LR':>10}"])
    print(" | ".join(header_parts))
    print(f"{'='*80}")
    
    best_val_gene_pcc = 0.0
    best_epoch = 0
    
    for epoch in range(epochs):
        # ====================================================================
        # Reload lazy samples for this epoch (different random sampling)
        # ====================================================================
        if large_samples and epoch > 0:
            print(f"\n  [LAZY] Reloading large samples for epoch {epoch}...")
            
            # Remove old lazy sample data from all_samples_data
            all_samples_data = [d for d in all_samples_data if not d.get('is_lazy', False)]
            
            # Reload with new sampling
            for large_sample in large_samples:
                large_sample.load_epoch_data(epoch=epoch, verbose=False)
                sample_dict = large_sample.get_data()
                sample_dict['sample_id'] = large_sample.sample_id
                sample_dict['n_genes'] = large_sample.n_genes
                sample_dict['latent_dim'] = large_sample.latent_dim
                sample_dict['subsampled'] = True
                sample_dict['n_hr_original'] = large_sample.n_hr_total
                sample_dict['is_lazy'] = True
                record_lazy_epoch_participation(sample_dict, epoch_index=epoch)
                all_samples_data.append(sample_dict)
            
            # Rebuild dataset and dataloader
            clear_memory()
            dataset = UnifiedFlowDataset(all_samples_data, verbose=False)
            collate_fn = partial(collate_flow_batch, dataset=dataset)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0
            )

        participation_tracker['epoch_stats'].append({
            'epoch': int(epoch + 1),
            'total_lr_spots': int(dataset.total_lr),
            'total_hr_spots': int(dataset.total_hr),
            'n_batches': int(len(dataloader)),
        })
        
        # v5.3: warmup ()
        trainer.current_epoch = epoch
        is_warmup = trainer._apply_warmup(epoch)
        
        epoch_metrics = {
            'loss': [], 'velocity_loss': [], 'kd_loss': []
        }
        
        for batch_idx, batch in enumerate(dataloader):
            loss, loss_dict = trainer.train_step(batch, batch_idx=batch_idx)
            
            epoch_metrics['loss'].append(loss)
            epoch_metrics['velocity_loss'].append(loss_dict['velocity_loss'])
            epoch_metrics['kd_loss'].append(loss_dict.get('kd_loss', 0.0))
            
            if (batch_idx + 1) % log_interval == 0 or batch_idx == 0:
                current_lr = trainer.get_current_lr()
                t_mean = loss_dict.get('t_mean', 0.5)
                # 
                log_parts = [f"{epoch+1:>6}", f"{batch_idx+1:>6}", f"{loss:>8.4f}"]
                if velocity_loss_enabled:
                    log_parts.append(f"{loss_dict['velocity_loss']:>8.4f}")
                if kd_enabled:
                    log_parts.append(f"{loss_dict.get('kd_loss', 0.0):>8.4f}")
                log_parts.extend([f"{t_mean:>5.3f}", f"{current_lr:>10.2e}"])
                print(" | ".join(log_parts))
        
        # v5.3: warmupscheduler.step()
        if not is_warmup:
            trainer.scheduler.step()
        
        # 
        trainer.history['learning_rate'].append(trainer.get_current_lr())
        
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        trainer.history['epoch_loss'].append(avg_metrics['loss'])
        
        # ====================================================================
        # validate_intervalepochODE
        # ====================================================================
        val_metrics = None
        validation_steps = flow_config.get('validation_steps', flow_config.get('num_steps', 50))
        if (epoch + 1) % validate_interval == 0 or epoch == epochs - 1:
            print(f"  [Validating with full ODE integration ({validate_batches} batches, {validation_steps} steps)...]")
            val_metrics = trainer.validate_epoch(
                dataloader, 
                max_batches=validate_batches,
                num_steps=validation_steps
            )
            
            # 
            if val_metrics['val_gene_pcc'] > best_val_gene_pcc:
                best_val_gene_pcc = val_metrics['val_gene_pcc']
                best_epoch = epoch + 1
                # 
                best_model_path = os.path.join(output_dir, 'flow_matching_model_best.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'val_gene_pcc': best_val_gene_pcc,
                    'val_spot_pcc': val_metrics['val_spot_pcc']
                }, best_model_path)
        
        print(f"{'-'*80}")
        # summary
        summary_parts = [f"Epoch {epoch+1:>3} Summary", f"Loss: {avg_metrics['loss']:.4f}"]
        if velocity_loss_enabled:
            summary_parts.append(f"Vel: {avg_metrics['velocity_loss']:.4f}")
        if kd_enabled:
            kd_avg = np.mean(epoch_metrics['kd_loss'])
            summary_parts.append(f"KD: {kd_avg:.4f}")
        summary_line = " | ".join(summary_parts)
        if val_metrics:
            summary_line += f" | [Val] GenePCC: {val_metrics['val_gene_pcc']:.4f}, SpotPCC: {val_metrics['val_spot_pcc']:.4f}"
        print(summary_line)
        
        # ====================================================================
        # Early Stopping  ()
        # ====================================================================
        if early_stop_enabled:
            current_train_loss = avg_metrics['loss']
            
            # 
            if current_train_loss < best_train_loss - early_stop_min_delta:
                best_train_loss = current_train_loss
                epochs_without_improvement = 0
                if early_stop_verbose:
                    print(f"  [EarlyStop] New best train loss: {best_train_loss:.6f}")
            else:
                epochs_without_improvement += 1
                if early_stop_verbose:
                    print(f"  [EarlyStop] No improvement for {epochs_without_improvement}/{early_stop_patience} epochs (best: {best_train_loss:.6f})")
            
            # 
            if epochs_without_improvement >= early_stop_patience:
                print(f"\n{'='*80}")
                print(f"[Early Stopping] Training stopped at epoch {epoch+1}")
                print(f"  - No improvement for {early_stop_patience} consecutive epochs")
                print(f"  - Best train loss: {best_train_loss:.6f}")
                print(f"{'='*80}")
                early_stopped = True
                break
        
        print(f"{'='*80}")
    
    # 
    if early_stopped:
        print(f"\n[Training] Stopped early at epoch {epoch+1}")
    else:
        print(f"\n[Training] Completed all {epochs} epochs")
    print(f"[Best Model] Epoch {best_epoch}, Val GenePCC: {best_val_gene_pcc:.4f}")

    # ======================================================================
    # Build run-level true spot participation statistics
    # ======================================================================
    epochs_completed = len(participation_tracker['epoch_stats'])
    samples_participation = {}

    total_unique_lr_spots_seen = 0
    total_unique_hr_spots_seen = 0
    total_lr_spot_exposures = 0
    total_hr_spot_exposures = 0

    for sample_id, info in participation_tracker['regular_samples'].items():
        sample_summary = {
            'sample_id': sample_id,
            'mode': 'regular',
            'n_lr_total': int(info['n_lr_total']),
            'n_hr_total': int(info['n_hr_total']),
            'n_lr_per_epoch': int(info['n_lr_per_epoch']),
            'n_hr_per_epoch': int(info['n_hr_per_epoch']),
            'n_lr_unique_seen': int(info['n_lr_unique_seen']),
            'n_hr_unique_seen': int(info['n_hr_unique_seen']),
            'lr_coverage_ratio': 1.0,
            'hr_coverage_ratio': 1.0,
            'n_lr_exposure_total': int(info['n_lr_per_epoch']) * epochs_completed,
            'n_hr_exposure_total': int(info['n_hr_per_epoch']) * epochs_completed,
        }
        samples_participation[sample_id] = sample_summary
        total_unique_lr_spots_seen += sample_summary['n_lr_unique_seen']
        total_unique_hr_spots_seen += sample_summary['n_hr_unique_seen']
        total_lr_spot_exposures += sample_summary['n_lr_exposure_total']
        total_hr_spot_exposures += sample_summary['n_hr_exposure_total']

    for sample_id, info in participation_tracker['lazy_samples'].items():
        n_lr_unique_seen = int(np.count_nonzero(info['lr_seen_mask']))
        n_hr_unique_seen = int(np.count_nonzero(info['hr_seen_mask']))

        n_lr_exposure_total = sum(int(v['n_lr_selected']) for v in info['epochs'].values())
        n_hr_exposure_total = sum(int(v['n_hr_selected']) for v in info['epochs'].values())

        sample_summary = {
            'sample_id': sample_id,
            'mode': 'lazy',
            'n_lr_total': int(info['n_lr_total']),
            'n_hr_total': int(info['n_hr_total']),
            'n_lr_per_epoch_target': int(info['n_lr_per_epoch_target']),
            'n_lr_unique_seen': n_lr_unique_seen,
            'n_hr_unique_seen': n_hr_unique_seen,
            'lr_coverage_ratio': float(n_lr_unique_seen / max(1, info['n_lr_total'])),
            'hr_coverage_ratio': float(n_hr_unique_seen / max(1, info['n_hr_total'])),
            'n_lr_exposure_total': int(n_lr_exposure_total),
            'n_hr_exposure_total': int(n_hr_exposure_total),
            'epochs': info['epochs'],
        }
        samples_participation[sample_id] = sample_summary
        total_unique_lr_spots_seen += sample_summary['n_lr_unique_seen']
        total_unique_hr_spots_seen += sample_summary['n_hr_unique_seen']
        total_lr_spot_exposures += sample_summary['n_lr_exposure_total']
        total_hr_spot_exposures += sample_summary['n_hr_exposure_total']

    last_epoch_total_lr_spots = int(participation_tracker['epoch_stats'][-1]['total_lr_spots']) if participation_tracker['epoch_stats'] else 0
    last_epoch_total_hr_spots = int(participation_tracker['epoch_stats'][-1]['total_hr_spots']) if participation_tracker['epoch_stats'] else 0

    spot_participation_summary = {
        'run_summary': {
            'epochs_completed': int(epochs_completed),
            'total_unique_lr_spots_seen': int(total_unique_lr_spots_seen),
            'total_unique_hr_spots_seen': int(total_unique_hr_spots_seen),
            'total_lr_spot_exposures': int(total_lr_spot_exposures),
            'total_hr_spot_exposures': int(total_hr_spot_exposures),
            'last_epoch_total_lr_spots': int(last_epoch_total_lr_spots),
            'last_epoch_total_hr_spots': int(last_epoch_total_hr_spots),
        },
        'epoch_stats': participation_tracker['epoch_stats'],
        'samples': samples_participation,
    }

    print(f"\nSpot Participation Summary:")
    print(f"  Epochs completed: {epochs_completed}")
    print(f"  Unique LR spots seen in run: {total_unique_lr_spots_seen:,}")
    print(f"  Unique HR spots seen in run: {total_unique_hr_spots_seen:,}")
    print(f"  Last epoch spots: {last_epoch_total_lr_spots:,} LR, {last_epoch_total_hr_spots:,} HR")
    
    # ==========================================================================
    # Phase 4: Save Model
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Phase 4: Saving Model")
    print("-" * 70)
    
    os.makedirs(output_dir, exist_ok=True)

    spot_stats_path = os.path.join(output_dir, 'training_spot_participation.json')
    with open(spot_stats_path, 'w') as f:
        json.dump(spot_participation_summary, f, indent=2)
    print(f"Spot participation stats saved: {spot_stats_path}")
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'n_genes': n_genes,
        'latent_dim': latent_dim,
        'flow_config': flow_config,
        'training_samples': [d['sample_id'] for d in all_samples_data],
        'test_samples': list(test_samples.keys()) if test_samples else [],
        'n_samples': len(all_samples_data),
        'total_lr_spots': total_lr,
        'total_hr_spots': total_hr,
        'history': trainer.history
    }
    
    model_path = os.path.join(output_dir, 'flow_matching_model.pt')
    torch.save(save_dict, model_path)
    print(f"Model saved: {model_path}")
    
    info = {
        'timestamp': datetime.now().isoformat(),
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'n_genes': n_genes,
        'latent_dim': latent_dim,
        'n_samples': len(all_samples_data),
        'training_samples': [d['sample_id'] for d in all_samples_data],
        'final_loss': float(trainer.history['epoch_loss'][-1]),
        # PCCt>0.7
        'final_gene_pcc_train': float(trainer.history['gene_pcc'][-1]) if trainer.history['gene_pcc'] else 0.0,
        # PCCODE
        'final_gene_pcc_val': float(trainer.history['val_gene_pcc'][-1]) if trainer.history['val_gene_pcc'] else 0.0,
        'final_spot_pcc_val': float(trainer.history['val_spot_pcc'][-1]) if trainer.history['val_spot_pcc'] else 0.0,
        'best_val_gene_pcc': float(best_val_gene_pcc),
        'best_epoch': best_epoch,
        'use_hr_spatial': flow_config.get('use_hr_spatial', True),
        'num_hr_neighbors': flow_config.get('num_hr_neighbors', 6),
        'use_latent_norm': use_latent_norm,  # 
        'epochs_completed': int(spot_participation_summary['run_summary']['epochs_completed']),
        'total_unique_lr_spots_seen': int(spot_participation_summary['run_summary']['total_unique_lr_spots_seen']),
        'total_unique_hr_spots_seen': int(spot_participation_summary['run_summary']['total_unique_hr_spots_seen']),
        'last_epoch_total_lr_spots': int(spot_participation_summary['run_summary']['last_epoch_total_lr_spots']),
        'last_epoch_total_hr_spots': int(spot_participation_summary['run_summary']['last_epoch_total_hr_spots']),
    }
    
    with open(os.path.join(output_dir, 'training_info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    
    # ==========================================================================
    # Phase 5: Testing
    # ==========================================================================
    all_test_metrics = {}
    
    if not args.skip_test and test_samples:
        print("\n" + "-" * 70)
        print("Phase 5: Testing")
        print("-" * 70)
        
        # :  >  > checkpoint > 50
        num_steps = args.num_steps or flow_config.get('num_steps', 50)
        print(f"Sampling steps: {num_steps}")
        
        for test_sample_id, test_cfg in test_samples.items():
            print(f"\nTesting: {test_sample_id}")
            
            try:
                test_data = load_sample_data(
                    sample_id=test_sample_id,
                    stage1_dir=stage1_dir,
                    hr_path=test_cfg.get('hr_path'),
                    config=config,
                    device=device,
                    use_latent_norm=use_latent_norm  # 
                )
                
                if test_data is None:
                    print(f"  [SKIP] Failed to load")
                    continue
                
                x_hr_pred = trainer.generate(
                    z_lr=test_data['z_lr'],
                    x_lr=test_data['x_lr'],
                    lr_coords=test_data['lr_coords'],
                    hr_coords=test_data['hr_coords'],
                    lr_hr_mapping=test_data['lr_hr_mapping'],
                    group_indices=test_data['group_indices'],
                    local_hr_indices=test_data['local_hr_indices'],
                    num_steps=num_steps
                )
                
                x_hr_true = test_data['x_hr'].numpy()
                
                if x_hr_pred.shape != x_hr_true.shape:
                    min_spots = min(x_hr_pred.shape[0], x_hr_true.shape[0])
                    min_genes = min(x_hr_pred.shape[1], x_hr_true.shape[1])
                    print(f"  [WARN] Shape mismatch: pred={x_hr_pred.shape}, true={x_hr_true.shape}")
                    x_hr_pred = x_hr_pred[:min_spots, :min_genes]
                    x_hr_true = x_hr_true[:min_spots, :min_genes]
                
                metrics = compute_metrics(x_hr_pred, x_hr_true)
                
                all_test_metrics[test_sample_id] = {
                    'metrics': metrics,
                    'n_lr': test_data['n_lr'],
                    'n_hr': test_data['n_hr']
                }
                
                print(f"  Results:")
                print(f"    PCC: {metrics['pcc']:.4f}, Spearman: {metrics['spearman']:.4f}")
                print(f"    Gene PCC: {metrics['gene_pcc']:.4f}, Spot PCC: {metrics['spot_pcc']:.4f}")
                print(f"    RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
                print(f"    SNR: {metrics['snr']:.2f} dB, SSIM: {metrics['ssim']:.4f}")
                
                pred_dir = os.path.join(output_dir, 'predictions', test_sample_id)
                os.makedirs(pred_dir, exist_ok=True)
                np.savez(
                    os.path.join(pred_dir, 'predictions.npz'),
                    x_hr_pred=x_hr_pred,
                    x_hr_true=x_hr_true
                )
                
            except Exception as e:
                print(f"  [ERROR] {e}")
                import traceback
                traceback.print_exc()
        
        if all_test_metrics:
            print("\n" + "-" * 40)
            print("Test Results Summary")
            print("-" * 40)
            
            avg_pcc = np.mean([m['metrics']['pcc'] for m in all_test_metrics.values()])
            avg_gene_pcc = np.mean([m['metrics']['gene_pcc'] for m in all_test_metrics.values()])
            avg_spot_pcc = np.mean([m['metrics']['spot_pcc'] for m in all_test_metrics.values()])
            avg_snr = np.mean([m['metrics']['snr'] for m in all_test_metrics.values()])
            avg_ssim = np.mean([m['metrics']['ssim'] for m in all_test_metrics.values()])
            
            print(f"Samples tested: {len(all_test_metrics)}")
            print(f"Average PCC: {avg_pcc:.4f}")
            print(f"Average Gene PCC: {avg_gene_pcc:.4f}")
            print(f"Average Spot PCC: {avg_spot_pcc:.4f}")
            print(f"Average SNR: {avg_snr:.2f} dB")
            print(f"Average SSIM: {avg_ssim:.4f}")
            
            with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
                json.dump(all_test_metrics, f, indent=2)
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"Final loss: {trainer.history['epoch_loss'][-1]:.6f}")
    print(f"Model saved to: {output_dir}")
    print(f"\nKey features of Flow Matching approach:")
    print(f"   OT-Flow: deterministic paths for better convergence")
    print(f"   HR Spatial Prior: neighborhood-aware predictions")
    print(f"   Faster sampling ({flow_config.get('num_steps', 50)} steps vs 100+ for diffusion)")
    print(f"   Gene-specific spatial patterns")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
