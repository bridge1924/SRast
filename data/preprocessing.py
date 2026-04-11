"""
SRast Data Preprocessing Module

This module handles data loading, downsampling, normalization, and PCA transformation
for spatial transcriptomics data.
"""

import numpy as np
import scanpy as sc
import anndata as ad
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy.sparse import issparse
from typing import Tuple, Dict, Optional, Union
import warnings
import os
import pickle


class SpatialDownsampler:
    """
    Downsampler for spatial transcriptomics data.
    
    Merges neighboring HR spots into LR spots based on spatial coordinates.
    """
    
    def __init__(self, ks: int = 4, method: str = "spatial_clustering"):
        """
        Initialize the downsampler.
        
        Args:
            ks: Number of HR spots to merge into one LR spot
            method: Downsampling method ('spatial_clustering' or 'grid')
        """
        self.ks = ks
        self.method = method
        self.hr_to_lr_mapping = None
        self.lr_to_hr_mapping = None
        self.lr_coordinates = None
        
    def downsample(self, adata: ad.AnnData) -> Tuple[ad.AnnData, Dict]:
        """
        Downsample HR data to LR data.
        
        Args:
            adata: AnnData object with HR data
            
        Returns:
            lr_adata: Downsampled LR AnnData
            mapping_info: Dictionary containing mapping information
        """
        if 'spatial' not in adata.obsm:
            raise ValueError("AnnData must contain spatial coordinates in .obsm['spatial']")
        
        spatial_coords = adata.obsm['spatial']
        n_spots = adata.n_obs
        n_lr_spots = max(1, n_spots // self.ks)
        
        if self.method == "spatial_clustering":
            lr_adata, mapping_info = self._downsample_clustering(adata, n_lr_spots)
        elif self.method == "grid":
            lr_adata, mapping_info = self._downsample_grid(adata, n_lr_spots)
        else:
            raise ValueError(f"Unknown downsampling method: {self.method}")
        
        self.hr_to_lr_mapping = mapping_info['hr_to_lr']
        self.lr_to_hr_mapping = mapping_info['lr_to_hr']
        self.lr_coordinates = lr_adata.obsm['spatial']
        
        return lr_adata, mapping_info
    
    def _downsample_clustering(self, adata: ad.AnnData, n_lr_spots: int) -> Tuple[ad.AnnData, Dict]:
        """
        Downsample using spatial clustering (KMeans).
        
        Args:
            adata: HR AnnData object
            n_lr_spots: Target number of LR spots
            
        Returns:
            lr_adata: Downsampled LR AnnData
            mapping_info: Mapping information
        """
        spatial_coords = adata.obsm['spatial']
        
        # Perform KMeans clustering on spatial coordinates
        kmeans = KMeans(n_clusters=n_lr_spots, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(spatial_coords)
        
        # Create mapping dictionaries
        hr_to_lr = cluster_labels.copy()
        lr_to_hr = {i: np.where(cluster_labels == i)[0].tolist() for i in range(n_lr_spots)}
        
        # Aggregate gene expression
        X = adata.X.toarray() if issparse(adata.X) else adata.X
        lr_X = np.zeros((n_lr_spots, X.shape[1]))
        lr_coords = np.zeros((n_lr_spots, spatial_coords.shape[1]))
        
        for i in range(n_lr_spots):
            hr_indices = lr_to_hr[i]
            if len(hr_indices) > 0:
                # Sum expression values for merged spots
                lr_X[i] = X[hr_indices].sum(axis=0)
                # Average spatial coordinates
                lr_coords[i] = spatial_coords[hr_indices].mean(axis=0)
        
        # Create LR AnnData
        lr_adata = ad.AnnData(X=lr_X, var=adata.var.copy())
        lr_adata.obsm['spatial'] = lr_coords
        lr_adata.obs_names = [f"LR_{i}" for i in range(n_lr_spots)]
        
        mapping_info = {
            'hr_to_lr': hr_to_lr,
            'lr_to_hr': lr_to_hr,
            'n_hr_spots': adata.n_obs,
            'n_lr_spots': n_lr_spots,
            'method': 'spatial_clustering'
        }
        
        return lr_adata, mapping_info
    
    def _downsample_grid(self, adata: ad.AnnData, n_lr_spots: int) -> Tuple[ad.AnnData, Dict]:
        """
        Downsample using grid-based method.
        
        Args:
            adata: HR AnnData object
            n_lr_spots: Target number of LR spots
            
        Returns:
            lr_adata: Downsampled LR AnnData
            mapping_info: Mapping information
        """
        spatial_coords = adata.obsm['spatial']
        
        # Calculate grid size
        grid_size = int(np.ceil(np.sqrt(n_lr_spots)))
        
        # Normalize coordinates to [0, 1]
        coords_min = spatial_coords.min(axis=0)
        coords_max = spatial_coords.max(axis=0)
        coords_range = coords_max - coords_min
        coords_range[coords_range == 0] = 1  # Avoid division by zero
        normalized_coords = (spatial_coords - coords_min) / coords_range
        
        # Assign spots to grid cells
        grid_indices = (normalized_coords * (grid_size - 1)).astype(int)
        grid_indices = np.clip(grid_indices, 0, grid_size - 1)
        cell_ids = grid_indices[:, 0] * grid_size + grid_indices[:, 1]
        
        # Get unique cells and remap
        unique_cells = np.unique(cell_ids)
        cell_remap = {old: new for new, old in enumerate(unique_cells)}
        hr_to_lr = np.array([cell_remap[c] for c in cell_ids])
        actual_n_lr = len(unique_cells)
        
        lr_to_hr = {i: np.where(hr_to_lr == i)[0].tolist() for i in range(actual_n_lr)}
        
        # Aggregate gene expression
        X = adata.X.toarray() if issparse(adata.X) else adata.X
        lr_X = np.zeros((actual_n_lr, X.shape[1]))
        lr_coords = np.zeros((actual_n_lr, spatial_coords.shape[1]))
        
        for i in range(actual_n_lr):
            hr_indices = lr_to_hr[i]
            if len(hr_indices) > 0:
                lr_X[i] = X[hr_indices].sum(axis=0)
                lr_coords[i] = spatial_coords[hr_indices].mean(axis=0)
        
        # Create LR AnnData
        lr_adata = ad.AnnData(X=lr_X, var=adata.var.copy())
        lr_adata.obsm['spatial'] = lr_coords
        lr_adata.obs_names = [f"LR_{i}" for i in range(actual_n_lr)]
        
        mapping_info = {
            'hr_to_lr': hr_to_lr,
            'lr_to_hr': lr_to_hr,
            'n_hr_spots': adata.n_obs,
            'n_lr_spots': actual_n_lr,
            'method': 'grid'
        }
        
        return lr_adata, mapping_info


class DataPreprocessor:
    """
    Preprocessor for spatial transcriptomics data.
    
    Handles normalization, log transformation, HVG selection, and PCA dimensionality reduction.
    Now supports returning HVG expression as reconstruction target.
    """
    
    def __init__(
        self,
        normalize: bool = True,
        log1p: bool = True,
        n_pca_components: int = 50,
        n_hvg: int = 3000,
        random_state: int = 42
    ):
        """
        Initialize the preprocessor.
        
        Args:
            normalize: Whether to normalize data
            log1p: Whether to apply log1p transformation
            n_pca_components: Number of PCA components
            n_hvg: Number of highly variable genes for reconstruction target
            random_state: Random state for reproducibility
        """
        self.normalize = normalize
        self.log1p = log1p
        self.n_pca_components = n_pca_components
        self.n_hvg = n_hvg
        self.random_state = random_state
        
        self.pca = None
        self.pca_components_ = None
        self.pca_mean_ = None
        self.hvg_names_ = None  # Store HVG names for reconstruction
        self.hvg_indices_ = None  # Store HVG indices
        # HVG-PCA: PCA model trained on HVG subset (for bilinear interpolation in Stage 2)
        self.hvg_pca = None
        self.hvg_pca_components_ = None
        self.hvg_pca_mean_ = None
        self._fitted = False
        
    def fit(self, adata: ad.AnnData) -> 'DataPreprocessor':
        """
        Fit the preprocessor on data.
        
        Args:
            adata: AnnData object to fit on
            
        Returns:
            self: Fitted preprocessor
        """
        # Create a copy to avoid modifying original
        adata_copy = adata.copy()
        
        # Normalize
        if self.normalize:
            sc.pp.normalize_total(adata_copy, target_sum=1e4)
        
        # Log1p
        if self.log1p:
            sc.pp.log1p(adata_copy)
        
        # Select highly variable genes
        #  n_hvg 
        if adata_copy.n_vars < self.n_hvg:
            raise ValueError(
                f"Sample has only {adata_copy.n_vars} genes, but n_hvg={self.n_hvg} is required. "
                f"Please use a sample with more genes or reduce n_hvg."
            )
        sc.pp.highly_variable_genes(adata_copy, n_top_genes=self.n_hvg, flavor='seurat_v3', span=0.3)
        
        #  n_hvg 
        hvg_mask = adata_copy.var['highly_variable'].values
        n_selected = hvg_mask.sum()
        
        if n_selected != self.n_hvg:
            print(f"[INFO] seurat_v3 selected {n_selected} HVG, forcing to {self.n_hvg}...")
            # 
            hvg_indices_all = np.where(hvg_mask)[0]
            
            if n_selected > self.n_hvg:
                #  highly_variable_rank  n_hvg 
                if 'highly_variable_rank' in adata_copy.var.columns:
                    ranks = adata_copy.var['highly_variable_rank'].values[hvg_indices_all]
                    sorted_idx = np.argsort(ranks)
                    hvg_indices_all = hvg_indices_all[sorted_idx[:self.n_hvg]]
                else:
                    #  rank  n_hvg 
                    hvg_indices_all = hvg_indices_all[:self.n_hvg]
            
            #  mask
            hvg_mask = np.zeros_like(hvg_mask, dtype=bool)
            hvg_mask[hvg_indices_all] = True
            adata_copy.var['highly_variable'] = hvg_mask
        
        self.hvg_names_ = adata_copy.var_names[adata_copy.var['highly_variable']].tolist()
        self.hvg_indices_ = np.where(adata_copy.var['highly_variable'].values)[0]
        
        # 
        assert len(self.hvg_names_) == self.n_hvg, f"HVG count mismatch: {len(self.hvg_names_)} vs {self.n_hvg}"
        
        print(f"Selected {len(self.hvg_names_)} highly variable genes for reconstruction target")
        
        # Memory-efficient processing: only extract and densify HVG subset
        # This avoids loading the full (n_spots x n_genes) dense matrix
        
        # Extract HVG subset FIRST (still sparse)
        X_hvg_sparse = adata_copy.X[:, self.hvg_indices_]
        X_hvg = X_hvg_sparse.toarray() if issparse(X_hvg_sparse) else np.asarray(X_hvg_sparse)
        
        print(f"      HVG matrix shape: {X_hvg.shape} (memory-efficient: only HVG extracted)")
        
        # Fit PCA on HVG genes only (memory-efficient approach)
        # Note: Previously we fitted PCA on all genes, but for large datasets this causes OOM
        # Using HVG for PCA is sufficient since reconstruction target is also HVG
        self.pca = PCA(n_components=self.n_pca_components, random_state=self.random_state)
        self.pca.fit(X_hvg)
        
        # Save components and mean for inverse transform
        self.pca_components_ = self.pca.components_.copy()
        self.pca_mean_ = self.pca.mean_.copy()
        
        # HVG-PCA is now the same as the main PCA (since both use HVG)
        self.hvg_pca = self.pca
        self.hvg_pca_components_ = self.pca_components_
        self.hvg_pca_mean_ = self.pca_mean_
        print(f"Fitted PCA on {X_hvg.shape[1]} HVG genes (n_components={self.n_pca_components})")
        
        self._fitted = True
        
        return self
    
    def transform(self, adata: ad.AnnData) -> np.ndarray:
        """
        Transform data using fitted preprocessor.
        
        Args:
            adata: AnnData object to transform
            
        Returns:
            pca_features: PCA-transformed features (based on HVG genes)
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        # Create a copy
        adata_copy = adata.copy()
        
        # Normalize
        if self.normalize:
            sc.pp.normalize_total(adata_copy, target_sum=1e4)
        
        # Log1p
        if self.log1p:
            sc.pp.log1p(adata_copy)
        
        # Memory-efficient: only extract HVG subset before converting to dense
        X_hvg_sparse = adata_copy.X[:, self.hvg_indices_]
        X_hvg = X_hvg_sparse.toarray() if issparse(X_hvg_sparse) else np.asarray(X_hvg_sparse)
        
        # Transform using PCA (fitted on HVG)
        pca_features = self.pca.transform(X_hvg)
        
        return pca_features
    
    def transform_hvg(self, hvg_expression: np.ndarray) -> np.ndarray:
        """
        Transform HVG expression using HVG-PCA model.
        
        This method is used for bilinear interpolation results in Stage 2,
        where we only have HVG expression (not full gene expression).
        
        Args:
            hvg_expression: HVG expression matrix (N x n_hvg)
            
        Returns:
            pca_features: PCA-transformed features (N x n_components)
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before transform_hvg")
        
        if self.hvg_pca is None:
            raise RuntimeError("HVG-PCA model not available. Please re-run Stage 1 training.")
        
        return self.hvg_pca.transform(hvg_expression)

    def get_hvg_expression(self, adata: ad.AnnData) -> np.ndarray:
        """
        Get HVG expression values (for use as reconstruction target).
        
        Args:
            adata: AnnData object
            
        Returns:
            hvg_expression: Expression of HVG genes (N x n_hvg)
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before getting HVG expression")
        
        # Create a copy and apply same preprocessing
        adata_copy = adata.copy()
        
        # Normalize
        if self.normalize:
            sc.pp.normalize_total(adata_copy, target_sum=1e4)
        
        # Log1p
        if self.log1p:
            sc.pp.log1p(adata_copy)
        
        # Extract HVG columns FIRST (before converting to dense)
        # This is memory-efficient: only extract n_hvg columns instead of all genes
        X_hvg = adata_copy.X[:, self.hvg_indices_]
        
        # Convert to dense array
        hvg_expression = X_hvg.toarray() if issparse(X_hvg) else np.asarray(X_hvg)
        
        return hvg_expression
    
    def fit_transform(self, adata: ad.AnnData) -> np.ndarray:
        """
        Fit and transform data.
        
        Args:
            adata: AnnData object
            
        Returns:
            pca_features: PCA-transformed features
        """
        self.fit(adata)
        return self.transform(adata)
    
    def inverse_transform(self, pca_features: np.ndarray) -> np.ndarray:
        """
        Inverse transform PCA features back to original gene expression space.
        
        Args:
            pca_features: PCA features (N x n_components)
            
        Returns:
            gene_expression: Reconstructed gene expression
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before inverse_transform")
        
        # Manual inverse transform using saved components
        # X_reconstructed = X_pca @ components + mean
        gene_expression = pca_features @ self.pca_components_ + self.pca_mean_
        
        return gene_expression
    
    def inverse_transform_full(self, pca_features: np.ndarray) -> np.ndarray:
        """
        Full inverse transform including exp-1 to reverse log1p.
        
        Args:
            pca_features: PCA features
            
        Returns:
            gene_counts: Reconstructed gene counts
        """
        gene_expression = self.inverse_transform(pca_features)
        
        # Reverse log1p if applied
        if self.log1p:
            gene_counts = np.expm1(gene_expression)
            gene_counts = np.clip(gene_counts, 0, None)  # Ensure non-negative
        else:
            gene_counts = gene_expression
        
        return gene_counts
    
    def save(self, path: str):
        """
        Save preprocessor state.
        
        Args:
            path: Path to save file
        """
        state = {
            'normalize': self.normalize,
            'log1p': self.log1p,
            'n_pca_components': self.n_pca_components,
            'n_hvg': self.n_hvg,
            'random_state': self.random_state,
            'pca_components_': self.pca_components_,
            'pca_mean_': self.pca_mean_,
            'hvg_names_': self.hvg_names_,
            'hvg_indices_': self.hvg_indices_,
            # HVG-PCA for Stage 2 bilinear interpolation
            'hvg_pca_components_': self.hvg_pca_components_,
            'hvg_pca_mean_': self.hvg_pca_mean_,
            '_fitted': self._fitted
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, path: str) -> 'DataPreprocessor':
        """
        Load preprocessor from file.
        
        Args:
            path: Path to saved file
            
        Returns:
            preprocessor: Loaded preprocessor
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        preprocessor = cls(
            normalize=state['normalize'],
            log1p=state['log1p'],
            n_pca_components=state['n_pca_components'],
            n_hvg=state.get('n_hvg', 3000),
            random_state=state['random_state']
        )
        preprocessor.pca_components_ = state['pca_components_']
        preprocessor.pca_mean_ = state['pca_mean_']
        preprocessor.hvg_names_ = state.get('hvg_names_', None)
        preprocessor.hvg_indices_ = state.get('hvg_indices_', None)
        preprocessor._fitted = state['_fitted']
        
        # Load HVG-PCA if available
        preprocessor.hvg_pca_components_ = state.get('hvg_pca_components_', None)
        preprocessor.hvg_pca_mean_ = state.get('hvg_pca_mean_', None)
        
        # Reconstruct PCA object
        if preprocessor._fitted:
            preprocessor.pca = PCA(n_components=preprocessor.n_pca_components)
            preprocessor.pca.components_ = preprocessor.pca_components_
            preprocessor.pca.mean_ = preprocessor.pca_mean_
            preprocessor.pca.n_features_in_ = preprocessor.pca_mean_.shape[0]
            
            # Reconstruct HVG-PCA object if available
            if preprocessor.hvg_pca_components_ is not None:
                preprocessor.hvg_pca = PCA(n_components=preprocessor.n_pca_components)
                preprocessor.hvg_pca.components_ = preprocessor.hvg_pca_components_
                preprocessor.hvg_pca.mean_ = preprocessor.hvg_pca_mean_
                preprocessor.hvg_pca.n_features_in_ = preprocessor.hvg_pca_mean_.shape[0]
        
        return preprocessor


def prepare_training_data(
    adata: ad.AnnData,
    ks: int = 4,
    n_pca_components: int = 50,
    n_hvg: int = 3000,
    normalize: bool = True,
    log1p: bool = True,
    save_dir: Optional[str] = None
) -> Dict:
    """
    Prepare training data for SRast model.
    
    This function performs the complete data preparation pipeline:
    1. Downsampling to create LR data
    2. Normalization and log transformation
    3. PCA dimensionality reduction for encoder input
    4. HVG expression extraction for decoder reconstruction target
    
    Args:
        adata: Input AnnData object with HR data
        ks: Number of HR spots to merge
        n_pca_components: Number of PCA components (encoder input)
        n_hvg: Number of highly variable genes (decoder output target)
        normalize: Whether to normalize
        log1p: Whether to apply log1p
        save_dir: Directory to save preprocessor state
        
    Returns:
        data_dict: Dictionary containing:
            - hr_features: HR PCA features (encoder input)
            - lr_features: LR PCA features (encoder input)
            - hr_hvg_expression: HR HVG expression (reconstruction target)
            - lr_hvg_expression: LR HVG expression (reconstruction target)
            - hr_coords: HR spatial coordinates
            - lr_coords: LR spatial coordinates
            - mapping_info: HR-LR mapping information
            - preprocessor: Fitted DataPreprocessor
    """
    print("=" * 50)
    print("Preparing training data...")
    print("=" * 50)
    
    # Store original HR coordinates
    hr_coords = adata.obsm['spatial'].copy()
    print(f"HR data: {adata.n_obs} spots, {adata.n_vars} genes")
    
    # Step 1: Downsampling
    print(f"\nStep 1: Downsampling (Ks={ks})...")
    downsampler = SpatialDownsampler(ks=ks, method="spatial_clustering")
    lr_adata, mapping_info = downsampler.downsample(adata)
    lr_coords = lr_adata.obsm['spatial'].copy()
    print(f"LR data: {lr_adata.n_obs} spots")
    
    # Step 2: Fit preprocessor on LR data (for consistency)
    print(f"\nStep 2: Fitting preprocessor on LR data (n_hvg={n_hvg})...")
    preprocessor = DataPreprocessor(
        normalize=normalize,
        log1p=log1p,
        n_pca_components=n_pca_components,
        n_hvg=n_hvg
    )
    lr_features = preprocessor.fit_transform(lr_adata)
    print(f"LR PCA features shape: {lr_features.shape}")
    
    # Step 3: Transform HR data using same preprocessor
    print(f"\nStep 3: Transforming HR data...")
    hr_features = preprocessor.transform(adata)
    print(f"HR PCA features shape: {hr_features.shape}")
    
    # Step 4: Get HVG expression for reconstruction target
    print(f"\nStep 4: Extracting HVG expression as reconstruction target...")
    lr_hvg_expression = preprocessor.get_hvg_expression(lr_adata)
    hr_hvg_expression = preprocessor.get_hvg_expression(adata)
    print(f"LR HVG expression shape: {lr_hvg_expression.shape}")
    print(f"HR HVG expression shape: {hr_hvg_expression.shape}")
    
    # Save preprocessor if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        preprocessor.save(os.path.join(save_dir, 'preprocessor.pkl'))
        
        # Also save mapping info
        with open(os.path.join(save_dir, 'mapping_info.pkl'), 'wb') as f:
            pickle.dump(mapping_info, f)
        print(f"\nPreprocessor and mapping saved to {save_dir}")
    
    data_dict = {
        'hr_features': hr_features,
        'lr_features': lr_features,
        'hr_hvg_expression': hr_hvg_expression,
        'lr_hvg_expression': lr_hvg_expression,
        'hr_coords': hr_coords,
        'lr_coords': lr_coords,
        'mapping_info': mapping_info,
        'preprocessor': preprocessor,
        'hr_adata': adata,
        'lr_adata': lr_adata
    }
    
    print("\nData preparation complete!")
    print("=" * 50)
    
    return data_dict


def load_h5ad(path: str) -> ad.AnnData:
    """
    Load h5ad file with validation.
    
    Args:
        path: Path to h5ad file
        
    Returns:
        adata: Loaded AnnData object
    """
    print(f"Loading data from {path}...")
    adata = sc.read_h5ad(path)
    
    # Validate required fields
    if 'spatial' not in adata.obsm:
        raise ValueError("AnnData must contain spatial coordinates in .obsm['spatial']")
    
    print(f"Loaded: {adata.n_obs} spots, {adata.n_vars} genes")
    print(f"Spatial coords shape: {adata.obsm['spatial'].shape}")
    
    return adata
