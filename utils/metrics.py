"""
Evaluation Metrics Module

This module provides metrics for evaluating spatial transcriptomics super-resolution.
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.special import rel_entr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Optional, Tuple, Union, List
import warnings


# =============================================================================
# : Fractional Deviation & JS Divergence
# =============================================================================

def compute_fractional_deviation(
    x_lr: np.ndarray,
    x_hr_pred: np.ndarray,
    lr_hr_mapping: np.ndarray
) -> float:
    """
     Fractional Deviation ()
    
     sum(x_hr_pred)  x_lr 
    
    Dev = (1/N_LR) * sum_{i=1}^{N_LR} ||x_i^LR - sum_{j in C(i)} x_j^HR_pred||_1 / ||x_i^LR||_1
    
    Args:
        x_lr: LR  (N_LR x N_genes)
        x_hr_pred:  HR  (N_HR x N_genes)
        lr_hr_mapping: LR-HR shape (2, N_HR) LR  HR 
    
    Returns:
        fractional_deviation: 
    """
    eps = 1e-8
    n_lr = x_lr.shape[0]
    
    #  LR -> HR 
    lr_to_hr = {}
    for hr_idx in range(lr_hr_mapping.shape[1]):
        lr_idx = lr_hr_mapping[0, hr_idx]
        if lr_idx not in lr_to_hr:
            lr_to_hr[lr_idx] = []
        lr_to_hr[lr_idx].append(hr_idx)
    
    deviations = []
    for i in range(n_lr):
        if i not in lr_to_hr:
            continue
        hr_indices = lr_to_hr[i]
        
        #  LR spot  HR spots 
        hr_sum = np.sum(x_hr_pred[hr_indices, :], axis=0)
        
        #  L1 
        lr_vec = x_lr[i, :]
        l1_diff = np.sum(np.abs(lr_vec - hr_sum))
        l1_lr = np.sum(np.abs(lr_vec)) + eps
        
        deviation = l1_diff / l1_lr
        deviations.append(deviation)
    
    return float(np.mean(deviations)) if deviations else 0.0


def compute_js_divergence(
    pred: np.ndarray,
    true: np.ndarray,
    n_bins: int = 100
) -> float:
    """
     JS Divergence (JS )
    
    
    
    D_JS(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)
     M = 0.5 * (P + Q)
    
    Args:
        pred:  HR  (N_HR x N_genes)
        true:  HR  (N_HR x N_genes)
        n_bins:  bin 
    
    Returns:
        js_divergence: JS  ( [0, 1]0 )
    """
    eps = 1e-10
    
    #  NaN
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    
    valid_mask = ~(np.isnan(pred_flat) | np.isnan(true_flat) | 
                   np.isinf(pred_flat) | np.isinf(true_flat))
    pred_valid = pred_flat[valid_mask]
    true_valid = true_flat[valid_mask]
    
    if len(pred_valid) < 2:
        return 1.0
    
    #  bin 
    min_val = min(pred_valid.min(), true_valid.min())
    max_val = max(pred_valid.max(), true_valid.max())
    
    if max_val <= min_val:
        return 0.0
    
    bins = np.linspace(min_val, max_val, n_bins + 1)
    
    #  ()
    p_hist, _ = np.histogram(true_valid, bins=bins, density=True)
    q_hist, _ = np.histogram(pred_valid, bins=bins, density=True)
    
    # 
    p_hist = p_hist / (p_hist.sum() + eps)
    q_hist = q_hist / (q_hist.sum() + eps)
    
    #  eps  log(0)
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    
    #  M = 0.5 * (P + Q)
    m_hist = 0.5 * (p_hist + q_hist)
    
    #  KL 
    kl_pm = np.sum(rel_entr(p_hist, m_hist))
    kl_qm = np.sum(rel_entr(q_hist, m_hist))
    
    # JS 
    js_div = 0.5 * kl_pm + 0.5 * kl_qm
    
    return float(js_div) if not np.isnan(js_div) else 1.0


def compute_js_divergence_per_gene(
    pred: np.ndarray,
    true: np.ndarray,
    n_bins: int = 50
) -> float:
    """
     JS 
    
    Args:
        pred:  HR  (N_HR x N_genes)
        true:  HR  (N_HR x N_genes)
        n_bins:  bin 
    
    Returns:
        mean_js:  JS 
    """
    eps = 1e-10
    n_genes = min(pred.shape[1], true.shape[1])
    js_values = []
    
    for g in range(n_genes):
        pred_g = pred[:, g]
        true_g = true[:, g]
        
        valid_mask = ~(np.isnan(pred_g) | np.isnan(true_g) | 
                       np.isinf(pred_g) | np.isinf(true_g))
        pred_valid = pred_g[valid_mask]
        true_valid = true_g[valid_mask]
        
        if len(pred_valid) < 10:  # 
            continue
        
        min_val = min(pred_valid.min(), true_valid.min())
        max_val = max(pred_valid.max(), true_valid.max())
        
        if max_val <= min_val:
            js_values.append(0.0)
            continue
        
        bins = np.linspace(min_val, max_val, n_bins + 1)
        
        p_hist, _ = np.histogram(true_valid, bins=bins, density=True)
        q_hist, _ = np.histogram(pred_valid, bins=bins, density=True)
        
        p_hist = p_hist / (p_hist.sum() + eps) + eps
        q_hist = q_hist / (q_hist.sum() + eps) + eps
        
        m_hist = 0.5 * (p_hist + q_hist)
        
        kl_pm = np.sum(rel_entr(p_hist, m_hist))
        kl_qm = np.sum(rel_entr(q_hist, m_hist))
        
        js_g = 0.5 * kl_pm + 0.5 * kl_qm
        if not np.isnan(js_g):
            js_values.append(js_g)
    
    return float(np.mean(js_values)) if js_values else 1.0


# =============================================================================
# 
# =============================================================================

def compute_snr(pred: np.ndarray, true: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute Signal-to-Noise Ratio in dB.
    
    SNR = 10 * log10(signal_power / noise_power)
    
    Args:
        pred: Predicted values
        true: Ground truth values
        eps: Small constant to avoid division by zero
    
    Returns:
        SNR in dB
    """
    valid_mask = ~(np.isnan(pred) | np.isnan(true) | np.isinf(pred) | np.isinf(true))
    
    pred_flat = pred.flatten()[valid_mask.flatten()]
    true_flat = true.flatten()[valid_mask.flatten()]
    
    if len(pred_flat) < 2:
        return 0.0
    
    signal_power = np.mean(true_flat ** 2)
    noise_power = np.mean((pred_flat - true_flat) ** 2)
    
    if noise_power < eps:
        return 100.0  # Very high SNR
    
    snr = 10 * np.log10(signal_power / (noise_power + eps))
    return float(snr) if not np.isnan(snr) else 0.0


def compute_ssim(pred: np.ndarray, true: np.ndarray, C1: float = 0.01**2, C2: float = 0.03**2) -> float:
    """
    Compute Structural Similarity Index per spot and average.
    
    Args:
        pred: Predicted expression (N_spots x N_genes)
        true: Ground truth expression (N_spots x N_genes)
        C1, C2: Stability constants
        
    Returns:
        Average SSIM across spots
    """
    eps = 1e-8
    
    if pred.shape != true.shape:
        min_spots = min(pred.shape[0], true.shape[0])
        min_genes = min(pred.shape[1], true.shape[1])
        pred = pred[:min_spots, :min_genes]
        true = true[:min_spots, :min_genes]
    
    valid_mask = ~(np.isnan(pred) | np.isnan(true) | np.isinf(pred) | np.isinf(true))
    pred_valid = np.where(valid_mask, pred, 0)
    true_valid = np.where(valid_mask, true, 0)
    
    # Normalize
    pred_min, pred_max = pred_valid.min(), pred_valid.max()
    true_min, true_max = true_valid.min(), true_valid.max()
    
    pred_norm = (pred_valid - pred_min) / (pred_max - pred_min + eps) if pred_max > pred_min else pred_valid
    true_norm = (true_valid - true_min) / (true_max - true_min + eps) if true_max > true_min else true_valid
    
    # Per-spot SSIM
    spot_ssims = []
    for i in range(pred.shape[0]):
        pred_i, true_i = pred_norm[i], true_norm[i]
        mu_x, mu_y = np.mean(pred_i), np.mean(true_i)
        sigma_x_sq, sigma_y_sq = np.var(pred_i), np.var(true_i)
        sigma_xy = np.mean((pred_i - mu_x) * (true_i - mu_y))
        
        num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        den = (mu_x**2 + mu_y**2 + C1) * (sigma_x_sq + sigma_y_sq + C2)
        ssim_i = num / (den + eps)
        
        if not np.isnan(ssim_i):
            spot_ssims.append(ssim_i)
    
    return float(np.mean(spot_ssims)) if spot_ssims else 0.0


def compute_all_metrics(
    pred: np.ndarray, 
    target: np.ndarray,
    x_lr: np.ndarray = None,
    lr_hr_mapping_np: np.ndarray = None
) -> Dict[str, float]:
    """
    Compute all evaluation metrics including new metrics.
    
    This is the unified function for comprehensive evaluation,
    including Fractional Deviation and JS Divergence.
    
    Args:
        pred: Predicted expression (N_HR x N_genes)
        target: Ground truth expression (N_HR x N_genes)
        x_lr: LR expression (N_LR x N_genes), optional
        lr_hr_mapping_np: LR-HR mapping numpy array, optional
        
    Returns:
        Dict with all metrics
    """
    # Get basic metrics
    basic_metrics = compute_metrics(pred, target)
    
    # Add Fractional Deviation (if data available)
    if x_lr is not None and lr_hr_mapping_np is not None:
        frac_dev = compute_fractional_deviation(x_lr, pred, lr_hr_mapping_np)
    else:
        frac_dev = -1.0  # Mark as unavailable
    
    # Add JS Divergence
    js_div = compute_js_divergence(pred, target)
    js_div_gene = compute_js_divergence_per_gene(pred, target)
    
    basic_metrics.update({
        'fractional_deviation': float(frac_dev),
        'js_divergence': float(js_div),
        'js_divergence_gene': float(js_div_gene)
    })
    
    return basic_metrics


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    """
    Compute evaluation metrics in expression space.
    
    Unified function used by both training and inference.
    
    Returns:
        Dict with keys: pcc, spearman, rmse, mae, gene_pcc, spot_pcc, snr, ssim
    """
    results = {}
    
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    
    valid = ~(np.isnan(pred_flat) | np.isnan(true_flat) | 
              np.isinf(pred_flat) | np.isinf(true_flat))
    pred_valid = pred_flat[valid]
    true_valid = true_flat[valid]
    
    if len(pred_valid) < 3:
        return {'pcc': 0.0, 'spearman': 0.0, 'rmse': 0.0, 'mae': 0.0, 
                'spot_pcc': 0.0, 'gene_pcc': 0.0, 'snr': 0.0, 'ssim': 0.0}
    
    # Overall PCC
    try:
        pcc, _ = pearsonr(pred_valid, true_valid)
        results['pcc'] = float(pcc) if not np.isnan(pcc) else 0.0
    except:
        results['pcc'] = 0.0
    
    # Spearman
    try:
        spearman, _ = spearmanr(pred_valid, true_valid)
        results['spearman'] = float(spearman) if not np.isnan(spearman) else 0.0
    except:
        results['spearman'] = 0.0
    
    # RMSE & MAE
    results['rmse'] = float(np.sqrt(mean_squared_error(true_valid, pred_valid)))
    results['mae'] = float(mean_absolute_error(true_valid, pred_valid))
    
    # SNR & SSIM
    results['snr'] = compute_snr(pred, true)
    results['ssim'] = compute_ssim(pred, true)
    
    # Per-spot PCC
    spot_pccs = []
    for i in range(min(pred.shape[0], true.shape[0])):
        pred_i, true_i = pred[i], true[i]
        valid_i = ~(np.isnan(pred_i) | np.isnan(true_i))
        if valid_i.sum() > 2 and np.std(pred_i[valid_i]) > 1e-8 and np.std(true_i[valid_i]) > 1e-8:
            try:
                pcc_i, _ = pearsonr(pred_i[valid_i], true_i[valid_i])
                if not np.isnan(pcc_i):
                    spot_pccs.append(pcc_i)
            except:
                pass
    results['spot_pcc'] = float(np.mean(spot_pccs)) if spot_pccs else 0.0
    
    # Per-gene PCC
    gene_pccs = []
    for j in range(min(pred.shape[1], true.shape[1])):
        pred_j, true_j = pred[:, j], true[:, j]
        valid_j = ~(np.isnan(pred_j) | np.isnan(true_j))
        if valid_j.sum() > 2 and np.std(pred_j[valid_j]) > 1e-8 and np.std(true_j[valid_j]) > 1e-8:
            try:
                pcc_j, _ = pearsonr(pred_j[valid_j], true_j[valid_j])
                if not np.isnan(pcc_j):
                    gene_pccs.append(pcc_j)
            except:
                pass
    results['gene_pcc'] = float(np.mean(gene_pccs)) if gene_pccs else 0.0
    
    return results


# =============================================================================
# 
# =============================================================================

def pearson_correlation_coefficient(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    per_gene: bool = True,
    per_spot: bool = True
) -> Dict[str, float]:
    """
    Compute Pearson Correlation Coefficient.
    
    Args:
        y_true: Ground truth expression (N_spots x N_genes)
        y_pred: Predicted expression (N_spots x N_genes)
        per_gene: Compute PCC per gene and return average
        per_spot: Compute PCC per spot and return average
        
    Returns:
        pcc_dict: Dictionary with PCC values
    """
    results = {}
    
    # Flatten and compute overall PCC
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Remove NaN values
    mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    if mask.sum() > 0:
        overall_pcc, _ = pearsonr(y_true_flat[mask], y_pred_flat[mask])
        results['pcc_overall'] = overall_pcc
    else:
        results['pcc_overall'] = np.nan
    
    # Per-gene PCC
    if per_gene:
        gene_pccs = []
        for g in range(y_true.shape[1]):
            true_g = y_true[:, g]
            pred_g = y_pred[:, g]
            mask = ~(np.isnan(true_g) | np.isnan(pred_g))
            
            if mask.sum() > 2 and np.std(true_g[mask]) > 1e-8 and np.std(pred_g[mask]) > 1e-8:
                pcc, _ = pearsonr(true_g[mask], pred_g[mask])
                if not np.isnan(pcc):
                    gene_pccs.append(pcc)
        
        results['pcc_per_gene_mean'] = np.mean(gene_pccs) if gene_pccs else np.nan
        results['pcc_per_gene_median'] = np.median(gene_pccs) if gene_pccs else np.nan
        results['pcc_per_gene_std'] = np.std(gene_pccs) if gene_pccs else np.nan
    
    # Per-spot PCC
    if per_spot:
        spot_pccs = []
        for s in range(y_true.shape[0]):
            true_s = y_true[s, :]
            pred_s = y_pred[s, :]
            mask = ~(np.isnan(true_s) | np.isnan(pred_s))
            
            if mask.sum() > 2 and np.std(true_s[mask]) > 1e-8 and np.std(pred_s[mask]) > 1e-8:
                pcc, _ = pearsonr(true_s[mask], pred_s[mask])
                if not np.isnan(pcc):
                    spot_pccs.append(pcc)
        
        results['pcc_per_spot_mean'] = np.mean(spot_pccs) if spot_pccs else np.nan
        results['pcc_per_spot_median'] = np.median(spot_pccs) if spot_pccs else np.nan
        results['pcc_per_spot_std'] = np.std(spot_pccs) if spot_pccs else np.nan
    
    return results


def root_mean_square_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    per_gene: bool = True,
    per_spot: bool = True
) -> Dict[str, float]:
    """
    Compute Root Mean Square Error.
    
    Args:
        y_true: Ground truth expression
        y_pred: Predicted expression
        per_gene: Compute RMSE per gene
        per_spot: Compute RMSE per spot
        
    Returns:
        rmse_dict: Dictionary with RMSE values
    """
    results = {}
    
    # Overall RMSE
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() > 0:
        results['rmse_overall'] = np.sqrt(mean_squared_error(
            y_true[mask], y_pred[mask]
        ))
    else:
        results['rmse_overall'] = np.nan
    
    # Per-gene RMSE
    if per_gene:
        gene_rmses = []
        for g in range(y_true.shape[1]):
            true_g = y_true[:, g]
            pred_g = y_pred[:, g]
            mask = ~(np.isnan(true_g) | np.isnan(pred_g))
            
            if mask.sum() > 0:
                rmse = np.sqrt(mean_squared_error(true_g[mask], pred_g[mask]))
                gene_rmses.append(rmse)
        
        results['rmse_per_gene_mean'] = np.mean(gene_rmses) if gene_rmses else np.nan
        results['rmse_per_gene_median'] = np.median(gene_rmses) if gene_rmses else np.nan
    
    # Per-spot RMSE
    if per_spot:
        spot_rmses = []
        for s in range(y_true.shape[0]):
            true_s = y_true[s, :]
            pred_s = y_pred[s, :]
            mask = ~(np.isnan(true_s) | np.isnan(pred_s))
            
            if mask.sum() > 0:
                rmse = np.sqrt(mean_squared_error(true_s[mask], pred_s[mask]))
                spot_rmses.append(rmse)
        
        results['rmse_per_spot_mean'] = np.mean(spot_rmses) if spot_rmses else np.nan
        results['rmse_per_spot_median'] = np.median(spot_rmses) if spot_rmses else np.nan
    
    return results


def mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-8
) -> float:
    """
    Compute Mean Absolute Percentage Error.
    
    Args:
        y_true: Ground truth
        y_pred: Predictions
        epsilon: Small constant to avoid division by zero
        
    Returns:
        mape: MAPE value
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred)) & (np.abs(y_true) > epsilon)
    
    if mask.sum() == 0:
        return np.nan
    
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + epsilon))) * 100
    return mape


def spearman_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute Spearman correlation coefficient.
    
    Args:
        y_true: Ground truth
        y_pred: Predictions
        
    Returns:
        spearman_dict: Dictionary with Spearman correlation values
    """
    results = {}
    
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    if mask.sum() > 2:
        rho, pval = spearmanr(y_true_flat[mask], y_pred_flat[mask])
        results['spearman_overall'] = rho
        results['spearman_pvalue'] = pval
    else:
        results['spearman_overall'] = np.nan
        results['spearman_pvalue'] = np.nan
    
    return results


def structural_similarity_spatial(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    coords: np.ndarray,
    window_size: int = 5
) -> float:
    """
    Compute structural similarity considering spatial context.
    
    This is a simplified version that computes local correlations
    in spatial neighborhoods.
    
    Args:
        y_true: Ground truth expression
        y_pred: Predicted expression
        coords: Spatial coordinates
        window_size: Number of neighbors for local computation
        
    Returns:
        ssim: Spatial structural similarity score
    """
    from sklearn.neighbors import NearestNeighbors
    
    n_spots = y_true.shape[0]
    k = min(window_size, n_spots - 1)
    
    if k < 1:
        return np.nan
    
    # Find spatial neighbors
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(coords)
    _, indices = nn.kneighbors(coords)
    
    # Compute local correlations
    local_corrs = []
    
    for i in range(n_spots):
        neighbor_idx = indices[i]  # Include self
        
        true_local = y_true[neighbor_idx].flatten()
        pred_local = y_pred[neighbor_idx].flatten()
        
        mask = ~(np.isnan(true_local) | np.isnan(pred_local))
        
        if mask.sum() > 2 and np.std(true_local[mask]) > 1e-8:
            corr, _ = pearsonr(true_local[mask], pred_local[mask])
            if not np.isnan(corr):
                local_corrs.append(corr)
    
    return np.mean(local_corrs) if local_corrs else np.nan


def evaluate_reconstruction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    coords: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Comprehensive evaluation of reconstruction quality.
    
    Args:
        y_true: Ground truth expression (N_spots x N_genes)
        y_pred: Predicted expression (N_spots x N_genes)
        coords: Optional spatial coordinates for SSIM
        verbose: Print results
        
    Returns:
        metrics: Dictionary with all evaluation metrics
    """
    metrics = {}
    
    # Ensure numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # PCC
    pcc_results = pearson_correlation_coefficient(y_true, y_pred)
    metrics.update(pcc_results)
    
    # RMSE
    rmse_results = root_mean_square_error(y_true, y_pred)
    metrics.update(rmse_results)
    
    # MAE
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() > 0:
        metrics['mae_overall'] = mean_absolute_error(y_true[mask], y_pred[mask])
    
    # MAPE
    metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
    
    # Spearman
    spearman_results = spearman_correlation(y_true, y_pred)
    metrics.update(spearman_results)
    
    # Spatial SSIM
    if coords is not None:
        metrics['ssim_spatial'] = structural_similarity_spatial(y_true, y_pred, coords)
    
    if verbose:
        print("\n" + "=" * 50)
        print("Evaluation Results")
        print("=" * 50)
        print(f"Overall PCC:       {metrics.get('pcc_overall', 'N/A'):.4f}")
        print(f"Per-gene PCC:      {metrics.get('pcc_per_gene_mean', 'N/A'):.4f}  {metrics.get('pcc_per_gene_std', 'N/A'):.4f}")
        print(f"Per-spot PCC:      {metrics.get('pcc_per_spot_mean', 'N/A'):.4f}  {metrics.get('pcc_per_spot_std', 'N/A'):.4f}")
        print(f"Overall RMSE:      {metrics.get('rmse_overall', 'N/A'):.4f}")
        print(f"Overall MAE:       {metrics.get('mae_overall', 'N/A'):.4f}")
        print(f"Spearman:          {metrics.get('spearman_overall', 'N/A'):.4f}")
        if coords is not None:
            print(f"Spatial SSIM:      {metrics.get('ssim_spatial', 'N/A'):.4f}")
        print("=" * 50)
    
    return metrics


def gene_wise_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gene_names: Optional[list] = None,
    top_k: int = 10
) -> Tuple[Dict, Dict]:
    """
    Compute metrics for individual genes.
    
    Args:
        y_true: Ground truth
        y_pred: Predictions
        gene_names: List of gene names
        top_k: Number of top/bottom genes to return
        
    Returns:
        gene_metrics: Dictionary mapping gene index to metrics
        summary: Summary with best and worst genes
    """
    n_genes = y_true.shape[1]
    
    if gene_names is None:
        gene_names = [f"Gene_{i}" for i in range(n_genes)]
    
    gene_metrics = {}
    
    for g in range(n_genes):
        true_g = y_true[:, g]
        pred_g = y_pred[:, g]
        mask = ~(np.isnan(true_g) | np.isnan(pred_g))
        
        metrics = {}
        
        if mask.sum() > 2 and np.std(true_g[mask]) > 1e-8:
            pcc, _ = pearsonr(true_g[mask], pred_g[mask])
            metrics['pcc'] = pcc
        else:
            metrics['pcc'] = np.nan
        
        if mask.sum() > 0:
            metrics['rmse'] = np.sqrt(mean_squared_error(true_g[mask], pred_g[mask]))
            metrics['mae'] = mean_absolute_error(true_g[mask], pred_g[mask])
        else:
            metrics['rmse'] = np.nan
            metrics['mae'] = np.nan
        
        metrics['name'] = gene_names[g]
        gene_metrics[g] = metrics
    
    # Sort genes by PCC
    valid_genes = [(g, m['pcc']) for g, m in gene_metrics.items() if not np.isnan(m['pcc'])]
    sorted_genes = sorted(valid_genes, key=lambda x: x[1], reverse=True)
    
    summary = {
        'best_genes': [(gene_metrics[g[0]]['name'], g[1]) for g in sorted_genes[:top_k]],
        'worst_genes': [(gene_metrics[g[0]]['name'], g[1]) for g in sorted_genes[-top_k:]]
    }
    
    return gene_metrics, summary
