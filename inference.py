"""
SRast Stage 2: Flow Matching Model Inference Script

Zero-shot inference for spatial transcriptomics super-resolution.

Usage:
    # Test all samples
    python inference.py --config configs/stage2_config.yaml --test_all
    
    # Test specific sample
    python inference.py --sample_id HLN_D1
    
    # Use specific model
    python inference.py --model_path results/experiments/xxx/checkpoints/flow_matching_model.pt --test_all
"""

import os
import sys
import argparse
import json
import yaml
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import anndata as ad
from scipy.sparse import csr_matrix
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.flow_matching import FlowMatchingRatio
from data.preprocessing import DataPreprocessor, load_h5ad
from utils import (
    compute_all_metrics,
    build_lr_hr_mapping,
    load_test_sample
)

REQUIRED_N_HVG = 3000


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data config if referenced
    if 'data_config' in config:
        data_config_path = config['data_config']
        if os.path.exists(data_config_path):
            with open(data_config_path, 'r') as f:
                data_config = yaml.safe_load(f)
            config['_data_config'] = data_config
    
    return config


def run_inference(
    model: FlowMatchingRatio,
    sample_data: Dict,
    num_steps: int = 50,
    device: str = 'cuda',
    use_raw_expression: bool = True
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Run inference on a single sample.
    
    Args:
        model: Flow matching model
        sample_data: Sample data dict from load_test_sample
        num_steps: Number of ODE integration steps
        device: Device to use
        use_raw_expression: If True, use raw expression for final reconstruction and metrics
                           
    Returns:
        x_hr_pred: Predicted HR expression
        metrics: Evaluation metrics dictionary
    """
    model.eval()
    
    # Move data to device
    z_lr = sample_data['z_lr'].to(device)
    lr_coords = sample_data['lr_coords'].to(device)
    hr_coords = sample_data['hr_coords'].to(device)
    lr_hr_mapping = sample_data['lr_hr_mapping'].to(device)
    group_indices = sample_data['group_indices'].to(device)
    local_hr_indices = sample_data['local_hr_indices'].to(device)
    
    lr_hr_mapping_np = sample_data.get('lr_hr_mapping_np')
    
    # Choose expression type for reconstruction
    if use_raw_expression and 'lr_raw_hvg' in sample_data and 'hr_raw_hvg' in sample_data:
        print("    Using RAW expression for reconstruction and metrics...")
        lr_raw_hvg = sample_data['lr_raw_hvg']
        hr_raw_hvg = sample_data['hr_raw_hvg']
        
        x_lr = torch.tensor(lr_raw_hvg, dtype=torch.float32).to(device)
        x_hr_true = hr_raw_hvg
        x_lr_np = lr_raw_hvg
    else:
        print("    Using PREPROCESSED expression for reconstruction and metrics...")
        x_lr = sample_data['x_lr'].to(device)
        x_hr_true = sample_data['x_hr'].numpy()
        x_lr_np = sample_data.get('x_lr_np')
    
    # Run model sampling
    with torch.no_grad():
        x_hr_pred = model.sample(
            z_lr=z_lr,
            x_lr=x_lr,
            lr_coords=lr_coords,
            hr_coords=hr_coords,
            lr_hr_mapping=lr_hr_mapping,
            group_indices=group_indices,
            local_hr_indices=local_hr_indices,
            num_steps=num_steps,
            verbose=True
        )
    
    x_hr_pred_np = x_hr_pred.cpu().numpy()
    
    # Compute metrics
    metrics = compute_all_metrics(
        x_hr_pred_np, 
        x_hr_true,
        x_lr=x_lr_np,
        lr_hr_mapping_np=lr_hr_mapping_np
    )
    
    return x_hr_pred_np, metrics


def main():
    parser = argparse.ArgumentParser(description='SRast Flow Matching Inference')
    parser.add_argument('--config', type=str, default='configs/stage2_config.yaml',
                        help='Path to config file')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model (overrides config)')
    parser.add_argument('--output_dir', type=str, default='results/inference',
                        help='Output directory for predictions')
    parser.add_argument('--sample_id', type=str, default=None,
                        help='Specific sample to test')
    parser.add_argument('--test_all', action='store_true',
                        help='Test all samples in test_datasets')
    parser.add_argument('--num_steps', type=int, default=None,
                        help='Number of ODE integration steps (default: from config or 50)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save prediction arrays to npz files')
    parser.add_argument('--save_h5ad', action='store_true',
                        help='Save SR results as h5ad files with spatial coordinates')
    parser.add_argument('--h5ad_output_dir', type=str, default='/home/hxl/ST_SR_result',
                        help='Output directory for h5ad files')
    parser.add_argument('--method_name', type=str, default='SRast_v5',
                        help='Method name for h5ad output directory structure')
    parser.add_argument('--downscale', type=int, default=None,
                        help='Downscale factor (4 or 10) for naming h5ad output files')
    parser.add_argument('--use_raw', action='store_true', default=True,
                        help='Use raw expression for reconstruction and metrics (default: True)')
    parser.add_argument('--no_use_raw', dest='use_raw', action='store_false',
                        help='Use preprocessed expression instead of raw')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    data_config = config.get('_data_config', {})
    flow_config = config.get('flow_matching', {})
    
    # Determine num_steps
    num_steps = args.num_steps or flow_config.get('num_steps', 50)
    print(f"Sampling steps: {num_steps}")
    
    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Stage 1 directory
    stage1_dir = config.get('stage1_dir', 'checkpoints/stage1')
    
    # Find model path
    model_path = args.model_path
    if model_path is None:
        default_paths = [
            'checkpoints/stage2/flow_matching/flow_matching_model.pt',
            'checkpoints/stage2/flow_matching/flow_matching_model_best.pt',
        ]
        exp_dir = 'results/experiments'
        if os.path.exists(exp_dir):
            experiments = sorted(os.listdir(exp_dir), reverse=True)
            for exp in experiments:
                exp_model = os.path.join(exp_dir, exp, 'checkpoints', 'flow_matching_model.pt')
                exp_model_best = os.path.join(exp_dir, exp, 'checkpoints', 'flow_matching_model_best.pt')
                default_paths.insert(0, exp_model_best)
                default_paths.insert(1, exp_model)
        
        for path in default_paths:
            if os.path.exists(path):
                model_path = path
                break
    
    if model_path is None or not os.path.exists(model_path):
        print(f"[ERROR] Model not found. Tried: {default_paths[:3]}...")
        print("Please specify --model_path or train a model first.")
        sys.exit(1)
    
    print(f"Loading model from: {model_path}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    n_genes = checkpoint.get('n_genes', REQUIRED_N_HVG)
    latent_dim = checkpoint.get('latent_dim', 128)
    
    use_dit = flow_config.get('use_dit', False)
    mlp_ratio = flow_config.get('mlp_ratio', 4.0)
    
    model = FlowMatchingRatio(
        n_genes=n_genes,
        latent_dim=latent_dim,
        hidden_dim=flow_config.get('hidden_dim', 256),
        num_heads=flow_config.get('num_heads', 8),
        num_layers=flow_config.get('num_layers', 4),
        dropout=flow_config.get('dropout', 0.1),
        use_hr_spatial=flow_config.get('use_hr_spatial', True),
        num_hr_neighbors=flow_config.get('num_hr_neighbors', 6),
        sigma_min=flow_config.get('sigma_min', 0.001),
        use_dit=use_dit,
        mlp_ratio=mlp_ratio
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded: n_genes={n_genes}, latent_dim={latent_dim}")
    
    # Determine samples to test
    if args.sample_id:
        test_samples = [args.sample_id]
    elif args.test_all:
        test_samples = config.get('test_datasets', [])
    else:
        print("[ERROR] Please specify --sample_id or --test_all")
        sys.exit(1)
    
    if not test_samples:
        print("[ERROR] No test samples found")
        sys.exit(1)
    
    print(f"\nTesting {len(test_samples)} samples: {test_samples}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run inference
    results = {}
    all_metrics = []
    
    for sample_id in test_samples:
        print(f"\n{'='*60}")
        print(f"Testing: {sample_id}")
        print('='*60)
        
        # Load sample using shared module
        sample_data = load_test_sample(
            sample_id, stage1_dir, data_config, device,
            include_raw=args.use_raw
        )
        if sample_data is None:
            print(f"[SKIP] {sample_id}")
            continue
        
        print(f"  LR spots: {sample_data['n_lr']}")
        print(f"  HR spots: {sample_data['n_hr']}")
        print(f"  Genes: {sample_data['n_genes']}")
        print(f"  Use raw expression: {args.use_raw}")
        
        # Run inference
        x_hr_pred, metrics = run_inference(
            model, sample_data, num_steps=num_steps, device=device,
            use_raw_expression=args.use_raw
        )
        
        print(f"\nResults for {sample_id}:")
        print(f"  PCC:      {metrics['pcc']:.4f}")
        print(f"  Spearman: {metrics['spearman']:.4f}")
        print(f"  Gene PCC: {metrics['gene_pcc']:.4f}")
        print(f"  Spot PCC: {metrics['spot_pcc']:.4f}")
        print(f"  RMSE:     {metrics['rmse']:.4f}")
        print(f"  MAE:      {metrics['mae']:.4f}")
        print(f"  SNR:      {metrics['snr']:.2f} dB")
        print(f"  SSIM:     {metrics['ssim']:.4f}")
        print(f"  FracDev:  {metrics.get('fractional_deviation', float('nan')):.6e}")
        print(f"  JS Div:   {metrics.get('js_divergence', float('nan')):.4f}")
        print(f"  JS Gene:  {metrics.get('js_divergence_gene', float('nan')):.4f}")
        
        results[sample_id] = {
            'metrics': metrics,
            'n_hr_spots': sample_data['n_hr'],
            'n_lr_spots': sample_data['n_lr']
        }
        all_metrics.append(metrics)
        
        # Save predictions if requested
        if args.save_predictions:
            pred_path = os.path.join(args.output_dir, f'{sample_id}_predictions.npz')
            np.savez_compressed(
                pred_path,
                x_hr_pred=x_hr_pred,
                x_hr_true=sample_data['x_hr'].numpy(),
                hr_coords=sample_data['hr_coords'].numpy()
            )
            print(f"  Saved predictions to: {pred_path}")
        
        # Save h5ad if requested
        if args.save_h5ad:
            h5ad_method_dir = os.path.join(args.h5ad_output_dir, args.method_name)
            os.makedirs(h5ad_method_dir, exist_ok=True)
            
            preprocessor = sample_data['preprocessor']
            gene_names = preprocessor.hvg_names_ if hasattr(preprocessor, 'hvg_names_') else None
            hr_coords_np = sample_data['hr_coords'].numpy()
            
            n_spots = x_hr_pred.shape[0]
            obs_df = {'spot_id': [f'spot_{i}' for i in range(n_spots)]}
            
            if gene_names is not None:
                var_df = {'gene_name': gene_names}
                adata_sr = ad.AnnData(X=csr_matrix(x_hr_pred), obs=obs_df, var=var_df)
                adata_sr.var_names = gene_names
            else:
                adata_sr = ad.AnnData(X=csr_matrix(x_hr_pred), obs=obs_df)
                adata_sr.var_names = [f'gene_{i}' for i in range(x_hr_pred.shape[1])]
            
            adata_sr.obs_names = [f'spot_{i}' for i in range(n_spots)]
            adata_sr.obsm['spatial'] = hr_coords_np
            adata_sr.uns['metrics'] = metrics
            adata_sr.uns['method'] = args.method_name
            adata_sr.uns['sample_id'] = sample_id
            adata_sr.uns['n_lr_spots'] = sample_data['n_lr']
            adata_sr.uns['n_hr_spots'] = sample_data['n_hr']
            
            if args.downscale is not None:
                h5ad_path = os.path.join(h5ad_method_dir, f'{sample_id}_{args.downscale}x.h5ad')
            else:
                h5ad_path = os.path.join(h5ad_method_dir, f'{sample_id}.h5ad')
            adata_sr.write_h5ad(h5ad_path)
            print(f"  Saved h5ad to: {h5ad_path}")
    
    # Summary
    if all_metrics:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)
        
        avg_metrics = {
            'pcc': np.mean([m['pcc'] for m in all_metrics]),
            'spearman': np.mean([m['spearman'] for m in all_metrics]),
            'rmse': np.mean([m['rmse'] for m in all_metrics]),
            'mae': np.mean([m['mae'] for m in all_metrics]),
            'gene_pcc': np.mean([m['gene_pcc'] for m in all_metrics]),
            'spot_pcc': np.mean([m['spot_pcc'] for m in all_metrics]),
            'snr': np.mean([m['snr'] for m in all_metrics]),
            'ssim': np.mean([m['ssim'] for m in all_metrics]),
            'fractional_deviation': np.mean([m.get('fractional_deviation', float('nan')) for m in all_metrics]),
            'js_divergence': np.mean([m.get('js_divergence', float('nan')) for m in all_metrics]),
            'js_divergence_gene': np.mean([m.get('js_divergence_gene', float('nan')) for m in all_metrics])
        }
        
        print(f"\n{'Sample':<20} {'PCC':>8} {'GenePCC':>8} {'SpotPCC':>8} {'RMSE':>8} {'SNR':>8} {'SSIM':>8} {'FracDev':>12} {'JSDivGene':>10}")
        print('-' * 110)
        for sample_id, data in results.items():
            m = data['metrics']
            frac_dev = m.get('fractional_deviation', float('nan'))
            js_gene = m.get('js_divergence_gene', float('nan'))
            print(f"{sample_id:<20} {m['pcc']:>8.4f} {m['gene_pcc']:>8.4f} {m['spot_pcc']:>8.4f} {m['rmse']:>8.4f} {m['snr']:>8.2f} {m['ssim']:>8.4f} {frac_dev:>12.4e} {js_gene:>10.4f}")
        print('-' * 110)
        print(f"{'Average':<20} {avg_metrics['pcc']:>8.4f} {avg_metrics['gene_pcc']:>8.4f} {avg_metrics['spot_pcc']:>8.4f} {avg_metrics['rmse']:>8.4f} {avg_metrics['snr']:>8.2f} {avg_metrics['ssim']:>8.4f} {avg_metrics['fractional_deviation']:>12.4e} {avg_metrics['js_divergence_gene']:>10.4f}")
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'results': results,
            'average': avg_metrics
        }
        
        summary_path = os.path.join(args.output_dir, f'summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")
    
    print("\nInference complete!")


if __name__ == '__main__':
    main()
