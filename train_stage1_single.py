"""
Stage 1 Single Sample Training Script
GraphVAE Self-supervised Training for ONE sample

This script trains a GraphVAE model on a single sample to avoid memory issues.
Use the shell script to batch process multiple samples sequentially.

Usage:
    # Train a single sample by ID (reads path from config)
    python train_stage1_single.py --sample_id HLN_A1
    
    # Train with custom data path (ignores config)
    python train_stage1_single.py --sample_id MyData --data_path /path/to/data.h5ad
    
    # Override training parameters
    python train_stage1_single.py --sample_id HLN_A1 --epochs 500 --device cuda:0
"""

import os
import sys
import argparse
import yaml
import json
import torch
import numpy as np
import gc
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from data import load_h5ad
from models.graph_vae import GraphVAE, GraphVAETrainer, MiniBatchGraphVAETrainer
from utils import (
    build_heterogeneous_graph,
    save_checkpoint,
    save_config,
    save_training_history,
    save_numpy_arrays,
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Stage 1: GraphVAE Training for a Single Sample',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train using sample ID from config
  python train_stage1_single.py --sample_id HLN_A1
  
  # Train with custom data path
  python train_stage1_single.py --sample_id CustomSample --data_path /path/to/data.h5ad
  
  # Override parameters
  python train_stage1_single.py --sample_id HLN_A1 --epochs 500 --device cuda:0
        """
    )
    
    # Required: sample identifier
    parser.add_argument('--sample_id', type=str, required=True,
                        help='Sample ID (used for output directory naming)')
    
    # Optional: data path (if not provided, will be read from config)
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to LR data file (overrides config)')
    
    # Config file
    parser.add_argument('--config', type=str, default='configs/stage1_config.yaml',
                        help='Path to stage 1 config file')
    
    # Data config override (for selecting 4x or 10x data)
    parser.add_argument('--data_config', type=str, default=None,
                        help='Override data config file path (e.g., configs/data_config_10x.yaml)')
    
    # Override options
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override output directory')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of training epochs')
    parser.add_argument('--device', type=str, default=None,
                        help='Override training device (cuda/cpu/auto)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override random seed')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    
    # Other options
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip if checkpoint already exists')
    parser.add_argument('--sample_name', type=str, default=None,
                        help='Display name for the sample (optional)')
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_memory():
    """Clear GPU and CPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_single_sample(
    sample_id: str,
    data_path: str,
    config: Dict,
    device: str,
    output_dir: str,
    sample_name: Optional[str] = None,
    hr_path: Optional[str] = None
) -> Dict:
    """
    Train GraphVAE model for a single sample
    
    Args:
        sample_id: Sample identifier (used for output naming)
        data_path: Path to LR data file
        config: Configuration dictionary
        device: Training device
        output_dir: Base output directory
        sample_name: Display name for the sample
        hr_path: Path to HR data file (for HVG selection)
        
    Returns:
        Training result dictionary
    """
    if sample_name is None:
        sample_name = sample_id
    
    # Create sample-specific output directory
    sample_output_dir = os.path.join(output_dir, sample_id)
    os.makedirs(sample_output_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print(f"Training Sample: {sample_id}")
    print(f"Display Name: {sample_name}")
    print(f"Data Path: {data_path}")
    print(f"Output Dir: {sample_output_dir}")
    print(f"Device: {device}")
    print("=" * 70)
    
    # Check if data file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # [1] Load and preprocess data
    print("\n[1/6] Loading and preprocessing data...")
    adata = load_h5ad(data_path)
    
    # Get preprocessing parameters
    use_pregenerated = config.get('downsampling', {}).get('use_pregenerated', True)
    hvg_source = config.get('hvg', {}).get('source', 'hr')  #  HR  HVG
    
    if use_pregenerated:
        # Using pre-generated LR data, perform preprocessing directly
        print("      Using pre-generated LR data (no downsampling needed)")
        
        from data.preprocessing import DataPreprocessor
        
        lr_coords = adata.obsm['spatial'].copy()
        
        # Create preprocessor
        preprocessor = DataPreprocessor(
            normalize=config['preprocessing']['normalize'],
            log1p=config['preprocessing']['log1p'],
            n_pca_components=config['pca']['n_components'],
            n_hvg=config.get('hvg', {}).get('n_genes', 3000)
        )
        
        # Determine HVG source
        if hvg_source == 'hr' and hr_path is not None and os.path.exists(hr_path):
            # Fit HVG on HR data (recommended: more spots, more stable HVG selection)
            print(f"      Fitting HVG from HR data: {hr_path}")
            adata_hr = load_h5ad(hr_path)
            preprocessor.fit(adata_hr)
            del adata_hr  # Release memory
            
            # Transform LR data using HR-fitted preprocessor
            lr_features = preprocessor.transform(adata)
            lr_hvg_expression = preprocessor.get_hvg_expression(adata)
            print(f"      HVG source: HR ({len(preprocessor.hvg_names_)} genes)")
        else:
            # Fallback: Fit and transform LR data (original behavior)
            if hvg_source == 'hr' and (hr_path is None or not os.path.exists(hr_path)):
                print(f"      [WARNING] HR path not available, falling back to LR for HVG selection")
            print(f"      Fitting HVG from LR data")
            lr_features = preprocessor.fit_transform(adata)
            lr_hvg_expression = preprocessor.get_hvg_expression(adata)
            print(f"      HVG source: LR ({len(preprocessor.hvg_names_)} genes)")
        
        # Save preprocessor
        preprocessor.save(os.path.join(sample_output_dir, 'preprocessor.pkl'))
        
        print(f"      LR PCA features shape: {lr_features.shape}")
        print(f"      LR HVG expression shape: {lr_hvg_expression.shape}")
        
    else:
        # Downsample HR data
        from data import prepare_training_data
        data_dict = prepare_training_data(
            adata,
            ks=config['downsampling']['ks'],
            n_pca_components=config['pca']['n_components'],
            n_hvg=config.get('hvg', {}).get('n_genes', 3000),
            normalize=config['preprocessing']['normalize'],
            log1p=config['preprocessing']['log1p'],
            save_dir=sample_output_dir
        )
        
        lr_features = data_dict['lr_features']
        lr_hvg_expression = data_dict['lr_hvg_expression']
        lr_coords = data_dict['lr_coords']
        preprocessor = data_dict['preprocessor']
    
    n_hvg = lr_hvg_expression.shape[1]
    
    print(f"      LR spots: {lr_features.shape[0]}")
    print(f"      PCA features dim: {lr_features.shape[1]}")
    print(f"      HVG genes: {n_hvg}")
    
    # [2] Build heterogeneous graph
    print("\n[2/6] Building heterogeneous graph...")
    edge_index, edge_weight = build_heterogeneous_graph(
        coords=lr_coords,
        features=lr_features,
        spatial_k=config['graph']['spatial_k'],
        feature_k=config['graph']['feature_k'],
        include_self_loops=True
    )
    print(f"      Nodes: {lr_features.shape[0]}, Edges: {edge_index.shape[1]}")
    
    # Convert to tensors
    x = torch.tensor(lr_features, dtype=torch.float32)
    x_target = torch.tensor(lr_hvg_expression, dtype=torch.float32)
    edge_attr = edge_weight.unsqueeze(-1) if edge_weight is not None else None
    
    # [3] Initialize model
    print("\n[3/6] Initializing GraphVAE model...")
    decoder_hidden_dims = tuple(config['graphvae'].get('decoder_hidden_dims', [256, 512, 1024]))
    use_latent_norm = config['graphvae'].get('use_latent_norm', True)  # 
    model = GraphVAE(
        input_dim=config['pca']['n_components'],
        hidden_dim=config['graphvae']['hidden_dim'],
        latent_dim=config['graphvae']['latent_dim'],
        decoder_hidden_dims=decoder_hidden_dims,
        output_dim=n_hvg,
        num_heads=config['graphvae']['gat_heads'],
        dropout=config['graphvae']['dropout'],
        use_latent_norm=use_latent_norm,
        latent_norm_momentum=config['graphvae'].get('latent_norm_momentum', 0.1)
    )
    print(f"      Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"      LatentNorm: {'Enabled' if use_latent_norm else 'Disabled'}")
    
    # [4] Initialize trainer and train
    print("\n[4/6] Training GraphVAE...")
    
    # Auto-select trainer based on dataset size
    n_spots = lr_features.shape[0]
    minibatch_threshold = config.get('training', {}).get('minibatch_threshold', 50000)
    use_minibatch = n_spots > minibatch_threshold
    
    if use_minibatch:
        # Use mini-batch trainer for large datasets
        print(f"      [INFO] Large dataset ({n_spots:,} spots > {minibatch_threshold:,}), using MiniBatchGraphVAETrainer")
        
        # Calculate optimal num_parts based on GPU memory and dataset size
        # Target ~5000-10000 spots per cluster for 16GB GPU
        target_cluster_size = config.get('training', {}).get('target_cluster_size', 5000)
        num_parts = max(10, n_spots // target_cluster_size)
        batch_size = config.get('training', {}).get('minibatch_clusters', 5)
        gradient_accumulation = config.get('training', {}).get('gradient_accumulation', 2)
        
        print(f"      Clusters: {num_parts}, Batch size: {batch_size}, Gradient accum: {gradient_accumulation}")
        
        trainer = MiniBatchGraphVAETrainer(
            model=model,
            learning_rate=config['graphvae']['learning_rate'],
            weight_decay=config['graphvae']['weight_decay'],
            recon_weight=config['graphvae'].get('recon_weight', 1.0),
            kl_weight=config['graphvae']['kl_weight'],
            cosine_weight=config['graphvae'].get('cosine_weight', 1.0),
            pcc_weight=config['graphvae'].get('pcc_weight', 100.0),
            pcc_gene_weight=config['graphvae'].get('pcc_gene_weight', 0.7),
            pcc_spot_weight=config['graphvae'].get('pcc_spot_weight', 0.3),
            scheduler_factor=config['graphvae'].get('scheduler_factor', 0.5),
            scheduler_patience=config['graphvae'].get('scheduler_patience', 10),
            grad_clip_max_norm=config['graphvae'].get('grad_clip_max_norm', 1.0),
            device=device,
            num_parts=num_parts,
            batch_size=batch_size,
            gradient_accumulation=gradient_accumulation,
            use_amp=config.get('training', {}).get('use_amp', True)
        )
    else:
        # Use full-batch trainer for smaller datasets
        print(f"      [INFO] Small dataset ({n_spots:,} spots), using full-batch GraphVAETrainer")
        trainer = GraphVAETrainer(
            model=model,
            learning_rate=config['graphvae']['learning_rate'],
            weight_decay=config['graphvae']['weight_decay'],
            recon_weight=config['graphvae'].get('recon_weight', 1.0),
            kl_weight=config['graphvae']['kl_weight'],
            cosine_weight=config['graphvae'].get('cosine_weight', 1.0),
            pcc_weight=config['graphvae'].get('pcc_weight', 100.0),
            pcc_gene_weight=config['graphvae'].get('pcc_gene_weight', 0.7),
            pcc_spot_weight=config['graphvae'].get('pcc_spot_weight', 0.3),
            scheduler_factor=config['graphvae'].get('scheduler_factor', 0.5),
            scheduler_patience=config['graphvae'].get('scheduler_patience', 10),
            grad_clip_max_norm=config['graphvae'].get('grad_clip_max_norm', 1.0),
            device=device
        )
    
    history = trainer.train(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        x_target=x_target,
        epochs=config['graphvae']['epochs'],
        early_stopping_patience=config['graphvae']['early_stopping_patience'],
        verbose=True
    )
    
    # [5] Extract latent representations
    print("\n[5/6] Extracting latent representations...")
    model.freeze_decoder()
    z_lr = trainer.get_latent_representations(x, edge_index, edge_attr)
    print(f"      LR latent shape: {z_lr.shape}")
    
    # [6] Save model and results
    print("\n[6/6] Saving model and results...")
    
    # Save complete checkpoint
    checkpoint_path = os.path.join(sample_output_dir, 'graphvae_checkpoint.pt')
    save_checkpoint(
        model=model,
        optimizer=trainer.optimizer,
        epoch=len(history['total_loss']),
        loss=history['total_loss'][-1],
        path=checkpoint_path,
        additional_info={
            'config': config,
            'data_path': data_path,
            'sample_id': sample_id,
            'sample_name': sample_name,
            'n_spots': lr_features.shape[0],
            'n_hvg': n_hvg,
            'hvg_names': preprocessor.hvg_names_
        }
    )
    print(f"      Checkpoint: {checkpoint_path}")
    
    # Save encoder weights (for stage 2)
    encoder_path = os.path.join(sample_output_dir, 'encoder_weights.pt')
    torch.save({
        'encoder_state_dict': model.encoder.state_dict(),
        'input_dim': config['pca']['n_components'],
        'hidden_dim': config['graphvae']['hidden_dim'],
        'latent_dim': config['graphvae']['latent_dim'],
        'num_heads': config['graphvae']['gat_heads'],
        'sample_id': sample_id
    }, encoder_path)
    print(f"      Encoder weights: {encoder_path}")
    
    # Save decoder weights (for inference)
    decoder_path = os.path.join(sample_output_dir, 'decoder_weights.pt')
    decoder_save_dict = {
        'decoder_state_dict': model.decoder.state_dict(),
        'latent_dim': config['graphvae']['latent_dim'],
        'output_dim': n_hvg,
        'hvg_names': preprocessor.hvg_names_,
        'sample_id': sample_id,
        'use_latent_norm': model.use_latent_norm
    }
    #  LatentNorm Stage 2 
    if model.use_latent_norm and model.latent_norm is not None:
        decoder_save_dict['latent_norm_state_dict'] = model.latent_norm.state_dict()
        decoder_save_dict['latent_norm_stats'] = model.get_latent_norm_stats()
        print(f"      LatentNorm stats saved: mean={model.latent_norm.running_mean.mean():.4f}, "
              f"var={model.latent_norm.running_var.mean():.4f}")
    torch.save(decoder_save_dict, decoder_path)
    print(f"      Decoder weights: {decoder_path}")
    
    # Save latent representations
    latent_path = os.path.join(sample_output_dir, 'latent_representations.npz')
    save_numpy_arrays({
        'z_lr': z_lr,
        'lr_features': lr_features,
        'lr_hvg_expression': lr_hvg_expression,
        'lr_coords': lr_coords
    }, latent_path)
    print(f"      Latent representations: {latent_path}")
    
    # Save graph data
    graph_path = os.path.join(sample_output_dir, 'graph_data.pt')
    torch.save({
        'edge_index': edge_index,
        'edge_weight': edge_weight
    }, graph_path)
    print(f"      Graph data: {graph_path}")
    
    # Save training history
    history_path = os.path.join(sample_output_dir, 'training_history.json')
    save_training_history(history, history_path)
    print(f"      Training history: {history_path}")
    
    # Save config backup
    config_path = os.path.join(sample_output_dir, 'config.json')
    save_config(config, config_path)
    print(f"      Config: {config_path}")
    
    # Save sample info (for stage 2 reference)
    sample_info_path = os.path.join(sample_output_dir, 'sample_info.json')
    with open(sample_info_path, 'w') as f:
        json.dump({
            'sample_id': sample_id,
            'sample_name': sample_name,
            'data_path': data_path,
            'n_spots': int(lr_features.shape[0]),
            'n_hvg': int(n_hvg),
            'latent_dim': config['graphvae']['latent_dim'],
            'final_loss': float(history['total_loss'][-1]),
            'trained_epochs': len(history['total_loss']),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    print(f"      Sample info: {sample_info_path}")
    
    print("\n" + "-" * 70)
    print(f"Training Complete: {sample_id}")
    print(f"Final loss: {history['total_loss'][-1]:.4f}")
    print(f"Trained epochs: {len(history['total_loss'])}")
    print("-" * 70)
    
    return {
        'sample_id': sample_id,
        'sample_name': sample_name,
        'output_dir': sample_output_dir,
        'final_loss': history['total_loss'][-1],
        'n_spots': lr_features.shape[0],
        'latent_shape': z_lr.shape,
        'status': 'success'
    }


def main():
    """Main function"""
    args = parse_args()
    
    print("\n" + "=" * 70)
    print("SRast Stage 1: GraphVAE Single Sample Training")
    print("=" * 70)
    
    # Load config
    config = load_config(args.config)
    print(f"\nConfig file: {args.config}")
    print(f"Sample ID: {args.sample_id}")
    
    # Determine data config path (command line override or from config)
    data_config = None
    data_config_path = args.data_config if args.data_config else config.get('data_config')
    if data_config_path:
        if not os.path.isabs(data_config_path):
            base_dir = os.path.dirname(os.path.abspath(args.config))
            if os.path.basename(base_dir) == 'configs':
                base_dir = os.path.dirname(base_dir)
            data_config_path = os.path.join(base_dir, data_config_path)
        
        if os.path.exists(data_config_path):
            from utils.config_loader import load_unified_data_config
            data_config = load_unified_data_config(data_config_path)
            print(f"Loaded unified data config: {data_config_path}")
        else:
            print(f"[WARNING] Data config not found: {data_config_path}")
    
    # Determine data path
    hr_path = None  # HR path for HVG selection
    if args.data_path is not None:
        # Use provided data path
        data_path = args.data_path
        sample_name = args.sample_name or args.sample_id
        # Try to infer HR path from LR path
        if '_LR_' in data_path:
            potential_hr_path = data_path.replace('_LR_ks4.h5ad', '_HR.h5ad').replace('_LR_', '_HR_')
            if os.path.exists(potential_hr_path):
                hr_path = potential_hr_path
    elif data_config is not None:
        # Look up from unified data config (new format)
        dataset_info = data_config.get_dataset(args.sample_id)
        if dataset_info is None:
            print(f"\n[ERROR] Sample '{args.sample_id}' not found in data config!")
            print(f"Available samples: {data_config.list_all_datasets()[:10]}...")
            sys.exit(1)
        
        data_path = dataset_info.lr_path
        hr_path = dataset_info.hr_path  # Get HR path from config
        sample_name = args.sample_name or dataset_info.name
    else:
        # Look up from old config format (backward compatibility)
        training_samples = config.get('training_samples', {})
        if args.sample_id not in training_samples:
            print(f"\n[ERROR] Sample '{args.sample_id}' not found in config!")
            print(f"Available samples: {list(training_samples.keys())}")
            sys.exit(1)
        
        sample_info = training_samples[args.sample_id]
        data_path = sample_info['path']
        sample_name = args.sample_name or sample_info.get('name', args.sample_id)
    
    print(f"Data path (LR): {data_path}")
    print(f"HR path (for HVG): {hr_path}")
    print(f"Sample name: {sample_name}")
    
    # Override config parameters
    if args.output_dir is not None:
        config['paths']['output_dir'] = args.output_dir
    if args.epochs is not None:
        config['graphvae']['epochs'] = args.epochs
    if args.device is not None:
        config['training']['device'] = args.device
    if args.seed is not None:
        config['training']['seed'] = args.seed
    if args.lr is not None:
        config['graphvae']['learning_rate'] = args.lr
    
    # Set random seed
    set_seed(config['training']['seed'])
    
    # Determine device
    if config['training']['device'] == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = config['training']['device']
    print(f"Device: {device}")
    
    # Output directory
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Check if should skip existing
    sample_output_dir = os.path.join(output_dir, args.sample_id)
    checkpoint_path = os.path.join(sample_output_dir, 'graphvae_checkpoint.pt')
    
    if args.skip_existing and os.path.exists(checkpoint_path):
        print(f"\n[SKIP] Checkpoint already exists: {checkpoint_path}")
        print("Use --skip_existing=false to retrain.")
        sys.exit(0)
    
    # Train the sample
    try:
        result = train_single_sample(
            sample_id=args.sample_id,
            data_path=data_path,
            config=config,
            device=device,
            output_dir=output_dir,
            sample_name=sample_name,
            hr_path=hr_path
        )
        
        print("\n" + "=" * 70)
        print("SUCCESS")
        print("=" * 70)
        print(f"Sample: {result['sample_id']}")
        print(f"Final loss: {result['final_loss']:.4f}")
        print(f"Output: {result['output_dir']}")
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Clear memory
        clear_memory()
    
    print("\nDone!")


if __name__ == '__main__':
    main()
