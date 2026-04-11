"""
Data Configuration Loader

This module provides utilities for loading and validating data configuration files.
Supports the unified data_config.yaml format with:
- Centralized dataset definitions (LR/HR pairs)
- Dataset groups for easy selection
- Stage 1 and Stage 2 specific loading functions
"""

import os
import yaml
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field


@dataclass
class DatasetInfo:
    """Represents a single dataset with LR and HR paths."""
    dataset_id: str
    name: str
    lr_path: str
    hr_path: str
    group: Optional[str] = None
    
    def lr_exists(self) -> bool:
        """Check if LR file exists."""
        return os.path.exists(self.lr_path)
    
    def hr_exists(self) -> bool:
        """Check if HR file exists."""
        return os.path.exists(self.hr_path)
    
    def __repr__(self):
        return f"DatasetInfo(id='{self.dataset_id}', name='{self.name}')"


@dataclass
class UnifiedDataConfig:
    """Unified data configuration containing all datasets and groups."""
    datasets: Dict[str, DatasetInfo] = field(default_factory=dict)
    dataset_groups: Dict[str, List[str]] = field(default_factory=dict)
    
    def get_dataset(self, dataset_id: str) -> Optional[DatasetInfo]:
        """Get a dataset by ID."""
        return self.datasets.get(dataset_id)
    
    def get_group_datasets(self, group_name: str) -> List[DatasetInfo]:
        """Get all datasets in a group."""
        if group_name not in self.dataset_groups:
            return []
        return [self.datasets[did] for did in self.dataset_groups[group_name] 
                if did in self.datasets]
    
    def expand_dataset_list(self, items: Union[str, List[str]]) -> List[str]:
        """
        Expand a list of dataset IDs or group names into dataset IDs.
        
        Args:
            items: "ALL", a single group/dataset name, or list of names
            
        Returns:
            List of dataset IDs
        """
        if isinstance(items, str):
            if items == "ALL":
                return list(self.datasets.keys())
            elif items in self.dataset_groups:
                return self.dataset_groups[items]
            elif items in self.datasets:
                return [items]
            else:
                return []
        
        result = []
        seen = set()
        for item in items:
            if item in self.dataset_groups:
                # Expand group
                for did in self.dataset_groups[item]:
                    if did not in seen:
                        result.append(did)
                        seen.add(did)
            elif item in self.datasets:
                if item not in seen:
                    result.append(item)
                    seen.add(item)
        return result
    
    def get_datasets_info(self, dataset_ids: List[str]) -> Dict[str, DatasetInfo]:
        """Get DatasetInfo objects for a list of dataset IDs."""
        return {did: self.datasets[did] for did in dataset_ids if did in self.datasets}
    
    def list_all_datasets(self) -> List[str]:
        """List all dataset IDs."""
        return list(self.datasets.keys())
    
    def list_groups(self) -> List[str]:
        """List all group names."""
        return list(self.dataset_groups.keys())
    
    def summary(self) -> str:
        """Get a summary of the data configuration."""
        lines = [
            "Unified Data Configuration Summary",
            "=" * 50,
            f"Total datasets: {len(self.datasets)}",
            f"Dataset groups: {len(self.dataset_groups)}",
            ""
        ]
        
        # Group by category
        by_group = {}
        for did, info in self.datasets.items():
            group = info.group or "Uncategorized"
            if group not in by_group:
                by_group[group] = []
            by_group[group].append(info)
        
        for group, datasets in sorted(by_group.items()):
            lines.append(f"\n{group} ({len(datasets)} datasets):")
            for info in datasets:
                lr_status = "" if info.lr_exists() else ""
                hr_status = "" if info.hr_exists() else ""
                lines.append(f"  {info.dataset_id}: LR[{lr_status}] HR[{hr_status}]")
        
        return "\n".join(lines)


def load_unified_data_config(config_path: str) -> UnifiedDataConfig:
    """
    Load unified data configuration from YAML file.
    
    Args:
        config_path: Path to data_config.yaml
        
    Returns:
        UnifiedDataConfig object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Data config not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # Parse datasets
    datasets = {}
    for dataset_id, info in config_dict.get('datasets', {}).items():
        datasets[dataset_id] = DatasetInfo(
            dataset_id=dataset_id,
            name=info.get('name', dataset_id),
            lr_path=info.get('lr_path', ''),
            hr_path=info.get('hr_path', ''),
            group=info.get('group', None)
        )
    
    # Parse dataset groups
    dataset_groups = config_dict.get('dataset_groups', {})
    
    return UnifiedDataConfig(datasets=datasets, dataset_groups=dataset_groups)


def load_stage1_config(
    config_path: str = "configs/stage1_config.yaml",
    base_dir: Optional[str] = None
) -> Tuple[Dict[str, Any], UnifiedDataConfig, List[str]]:
    """
    Load Stage 1 configuration and resolve dataset information.
    
    Args:
        config_path: Path to stage1_config.yaml
        base_dir: Base directory for resolving paths
        
    Returns:
        config: Stage 1 configuration dictionary
        data_config: UnifiedDataConfig object
        training_dataset_ids: List of dataset IDs for training
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Stage 1 config not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Determine base directory
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(config_path))
        if os.path.basename(base_dir) == 'configs':
            base_dir = os.path.dirname(base_dir)
    
    # Load data config
    data_config_path = config.get('data_config', 'configs/data_config.yaml')
    if not os.path.isabs(data_config_path):
        data_config_path = os.path.join(base_dir, data_config_path)
    
    data_config = load_unified_data_config(data_config_path)
    
    # Expand training datasets
    training_datasets = config.get('training_datasets', 'ALL')
    training_dataset_ids = data_config.expand_dataset_list(training_datasets)
    
    return config, data_config, training_dataset_ids


def load_stage2_config(
    config_path: str = "configs/stage2_config.yaml",
    experiment_config_path: Optional[str] = None,
    base_dir: Optional[str] = None
) -> Tuple[Dict[str, Any], UnifiedDataConfig, List[str], List[str]]:
    """
    Load Stage 2 configuration with optional experiment config override.
    
    Args:
        config_path: Path to stage2_config.yaml
        experiment_config_path: Optional path to experiment config that overrides train/test sets
        base_dir: Base directory for resolving paths
        
    Returns:
        config: Merged configuration dictionary
        data_config: UnifiedDataConfig object
        training_dataset_ids: List of dataset IDs for training
        test_dataset_ids: List of dataset IDs for testing
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Stage 2 config not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Load experiment config and merge if provided
    experiment_info = None
    if experiment_config_path and os.path.exists(experiment_config_path):
        with open(experiment_config_path, 'r', encoding='utf-8') as f:
            exp_config = yaml.safe_load(f)
        
        # Store experiment metadata
        experiment_info = exp_config.get('experiment', {})
        
        # Override training_datasets and test_datasets
        if 'training_datasets' in exp_config:
            config['training_datasets'] = exp_config['training_datasets']
        if 'test_datasets' in exp_config:
            config['test_datasets'] = exp_config['test_datasets']
        
        # Override other parameters if specified
        for key in ['diffusion', 'training', 'graph_transformer']:
            if key in exp_config:
                if key in config:
                    config[key].update(exp_config[key])
                else:
                    config[key] = exp_config[key]
    
    # Store experiment info in config
    config['_experiment_info'] = experiment_info
    
    # Determine base directory
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(config_path))
        if os.path.basename(base_dir) == 'configs':
            base_dir = os.path.dirname(base_dir)
    
    # Load data config
    data_config_path = config.get('data_config', 'configs/data_config.yaml')
    if not os.path.isabs(data_config_path):
        data_config_path = os.path.join(base_dir, data_config_path)
    
    data_config = load_unified_data_config(data_config_path)
    
    # Expand training and test datasets
    training_datasets = config.get('training_datasets', [])
    test_datasets = config.get('test_datasets', [])
    
    training_dataset_ids = data_config.expand_dataset_list(training_datasets)
    test_dataset_ids = data_config.expand_dataset_list(test_datasets)
    
    return config, data_config, training_dataset_ids, test_dataset_ids


def get_training_samples_dict(
    data_config: UnifiedDataConfig,
    dataset_ids: List[str],
    use_lr_path: bool = True,
    use_hr_path: bool = False
) -> Dict[str, Dict[str, str]]:
    """
    Convert dataset IDs to the training_samples dict format expected by existing code.
    
    Args:
        data_config: UnifiedDataConfig object
        dataset_ids: List of dataset IDs
        use_lr_path: Include LR path
        use_hr_path: Include HR path
        
    Returns:
        Dictionary in the format {sample_id: {path/lr_path/hr_path: ..., name: ...}}
    """
    result = {}
    for did in dataset_ids:
        info = data_config.get_dataset(did)
        if info is None:
            continue
        
        sample_dict = {'name': info.name}
        
        if use_lr_path and use_hr_path:
            sample_dict['lr_path'] = info.lr_path
            sample_dict['hr_path'] = info.hr_path
        elif use_lr_path:
            sample_dict['path'] = info.lr_path
        elif use_hr_path:
            sample_dict['path'] = info.hr_path
        
        result[did] = sample_dict
    
    return result


class ExperimentResultSaver:
    """
    Utility class for saving experiment results with timestamp-based naming.
    
    Directory structure:
        results/experiments/{YYYYMMDD_HHMMSS}/
             config/
                experiment_config.yaml
                stage2_config.yaml
                data_config.yaml
             checkpoints/
                diffusion_model.pt
             predictions/
                {sample_id}/
                    z_hr_pred.npy
                    gene_pred.h5ad
                    metrics.json
             summary/
                 metrics_summary.csv
                 training_history.json
    """
    
    def __init__(
        self,
        base_dir: str = "results/experiments",
        timestamp_format: str = "%Y%m%d_%H%M%S",
        experiment_name: Optional[str] = None
    ):
        """
        Initialize experiment result saver.
        
        Args:
            base_dir: Base directory for experiments
            timestamp_format: Format for timestamp in directory name
            experiment_name: Optional name to append to timestamp
        """
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime(timestamp_format)
        
        if experiment_name:
            self.experiment_dir = os.path.join(base_dir, f"{self.timestamp}_{experiment_name}")
        else:
            self.experiment_dir = os.path.join(base_dir, self.timestamp)
        
        # Create subdirectories
        self.config_dir = os.path.join(self.experiment_dir, "config")
        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.predictions_dir = os.path.join(self.experiment_dir, "predictions")
        self.summary_dir = os.path.join(self.experiment_dir, "summary")
        
        self._created = False
    
    def create_directories(self):
        """Create all necessary directories."""
        if self._created:
            return
        
        for dir_path in [self.config_dir, self.checkpoint_dir, 
                         self.predictions_dir, self.summary_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        self._created = True
        print(f"Created experiment directory: {self.experiment_dir}")
    
    def save_config(
        self,
        stage2_config: Dict[str, Any],
        data_config_path: str,
        experiment_config_path: Optional[str] = None,
        training_dataset_ids: Optional[List[str]] = None,
        test_dataset_ids: Optional[List[str]] = None
    ):
        """
        Save configuration files.
        
        Args:
            stage2_config: Stage 2 configuration dictionary
            data_config_path: Path to original data_config.yaml
            experiment_config_path: Optional path to experiment config
            training_dataset_ids: Resolved training dataset IDs
            test_dataset_ids: Resolved test dataset IDs
        """
        self.create_directories()
        
        # Save stage2 config with resolved dataset IDs
        config_to_save = stage2_config.copy()
        config_to_save['_resolved_training_datasets'] = training_dataset_ids
        config_to_save['_resolved_test_datasets'] = test_dataset_ids
        config_to_save['_timestamp'] = self.timestamp
        
        stage2_path = os.path.join(self.config_dir, "stage2_config.yaml")
        with open(stage2_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_to_save, f, default_flow_style=False, allow_unicode=True)
        
        # Copy data config
        if os.path.exists(data_config_path):
            shutil.copy2(data_config_path, os.path.join(self.config_dir, "data_config.yaml"))
        
        # Copy experiment config if provided
        if experiment_config_path and os.path.exists(experiment_config_path):
            shutil.copy2(experiment_config_path, 
                        os.path.join(self.config_dir, "experiment_config.yaml"))
        
        print(f"Saved configuration to: {self.config_dir}")
    
    def save_model(self, model_state: Dict[str, Any], filename: str = "diffusion_model.pt"):
        """
        Save model checkpoint.
        
        Args:
            model_state: Model state dictionary
            filename: Checkpoint filename
        """
        import torch
        self.create_directories()
        
        model_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(model_state, model_path)
        print(f"Saved model checkpoint: {model_path}")
        
        return model_path
    
    def get_prediction_dir(self, sample_id: str) -> str:
        """
        Get prediction directory for a sample.
        
        Args:
            sample_id: Sample identifier
            
        Returns:
            Path to sample prediction directory
        """
        self.create_directories()
        sample_dir = os.path.join(self.predictions_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)
        return sample_dir
    
    def save_sample_metrics(self, sample_id: str, metrics: Dict[str, float]):
        """
        Save metrics for a single sample.
        
        Args:
            sample_id: Sample identifier
            metrics: Dictionary of metric values
        """
        import json
        
        sample_dir = self.get_prediction_dir(sample_id)
        metrics_path = os.path.join(sample_dir, "metrics.json")
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def save_metrics_summary(self, all_metrics: Dict[str, Any]):
        """
        Save metrics summary for all samples.
        
        Supports two formats:
        1. New format (inference_universal compatible):
           {
               "timestamp": "...",
               "model_path": "...", 
               "results": {
                   "sample_id": {
                       "metrics": {...},
                       "n_lr_spots": ...,
                       "n_hr_spots": ...
                   }
               }
           }
        2. Legacy format:
           {"sample_id": {metric_name: value}}
        
        Args:
            all_metrics: Dictionary of metrics (new or legacy format)
        """
        import json
        import csv
        
        self.create_directories()
        
        # Save as JSON (preserves original format)
        json_path = os.path.join(self.summary_dir, "metrics_summary.json")
        with open(json_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        # Detect format and extract results for CSV
        if 'results' in all_metrics and isinstance(all_metrics.get('results'), dict):
            # New format: extract nested metrics
            results_dict = all_metrics['results']
            flat_metrics = {}
            for sample_id, sample_data in results_dict.items():
                if isinstance(sample_data, dict) and 'metrics' in sample_data:
                    # Flatten: combine metrics and spot counts
                    flat_entry = sample_data.get('metrics', {}).copy()
                    if 'n_lr_spots' in sample_data:
                        flat_entry['n_lr_spots'] = sample_data['n_lr_spots']
                    if 'n_hr_spots' in sample_data:
                        flat_entry['n_hr_spots'] = sample_data['n_hr_spots']
                    flat_metrics[sample_id] = flat_entry
                else:
                    flat_metrics[sample_id] = sample_data
        else:
            # Legacy format: use directly
            flat_metrics = {k: v for k, v in all_metrics.items() 
                          if isinstance(v, dict)}
        
        # Save as CSV
        if flat_metrics:
            csv_path = os.path.join(self.summary_dir, "metrics_summary.csv")
            
            # Get all metric names
            metric_names = set()
            for metrics in flat_metrics.values():
                if isinstance(metrics, dict):
                    metric_names.update(metrics.keys())
            metric_names = sorted(metric_names)
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['sample_id'] + list(metric_names))
                
                for sample_id, metrics in sorted(flat_metrics.items()):
                    if isinstance(metrics, dict):
                        row = [sample_id] + [metrics.get(m, '') for m in metric_names]
                        writer.writerow(row)
            
            print(f"Saved metrics summary: {csv_path}")
    
    def save_training_history(self, history: Dict[str, List[float]]):
        """
        Save training history.
        
        Args:
            history: Training history dictionary
        """
        import json
        
        self.create_directories()
        
        history_path = os.path.join(self.summary_dir, "training_history.json")
        
        # Convert numpy arrays if present
        history_serializable = {}
        for k, v in history.items():
            if hasattr(v, 'tolist'):
                history_serializable[k] = v.tolist()
            else:
                history_serializable[k] = v
        
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        print(f"Saved training history: {history_path}")
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Get experiment summary information.
        
        Returns:
            Dictionary with experiment metadata
        """
        return {
            'experiment_dir': self.experiment_dir,
            'timestamp': self.timestamp,
            'config_dir': self.config_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'predictions_dir': self.predictions_dir,
            'summary_dir': self.summary_dir
        }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Data configuration utilities')
    parser.add_argument('--unified', type=str, help='Load and display unified data config')
    
    args = parser.parse_args()
    
    if args.unified:
        config = load_unified_data_config(args.unified)
        print(config.summary())
