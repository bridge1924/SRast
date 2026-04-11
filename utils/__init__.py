"""
Utils module initialization
"""

from .graph_utils import (
    build_spatial_graph,
    build_feature_graph,
    build_heterogeneous_graph,
    build_lr_hr_bipartite_graph,
    create_pyg_data,
    compute_graph_statistics
)

from .metrics import (
    compute_metrics,
    compute_all_metrics,
    compute_snr,
    compute_ssim,
    compute_fractional_deviation,
    compute_js_divergence,
    compute_js_divergence_per_gene,
    pearson_correlation_coefficient,
    root_mean_square_error,
    mean_absolute_percentage_error,
    spearman_correlation,
    structural_similarity_spatial,
    evaluate_reconstruction,
    gene_wise_metrics
)

from .io_utils import (
    save_checkpoint,
    load_checkpoint,
    save_config,
    load_config,
    save_results_h5ad,
    save_training_history,
    load_training_history,
    create_output_directory,
    save_numpy_arrays,
    load_numpy_arrays
)

from .config_loader import (
    # Unified config support
    DatasetInfo,
    UnifiedDataConfig,
    load_unified_data_config,
    load_stage1_config,
    load_stage2_config,
    get_training_samples_dict,
    ExperimentResultSaver
)

from .training_utils import (
    clear_memory,
    set_seed,
    get_device,
    estimate_memory_usage,
    spatial_uniform_subsample,
    EarlyStopping,
    AverageMeter
)

from .data_loading import (
    build_lr_hr_mapping,
    load_stage1_latent_norm,
    apply_latent_norm,
    load_test_sample,
    load_training_sample
)

__all__ = [
    # Graph utilities
    'build_spatial_graph',
    'build_feature_graph',
    'build_heterogeneous_graph',
    'build_lr_hr_bipartite_graph',
    'create_pyg_data',
    'compute_graph_statistics',
    # Metrics
    'compute_metrics',
    'compute_all_metrics',
    'compute_snr',
    'compute_ssim',
    'compute_fractional_deviation',
    'compute_js_divergence',
    'compute_js_divergence_per_gene',
    'pearson_correlation_coefficient',
    'root_mean_square_error',
    'mean_absolute_percentage_error',
    'spearman_correlation',
    'structural_similarity_spatial',
    'evaluate_reconstruction',
    'gene_wise_metrics',
    # I/O utilities
    'save_checkpoint',
    'load_checkpoint',
    'save_config',
    'load_config',
    'save_results_h5ad',
    'save_training_history',
    'load_training_history',
    'create_output_directory',
    'save_numpy_arrays',
    'load_numpy_arrays',
    # Config utilities (unified format)
    'DatasetInfo',
    'UnifiedDataConfig',
    'load_unified_data_config',
    'load_stage1_config',
    'load_stage2_config',
    'get_training_samples_dict',
    'ExperimentResultSaver',
    # Training utilities
    'clear_memory',
    'set_seed',
    'get_device',
    'estimate_memory_usage',
    'spatial_uniform_subsample',
    'EarlyStopping',
    'AverageMeter',
    # Data loading utilities
    'build_lr_hr_mapping',
    'load_stage1_latent_norm',
    'apply_latent_norm',
    'load_test_sample',
    'load_training_sample'
]
