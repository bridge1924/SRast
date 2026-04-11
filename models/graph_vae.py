"""
Graph Variational Autoencoder (GraphVAE) Module

This module implements the GraphVAE architecture for Stage 1 training.
The encoder uses GATConv layers to map features to a latent space,
and the decoder uses MLP layers to reconstruct the original features.

Key Feature: LatentNorm layer normalizes latent vectors before decoding,
ensuring decoder always receives standardized Gaussian input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Tuple, Optional, Dict
import numpy as np


class LatentNorm(nn.Module):
    """
    Latent Vector Normalization Layer.
    
     decoder  latent vector  decoder 
     Stage 2 
    
    
    1. /running statistics
    2. 
    3.  gamma/beta
    
    z_norm = gamma * (z - mean) / std + beta
    """
    
    def __init__(
        self, 
        latent_dim: int,
        momentum: float = 0.1,
        eps: float = 1e-6,
        affine: bool = True
    ):
        """
        Args:
            latent_dim: 
            momentum: running statistics  ( BatchNorm)
            eps: 
            affine:  gamma/beta
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.momentum = momentum
        self.eps = eps
        self.affine = affine
        
        # 
        if affine:
            self.gamma = nn.Parameter(torch.ones(latent_dim))
            self.beta = nn.Parameter(torch.zeros(latent_dim))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
        
        # Running statistics ()
        self.register_buffer('running_mean', torch.zeros(latent_dim))
        self.register_buffer('running_var', torch.ones(latent_dim))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
    
    def reset_running_stats(self):
        """"""
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        
        
         running statistics running 
         running statistics
        
        Args:
            z:  (N, latent_dim)
            
        Returns:
            z_norm:  (N, latent_dim)
        """
        if self.training:
            # 
            batch_mean = z.mean(dim=0)
            batch_var = z.var(dim=0, unbiased=False)
            
            #  running statistics (EMA)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
                self.num_batches_tracked += 1
        
        #  running statistics /
        mean = self.running_mean
        var = self.running_var
        
        # 
        z_norm = (z - mean) / torch.sqrt(var + self.eps)
        
        # 
        if self.affine:
            z_norm = self.gamma * z_norm + self.beta
        
        return z_norm
    
    def normalize_external(self, z: torch.Tensor) -> torch.Tensor:
        """
        Stage 2 
        
         running statistics
        
        Args:
            z:  (N, latent_dim)
            
        Returns:
            z_norm: 
        """
        z_norm = (z - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        if self.affine:
            z_norm = self.gamma * z_norm + self.beta
        return z_norm
    
    def get_stats(self) -> Dict[str, torch.Tensor]:
        """"""
        stats = {
            'running_mean': self.running_mean.clone(),
            'running_var': self.running_var.clone(),
            'num_batches_tracked': self.num_batches_tracked.clone()
        }
        if self.affine:
            stats['gamma'] = self.gamma.clone()
            stats['beta'] = self.beta.clone()
        return stats
    
    def extra_repr(self) -> str:
        return f'latent_dim={self.latent_dim}, momentum={self.momentum}, affine={self.affine}'


class GATEncoder(nn.Module):
    """
    Graph Attention Network Encoder.
    
    Uses two GATConv layers to encode node features into a latent space
    with variational inference (outputs mu and log_var).
    """
    
    def __init__(
        self,
        input_dim: int = 50,
        hidden_dim: int = 64,
        latent_dim: int = 30,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat_heads: bool = True
    ):
        """
        Initialize the GAT Encoder.
        
        Args:
            input_dim: Input feature dimension (PCA components)
            hidden_dim: Hidden layer dimension
            latent_dim: Latent space dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            concat_heads: Whether to concatenate attention heads
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.concat_heads = concat_heads
        
        # First GAT layer
        self.gat1 = GATConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            concat=concat_heads,
            dropout=dropout,
            add_self_loops=True
        )
        
        # Compute intermediate dimension
        gat1_out_dim = hidden_dim * num_heads if concat_heads else hidden_dim
        
        # Second GAT layer (for mu)
        self.gat2_mu = GATConv(
            in_channels=gat1_out_dim,
            out_channels=latent_dim,
            heads=1,
            concat=False,
            dropout=dropout,
            add_self_loops=True
        )
        
        # Second GAT layer (for log_var)
        self.gat2_logvar = GATConv(
            in_channels=gat1_out_dim,
            out_channels=latent_dim,
            heads=1,
            concat=False,
            dropout=dropout,
            add_self_loops=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(gat1_out_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.
        
        Args:
            x: Node features (N, input_dim)
            edge_index: Edge indices (2, E)
            edge_attr: Optional edge attributes
            
        Returns:
            mu: Mean of latent distribution (N, latent_dim)
            log_var: Log variance of latent distribution (N, latent_dim)
        """
        # First GAT layer with ELU activation
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = self.norm1(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Second GAT layers for mu and log_var
        mu = self.gat2_mu(h, edge_index)
        log_var = self.gat2_logvar(h, edge_index)
        
        return mu, log_var


class MLPDecoder(nn.Module):
    """
    Multi-Layer Perceptron Decoder.
    
    Uses three fully connected layers to reconstruct features from latent space.
    """
    
    def __init__(
        self,
        latent_dim: int = 30,
        hidden_dims: Tuple[int, ...] = (64, 128),
        output_dim: int = 50,
        dropout: float = 0.1
    ):
        """
        Initialize the MLP Decoder.
        
        Args:
            latent_dim: Latent space dimension
            hidden_dims: Hidden layer dimensions
            output_dim: Output dimension (should match input_dim of encoder)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Build decoder layers
        layers = []
        in_dim = latent_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ELU(),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.
        
        Args:
            z: Latent representations (N, latent_dim)
            
        Returns:
            x_recon: Reconstructed features (N, output_dim)
        """
        return self.decoder(z)


class GraphVAE(nn.Module):
    """
    Graph Variational Autoencoder.
    
    Combines GATEncoder and MLPDecoder with variational inference
    for learning latent representations of spatial transcriptomics data.
    
    Key Feature: LatentNorm layer between encoder and decoder ensures
    the decoder always receives standardized Gaussian input, solving
    the distribution drift problem in Stage 2 zero-shot inference.
    
    NOTE: Now supports different input_dim (PCA features) and output_dim (HVG expression).
    """
    
    def __init__(
        self,
        input_dim: int = 50,
        hidden_dim: int = 64,
        latent_dim: int = 30,
        decoder_hidden_dims: Tuple[int, ...] = (64, 128),
        output_dim: int = None,  # If None, defaults to input_dim (original behavior)
        num_heads: int = 4,
        dropout: float = 0.1,
        use_latent_norm: bool = True,  #  LatentNorm
        latent_norm_momentum: float = 0.1
    ):
        """
        Initialize GraphVAE.
        
        Args:
            input_dim: Input feature dimension (PCA components for encoder)
            hidden_dim: Encoder hidden dimension
            latent_dim: Latent space dimension
            decoder_hidden_dims: Decoder hidden dimensions
            output_dim: Output dimension (HVG genes for decoder). If None, equals input_dim.
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_latent_norm: Whether to use LatentNorm before decoder (recommended: True)
            latent_norm_momentum: Momentum for LatentNorm running statistics
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.use_latent_norm = use_latent_norm
        
        # Encoder
        self.encoder = GATEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # LatentNorm -  decoder  latent
        if use_latent_norm:
            self.latent_norm = LatentNorm(
                latent_dim=latent_dim,
                momentum=latent_norm_momentum,
                affine=True
            )
        else:
            self.latent_norm = None
        
        # Decoder - output to HVG expression space
        self.decoder = MLPDecoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
            output_dim=self.output_dim,
            dropout=dropout
        )
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for variational inference.
        
        Args:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
            
        Returns:
            z: Sampled latent vector
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input features to latent space.
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Optional edge attributes
            
        Returns:
            z: Sampled latent vector
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
        """
        mu, log_var = self.encoder(x, edge_index, edge_attr)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var
    
    def decode(self, z: torch.Tensor, apply_latent_norm: bool = True) -> torch.Tensor:
        """
        Decode latent vector to reconstructed features.
        
        Args:
            z: Latent vector (N, latent_dim)
            apply_latent_norm: Whether to apply LatentNorm before decoding
                              (default: True, set False if z is already normalized)
            
        Returns:
            x_recon: Reconstructed features (N, output_dim)
        """
        if apply_latent_norm and self.latent_norm is not None:
            z = self.latent_norm(z)
        return self.decoder(z)
    
    def decode_normalized(self, z_normalized: torch.Tensor) -> torch.Tensor:
        """
        Decode already-normalized latent vector (for Stage 2 inference).
        
         Stage 2  z 
         LatentNorm
        
        Args:
            z_normalized: Already normalized latent vector
            
        Returns:
            x_recon: Reconstructed features
        """
        return self.decoder(z_normalized)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass through GraphVAE.
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Optional edge attributes
            
        Returns:
            x_recon: Reconstructed features
            z: Latent vector
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
        """
        z, mu, log_var = self.encode(x, edge_index, edge_attr)
        x_recon = self.decode(z)
        return x_recon, z, mu, log_var
    
    def freeze_decoder(self):
        """Freeze decoder parameters after Stage 1 training."""
        for param in self.decoder.parameters():
            param.requires_grad = False
        print("Decoder parameters frozen.")
    
    def unfreeze_decoder(self):
        """Unfreeze decoder parameters."""
        for param in self.decoder.parameters():
            param.requires_grad = True
        print("Decoder parameters unfrozen.")
    
    def get_latent(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get latent representation without reconstruction.
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Optional edge attributes
            
        Returns:
            mu: Mean latent representation (deterministic)
        """
        mu, _ = self.encoder(x, edge_index, edge_attr)
        return mu
    
    def get_latent_norm_stats(self) -> Optional[Dict[str, torch.Tensor]]:
        """
         LatentNorm  Stage 2 
        
        Returns:
            Dict containing running_mean, running_var, gamma, beta
            or None if LatentNorm is not used
        """
        if self.latent_norm is not None:
            return self.latent_norm.get_stats()
        return None
    
    def normalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
         LatentNorm  latent vector
        
         Stage 2 
         z_hr_pred     decoder
        
        Args:
            z:  latent vector (N, latent_dim)
            
        Returns:
            z_norm:  latent vector
        """
        if self.latent_norm is not None:
            return self.latent_norm.normalize_external(z)
        return z
    
    def reset_latent_norm_stats(self):
        """ LatentNorm """
        if self.latent_norm is not None:
            self.latent_norm.reset_running_stats()
            print("LatentNorm running statistics reset.")


class GraphVAELoss(nn.Module):
    """
    Loss function for GraphVAE training.
    
    
    1. ELBO = MSE reconstruction loss + KL divergence
    2. Cosine similarity loss (gene - )
    3. PCC Loss (gene + spot )
    
    
    """
    
    def __init__(
        self, 
        recon_weight: float = 1.0,
        kl_weight: float = 0.001, 
        cosine_weight: float = 1.0, 
        pcc_weight: float = 10.0,
        pcc_gene_weight: float = 0.7,
        pcc_spot_weight: float = 0.3
    ):
        """
        Initialize loss function.
        
        Args:
            recon_weight: Weight for MSE reconstruction loss (default: 1.0)
            kl_weight: Weight for KL divergence term (beta in beta-VAE)
            cosine_weight: Weight for cosine similarity loss
            pcc_weight: Weight for Pearson Correlation loss (recommended: 10.0)
            pcc_gene_weight: Weight for gene-direction PCC loss (default: 0.7)
            pcc_spot_weight: Weight for spot-direction PCC loss (default: 0.3)
        """
        super().__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.cosine_weight = cosine_weight
        self.pcc_weight = pcc_weight
        self.pcc_gene_weight = pcc_gene_weight
        self.pcc_spot_weight = pcc_spot_weight
        self.mse = nn.MSELoss(reduction='mean')
    
    def pearson_correlation_loss(self, x: torch.Tensor, y: torch.Tensor, dim: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Pearson Correlation Loss: 1 - mean(PCC)
        
        PCC is computed along the specified dimension.
        For per-gene PCC, we compute correlation across spots (dim=0).
        
        Args:
            x: Predicted values [N_spots, N_genes]
            y: Target values [N_spots, N_genes]
            dim: Dimension along which to compute correlation
                 dim=0: per-gene PCC (correlation across spots for each gene)
                 dim=1: per-spot PCC (correlation across genes for each spot)
        
        Returns:
            loss: 1 - mean(PCC)
            mean_pcc: Mean PCC value for logging
        """
        # Center the data (subtract mean along the specified dimension)
        x_centered = x - torch.mean(x, dim=dim, keepdim=True)
        y_centered = y - torch.mean(y, dim=dim, keepdim=True)
        
        # Compute covariance
        cov = torch.sum(x_centered * y_centered, dim=dim)
        
        # Compute standard deviations with epsilon for numerical stability
        eps = 1e-8
        std_x = torch.sqrt(torch.sum(x_centered ** 2, dim=dim) + eps)
        std_y = torch.sqrt(torch.sum(y_centered ** 2, dim=dim) + eps)
        
        # Compute Pearson correlation coefficient
        pcc = cov / (std_x * std_y)
        
        # Clamp to valid range [-1, 1] to handle numerical issues
        pcc = torch.clamp(pcc, -1.0, 1.0)
        
        # Loss is 1 - mean(PCC), so we minimize loss to maximize PCC
        mean_pcc = torch.mean(pcc)
        loss = 1.0 - mean_pcc
        
        return loss, mean_pcc
    
    def forward(
        self,
        x_recon: torch.Tensor,
        x_target: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GraphVAE loss.
        
        
        1. ELBO = (MSE) + KL
        2. gene
        3. PCCgene + spot 0.7:0.3
        
        Args:
            x_recon: Reconstructed features (decoder output) [N_spots, N_genes]
            x_target: Target features for reconstruction [N_spots, N_genes]
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
            
        Returns:
            loss: Total loss
            loss_dict: Dictionary with individual loss components
        """
        # =====================================================================
        # 1. ELBO: Reconstruction loss (MSE) + KL divergence
        # =====================================================================
        # Reconstruction loss - for magnitude accuracy
        recon_loss = self.mse(x_recon, x_target)
        
        # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        
        # =====================================================================
        # 2. Cosine similarity loss (gene) - for gene expression pattern
        # =====================================================================
        cos_sim_gene = F.cosine_similarity(x_recon.t(), x_target.t(), dim=1)  # [N_genes]
        cosine_loss = 1.0 - cos_sim_gene.mean()
        
        # =====================================================================
        # 3. PCC loss - optimizes both gene and spot correlation
        # =====================================================================
        # Per-gene PCC: correlation across spots for each gene (dim=0)
        pcc_loss_gene, mean_pcc_gene = self.pearson_correlation_loss(x_recon, x_target, dim=0)
        
        # Per-spot PCC: correlation across genes for each spot (dim=1)
        pcc_loss_spot, mean_pcc_spot = self.pearson_correlation_loss(x_recon, x_target, dim=1)
        
        # Combined PCC loss (weighted by configurable pcc_gene_weight and pcc_spot_weight)
        pcc_loss = self.pcc_gene_weight * pcc_loss_gene + self.pcc_spot_weight * pcc_loss_spot
        
        # =====================================================================
        # Total loss = ELBO + Cosine + PCC (gene + spot)
        # 
        # =====================================================================
        total_loss = (self.recon_weight * recon_loss + 
                      self.kl_weight * kl_loss + 
                      self.cosine_weight * cosine_loss + 
                      self.pcc_weight * pcc_loss)
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'cosine_loss': cosine_loss.item(),
            'pcc_loss': pcc_loss.item(),  # Combined PCC loss
            'pcc_gene': mean_pcc_gene.item(),  # This is what we want to maximize!
            'pcc_spot': mean_pcc_spot.item(),  # 
            'cos_sim_gene': cos_sim_gene.mean().item()
        }
        
        return total_loss, loss_dict


class GraphVAETrainer:
    """
    Trainer class for GraphVAE Stage 1 training.
    """
    
    def __init__(
        self,
        model: GraphVAE,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        recon_weight: float = 1.0,
        kl_weight: float = 0.001,
        cosine_weight: float = 1.0,
        pcc_weight: float = 10.0,
        pcc_gene_weight: float = 0.7,
        pcc_spot_weight: float = 0.3,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 10,
        grad_clip_max_norm: float = 1.0,
        device: str = 'cuda'
    ):
        """
        Initialize trainer.
        
        Args:
            model: GraphVAE model
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            recon_weight: MSE reconstruction loss weight (default: 1.0)
            kl_weight: KL divergence weight
            cosine_weight: Cosine similarity loss weight
            pcc_weight: Pearson correlation loss weight (high value recommended)
            pcc_gene_weight: Weight for gene-direction PCC loss
            pcc_spot_weight: Weight for spot-direction PCC loss
            scheduler_factor: Factor by which the learning rate will be reduced
            scheduler_patience: Number of epochs with no improvement after which lr is reduced
            grad_clip_max_norm: Max norm for gradient clipping
            device: Device to train on
        """
        self.model = model.to(device)
        self.device = device
        self.grad_clip_max_norm = grad_clip_max_norm
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.loss_fn = GraphVAELoss(
            recon_weight=recon_weight,
            kl_weight=kl_weight, 
            cosine_weight=cosine_weight,
            pcc_weight=pcc_weight,
            pcc_gene_weight=pcc_gene_weight,
            pcc_spot_weight=pcc_spot_weight
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True
        )
        
        self.history = {
            'total_loss': [],
            'recon_loss': [],
            'kl_loss': [],
            'cosine_loss': [],
            'pcc_loss': [],
            'pcc_gene': [],
            'pcc_spot': [],
            'cos_sim_gene': []
        }
    
    def train_epoch(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        x_target: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            x: Node features (encoder input, e.g., PCA features)
            edge_index: Edge indices
            edge_attr: Optional edge attributes
            x_target: Reconstruction target (e.g., HVG expression). If None, uses x.
            
        Returns:
            loss_dict: Dictionary with loss values
        """
        self.model.train()
        
        # Move data to device
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        if edge_attr is not None:
            edge_attr = edge_attr.to(self.device)
        
        # Use x as target if not specified
        if x_target is None:
            x_target = x
        else:
            x_target = x_target.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        x_recon, z, mu, log_var = self.model(x, edge_index, edge_attr)
        
        # Compute loss - use x_target instead of x
        loss, loss_dict = self.loss_fn(x_recon, x_target, mu, log_var)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_max_norm)
        
        self.optimizer.step()
        
        return loss_dict
    
    def train(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        x_target: Optional[torch.Tensor] = None,
        epochs: int = 200,
        early_stopping_patience: int = 20,
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        Full training loop.
        
        Args:
            x: Node features (encoder input)
            edge_index: Edge indices
            edge_attr: Optional edge attributes
            x_target: Reconstruction target (if None, uses x)
            epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            verbose: Print training progress
            
        Returns:
            history: Training history
        """
        best_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            loss_dict = self.train_epoch(x, edge_index, edge_attr, x_target)
            
            # Record history
            for key, value in loss_dict.items():
                if key in self.history:
                    self.history[key].append(value)
            
            # Learning rate scheduling
            self.scheduler.step(loss_dict['total_loss'])
            
            # Early stopping check
            if loss_dict['total_loss'] < best_loss:
                best_loss = loss_dict['total_loss']
                patience_counter = 0
                best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | "
                      f"Loss: {loss_dict['total_loss']:.4f} | "
                      f"Recon: {loss_dict['recon_loss']:.4f} | "
                      f"PCC_loss: {loss_dict['pcc_loss']:.4f} | "
                      f"PCC(gene): {loss_dict['pcc_gene']:.4f} | "
                      f"PCC(spot): {loss_dict['pcc_spot']:.4f}")
            
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return self.history
    
    @torch.no_grad()
    def get_latent_representations(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch_size: int = 5000
    ) -> np.ndarray:
        """
        Get latent representations for all nodes.
        
         mini-batch  OOM
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Optional edge attributes
            batch_size: Batch size for large datasets
            
        Returns:
            z: Latent representations as numpy array
        """
        self.model.eval()
        n_nodes = x.shape[0]
        
        # 
        if n_nodes <= batch_size:
            x = x.to(self.device)
            edge_index = edge_index.to(self.device)
            if edge_attr is not None:
                edge_attr = edge_attr.to(self.device)
            z = self.model.get_latent(x, edge_index, edge_attr)
            return z.cpu().numpy()
        
        # 
        print(f"      Using mini-batch inference ({n_nodes} nodes, batch_size={batch_size})")
        
        try:
            from torch_geometric.loader import NeighborLoader
            from torch_geometric.data import Data
            
            data = Data(x=x, edge_index=edge_index)
            if edge_attr is not None:
                data.edge_attr = edge_attr
            
            loader = NeighborLoader(
                data,
                num_neighbors=[10, 10],
                batch_size=batch_size,
                input_nodes=torch.arange(n_nodes),
                shuffle=False
            )
            
            all_z = []
            for batch_data in loader:
                batch_data = batch_data.to(self.device)
                batch_z = self.model.get_latent(
                    batch_data.x,
                    batch_data.edge_index,
                    batch_data.edge_attr if hasattr(batch_data, 'edge_attr') else None
                )
                n_batch = min(batch_size, batch_data.batch_size if hasattr(batch_data, 'batch_size') else batch_z.shape[0])
                all_z.append(batch_z[:n_batch].cpu())
                del batch_data, batch_z
                torch.cuda.empty_cache()
            
            z = torch.cat(all_z, dim=0)
            return z.numpy()
            
        except Exception as e:
            print(f"      NeighborLoader failed: {e}, using simple batching")
            
            all_z = []
            x = x.to(self.device)
            
            for i in range(0, n_nodes, batch_size):
                end_idx = min(i + batch_size, n_nodes)
                batch_x = x[i:end_idx]
                
                mask = ((edge_index[0] >= i) & (edge_index[0] < end_idx) &
                        (edge_index[1] >= i) & (edge_index[1] < end_idx))
                batch_edge_index = edge_index[:, mask] - i
                batch_edge_index = batch_edge_index.to(self.device)
                
                batch_edge_attr = None
                if edge_attr is not None and mask.sum() > 0:
                    batch_edge_attr = edge_attr[mask].to(self.device)
                
                if batch_edge_index.shape[1] == 0:
                    batch_n = batch_x.shape[0]
                    batch_edge_index = torch.stack([
                        torch.arange(batch_n, device=self.device),
                        torch.arange(batch_n, device=self.device)
                    ])
                
                batch_z = self.model.get_latent(batch_x, batch_edge_index, batch_edge_attr)
                all_z.append(batch_z.cpu())
                del batch_x, batch_edge_index, batch_z
                torch.cuda.empty_cache()
            
            z = torch.cat(all_z, dim=0)
            return z.numpy()


class MiniBatchGraphVAETrainer:
    """
    Memory-efficient trainer using graph clustering for large datasets.
    
    Uses PyG's ClusterData to partition the graph into subgraphs,
    enabling training on datasets that don't fit in GPU memory.
    
    Key features:
    - Graph clustering to preserve local structure
    - Gradient accumulation for stable training
    - Mixed precision training support
    - Memory-efficient forward passes
    """
    
    def __init__(
        self,
        model: GraphVAE,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        recon_weight: float = 1.0,
        kl_weight: float = 0.001,
        cosine_weight: float = 1.0,
        pcc_weight: float = 10.0,
        pcc_gene_weight: float = 0.7,
        pcc_spot_weight: float = 0.3,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 10,
        grad_clip_max_norm: float = 1.0,
        device: str = 'cuda',
        num_parts: int = 100,           # Number of graph partitions
        batch_size: int = 10,           # Clusters per batch
        gradient_accumulation: int = 1, # Accumulation steps
        use_amp: bool = True            # Mixed precision
    ):
        """
        Initialize mini-batch trainer.
        
        Args:
            model: GraphVAE model
            learning_rate: Learning rate
            weight_decay: Weight decay
            recon_weight: MSE reconstruction loss weight (default: 1.0)
            kl_weight: KL loss weight
            cosine_weight: Cosine loss weight  
            pcc_weight: PCC loss weight
            pcc_gene_weight: Weight for gene-direction PCC loss
            pcc_spot_weight: Weight for spot-direction PCC loss
            scheduler_factor: Factor by which the learning rate will be reduced
            scheduler_patience: Number of epochs with no improvement after which lr is reduced
            grad_clip_max_norm: Max norm for gradient clipping
            device: Training device
            num_parts: Number of graph partitions (more = smaller subgraphs)
            batch_size: Number of clusters per mini-batch
            gradient_accumulation: Number of steps to accumulate gradients
            use_amp: Whether to use automatic mixed precision
        """
        self.model = model.to(device)
        self.device = device
        self.num_parts = num_parts
        self.batch_size = batch_size
        self.gradient_accumulation = gradient_accumulation
        self.use_amp = use_amp and torch.cuda.is_available()
        self.grad_clip_max_norm = grad_clip_max_norm
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.loss_fn = GraphVAELoss(
            recon_weight=recon_weight,
            kl_weight=kl_weight,
            cosine_weight=cosine_weight,
            pcc_weight=pcc_weight,
            pcc_gene_weight=pcc_gene_weight,
            pcc_spot_weight=pcc_spot_weight
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True
        )
        
        # Mixed precision scaler (compatible with older PyTorch versions)
        if self.use_amp:
            try:
                # PyTorch >= 2.0
                self.scaler = torch.amp.GradScaler('cuda')
            except (AttributeError, TypeError):
                # PyTorch < 2.0
                self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        self.history = {
            'total_loss': [], 'recon_loss': [], 'kl_loss': [],
            'cosine_loss': [], 'pcc_loss': [], 'pcc_gene': [],
            'pcc_spot': [], 'cos_sim_gene': []
        }
    
    def _create_cluster_loader(self, data: Data) -> 'ClusterLoader':
        """Create ClusterLoader for the graph."""
        from torch_geometric.loader import ClusterData, ClusterLoader
        
        # Partition graph into clusters
        cluster_data = ClusterData(
            data,
            num_parts=self.num_parts,
            recursive=False,
            log=False
        )
        
        # Create loader
        loader = ClusterLoader(
            cluster_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        return loader
    
    def train_epoch(
        self,
        loader: 'ClusterLoader',
        x_target_full: torch.Tensor
    ) -> Dict[str, float]:
        """
        Train for one epoch using mini-batches.
        
        Args:
            loader: ClusterLoader with partitioned graph
            x_target_full: Full target tensor (on CPU)
            
        Returns:
            Averaged loss dict for the epoch
        """
        self.model.train()
        
        epoch_losses = {k: 0.0 for k in self.history.keys()}
        num_batches = 0
        accumulation_count = 0
        
        self.optimizer.zero_grad()
        
        for batch in loader:
            batch = batch.to(self.device)
            
            # Get target for this batch using original indices
            if hasattr(batch, 'original_indices'):
                batch_indices = batch.original_indices.cpu()  # Keep on CPU for indexing
            else:
                # Fallback: ClusterData stores mapping
                batch_indices = batch.n_id.cpu() if hasattr(batch, 'n_id') else None
            
            if batch_indices is not None:
                x_target = x_target_full[batch_indices].to(self.device)
            else:
                x_target = batch.x_target.to(self.device) if hasattr(batch, 'x_target') else batch.x.to(self.device)
            
            # Forward pass with optional AMP
            if self.use_amp:
                try:
                    # PyTorch >= 2.0
                    amp_context = torch.amp.autocast('cuda')
                except (AttributeError, TypeError):
                    # PyTorch < 2.0
                    amp_context = torch.cuda.amp.autocast()
                
                with amp_context:
                    x_recon, z, mu, log_var = self.model(
                        batch.x, batch.edge_index, 
                        batch.edge_attr if hasattr(batch, 'edge_attr') else None
                    )
                    loss, loss_dict = self.loss_fn(x_recon, x_target, mu, log_var)
                    loss = loss / self.gradient_accumulation
                
                self.scaler.scale(loss).backward()
            else:
                x_recon, z, mu, log_var = self.model(
                    batch.x, batch.edge_index,
                    batch.edge_attr if hasattr(batch, 'edge_attr') else None
                )
                loss, loss_dict = self.loss_fn(x_recon, x_target, mu, log_var)
                loss = loss / self.gradient_accumulation
                loss.backward()
            
            accumulation_count += 1
            
            # Update weights after accumulation
            if accumulation_count >= self.gradient_accumulation:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_max_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_max_norm)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                accumulation_count = 0
            
            # Accumulate losses
            for k, v in loss_dict.items():
                if k in epoch_losses:
                    epoch_losses[k] += v
            num_batches += 1
            
            # Free memory
            del batch, x_target, x_recon, z, mu, log_var, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Handle remaining gradients
        if accumulation_count > 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_max_norm)
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= max(num_batches, 1)
        
        return epoch_losses
    
    def train(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        x_target: Optional[torch.Tensor] = None,
        epochs: int = 200,
        early_stopping_patience: int = 20,
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        Full training loop with mini-batches.
        
        Args:
            x: Node features
            edge_index: Edge indices  
            edge_attr: Edge attributes
            x_target: Reconstruction target
            epochs: Training epochs
            early_stopping_patience: Early stopping patience
            verbose: Print progress
            
        Returns:
            Training history
        """
        import gc
        
        # Keep target on CPU to save GPU memory
        if x_target is None:
            x_target = x
        x_target_cpu = x_target.cpu() if x_target.is_cuda else x_target
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=x.shape[0]
        )
        
        # Store original indices for target mapping
        data.n_id = torch.arange(x.shape[0])
        
        # Create cluster loader
        print(f"      Creating graph clusters (num_parts={self.num_parts})...")
        loader = self._create_cluster_loader(data)
        print(f"      Clusters created. Batches per epoch: {len(loader)}")
        
        # Free original data tensors from GPU
        del data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        best_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            loss_dict = self.train_epoch(loader, x_target_cpu)
            
            # Record history
            for key, value in loss_dict.items():
                if key in self.history:
                    self.history[key].append(value)
            
            # Learning rate scheduling
            self.scheduler.step(loss_dict['total_loss'])
            
            # Early stopping
            if loss_dict['total_loss'] < best_loss:
                best_loss = loss_dict['total_loss']
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | "
                      f"Loss: {loss_dict['total_loss']:.4f} | "
                      f"Recon: {loss_dict['recon_loss']:.4f} | "
                      f"PCC(gene): {loss_dict['pcc_gene']:.4f} | "
                      f"PCC(spot): {loss_dict['pcc_spot']:.4f}")
            
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return self.history
    
    @torch.no_grad()
    def get_latent_representations(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch_size: int = 5000
    ) -> np.ndarray:
        """
        Get latent representations in batches to save memory.
        
         mini-batch  OOM
         NeighborLoader 
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Optional edge attributes
            batch_size: Number of nodes per batch (default: 5000)
            
        Returns:
            Latent representations as numpy array
        """
        self.model.eval()
        
        n_nodes = x.shape[0]
        
        # 
        if n_nodes <= batch_size:
            x = x.to(self.device)
            edge_index = edge_index.to(self.device)
            if edge_attr is not None:
                edge_attr = edge_attr.to(self.device)
            z = self.model.get_latent(x, edge_index, edge_attr)
            return z.cpu().numpy()
        
        #  mini-batch 
        print(f"      Using mini-batch inference ({n_nodes} nodes, batch_size={batch_size})")
        
        try:
            from torch_geometric.loader import NeighborLoader
            from torch_geometric.data import Data
            
            #  PyG Data 
            data = Data(x=x, edge_index=edge_index)
            if edge_attr is not None:
                data.edge_attr = edge_attr
            
            #  NeighborLoader 
            # num_neighbors=-1 
            # 
            loader = NeighborLoader(
                data,
                num_neighbors=[10, 10],  # 210
                batch_size=batch_size,
                input_nodes=torch.arange(n_nodes),
                shuffle=False
            )
            
            all_z = []
            for batch_data in loader:
                batch_data = batch_data.to(self.device)
                #  latent
                batch_z = self.model.get_latent(
                    batch_data.x, 
                    batch_data.edge_index,
                    batch_data.edge_attr if hasattr(batch_data, 'edge_attr') else None
                )
                #  batch  batch_size 
                n_batch = min(batch_size, batch_data.batch_size if hasattr(batch_data, 'batch_size') else batch_z.shape[0])
                all_z.append(batch_z[:n_batch].cpu())
                
                # 
                del batch_data, batch_z
                torch.cuda.empty_cache()
            
            z = torch.cat(all_z, dim=0)
            return z.numpy()
            
        except Exception as e:
            print(f"      NeighborLoader failed: {e}")
            print(f"      Falling back to simple batching (may lose some graph info)")
            
            # 
            # 
            all_z = []
            x = x.to(self.device)
            
            for i in range(0, n_nodes, batch_size):
                end_idx = min(i + batch_size, n_nodes)
                batch_x = x[i:end_idx]
                
                # 
                mask = ((edge_index[0] >= i) & (edge_index[0] < end_idx) & 
                        (edge_index[1] >= i) & (edge_index[1] < end_idx))
                batch_edge_index = edge_index[:, mask] - i  # 
                batch_edge_index = batch_edge_index.to(self.device)
                
                batch_edge_attr = None
                if edge_attr is not None and mask.sum() > 0:
                    batch_edge_attr = edge_attr[mask].to(self.device)
                
                # 
                if batch_edge_index.shape[1] == 0:
                    batch_n = batch_x.shape[0]
                    batch_edge_index = torch.stack([
                        torch.arange(batch_n, device=self.device),
                        torch.arange(batch_n, device=self.device)
                    ])
                
                batch_z = self.model.get_latent(batch_x, batch_edge_index, batch_edge_attr)
                all_z.append(batch_z.cpu())
                
                # 
                del batch_x, batch_edge_index, batch_z
                if batch_edge_attr is not None:
                    del batch_edge_attr
                torch.cuda.empty_cache()
            
            z = torch.cat(all_z, dim=0)
            return z.numpy()
