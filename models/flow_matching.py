"""
Flow Matching Model with HR Spatial Prior for Spatial Transcriptomics Super-Resolution

This module implements a Flow Matching approach () combined with 
HR spatial prior (HR spots).

Key Improvements over Ratio Diffusion:
1. Flow Matching: OT flow
   - 
   - 
   
2. HR Spatial Prior: HR spots
   - HR spots
   - 

References:
- Lipman et al. "Flow Matching for Generative Modeling" (2023)
- Optimal Transport Flow Matching (OT-FM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


class HRSpatialGraph(nn.Module):
    """
    Builds and processes HR spatial neighborhood graph.
    
    HR spots
    - spots
    - smooth noisy predictions
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_neighbors: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_neighbors = num_neighbors
        self.num_heads = num_heads
        
        # Edge feature encoder (distance-based)
        self.edge_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )
        
        # Spatial attention layer
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        
        # Apply Xavier initialization
        self._init_weights()
    
    def _init_weights(self):
        """Apply Xavier initialization to all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def build_knn_graph(
        self,
        hr_coords: torch.Tensor,
        k: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build k-NN graph for HR spots based on spatial coordinates.
        
        Args:
            hr_coords: HR spot coordinates (N_hr, 2)
            k: Number of neighbors (default: self.num_neighbors)
            
        Returns:
            edge_index: Edge indices (2, num_edges)
            edge_weight: Edge weights based on distance (num_edges,)
        """
        if k is None:
            k = self.num_neighbors
            
        n_hr = hr_coords.size(0)
        k = min(k, n_hr - 1)  # Cannot have more neighbors than nodes
        
        if k <= 0:
            # No neighbors possible, return empty
            device = hr_coords.device
            return (torch.zeros(2, 0, dtype=torch.long, device=device),
                    torch.zeros(0, device=device))
        
        # Use sklearn for efficient k-NN
        coords_np = hr_coords.detach().cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(coords_np)
        distances, indices = nbrs.kneighbors(coords_np)
        
        # Remove self-loops (first neighbor is always self)
        distances = distances[:, 1:]  # (N_hr, k)
        indices = indices[:, 1:]  # (N_hr, k)
        
        # Build edge index
        device = hr_coords.device
        src = torch.arange(n_hr, device=device).unsqueeze(1).expand(-1, k).flatten()
        dst = torch.tensor(indices.flatten(), device=device)
        edge_index = torch.stack([src, dst], dim=0)
        
        # Edge weights (inverse distance, normalized)
        distances_tensor = torch.tensor(distances.flatten(), device=device, dtype=torch.float32)
        edge_weight = 1.0 / (distances_tensor + 1e-6)
        edge_weight = edge_weight / edge_weight.max()  # Normalize to [0, 1]
        
        return edge_index, edge_weight
    
    def forward(
        self,
        h: torch.Tensor,
        hr_coords: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply spatial attention based on HR neighborhood.
        
        Args:
            h: HR spot features (N_hr, hidden_dim)
            hr_coords: HR coordinates (N_hr, 2)
            edge_index: Pre-computed edge index (optional)
            edge_weight: Pre-computed edge weights (optional)
            
        Returns:
            h_out: Updated features with spatial context (N_hr, hidden_dim)
        """
        n_hr = h.size(0)
        device = h.device
        
        # Build graph if not provided
        if edge_index is None:
            edge_index, edge_weight = self.build_knn_graph(hr_coords)
        
        if edge_index.size(1) == 0:
            # No edges, return input
            return h
        
        # Simple approach: use full attention with spatial bias
        # (For efficiency, can use sparse attention in production)
        
        # Create attention mask based on k-NN graph
        attn_mask = torch.zeros(n_hr, n_hr, device=device)
        attn_mask[edge_index[0], edge_index[1]] = edge_weight
        attn_mask = attn_mask + torch.eye(n_hr, device=device)  # Self-connection
        
        # Convert to attention bias (mask out non-neighbors)
        attn_bias = torch.where(
            attn_mask > 0,
            torch.zeros_like(attn_mask),
            torch.full_like(attn_mask, float('-inf'))
        )
        
        # Apply attention
        h_attn, _ = self.spatial_attn(
            h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0),
            attn_mask=attn_bias
        )
        h_attn = h_attn.squeeze(0)
        
        # Residual connection
        h_out = h + self.output_proj(h_attn)
        
        return h_out


# ============================================================================
# AdaLN (Adaptive Layer Normalization) - DiT Style
# ============================================================================

class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN) from DiT paper.
    
    :  ( + LR)  LayerNorm  scale  shift
    
    
    : AdaLN(h, c) = (1 + scale) * LayerNorm(h) + shift
     scale, shift = MLP(c)
    """
    
    def __init__(self, hidden_dim: int, condition_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        #  scale  shift
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 2 * hidden_dim)
        )
        #  AdaLN  LayerNorm
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:  (N, hidden_dim)
            c:  (N, condition_dim)
        Returns:
             (N, hidden_dim)
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return (1 + scale) * self.norm(x) + shift


class AdaLNZero(nn.Module):
    """
    AdaLN-Zero:  gate 
    
    : output = x + gate * sublayer(AdaLN(x, c))
     gate=0
    """
    
    def __init__(self, hidden_dim: int, condition_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        #  scale, shift, gate
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 3 * hidden_dim)
        )
        # 
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            normalized: AdaLN 
            gate: 
        """
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=-1)
        normalized = (1 + scale) * self.norm(x) + shift
        return normalized, gate


class DiTBlock(nn.Module):
    """
    DiT-style Transformer Block with AdaLN.
    
    :
    1. AdaLN -> Self-Attention -> Gate -> Residual
    2. AdaLN -> Cross-Attention () -> Gate -> Residual
    3. AdaLN -> MLP -> Gate -> Residual
    
     Transformer :
    -  AdaLN  LayerNorm
    -  scale/shift 
    -  gate 
    """
    
    def __init__(
        self,
        hidden_dim: int,
        condition_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_cross_attention: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_cross_attention = use_cross_attention
        
        # Self-Attention with AdaLN
        self.adaLN_self = AdaLNZero(hidden_dim, condition_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Cross-Attention with AdaLN (optional)
        if use_cross_attention:
            self.adaLN_cross = AdaLNZero(hidden_dim, condition_dim)
            self.cross_attn = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
        
        # MLP with AdaLN
        self.adaLN_mlp = AdaLNZero(hidden_dim, condition_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x:  (N, hidden_dim)
            c:  (N, condition_dim) - LR
            context: Cross-attention  key/value (N, hidden_dim)
            
        Returns:
             (N, hidden_dim)
        """
        # Self-Attention
        x_norm, gate_sa = self.adaLN_self(x, c)
        attn_out, _ = self.self_attn(
            x_norm.unsqueeze(0), x_norm.unsqueeze(0), x_norm.unsqueeze(0)
        )
        x = x + gate_sa * attn_out.squeeze(0)
        
        # Cross-Attention (if enabled and context provided)
        if self.use_cross_attention and context is not None:
            x_norm, gate_ca = self.adaLN_cross(x, c)
            cross_out, _ = self.cross_attn(
                x_norm.unsqueeze(0), context.unsqueeze(0), context.unsqueeze(0)
            )
            x = x + gate_ca * cross_out.squeeze(0)
        
        # MLP
        x_norm, gate_mlp = self.adaLN_mlp(x, c)
        x = x + gate_mlp * self.mlp(x_norm)
        
        return x


class DiTVelocityNet(nn.Module):
    """
    DiT-style Velocity Network for Flow Matching.
    
    :
    1. AdaLN :  scale/shift 
    2. Gate : 
    3. : 
    
     FlowMatchingVelocityNet :
    - : 
    - DiT:  AdaLN  normalization
    """
    
    def __init__(
        self,
        n_genes: int,
        hidden_dim: int = 256,
        condition_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,  # DiT 
        dropout: float = 0.1,
        max_hr_per_lr: int = 16,
        use_hr_spatial: bool = True,
        num_hr_neighbors: int = 6,
        mlp_ratio: float = 4.0
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim
        self.max_hr_per_lr = max_hr_per_lr
        self.use_hr_spatial = use_hr_spatial
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Time embedding with learnable frequency
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Condition projection (LR latent -> hidden)
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Position embeddings
        self.rel_pos_embed = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        self.local_idx_embed = nn.Embedding(max_hr_per_lr, hidden_dim)
        
        # Condition fusion MLP (time + condition -> modulation vector)
        self.condition_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # DiT Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=hidden_dim,
                condition_dim=hidden_dim,  # fused condition
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_cross_attention=True
            )
            for _ in range(num_layers)
        ])
        
        # HR spatial prior module
        if use_hr_spatial:
            self.hr_spatial = HRSpatialGraph(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_neighbors=num_hr_neighbors,
                dropout=dropout
            )
            self.spatial_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
        
        # Final layer with AdaLN
        self.final_adaLN = AdaLN(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, n_genes)
        
        # Skip connection
        self.skip_proj = nn.Linear(n_genes, n_genes)
        
        # Apply Xavier initialization to all linear layers (except AdaLN which uses zero init)
        self._init_weights()
        
        # Initialize output to zero for stable training (override Xavier for output)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def _init_weights(self):
        """Apply Xavier initialization to linear layers."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Skip AdaLN modulation layers (they use zero init)
                if 'adaLN' in name or 'adaLN_modulation' in name:
                    continue
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
    
    def get_time_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        """Sinusoidal time embedding."""
        device = t.device
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        t_scaled = t.float() * 1000.0
        emb = t_scaled.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    def forward(
        self,
        x_t: torch.Tensor,
        h_condition: torch.Tensor,
        t: torch.Tensor,
        lr_hr_mapping: torch.Tensor,
        hr_coords: torch.Tensor,
        hr_rel_coords: torch.Tensor,
        local_hr_indices: torch.Tensor,
        group_indices: torch.Tensor,
        hr_edge_index: Optional[torch.Tensor] = None,
        hr_edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict velocity using DiT architecture.
        """
        n_hr = x_t.size(0)
        device = x_t.device
        
        # Project input
        h = self.input_proj(x_t)
        
        # Get LR condition for each HR spot
        lr_idx = lr_hr_mapping[0]
        hr_idx = lr_hr_mapping[1]
        
        h_cond_expanded = torch.zeros(n_hr, h_condition.size(-1), device=device)
        count = torch.zeros(n_hr, 1, device=device)
        h_cond_expanded.scatter_add_(
            0, hr_idx.unsqueeze(-1).expand(-1, h_condition.size(-1)),
            h_condition[lr_idx]
        )
        count.scatter_add_(
            0, hr_idx.unsqueeze(-1),
            torch.ones_like(hr_idx, dtype=torch.float).unsqueeze(-1)
        )
        h_cond = h_cond_expanded / count.clamp(min=1)
        h_cond = self.condition_proj(h_cond)  # (N_hr, hidden_dim)
        
        # Time embedding -  per-sample  t
        # t: (n_samples,)  (1,)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        t_emb = self.get_time_embedding(t, self.hidden_dim)  # (n_t, hidden_dim)
        t_emb = self.time_embed(t_emb)  # (n_t, hidden_dim)
        
        # tper-sampleexpand
        if t_emb.size(0) == n_hr:
            pass  # per-sample
        elif t_emb.size(0) == 1:
            t_emb = t_emb.expand(n_hr, -1)  # (N_hr, hidden_dim)
        else:
            # tper-samplegroup
            # thr_idx
            t_emb = t_emb.expand(n_hr, -1)  # fallback: broadcast
        
        # Fuse time and condition for AdaLN modulation
        c = self.condition_fusion(torch.cat([t_emb, h_cond], dim=-1))  # (N_hr, hidden_dim)
        
        # Position embeddings (added to input, not to condition)
        rel_pos_emb = self.rel_pos_embed(hr_rel_coords)
        local_idx_emb = self.local_idx_embed(
            local_hr_indices.clamp(0, self.max_hr_per_lr - 1)
        )
        h = h + rel_pos_emb + local_idx_emb
        
        # Apply DiT blocks
        for block in self.blocks:
            h = block(h, c, context=h_cond)
        
        # HR spatial prior
        if self.use_hr_spatial:
            h_spatial = self.hr_spatial(h, hr_coords, hr_edge_index, hr_edge_weight)
            gate = self.spatial_gate(torch.cat([h, h_spatial], dim=-1))
            h = h * (1 - gate) + h_spatial * gate
        
        # Final projection with AdaLN
        h = self.final_adaLN(h, c)
        velocity = self.output_proj(h) + self.skip_proj(x_t)
        
        return velocity


class FlowMatchingVelocityNet(nn.Module):
    """
    Velocity network for Flow Matching.
    
     x_0 ()  x_1 (logits)  v(x_t, t)
    LRHR
    """
    
    def __init__(
        self,
        n_genes: int,
        hidden_dim: int = 256,
        condition_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_hr_per_lr: int = 16,
        use_hr_spatial: bool = True,
        num_hr_neighbors: int = 6
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim
        self.max_hr_per_lr = max_hr_per_lr
        self.use_hr_spatial = use_hr_spatial
        
        # Input projection (logits)
        self.input_proj = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Condition projection
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Time embedding (continuous time in [0, 1])
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Position embeddings
        self.rel_pos_embed = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        self.local_idx_embed = nn.Embedding(max_hr_per_lr, hidden_dim)
        
        # Transformer layers with intra-group and HR spatial attention
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                # Intra-group attention (within same LR)
                'group_attn': nn.MultiheadAttention(
                    hidden_dim, num_heads, dropout=dropout, batch_first=True
                ),
                'group_norm': nn.LayerNorm(hidden_dim),
                
                # Cross-attention with LR condition
                'cross_attn': nn.MultiheadAttention(
                    hidden_dim, num_heads, dropout=dropout, batch_first=True
                ),
                'cross_norm': nn.LayerNorm(hidden_dim),
                
                # Feed-forward
                'ff': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout)
                ),
                'ff_norm': nn.LayerNorm(hidden_dim)
            })
            self.layers.append(layer)
        
        # HR spatial prior module
        if use_hr_spatial:
            self.hr_spatial = HRSpatialGraph(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_neighbors=num_hr_neighbors,
                dropout=dropout
            )
            self.spatial_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
        
        # Output projection (predict velocity)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_genes)
        )
        
        # Skip connection
        self.skip_proj = nn.Linear(n_genes, n_genes)
        
        # Apply Xavier initialization
        self._init_weights()
    
    def _init_weights(self):
        """Apply Xavier initialization to all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
    
    def get_time_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Sinusoidal time embedding for continuous time t in [0, 1].
        """
        device = t.device
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # Scale t from [0,1] to [0, 1000] for better embedding resolution
        t_scaled = t.float() * 1000.0
        emb = t_scaled.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    def forward(
        self,
        x_t: torch.Tensor,
        h_condition: torch.Tensor,
        t: torch.Tensor,
        lr_hr_mapping: torch.Tensor,
        hr_coords: torch.Tensor,
        hr_rel_coords: torch.Tensor,
        local_hr_indices: torch.Tensor,
        group_indices: torch.Tensor,
        hr_edge_index: Optional[torch.Tensor] = None,
        hr_edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict velocity v(x_t, t) for flow matching.
        
        Args:
            x_t: Interpolated state at time t (N_hr, n_genes)
            h_condition: LR condition features (N_lr, hidden_dim)
            t: Time in [0, 1], shape (1,)
            lr_hr_mapping: LR-HR mapping edges
            hr_coords: Absolute HR coordinates (N_hr, 2)
            hr_rel_coords: Relative HR coordinates (N_hr, 2)
            local_hr_indices: Local index within group (N_hr,)
            group_indices: LR group for each HR (N_hr,)
            hr_edge_index: Pre-computed HR neighbor graph (optional)
            hr_edge_weight: Pre-computed HR edge weights (optional)
            
        Returns:
            velocity: Predicted velocity (N_hr, n_genes)
        """
        n_hr = x_t.size(0)
        device = x_t.device
        
        # Project input
        h = self.input_proj(x_t)  # (N_hr, hidden_dim)
        
        # Get condition for each HR spot
        lr_idx = lr_hr_mapping[0]
        hr_idx = lr_hr_mapping[1]
        
        h_cond_expanded = torch.zeros(n_hr, h_condition.size(-1), device=device)
        count = torch.zeros(n_hr, 1, device=device)
        h_cond_expanded.scatter_add_(
            0, hr_idx.unsqueeze(-1).expand(-1, h_condition.size(-1)),
            h_condition[lr_idx]
        )
        count.scatter_add_(
            0, hr_idx.unsqueeze(-1),
            torch.ones_like(hr_idx, dtype=torch.float).unsqueeze(-1)
        )
        h_cond = h_cond_expanded / count.clamp(min=1)
        h_cond = self.condition_proj(h_cond)
        
        # Time embedding
        t_emb = self.get_time_embedding(t, self.hidden_dim)  # (1, hidden_dim)
        t_emb = self.time_mlp(t_emb)
        t_emb = t_emb.expand(n_hr, -1)
        
        # Position embeddings
        rel_pos_emb = self.rel_pos_embed(hr_rel_coords)
        local_idx_emb = self.local_idx_embed(
            local_hr_indices.clamp(0, self.max_hr_per_lr - 1)
        )
        
        # Combine embeddings
        h = h + t_emb + rel_pos_emb + local_idx_emb
        
        # Apply transformer layers
        for layer in self.layers:
            # Intra-group attention
            h_norm = layer['group_norm'](h)
            h_attn, _ = layer['group_attn'](
                h_norm.unsqueeze(0), h_norm.unsqueeze(0), h_norm.unsqueeze(0)
            )
            h = h + h_attn.squeeze(0)
            
            # Cross-attention with condition
            h_norm = layer['cross_norm'](h)
            h_cross, _ = layer['cross_attn'](
                h_norm.unsqueeze(0), h_cond.unsqueeze(0), h_cond.unsqueeze(0)
            )
            h = h + h_cross.squeeze(0)
            
            # Feed-forward
            h = h + layer['ff'](layer['ff_norm'](h))
        
        # HR spatial prior
        if self.use_hr_spatial:
            h_spatial = self.hr_spatial(h, hr_coords, hr_edge_index, hr_edge_weight)
            
            # Gated fusion
            gate = self.spatial_gate(torch.cat([h, h_spatial], dim=-1))
            h = h * (1 - gate) + h_spatial * gate
        
        # Output velocity with skip connection
        velocity = self.output_proj(h) + self.skip_proj(x_t)
        
        return velocity


class RatioConditionEncoder(nn.Module):
    """
    Encodes LR spot latent vectors for conditioning using MLP.
    
    v5.4 : MLPTransformer
    """
    
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 8,  # 
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Position encoder for spatial coordinates
        self.pos_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # MLP encoder (Transformer)
        # : latent_dim + hidden_dim ()
        mlp_layers = []
        input_dim = latent_dim + hidden_dim
        
        for i in range(num_layers):
            mlp_layers.extend([
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Apply Xavier initialization
        self._init_weights()
    
    def _init_weights(self):
        """Apply Xavier initialization to all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        z_lr: torch.Tensor,
        lr_coords: torch.Tensor
    ) -> torch.Tensor:
        # Encode position
        pos = self.pos_encoder(lr_coords)
        
        # Concatenate latent and position
        h = torch.cat([z_lr, pos], dim=-1)
        
        # MLP encoding
        h = self.mlp(h)
        
        # Output projection
        h = self.output_proj(h)
        return h


class FlowMatchingRatio(nn.Module):
    """
    Flow Matching model for ratio-based super-resolution.
    
    Key differences from standard diffusion:
    1. Uses OT-Flow: linear interpolation between noise and target
       x_t = (1-t) * x_0 + t * x_1, where x_0 ~ N(0,I), x_1 = target logits
       
    2. Learns velocity field v(x_t, t) such that dx/dt = v(x_t, t)
    
    3. Inference: integrate ODE from t=0 to t=1
       x_{t+dt} = x_t + v(x_t, t) * dt
       
    Advantages:
    - Deterministic paths (no stochasticity in forward process)
    - Faster sampling (fewer steps needed)
    - Better for constrained distributions
    
    v5.2 :  DiT (Diffusion Transformer) 
    - use_dit=True:  AdaLN  DiT 
    - use_dit=False: 
    """
    
    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_hr_per_lr: int = 16,
        use_hr_spatial: bool = True,
        num_hr_neighbors: int = 6,
        sigma_min: float = 0.001,
        # v5.2 : DiT 
        use_dit: bool = False,
        mlp_ratio: float = 4.0,
        # v5.4 : 
        kd_enabled: bool = True,
        kd_weight: float = 1.0,
        kd_temperature: float = 3.0,
        # v5.5 : velocity loss  ()
        velocity_loss_enabled: bool = True,
        velocity_loss_weight: float = 1.0
    ):
        """
        Args:
            n_genes: Number of HVG genes
            latent_dim: LR latent dimension
            hidden_dim: Hidden dimension
            num_heads: Attention heads
            num_layers: Number of layers
            dropout: Dropout rate
            max_hr_per_lr: Maximum HR per LR
            use_hr_spatial: Whether to use HR spatial prior
            num_hr_neighbors: Number of HR neighbors for spatial graph
            sigma_min: Minimum noise level (for numerical stability)
            use_dit: Whether to use DiT architecture with AdaLN (v5.2)
            mlp_ratio: MLP hidden dimension ratio for DiT blocks
            velocity_loss_enabled: Whether to use velocity loss (ablation)
            velocity_loss_weight: Weight for velocity loss
        """
        super().__init__()
        
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.sigma_min = sigma_min
        self.use_hr_spatial = use_hr_spatial
        self.use_dit = use_dit
        
        # v5.4 
        self.kd_enabled = kd_enabled
        self.kd_weight = kd_weight
        self.kd_temperature = kd_temperature
        
        # v5.5 velocity loss 
        self.velocity_loss_enabled = velocity_loss_enabled
        self.velocity_loss_weight = velocity_loss_weight
        
        # Condition encoder
        self.condition_encoder = RatioConditionEncoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=2,
            dropout=dropout
        )
        
        # Velocity network - 
        if use_dit:
            # DiT  with AdaLN
            self.velocity_net = DiTVelocityNet(
                n_genes=n_genes,
                hidden_dim=hidden_dim,
                condition_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                max_hr_per_lr=max_hr_per_lr,
                use_hr_spatial=use_hr_spatial,
                num_hr_neighbors=num_hr_neighbors,
                mlp_ratio=mlp_ratio
            )
        else:
            # 
            self.velocity_net = FlowMatchingVelocityNet(
                n_genes=n_genes,
                hidden_dim=hidden_dim,
                condition_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                max_hr_per_lr=max_hr_per_lr,
                use_hr_spatial=use_hr_spatial,
                num_hr_neighbors=num_hr_neighbors
            )
    
    def compute_ground_truth_logits(
        self,
        x_hr: torch.Tensor,
        group_indices: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Compute ground-truth logits for spatial distribution.
        
        
        1. 
        2. ratio
        3. centered logits
        """
        n_hr, n_genes = x_hr.shape
        device = x_hr.device
        
        # 
        x_hr_pos = torch.clamp(x_hr, min=0)
        
        n_groups = group_indices.max().item() + 1
        
        # group
        group_sums = torch.zeros(n_groups, n_genes, device=device)
        group_sums.scatter_add_(
            0, group_indices.unsqueeze(-1).expand(-1, n_genes), x_hr_pos
        )
        
        hr_group_sums = group_sums[group_indices]
        
        # ratios1
        ratios = x_hr_pos / (hr_group_sums + eps)
        
        # ratiolog(0)
        ratios_safe = torch.clamp(ratios, min=eps, max=1.0 - eps)
        
        # logits
        log_ratios = torch.log(ratios_safe)
        
        # Center logits per group (logits0)
        group_log_means = torch.zeros(n_groups, n_genes, device=device)
        group_counts = torch.zeros(n_groups, 1, device=device)
        group_log_means.scatter_add_(
            0, group_indices.unsqueeze(-1).expand(-1, n_genes), log_ratios
        )
        group_counts.scatter_add_(
            0, group_indices.unsqueeze(-1),
            torch.ones(n_hr, 1, device=device)
        )
        group_log_means = group_log_means / group_counts.clamp(min=1)
        
        hr_group_log_means = group_log_means[group_indices]
        logits = log_ratios - hr_group_log_means
        
        # NaN
        logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
        logits = torch.where(torch.isinf(logits), torch.zeros_like(logits), logits)
        
        return logits
    
    def _get_group_counts(
        self,
        group_indices: torch.Tensor,
        n_groups: int,
        n_genes: int,
        device: torch.device
    ) -> torch.Tensor:
        """Get count of HR spots per group, expanded to (N_hr, n_genes)."""
        group_counts = torch.zeros(n_groups, device=device)
        group_counts.scatter_add_(
            0, group_indices,
            torch.ones(group_indices.size(0), device=device)
        )
        hr_counts = group_counts[group_indices]
        return hr_counts.unsqueeze(-1).expand(-1, n_genes)
    
    def _sample_time(
        self,
        device: torch.device,
        strategy: str = 'uniform',
        beta: float = 2.0,
        n_samples: int = 1
    ) -> torch.Tensor:
        """
         t
        
        Args:
            device: 
            strategy: 
                - 'uniform':  t ~ U[0, 1]
                - 'importance':  t  1 ()
                - 'logit_normal': Logit-normal 
            beta: importance sampling  t=1
            n_samples:  (per-samplet)
            
        Returns:
            t:  (n_samples,)
        """
        if strategy == 'uniform':
            t = torch.rand(n_samples, device=device)
        elif strategy == 'importance':
            # Beta beta > 1  t=1
            # t ~ Beta(beta, 1)  p(t)  t^(beta-1)
            u = torch.rand(n_samples, device=device)
            t = u ** (1.0 / beta)  # CDF
        elif strategy == 'logit_normal':
            # Logit-normal  t=0.5 
            z = torch.randn(n_samples, device=device) * 0.5  #  0.5
            t = torch.sigmoid(z)
        else:
            t = torch.rand(n_samples, device=device)
        
        #  t  (0, 1) 
        t = t.clamp(min=0.001, max=0.999)
        return t
    
    def flow_matching_loss(
        self,
        x_hr: torch.Tensor,
        x_lr: torch.Tensor,
        z_lr: torch.Tensor,
        lr_coords: torch.Tensor,
        hr_coords: torch.Tensor,
        lr_hr_mapping: torch.Tensor,
        group_indices: torch.Tensor,
        local_hr_indices: torch.Tensor,
        time_sampling_config: Dict = None,
        batch_idx: int = 0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Flow Matching loss with improved training strategy.
        
        :
        1.  importance sampling  t 
        
        OT-Flow Matching objective:
        L = E_{t, x_0, x_1} [ || v(x_t, t) - (x_1 - x_0) ||^2 ]
        
        where x_t = (1-t) * x_0 + t * x_1
        """
        device = x_hr.device
        n_hr = x_hr.size(0)
        
        # 
        time_sampling_config = time_sampling_config or {}
        
        loss_dict = {
            'velocity_loss': 0.0,
            'gene_pcc': 0.0,
            'spot_pcc': 0.0
        }
        
        # Get ground-truth logits
        x_1 = self.compute_ground_truth_logits(x_hr, group_indices)
        
        # Sample noise (starting point)
        x_0 = torch.randn_like(x_1)
        
        # Sample time t with configurable strategy - HR spott
        time_strategy = time_sampling_config.get('strategy', 'uniform')
        importance_beta = time_sampling_config.get('importance_beta', 2.0)
        t = self._sample_time(device, strategy=time_strategy, beta=importance_beta, n_samples=n_hr)
        # t shape: (n_hr,)
        
        # Interpolate: x_t = (1-t) * x_0 + t * x_1 (per-sample t)
        sigma_t = self.sigma_min
        t_expanded = t.unsqueeze(-1)  # (n_hr, 1)
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1 + sigma_t * torch.randn_like(x_1)
        
        # Target velocity (conditional flow matching)
        target_velocity = x_1 - x_0
        
        # Encode LR condition
        h_condition = self.condition_encoder(z_lr, lr_coords)
        
        # Compute relative HR coordinates
        lr_idx, hr_idx = lr_hr_mapping[0], lr_hr_mapping[1]
        lr_centers = torch.zeros(n_hr, 2, device=device)
        lr_center_count = torch.zeros(n_hr, 1, device=device)
        lr_centers.scatter_add_(0, hr_idx.unsqueeze(-1).expand(-1, 2), lr_coords[lr_idx])
        lr_center_count.scatter_add_(
            0, hr_idx.unsqueeze(-1),
            torch.ones_like(hr_idx, dtype=torch.float).unsqueeze(-1)
        )
        lr_centers = lr_centers / lr_center_count.clamp(min=1)
        hr_rel_coords = hr_coords - lr_centers
        
        # Predict velocity
        velocity_pred = self.velocity_net(
            x_t, h_condition, t,
            lr_hr_mapping, hr_coords, hr_rel_coords,
            local_hr_indices, group_indices
        )
        
        # Velocity loss (main FM objective)
        velocity_loss = F.mse_loss(velocity_pred, target_velocity)
        loss_dict['velocity_loss'] = velocity_loss.item()
        
        #  ()
        if self.velocity_loss_enabled:
            total_loss = self.velocity_loss_weight * velocity_loss
        else:
            # :  velocity loss
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # ====================================================================
        # : HR
        # tKDt (t)
        #  self.kd_enabled, self.kd_weight, self.kd_temperature 
        # ====================================================================
        if self.kd_enabled:
            # logits: logits (per-sample t)
            # student_logits = x_t + (1 - t) * velocity_pred
            student_logits = x_t + (1 - t_expanded) * velocity_pred
            
            # logits: HRground-truth logits (x_1)
            teacher_logits = x_1
            
            # group-wise KL (per gene, per sample)
            kd_loss = self._compute_distillation_loss_weighted(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                group_indices=group_indices,
                t=t,  # per-sample t
                temperature=self.kd_temperature
            )
            
            total_loss = total_loss + self.kd_weight * kd_loss
            loss_dict['kd_loss'] = kd_loss.item()
        else:
            loss_dict['kd_loss'] = 0.0
        
        loss_dict['t_mean'] = t.mean().item()  # t
        
        # Prepare LR expression for HR spots (for reconstruction and PCC)
        x_lr_for_hr = torch.zeros(n_hr, self.n_genes, device=device)
        x_lr_count = torch.zeros(n_hr, 1, device=device)
        x_lr_for_hr.scatter_add_(
            0, hr_idx.unsqueeze(-1).expand(-1, self.n_genes), x_lr[lr_idx]
        )
        x_lr_count.scatter_add_(
            0, hr_idx.unsqueeze(-1),
            torch.ones_like(hr_idx, dtype=torch.float).unsqueeze(-1)
        )
        x_lr_for_hr = x_lr_for_hr / x_lr_count.clamp(min=1)
        
        # ====================================================================
        #  ()
        # t=1
        # ====================================================================
        with torch.no_grad():
            # per-sample tlogits
            logits_pred = x_t + (1 - t_expanded) * velocity_pred
            ratios_pred = self._normalize_logits_by_group(logits_pred, group_indices)
            x_hr_pred = ratios_pred * x_lr_for_hr
            gene_pcc, spot_pcc = self._compute_pcc(x_hr_pred, x_hr)
            loss_dict['gene_pcc'] = gene_pcc.item()
            loss_dict['spot_pcc'] = spot_pcc.item()
        
        return total_loss, loss_dict
    
    def _normalize_logits_by_group(
        self,
        logits: torch.Tensor,
        group_indices: torch.Tensor
    ) -> torch.Tensor:
        """Convert logits to normalized ratios using group-wise softmax."""
        n_hr, n_genes = logits.shape
        device = logits.device
        n_groups = group_indices.max().item() + 1
        
        # Log-sum-exp for stability
        group_max = torch.full((n_groups, n_genes), float('-inf'), device=device)
        group_max.scatter_reduce_(
            0, group_indices.unsqueeze(-1).expand(-1, n_genes),
            logits, reduce='amax', include_self=False
        )
        hr_group_max = group_max[group_indices]
        
        logits_exp = torch.exp(logits - hr_group_max)
        
        group_sum = torch.zeros(n_groups, n_genes, device=device)
        group_sum.scatter_add_(
            0, group_indices.unsqueeze(-1).expand(-1, n_genes), logits_exp
        )
        hr_group_sum = group_sum[group_indices]
        
        ratios = logits_exp / (hr_group_sum + 1e-8)
        return ratios
    
    def _compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        group_indices: torch.Tensor,
        temperature: float = 3.0
    ) -> torch.Tensor:
        """
         (Group-wise KL)
        
        HR
        KL
        
        Args:
            student_logits: logits (N_hr, n_genes)
            teacher_logits: ()logits (N_hr, n_genes)
            group_indices: HRLR (N_hr,)
            temperature: 
            
        Returns:
            kd_loss: 
        """
        n_hr, n_genes = student_logits.shape
        device = student_logits.device
        n_groups = group_indices.max().item() + 1
        
        # 
        student_scaled = student_logits / temperature
        teacher_scaled = teacher_logits / temperature
        
        # ====================================================================
        # Group-wise Softmax (per gene)
        # LRHR spotssoftmax
        # ====================================================================
        
        # : softmax(teacher / T)
        # group-wise max
        teacher_group_max = torch.full((n_groups, n_genes), float('-inf'), device=device)
        teacher_group_max.scatter_reduce_(
            0, group_indices.unsqueeze(-1).expand(-1, n_genes),
            teacher_scaled, reduce='amax', include_self=False
        )
        teacher_max_expanded = teacher_group_max[group_indices]
        
        teacher_exp = torch.exp(teacher_scaled - teacher_max_expanded)
        teacher_group_sum = torch.zeros(n_groups, n_genes, device=device)
        teacher_group_sum.scatter_add_(
            0, group_indices.unsqueeze(-1).expand(-1, n_genes), teacher_exp
        )
        teacher_sum_expanded = teacher_group_sum[group_indices]
        teacher_soft = teacher_exp / (teacher_sum_expanded + 1e-8)
        
        # log-softmax: log_softmax(student / T)
        student_group_max = torch.full((n_groups, n_genes), float('-inf'), device=device)
        student_group_max.scatter_reduce_(
            0, group_indices.unsqueeze(-1).expand(-1, n_genes),
            student_scaled, reduce='amax', include_self=False
        )
        student_max_expanded = student_group_max[group_indices]
        
        student_exp = torch.exp(student_scaled - student_max_expanded)
        student_group_sum = torch.zeros(n_groups, n_genes, device=device)
        student_group_sum.scatter_add_(
            0, group_indices.unsqueeze(-1).expand(-1, n_genes), student_exp
        )
        student_sum_expanded = student_group_sum[group_indices]
        student_log_soft = (student_scaled - student_max_expanded) - torch.log(student_sum_expanded + 1e-8)
        
        # ====================================================================
        # KL: KL(teacher || student) = sum(teacher * (log(teacher) - log(student)))
        # ====================================================================
        # log(0)
        teacher_soft_safe = torch.clamp(teacher_soft, min=1e-8)
        
        # Per-element KL
        kl_elements = teacher_soft_safe * (torch.log(teacher_soft_safe) - student_log_soft)
        
        # 
        # 
        kl_per_gene = kl_elements.sum(dim=0)  # (n_genes,) - spotsKL
        
        # : KL
        kl_loss = kl_per_gene.mean()  # 
        
        #  T ()
        kl_loss = kl_loss * (temperature ** 2)
        
        return kl_loss
    
    def _compute_distillation_loss_weighted(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        group_indices: torch.Tensor,
        t: torch.Tensor,
        temperature: float = 3.0
    ) -> torch.Tensor:
        """
         (per-samplet)
        
        PyTorchkl_div
        
        Args:
            student_logits: logits (N_hr, n_genes)
            teacher_logits: ()logits (N_hr, n_genes)
            group_indices: HRLR (N_hr,)
            t:  (N_hr,)
            temperature: 
            
        Returns:
            kd_loss: 
        """
        n_hr, n_genes = student_logits.shape
        device = student_logits.device
        n_groups = group_indices.max().item() + 1
        
        # 
        student_scaled = student_logits / temperature
        teacher_scaled = teacher_logits / temperature
        
        # ====================================================================
        # Group-wise Softmax - 
        # HR spotssoftmax
        # ====================================================================
        
        # group-wise softmax for teacher
        teacher_group_max = torch.full((n_groups, n_genes), float('-inf'), device=device)
        teacher_group_max.scatter_reduce_(
            0, group_indices.unsqueeze(-1).expand(-1, n_genes),
            teacher_scaled, reduce='amax', include_self=False
        )
        teacher_max_expanded = teacher_group_max[group_indices]
        
        teacher_exp = torch.exp(teacher_scaled - teacher_max_expanded)
        teacher_group_sum = torch.zeros(n_groups, n_genes, device=device)
        teacher_group_sum.scatter_add_(
            0, group_indices.unsqueeze(-1).expand(-1, n_genes), teacher_exp
        )
        teacher_sum_expanded = teacher_group_sum[group_indices]
        teacher_soft = teacher_exp / (teacher_sum_expanded + 1e-8)
        teacher_soft = torch.clamp(teacher_soft, min=1e-8, max=1.0)
        
        # group-wise log_softmax for student ()
        student_group_max = torch.full((n_groups, n_genes), float('-inf'), device=device)
        student_group_max.scatter_reduce_(
            0, group_indices.unsqueeze(-1).expand(-1, n_genes),
            student_scaled, reduce='amax', include_self=False
        )
        student_max_expanded = student_group_max[group_indices]
        
        student_exp = torch.exp(student_scaled - student_max_expanded)
        student_group_sum = torch.zeros(n_groups, n_genes, device=device)
        student_group_sum.scatter_add_(
            0, group_indices.unsqueeze(-1).expand(-1, n_genes), student_exp
        )
        student_sum_expanded = student_group_sum[group_indices]
        # log_softmax = x - max - log(sum(exp(x - max)))
        student_log_soft = (student_scaled - student_max_expanded) - torch.log(student_sum_expanded + 1e-8)
        
        # ====================================================================
        # KL: KL(teacher || student) = sum(teacher * (log(teacher) - log(student)))
        #  F.kl_div 
        # ====================================================================
        
        # Per-element KL: teacher * log(teacher) - teacher * log_student
        # = teacher * log(teacher/student)
        kl_elements = teacher_soft * (torch.log(teacher_soft + 1e-8) - student_log_soft)
        
        # Clamp to ensure non-negative (KL >= 0)
        kl_elements = torch.clamp(kl_elements, min=0.0)
        
        # KL ()
        kl_per_sample = kl_elements.sum(dim=1)  # (n_hr,)
        
        # t (t)
        weights = t ** 2  # (n_hr,)
        
        # 
        weighted_kl = (weights * kl_per_sample).sum() / (weights.sum() + 1e-8)
        
        #  T ()
        kd_loss = weighted_kl * (temperature ** 2)
        
        # 
        kd_loss = torch.clamp(kd_loss, min=0.0)
        
        return kd_loss
    
    def _compute_pcc(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        eps: float = 1e-8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Gene PCC and Spot PCC."""
        # Gene PCC
        pred_gene_centered = pred - pred.mean(dim=0, keepdim=True)
        target_gene_centered = target - target.mean(dim=0, keepdim=True)
        
        cov_gene = (pred_gene_centered * target_gene_centered).sum(dim=0)
        std_pred_gene = torch.sqrt((pred_gene_centered ** 2).sum(dim=0) + eps)
        std_target_gene = torch.sqrt((target_gene_centered ** 2).sum(dim=0) + eps)
        gene_pcc_per_gene = cov_gene / (std_pred_gene * std_target_gene + eps)
        
        valid_genes = (std_target_gene > 0.01) & (std_pred_gene > 0.01)
        gene_pcc = gene_pcc_per_gene[valid_genes].mean() if valid_genes.sum() > 0 else torch.tensor(0.0, device=pred.device)
        
        # Spot PCC
        pred_spot_centered = pred - pred.mean(dim=1, keepdim=True)
        target_spot_centered = target - target.mean(dim=1, keepdim=True)
        
        cov_spot = (pred_spot_centered * target_spot_centered).sum(dim=1)
        std_pred_spot = torch.sqrt((pred_spot_centered ** 2).sum(dim=1) + eps)
        std_target_spot = torch.sqrt((target_spot_centered ** 2).sum(dim=1) + eps)
        spot_pcc = cov_spot / (std_pred_spot * std_target_spot + eps)
        spot_pcc = spot_pcc.mean()
        
        return gene_pcc, spot_pcc
    
    @torch.no_grad()
    def sample(
        self,
        z_lr: torch.Tensor,
        x_lr: torch.Tensor,
        lr_coords: torch.Tensor,
        hr_coords: torch.Tensor,
        lr_hr_mapping: torch.Tensor,
        group_indices: torch.Tensor,
        local_hr_indices: torch.Tensor,
        num_steps: int = 50,
        verbose: bool = True
    ) -> torch.Tensor:
        """
        Generate HR expression by integrating the flow ODE.
        
        Uses Euler integration: x_{t+dt} = x_t + v(x_t, t) * dt
        
        Args:
            num_steps: Number of integration steps
            
        Returns:
            x_hr_pred: Predicted HR expression (N_hr, n_genes)
        """
        device = z_lr.device
        n_hr = hr_coords.size(0)
        
        # Encode LR condition
        h_condition = self.condition_encoder(z_lr, lr_coords)
        
        # Compute relative HR coordinates
        lr_idx, hr_idx = lr_hr_mapping[0], lr_hr_mapping[1]
        lr_centers = torch.zeros(n_hr, 2, device=device)
        lr_center_count = torch.zeros(n_hr, 1, device=device)
        lr_centers.scatter_add_(0, hr_idx.unsqueeze(-1).expand(-1, 2), lr_coords[lr_idx])
        lr_center_count.scatter_add_(
            0, hr_idx.unsqueeze(-1),
            torch.ones_like(hr_idx, dtype=torch.float).unsqueeze(-1)
        )
        lr_centers = lr_centers / lr_center_count.clamp(min=1)
        hr_rel_coords = hr_coords - lr_centers
        
        # Build HR spatial graph once (for efficiency)
        hr_edge_index, hr_edge_weight = None, None
        if self.use_hr_spatial:
            hr_edge_index, hr_edge_weight = self.velocity_net.hr_spatial.build_knn_graph(hr_coords)
        
        # Start from noise
        x_t = torch.randn(n_hr, self.n_genes, device=device)
        
        # Time steps
        dt = 1.0 / num_steps
        times = torch.linspace(0, 1 - dt, num_steps, device=device)
        
        if verbose:
            times = tqdm(times, desc="Flow sampling")
        
        for t in times:
            t_tensor = t.unsqueeze(0)
            
            # Predict velocity
            velocity = self.velocity_net(
                x_t, h_condition, t_tensor,
                lr_hr_mapping, hr_coords, hr_rel_coords,
                local_hr_indices, group_indices,
                hr_edge_index, hr_edge_weight
            )
            
            # Euler step
            x_t = x_t + velocity * dt
        
        # Final logits to ratios
        ratios_pred = self._normalize_logits_by_group(x_t, group_indices)
        
        # Get LR expression for HR
        x_lr_for_hr = torch.zeros(n_hr, self.n_genes, device=device)
        x_lr_count = torch.zeros(n_hr, 1, device=device)
        x_lr_for_hr.scatter_add_(
            0, hr_idx.unsqueeze(-1).expand(-1, self.n_genes), x_lr[lr_idx]
        )
        x_lr_count.scatter_add_(
            0, hr_idx.unsqueeze(-1),
            torch.ones_like(hr_idx, dtype=torch.float).unsqueeze(-1)
        )
        x_lr_for_hr = x_lr_for_hr / x_lr_count.clamp(min=1)
        
        # Reconstruct HR expression
        x_hr_pred = ratios_pred * x_lr_for_hr
        
        return x_hr_pred


class FlowMatchingTrainer:
    """
    Trainer for Flow Matching model (v5.3).
    
    :
    1.  importance sampling 
    2.  (reconstruction, gene consistency)
    3. 
    """
    
    def __init__(
        self,
        model: FlowMatchingRatio,
        learning_rate: float = 0.0001,
        weight_decay: float = 0.0001,
        device: str = 'cuda',
        validation_steps: int = 50,
        scheduler_config: dict = None,
        total_steps: int = 1000,
        time_sampling_config: dict = None
    ):
        self.model = model.to(device)
        self.device = device
        self.validation_steps = validation_steps
        
        # : 
        self.time_sampling_config = time_sampling_config or {}
        
        # 
        self.global_step = 0
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        #  ()
        self.scheduler = self._create_scheduler(
            scheduler_config or {},
            total_steps,
            learning_rate
        )
        self.warmup_epochs = (scheduler_config or {}).get('warmup_epochs', 0)
        self.warmup_start_lr = learning_rate / 10
        self.base_lr = learning_rate
        self.current_epoch = 0
        
        self.history = {
            'loss': [], 'velocity_loss': [], 'recon_loss': [], 'gene_consistency_loss': [],
            'kd_loss': [],  # 
            'gene_pcc': [], 'spot_pcc': [], 'epoch_loss': [],
            'val_gene_pcc': [], 'val_spot_pcc': [], 'val_pcc': [],
            'learning_rate': []
        }
    
    def _create_scheduler(
        self,
        scheduler_config: dict,
        total_steps: int,
        learning_rate: float
    ):
        """
        
        
        :
        - 'cosine': CosineAnnealingLR
        - 'step': StepLR  
        - 'plateau': ReduceLROnPlateau
        - 'none': 
        
        Args:
            scheduler_config: 
            total_steps:  (T_max)
            learning_rate: 
        """
        scheduler_type = scheduler_config.get('type', 'cosine')
        
        if scheduler_type == 'none':
            # 
            return torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: 1.0
            )
        
        elif scheduler_type == 'cosine':
            # CosineAnnealingLR
            t_max = scheduler_config.get('t_max')
            if t_max is None:
                t_max = total_steps  # 
            eta_min = scheduler_config.get('eta_min', 1e-6)
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=t_max, eta_min=eta_min
            )
        
        elif scheduler_type == 'step':
            # StepLR
            step_size = scheduler_config.get('step_size', 30)
            gamma = scheduler_config.get('gamma', 0.1)
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        
        elif scheduler_type == 'plateau':
            # ReduceLROnPlateau
            patience = scheduler_config.get('patience', 10)
            factor = scheduler_config.get('factor', 0.5)
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=patience, factor=factor
            )
        
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def _apply_warmup(self, epoch: int):
        """warmup"""
        if epoch < self.warmup_epochs:
            # warmup
            warmup_factor = (epoch + 1) / self.warmup_epochs
            lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * warmup_factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return True
        return False
    
    def get_current_lr(self) -> float:
        """"""
        return self.optimizer.param_groups[0]['lr']
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int = 0
    ) -> Tuple[float, Dict[str, float]]:
        """
        Single training step with improved loss functions.
        
        :
        1.  importance sampling 
        2.  (reconstruction, gene consistency)
        """
        self.model.train()
        
        # Move data to device
        x_hr = batch['x_hr'].to(self.device)
        x_lr = batch['x_lr'].to(self.device)
        z_lr = batch['z_lr'].to(self.device)
        lr_coords = batch['lr_coords'].to(self.device)
        hr_coords = batch['hr_coords'].to(self.device)
        lr_hr_mapping = batch['lr_hr_mapping'].to(self.device)
        group_indices = batch['group_indices'].to(self.device)
        local_hr_indices = batch['local_hr_indices'].to(self.device)
        
        self.optimizer.zero_grad()
        
        loss, loss_dict = self.model.flow_matching_loss(
            x_hr=x_hr,
            x_lr=x_lr,
            z_lr=z_lr,
            lr_coords=lr_coords,
            hr_coords=hr_coords,
            lr_hr_mapping=lr_hr_mapping,
            group_indices=group_indices,
            local_hr_indices=local_hr_indices,
            time_sampling_config=self.time_sampling_config,
            batch_idx=self.global_step
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.global_step += 1
        
        return loss.item(), loss_dict
    
    def train_epoch(
        self,
        dataloader,
        epoch_num: int = 0,
        verbose: bool = True
    ) -> Dict[str, float]:
        """Train one epoch."""
        epoch_losses = []
        epoch_velocity_losses = []
        epoch_kd_losses = []
        epoch_gene_pccs = []
        epoch_spot_pccs = []
        
        for batch_idx, batch in enumerate(dataloader):
            loss, loss_dict = self.train_step(batch)
            
            epoch_losses.append(loss)
            epoch_velocity_losses.append(loss_dict['velocity_loss'])
            epoch_kd_losses.append(loss_dict.get('kd_loss', 0.0))
            epoch_gene_pccs.append(loss_dict['gene_pcc'])
            epoch_spot_pccs.append(loss_dict['spot_pcc'])
            
            self.history['loss'].append(loss)
            self.history['velocity_loss'].append(loss_dict['velocity_loss'])
            self.history['kd_loss'].append(loss_dict.get('kd_loss', 0.0))
            self.history['gene_pcc'].append(loss_dict['gene_pcc'])
            self.history['spot_pcc'].append(loss_dict['spot_pcc'])
            
            if verbose and batch_idx % 50 == 0:
                kd_str = f", KD: {loss_dict.get('kd_loss', 0.0):.4f}" if loss_dict.get('kd_loss', 0.0) > 0 else ""
                print(f"  Epoch {epoch_num+1}, Batch {batch_idx}/{len(dataloader)} | "
                      f"Loss: {loss:.4f}, Vel: {loss_dict['velocity_loss']:.4f}{kd_str}, "
                      f"GenePCC: {loss_dict['gene_pcc']:.4f}, SpotPCC: {loss_dict['spot_pcc']:.4f}")
        
        self.scheduler.step()
        
        avg_metrics = {
            'loss': np.mean(epoch_losses),
            'velocity_loss': np.mean(epoch_velocity_losses),
            'kd_loss': np.mean(epoch_kd_losses),
            'gene_pcc': np.mean(epoch_gene_pccs),
            'spot_pcc': np.mean(epoch_spot_pccs)
        }
        
        self.history['epoch_loss'].append(avg_metrics['loss'])
        
        return avg_metrics
    
    @torch.no_grad()
    def validate_batch(
        self,
        batch: Dict[str, torch.Tensor],
        num_steps: int = None
    ) -> Dict[str, float]:
        """
        t=0ODE
        
        Args:
            batch: batch
            num_steps: ODE
            
        Returns:
            metrics: gene_pcc, spot_pcc, pcc
        """
        self.model.eval()
        
        if num_steps is None:
            num_steps = self.validation_steps
        
        # Move data to device
        x_hr = batch['x_hr'].to(self.device)
        x_lr = batch['x_lr'].to(self.device)
        z_lr = batch['z_lr'].to(self.device)
        lr_coords = batch['lr_coords'].to(self.device)
        hr_coords = batch['hr_coords'].to(self.device)
        lr_hr_mapping = batch['lr_hr_mapping'].to(self.device)
        group_indices = batch['group_indices'].to(self.device)
        local_hr_indices = batch['local_hr_indices'].to(self.device)
        
        # ODE
        x_hr_pred = self.model.sample(
            z_lr=z_lr,
            x_lr=x_lr,
            lr_coords=lr_coords,
            hr_coords=hr_coords,
            lr_hr_mapping=lr_hr_mapping,
            group_indices=group_indices,
            local_hr_indices=local_hr_indices,
            num_steps=num_steps,
            verbose=False  # 
        )
        
        # PCC
        gene_pcc, spot_pcc = self.model._compute_pcc(x_hr_pred, x_hr)
        
        # PCC
        x_hr_pred_flat = x_hr_pred.flatten()
        x_hr_flat = x_hr.flatten()
        
        # NaNInf
        valid_mask = ~(torch.isnan(x_hr_pred_flat) | torch.isnan(x_hr_flat) |
                       torch.isinf(x_hr_pred_flat) | torch.isinf(x_hr_flat))
        
        if valid_mask.sum() > 2:
            pred_valid = x_hr_pred_flat[valid_mask]
            true_valid = x_hr_flat[valid_mask]
            
            # Pearson correlation
            pred_centered = pred_valid - pred_valid.mean()
            true_centered = true_valid - true_valid.mean()
            
            cov = (pred_centered * true_centered).sum()
            std_pred = torch.sqrt((pred_centered ** 2).sum() + 1e-8)
            std_true = torch.sqrt((true_centered ** 2).sum() + 1e-8)
            
            overall_pcc = cov / (std_pred * std_true + 1e-8)
            overall_pcc = overall_pcc.clamp(-1, 1)
        else:
            overall_pcc = torch.tensor(0.0, device=self.device)
        
        return {
            'val_gene_pcc': gene_pcc.item(),
            'val_spot_pcc': spot_pcc.item(),
            'val_pcc': overall_pcc.item()
        }
    
    @torch.no_grad()
    def validate_epoch(
        self,
        dataloader,
        max_batches: int = 5,
        num_steps: int = None
    ) -> Dict[str, float]:
        """
        epoch
        
        batch
        
        Args:
            dataloader: 
            max_batches: batch
            num_steps: ODE
            
        Returns:
            avg_metrics: 
        """
        self.model.eval()
        
        val_gene_pccs = []
        val_spot_pccs = []
        val_pccs = []
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            
            metrics = self.validate_batch(batch, num_steps)
            val_gene_pccs.append(metrics['val_gene_pcc'])
            val_spot_pccs.append(metrics['val_spot_pcc'])
            val_pccs.append(metrics['val_pcc'])
        
        avg_metrics = {
            'val_gene_pcc': np.mean(val_gene_pccs) if val_gene_pccs else 0.0,
            'val_spot_pcc': np.mean(val_spot_pccs) if val_spot_pccs else 0.0,
            'val_pcc': np.mean(val_pccs) if val_pccs else 0.0
        }
        
        # history
        self.history['val_gene_pcc'].append(avg_metrics['val_gene_pcc'])
        self.history['val_spot_pcc'].append(avg_metrics['val_spot_pcc'])
        self.history['val_pcc'].append(avg_metrics['val_pcc'])
        
        return avg_metrics
    
    @torch.no_grad()
    def generate(
        self,
        z_lr: torch.Tensor,
        x_lr: torch.Tensor,
        lr_coords: torch.Tensor,
        hr_coords: torch.Tensor,
        lr_hr_mapping: torch.Tensor,
        group_indices: torch.Tensor,
        local_hr_indices: torch.Tensor,
        num_steps: int = 50
    ) -> np.ndarray:
        """Generate HR expression."""
        self.model.eval()
        
        x_hr_pred = self.model.sample(
            z_lr=z_lr.to(self.device),
            x_lr=x_lr.to(self.device),
            lr_coords=lr_coords.to(self.device),
            hr_coords=hr_coords.to(self.device),
            lr_hr_mapping=lr_hr_mapping.to(self.device),
            group_indices=group_indices.to(self.device),
            local_hr_indices=local_hr_indices.to(self.device),
            num_steps=num_steps
        )
        
        return x_hr_pred.cpu().numpy()
