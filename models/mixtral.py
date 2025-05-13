import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from flash_attn import flash_attn_func
from typing import Optional, Tuple, Dict, List
import math
from dataclasses import dataclass

# --------------------------
# 1. Core Components
# --------------------------
class SwiGLU(nn.Module):
    """Fused SwiGLU with optimal memory layout."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class RMSNorm(nn.Module):
    """Optimized RMSNorm with fused operations."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1, eps=self.eps) * self.weight * (x.shape[-1] ** 0.5)

# --------------------------
# 2. Parallel MoE Layer 
# --------------------------
class ParallelMoELayer(nn.Module):
    """Mixture of Experts layer matching Mixtral-8x7B specs."""
    def __init__(
        self,
        dim: int,
        num_experts: int = 8,
        expert_capacity_factor: float = 1.25,
        hidden_dim: int = 14336,
        top_k: int = 2,
        dropout: float = 0.0,
        router_z_loss_coef: float = 0.001,
        router_aux_loss_coef: float = 0.001,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity = expert_capacity
        self.router = nn.Linear(dim, num_experts, bias=False)
        
        # Parallel experts (fused SwiGLU)
        self.experts = nn.ModuleList([SwiGLU(dim, hidden_dim) for _ in range(num_experts)])
        self.dropout = nn.Dropout(dropout)
        
        # Loss coefficients
        self.router_z_loss_coef = router_z_loss_coef
        self.router_load_loss_coef = router_load_loss_coef

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # --- Router Logic (Optimized) ---
        router_logits = self.router(x)  # [batch, seq_len, num_experts]
        probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        
        # --- Expert Dispatch (Parallel) ---
        # Flatten for parallel processing
        flat_x = x.view(-1, x.shape[-1])
        flat_topk_indices = topk_indices.view(-1)
        
        # One-hot mask for expert assignment
        expert_mask = F.one_hot(flat_topk_indices, self.num_experts).float()
        expert_inputs = torch.einsum('be,bd->ebd', expert_mask, flat_x)
        
        # --- Expert Computation (Fused) ---
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_outputs.append(expert(expert_inputs[i]))
        expert_outputs = torch.stack(expert_outputs)
        
        # --- Combine Outputs ---
        output = torch.einsum('ebd,be->bd', expert_outputs, expert_mask).view_as(x)
        output = self.dropout(output)
        
        # --- Load Balancing Loss ---
        aux_loss = self._compute_aux_loss(router_logits, probs)
        return output, aux_loss

    def _compute_aux_loss(self, router_logits: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        router_z_loss = torch.logsumexp(router_logits, dim=-1).mean() ** 2
        expert_load = probs.sum(dim=0)
        load_loss = (expert_load.std() / expert_load.mean()) ** 2
        return self.router_z_loss_coef * router_z_loss + self.router_load_loss_coef * load_loss

# --------------------------
# 3. Rotary Embeddings & Attention
# --------------------------
def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute the frequency tensor for complex rotation."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to input tensors."""
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(x_complex * freqs_cis).flatten(-2)

class FlashAttentionBlock(nn.Module):
    """Grouped-query attention with flash attention."""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int = 32768,
        dropout: float = 0.0,
        window_size: Optional[int] = None
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.dropout = dropout
        
        # Compute repeats needed for grouped-query attention
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        
        # Linear projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.head_dim * self.num_kv_heads, bias=False)
        self.v_proj = nn.Linear(dim, self.head_dim * self.num_kv_heads, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
        # Setup rotary embeddings
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(self.head_dim, max_seq_len),
            persistent=False
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Linear projections and reshape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Apply rotary embeddings
        q_rot = apply_rotary_emb(q, self.freqs_cis[:seq_len])
        k_rot = apply_rotary_emb(k, self.freqs_cis[:seq_len])
        
        # Repeat KV heads for grouped-query attention
        if self.num_key_value_groups > 1:
            k_rot = repeat(k_rot, 'b s h d -> b s (h g) d', g=self.num_key_value_groups)
            v = repeat(v, 'b s h d -> b s (h g) d', g=self.num_key_value_groups)
        
        # Run flash attention
        attn_output = flash_attn_func(
            q_rot, k_rot, v,
            dropout_p=self.dropout if self.training else 0.0,
            causal=True,
            window_size=self.window_size
        )
        
        # Project output
        return self.o_proj(attn_output.reshape(batch_size, seq_len, self.dim))

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: int = 10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Build here to make `torch.jit` work
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)

# --------------------------
# 4. Full Mixtral Model (Production-Grade)
# --------------------------
@dataclass
class MixtralConfig:
    """Configuration class for Mixtral model, matching Mixtral-8x7B specs"""
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    num_experts: int = 8
    num_experts_per_tok: int = 2
    sliding_window: int = 4096
    attention_dropout: float = 0.0
    router_z_loss_coef: float = 0.001
    router_aux_loss_coef: float = 0.001

class Mixtral(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32000,  # Matches Mixtral tokenizer
        dim: int = 4096,          # d_model in Mixtral
        num_layers: int = 32,
        num_heads: int = 32,
        num_kv_heads: int = 8,    # Grouped-query attention
        num_experts: int = 8,     # Mixtral uses 8 experts
        hidden_dim: int = 14336,  # Increased FFN dim
        top_k: int = 2,
        max_seq_len: int = 32768,
        sliding_window: int = 4096,
        dropout: float = 0.0,     # Mixtral uses no dropout
        router_z_loss_coef: float = 0.001,
        router_aux_loss_coef: float = 0.001
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        
        # Layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attention": FlashAttentionBlock(
                    dim=dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    max_seq_len=max_seq_len,
                    dropout=dropout,
                    window_size=sliding_window
                ),
                "moe": ParallelMoELayer(
                    dim=dim,
                    num_experts=num_experts,
                    hidden_dim=hidden_dim,
                    top_k=top_k,
                    dropout=dropout,
                    router_z_loss_coef=router_z_loss_coef,
                    router_aux_loss_coef=router_aux_loss_coef
                ),
                "norm1": RMSNorm(dim),
                "norm2": RMSNorm(dim),
            }) for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # Initialize router weights to small values
        for layer in self.layers:
            nn.init.normal_(layer["moe"].router.weight, mean=0.0, std=0.02 / math.sqrt(dim))
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.token_emb(x)
        total_aux_loss = 0.0
        expert_counts = torch.zeros(self.num_experts, device=x.device)
        
        for layer in self.layers:
            # Attention with gradient scaling
            attn_scale = 1.0 / math.sqrt(2.0 * len(self.layers))
            x = x + attn_scale * layer["attention"](layer["norm1"](x))
            
            # MoE with expert tracking
            moe_out, aux_loss = layer["moe"](layer["norm2"](x))
            x = x + moe_out
            total_aux_loss += aux_loss
            
            # Track expert usage
            if hasattr(layer["moe"], "last_expert_counts"):
                expert_counts += layer["moe"].last_expert_counts
            
        return {
            "logits": self.lm_head(self.norm(x)),
            "aux_loss": total_aux_loss,
        }

