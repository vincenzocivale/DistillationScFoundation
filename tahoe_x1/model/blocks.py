# Copyright (C) Tahoe Therapeutics 2025. All rights reserved.
import logging
from functools import lru_cache
from typing import Any, Dict, Optional, Union, Tuple

import numpy as np
import torch
from composer.utils import dist # Keep for now as it's used in GeneEncoder and ChemEncoder for S3 downloads
# Removed llmfoundry imports
# from llmfoundry.layers_registry import attention_classes, norms
# from llmfoundry.models.layers.ffn import (
#     resolve_ffn_act_fn,
#     resolve_ffn_hidden_size,
# )
# from llmfoundry.models.mpt.modeling_mpt import gen_flash_attn_padding_info
from torch import Tensor, nn
from torch.nn.modules.transformer import _get_clones

from tahoe_x1.utils import download_file_from_s3_url

# --- Helper functions to replace llmfoundry utilities ---

def _get_activation_fn(activation_name: str) -> nn.Module:
    """Helper function to get activation module from string name."""
    if activation_name == "relu":
        return nn.ReLU()
    elif activation_name == "gelu":
        return nn.GELU()
    elif activation_name == "leaky_relu":
        return nn.LeakyReLU()
    elif activation_name == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")

def _get_norm_layer(norm_type: str, d_model: int, eps: float = 1e-5) -> nn.Module:
    """Helper function to get normalization layer from string name."""
    if norm_type == "layernorm" or norm_type == "low_precision_layernorm": # Treat low_precision as standard LayerNorm
        return nn.LayerNorm(d_model, eps=eps)
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")


# --- Custom MultiheadAttention for conditional Flash Attention ---
class _CustomMultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attn_impl: str = "torch", attn_pdrop: float = 0.0, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_impl = attn_impl
        self.attn_pdrop = attn_pdrop

        assert self.head_dim * n_heads == self.d_model, "d_model must be divisible by n_heads"

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(attn_pdrop)

        if self.attn_impl == "flash":
            if not hasattr(F, "scaled_dot_product_attention"):
                raise ImportError("Flash Attention requires PyTorch 2.0 or higher. Please upgrade PyTorch or set attn_impl='torch'.")

    def forward(
        self,
        query: Tensor, # (batch, seq_len, d_model)
        key: Tensor, # (batch, seq_len, d_model)
        value: Tensor, # (batch, seq_len, d_model)
        attn_mask: Optional[Tensor] = None, # (seq_len, seq_len) or (batch, seq_len, seq_len)
        key_padding_mask: Optional[Tensor] = None, # (batch, seq_len)
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]: # (output, attn_weights, past_key_value) - simplified return

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        if self.attn_impl == "flash":
            # Reshape for Flash Attention (batch, num_heads, seq_len, head_dim)
            q = q.view(query.shape[0], -1, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(key.shape[0], -1, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(value.shape[0], -1, self.n_heads, self.head_dim).transpose(1, 2)
            
            # F.scaled_dot_product_attention takes boolean attn_mask directly
            # key_padding_mask can be merged into attn_mask
            if key_padding_mask is not None:
                if attn_mask is None:
                    # Create a broadasting mask (batch, 1, 1, key_len)
                    attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).expand(-1, self.n_heads, query.shape[1], -1)
                else:
                    # Combine existing attn_mask with key_padding_mask
                    # Assuming attn_mask is already (batch, num_heads, query_len, key_len) or broadcastable
                    attn_mask = attn_mask | key_padding_mask.unsqueeze(1).unsqueeze(2)


            output = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attn_mask, # Should be boolean for F.scaled_dot_product_attention
                dropout_p=self.attn_pdrop if self.training else 0.0,
                is_causal=is_causal,
            )
            output = output.transpose(1, 2).reshape(query.shape[0], -1, self.d_model)
        else: # Standard PyTorch MultiheadAttention-like implementation
            q = q.view(query.shape[0], query.shape[1], self.n_heads, self.head_dim).transpose(1, 2) # (B, H, S, D_head)
            k = k.view(key.shape[0], key.shape[1], self.n_heads, self.head_dim).transpose(1, 2) # (B, H, S, D_head)
            v = v.view(value.shape[0], value.shape[1], self.n_heads, self.head_dim).transpose(1, 2) # (B, H, S, D_head)

            # Apply scaled dot-product attention manually
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            
            if attn_mask is not None:
                # Expand attn_mask to be broadcastable if it's (S,S) or (B,S,S)
                if attn_mask.dim() == 2: # (S, S)
                    scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
                elif attn_mask.dim() == 3: # (B, S, S)
                    scores = scores + attn_mask.unsqueeze(1)
            
            if key_padding_mask is not None:
                # Apply key_padding_mask
                scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            output = torch.matmul(attn_weights, v) # (B, H, S, D_head)
            output = output.transpose(1, 2).contiguous().view(query.shape[0], -1, self.d_model)

        output = self.out_proj(output)
        return output, None, None # Simplified return: output, (attn_weights, past_key_value are not returned directly)


attn_config_defaults: Dict = {
    "attn_type": "grouped_query_attention",
    "attn_pdrop": 0.0,
    "attn_impl": "torch",
    "use_attn_mask": True,
    "qk_ln": False,
    "qk_gn": False,
    "clip_qkv": None,
    "softmax_scale": None,
}

norm_config_defaults: Dict = {
    "norm_type": "low_precision_layernorm",
    "eps": 1e-5,
}

init_config_defaults: Dict = {
    "name": "kaiming_normal_",
    "fan_mode": "fan_in",
    "init_nonlinearity": "relu",
    "init_div_is_residual": True,
    "emb_init_std": None,
    "emb_init_uniform_lim": None,
    "init_std": None,
    "init_gain": 0.0,
}

gene_encoder_defaults: Dict = {
    "use_norm": False,
}

log = logging.getLogger(__name__)


class TXBlock(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    The class is modified from torch.nn.TransformerEncoderLayer to support the
    FlashAttention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        expansion_ratio: int,
        attn_config: Optional[Dict] = None,
        norm_config: Optional[Dict] = None,
        dropout: Optional[float] = 0.0,
        activation: Optional[str] = "gelu",
        device: Optional[str] = None,
        dtype=None,
        norm_scheme="pre",
        use_glu: bool = False,
        **kwargs: Any,
    ) -> None:
        if attn_config is None:
            attn_config = attn_config_defaults
        if norm_config is None:
            norm_config = norm_config_defaults
        del kwargs  # unused, just to capture any extra args from the config
        super().__init__()
        # factory_kwargs = {"device": device, "dtype": dtype} # device not directly passed to nn.Linear anymore

        self.d_model = d_model
        self.n_heads = n_heads
        self.device = device
        
        # Replaced llmfoundry attention_classes with custom MultiheadAttention
        self.self_attn = _CustomMultiheadAttention(
            d_model=d_model,
            n_heads=n_heads,
            attn_impl=attn_config.get("attn_impl", "torch"),
            attn_pdrop=attn_config.get("attn_pdrop", 0.0),
        )
        
        # Replaced llmfoundry resolve_ffn_hidden_size
        dim_feedforward = d_model * expansion_ratio 
        self.up_proj = nn.Linear(d_model, dim_feedforward)
        self.down_proj = nn.Linear(dim_feedforward, d_model)
        self.use_glu = use_glu
        if self.use_glu:
            self.gate_proj = nn.Linear(d_model, dim_feedforward)

        # Replaced llmfoundry norms with custom helper
        self.norm1 = _get_norm_layer(norm_config["norm_type"], d_model, eps=norm_config.get("eps", 1e-5))
        self.norm2 = _get_norm_layer(norm_config["norm_type"], d_model, eps=norm_config.get("eps", 1e-5))
        self.post_sa_dropout = nn.Dropout(dropout)
        self.post_ffn_dropout = nn.Dropout(dropout)

        # Replaced llmfoundry resolve_ffn_act_fn with custom helper
        self.activation = _get_activation_fn(activation)
        self.norm_scheme = norm_scheme
        if norm_scheme not in ["pre", "post"]:
            raise ValueError("norm_scheme must be either pre or post")

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None, # Corresponds to attn_bias in original
        key_padding_mask: Optional[Tensor] = None, # Used by CustomMultiheadAttention
        is_causal: bool = False,
    ) -> Tensor:
        
        if self.norm_scheme == "pre":
            x = x + self._sa_block(
                self.norm1(x),
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x
                + self._sa_block(
                    x,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    is_causal=is_causal,
                ),
            )
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        x, _, _ = self.self_attn(
            x, x, x, # Q, K, V are all x for self-attention
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
        )
        return self.post_sa_dropout(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        if self.use_glu:
            x = self.down_proj(self.activation(self.gate_proj(x)) * self.up_proj(x))
        else:
            x = self.down_proj(self.activation(self.up_proj(x)))
        return self.post_ffn_dropout(x)


class TXEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers.
    """

    def __init__(
        self,
        encoder_layer: TXBlock,
        num_layers: int,
        use_norm: bool = False,
        norm_config: Optional[Dict] = None,
        attn_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.use_norm = use_norm

        if attn_config is None:
            attn_config = attn_config_defaults
        self.use_attn_mask = attn_config.get("use_attn_mask", True)
        self.attn_impl = attn_config.get("attn_impl", "torch")

        if self.use_norm:
            if norm_config is None:
                norm_config = norm_config_defaults
            # Replaced llmfoundry norms with custom helper
            self.norm = _get_norm_layer(norm_config["norm_type"], encoder_layer.d_model, eps=norm_config.get("eps", 1e-5))

    def forward(
        self,
        total_embs: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        gen_mask: Optional[Tensor] = None,
    ) -> Tensor:

        attn_mask = None
        is_causal = False

        if self.use_attn_mask:
            # Recreating _make_mask logic here from original, but adjusting for F.scaled_dot_product_attention
            mask_dim = 2
            assert gen_mask.dtype == torch.bool and gen_mask.dim() == mask_dim
            _, S = gen_mask.shape
            device = gen_mask.device

            pcpt_cols = (~gen_mask).unsqueeze(1)  # (B, 1, S), broadcasts across rows
            eye = torch.eye(S, dtype=torch.bool, device=device)  # (S, S)
            gen_diag = gen_mask.unsqueeze(2) & eye  # (B, S, S)
            attn_mask_bool = pcpt_cols | gen_diag  # (B, S, S) bool

            if self.attn_impl == "flash":
                # F.scaled_dot_product_attention expects boolean mask where True means masked (attention ignored)
                attn_mask = ~attn_mask_bool
                # Key padding mask needs to be combined here for flash attention
                if key_padding_mask is not None:
                     # Expand key_padding_mask to (batch, 1, 1, key_len) and combine
                     attn_mask = attn_mask | key_padding_mask.unsqueeze(1).unsqueeze(2)

            else: # For torch/custom implementation, usually -inf for masked
                attn_mask = torch.zeros_like(
                    attn_mask_bool,
                    dtype=total_embs.dtype,
                    device=device,
                    requires_grad=False,
                ).masked_fill_(
                    ~attn_mask_bool,
                    torch.finfo(total_embs.dtype).min,
                )
                # key_padding_mask will be passed separately to CustomMHA if not flash

        for mod in self.layers:
            total_embs = mod(
                total_embs,
                attn_mask=attn_mask, # Pass the generated attn_mask
                key_padding_mask=key_padding_mask, # Pass key_padding_mask separately for non-flash
                is_causal=is_causal,
            )

        if self.use_norm:
            total_embs = self.norm(total_embs)

        return total_embs

    @torch.no_grad()
    @lru_cache(maxsize=1)
    def _make_mask(self, gen_mask: Tensor, device) -> Tensor:
        """
        gen_mask: (B, S) bool, True = generative token, False = perceptual token
        Returns: (B, S, S) bool, True = attention allowed.

        Rules:
        - pcpt rows (False) cannot attend to gen columns (True)
        - gen rows (True) can attend to all pcpt columns (False) and themselves (diagonal only among gen)
        """
        mask_dim = 2
        assert gen_mask.dtype == torch.bool and gen_mask.dim() == mask_dim
        _, S = gen_mask.shape
        # device = gen_mask.device # device is passed as an argument

        # Allow attending to all perceptual columns, for every row.
        pcpt_cols = (~gen_mask).unsqueeze(1)  # (B, 1, S), broadcasts across rows

        # Allow diagonal for gen rows only.
        eye = torch.eye(S, dtype=torch.bool, device=device)  # (S, S)
        gen_diag = gen_mask.unsqueeze(2) & eye  # (B, S, S)

        # Combine: allowed if column is pcpt OR (row is gen AND i==j)
        attention_mask = pcpt_cols | gen_diag  # (B, S, S) bool
        return attention_mask


class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        use_norm: bool = False,
        gene_encoder_cfg: Optional[Dict] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
        )
        self.use_norm = use_norm
        if not gene_encoder_cfg:
            gene_encoder_cfg = {}
        additional_embedding_cfg = gene_encoder_cfg.get("embeddings", {})
        self.extra_embeddings = nn.ModuleDict()
        self.extra_norms = nn.ModuleDict()

        for name, e_cfg in additional_embedding_cfg.items():
            local, remote = e_cfg["local"], e_cfg["remote"]
            if dist.get_local_rank() == 0:
                download_file_from_s3_url(remote, local)
            with dist.local_rank_zero_download_and_wait(local):
                dist.barrier()

            pretrained_weight = torch.load(local, weights_only=True)["embedding.weight"]
            pretrained_vocab_size, pretrained_dim = pretrained_weight.shape
            if pretrained_vocab_size < num_embeddings:
                log.warning(
                    f"[{name}] Pretrained embedding size ({pretrained_vocab_size}) is smaller than vocab size ({num_embeddings}). "
                    f"Filling remaining {num_embeddings - pretrained_vocab_size} rows with zeros.",
                )
            weight = torch.zeros(
                num_embeddings,
                pretrained_dim,
                dtype=pretrained_weight.dtype,
            )
            weight[:pretrained_vocab_size, :] = pretrained_weight
            emb = nn.Embedding.from_pretrained(
                weight,
                padding_idx=padding_idx,
                freeze=e_cfg.get("freeze", True),
            )
            # for m in emb.modules(): # composer specific attribute skip_init
            #     m.skip_init = True
            self.extra_embeddings[name] = emb

            if e_cfg.get("use_norm", False):
                self.extra_norms[name] = nn.LayerNorm(emb.embedding_dim)

        if self.extra_embeddings:
            concat_dim = embedding_dim + sum(
                emb.embedding_dim for emb in self.extra_embeddings.values()
            )
            self.project = nn.Linear(concat_dim, embedding_dim, bias=False)
        else:
            self.project = nn.Identity()

        if self.use_norm:
            self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        reps = [self.embedding(x)]
        for name, emb in self.extra_embeddings.items():
            y = emb(x)
            if name in self.extra_norms:
                y = self.extra_norms[name](y)
            reps.append(y)
        x = torch.cat(reps, dim=-1) if len(reps) > 1 else reps[0]
        x = self.project(x)
        if self.use_norm:
            x = self.enc_norm(x)
        return x


class ChemEncoder(nn.Module):
    def __init__(
        self,
        d_out: int,
        padding_idx: int = 0,
        activation: str = "leaky_relu",
        use_norm: bool = True,
        freeze: bool = False,
        drug_fps_path: Optional[dict] = None,
        num_drugs: Optional[int] = None,
        fp_dim: Optional[int] = None,
    ):
        super().__init__()

        # download pretrained drug embeddings if specified, otherwise use arguments
        if drug_fps_path is not None:
            if dist.get_local_rank() == 0:
                download_file_from_s3_url(
                    s3_url=drug_fps_path["remote"],
                    local_file_path=drug_fps_path["local"],
                )
            with dist.local_rank_zero_download_and_wait(drug_fps_path["local"]):
                dist.barrier()

            drug_fps = torch.as_as_tensor(
                np.load(drug_fps_path["local"]),
                dtype=torch.float32,
            )
            embedding_dim = drug_fps.shape[1]
        else:
            assert num_drugs is not None and fp_dim is not None
            embedding_dim = fp_dim
            drug_fps = torch.zeros((num_drugs, fp_dim), dtype=torch.float32)

        self.embedding = nn.Embedding.from_pretrained(
            drug_fps,
            padding_idx=padding_idx,
            freeze=freeze,
        )
        # for m in self.embedding.modules(): # composer specific attribute skip_init
        #     m.skip_init = True
        self.fc = nn.Linear(embedding_dim, d_out)
        self.activation = _get_activation_fn(activation) # Replaced llmfoundry helper
        self.proj = nn.Linear(d_out, d_out)

        self.use_norm = use_norm
        if self.use_norm:
            self.norm = nn.LayerNorm(d_out)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, d_out)
        x = self.activation(self.fc(x))
        x = self.proj(x)  # (batch, d_out)

        if self.use_norm:
            x = self.norm(x)
        return x


class ContinuousValueEncoder(nn.Module):
    """Encode real number values to a vector using neural nets projection."""

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_value: int = 512,
        activation: str = "relu",
        use_norm: bool = False,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = _get_activation_fn(activation) # Replaced llmfoundry helper
        self.linear2 = nn.Linear(d_model, d_model)
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        # expand last dimension
        x = x.unsqueeze(-1)
        # clip x to [-inf, max_value]
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        if self.use_norm:
            x = self.norm(x)
        return self.dropout(x)


class ExprDecoder(nn.Module):
    """Consists of three linear functions and leaky-relu as an activation
    function."""

    def __init__(
        self,
        d_model: int,
        n_outputs: int = 1,
        n_layers: int = 2,
        activation: str = "leaky_relu",
    ):
        super().__init__()
        d_in = d_model
        self.activation = _get_activation_fn(activation) # Replaced llmfoundry helper
        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_in, d_model) for _ in range(n_layers)],
        )
        self.out_proj = nn.Linear(d_model, n_outputs)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """X is the output of the transformer, (batch, seq_len, d_model)"""
        for layer in self.linear_layers:
            x = self.activation(layer(x))
        pred_value = self.out_proj(x)  # (batch, seq_len, n_outputs)
        if pred_value.shape[-1] == 1:
            pred_value = pred_value.squeeze(-1)
        return {"pred": pred_value}


class MVCDecoder(nn.Module):
    """Decoder for the masked value prediction for cell embeddings."""

    def __init__(
        self,
        d_model: int,
        arch_style: str = "inner product",
        query_activation: str = "sigmoid",
        scaled_dot_product: bool = False,
    ) -> None:
        """
        Args:
            d_model (:obj:`int`): dimension of the gene embedding.
            arch_style (:obj:`str`): architecture style of the decoder, choice from
                1. "inner product" or 2. "concat query" or 3. "sum query".
            query_activation (:obj:`nn.Module`): activation function for the query
                vectors.
            hidden_activation (:obj:`nn.Module`): activation function for the hidden
                layers.
        """
        super().__init__()
        d_in = d_model
        self.scaled_dot_product = scaled_dot_product
        if arch_style == "inner product":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = _get_activation_fn(query_activation) # Replaced llmfoundry helper
            self.W = nn.Linear(d_model, d_in, bias=False)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")

        self.arch_style = arch_style

    def forward(
        self,
        cell_emb: Tensor,
        gene_embs: Tensor,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Args:
            cell_emb: Tensor, shape (batch, embsize=d_model)
            gene_embs: Tensor, shape (batch, seq_len, embsize=d_model)
        """
        if self.arch_style == "inner product":
            query_vecs = self.query_activation(
                self.gene2query(gene_embs),
            )  # (batch, seq_len, embsize)
            inner_product_dimension = query_vecs.shape[-1]
            cell_emb = cell_emb.unsqueeze(2)  # (batch, embsize, 1)
            pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(
                2,
            )  # (batch, seq_len)
            if self.scaled_dot_product:
                pred_value = pred_value / torch.sqrt(
                    torch.tensor(inner_product_dimension, dtype=pred_value.dtype),
                )
            return {"pred": pred_value}
        else:
            raise ValueError(f"Unknown arch_style: {self.arch_style}")