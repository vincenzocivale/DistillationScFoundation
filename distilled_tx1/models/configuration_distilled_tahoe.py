"""
Distilled Tahoe X1 Encoder Configuration

HuggingFace-compatible configuration for the distilled student encoder.
"""

from transformers import PretrainedConfig
from typing import Optional, List


class DistilledTahoeConfig(PretrainedConfig):
    """
    Configuration class for Distilled Tahoe X1 Encoder.
    
    This is a student model that learns to replicate Tahoe X1 embeddings
    using a simpler, more deployment-friendly architecture.
    
    Args:
        vocab_size: Size of gene vocabulary (including special tokens)
        n_bins: Number of expression value bins (default: 51)
        hidden_size: Dimension of hidden layers and embeddings
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        intermediate_size: Dimension of FFN intermediate layer
        hidden_dropout_prob: Dropout probability
        attention_probs_dropout_prob: Attention dropout probability
        max_position_embeddings: Maximum sequence length
        layer_norm_eps: Layer normalization epsilon
        use_expression_embeddings: Whether to use expression value embeddings
        pooling_strategy: How to pool sequence ("cls", "mean", "max")
        tie_word_embeddings: Whether to tie input/output embeddings
    """
    
    model_type = "distilled_tahoe"
    
    def __init__(
        self,
        vocab_size: int = 30000,
        n_bins: int = 51,
        hidden_size: int = 512,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        intermediate_size: int = 2048,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 2048,
        layer_norm_eps: float = 1e-12,
        use_expression_embeddings: bool = True,
        pooling_strategy: str = "cls",
        tie_word_embeddings: bool = False,
        pad_token_id: int = 0,
        cls_token_id: int = 2,
        sep_token_id: int = 3,
        mask_token_id: int = 1,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            **kwargs
        )
        
        self.vocab_size = vocab_size
        self.n_bins = n_bins
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.use_expression_embeddings = use_expression_embeddings
        self.pooling_strategy = pooling_strategy
        self.tie_word_embeddings = tie_word_embeddings
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.mask_token_id = mask_token_id
    
    @property
    def attention_head_size(self):
        return self.hidden_size // self.num_attention_heads
