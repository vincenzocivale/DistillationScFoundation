"""
Distilled Tahoe X1 Encoder Model

A student transformer encoder that replicates Tahoe X1 embeddings.
Fully compatible with HuggingFace transformers library.
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from typing import Optional, Tuple, Union
import math

from .configuration_distilled_tahoe import DistilledTahoeConfig


class GeneExpressionEmbedding(nn.Module):
    """
    Embedding layer for gene sequences with expression values.
    
    Combines:
    - Gene identity embeddings (which gene)
    - Expression value embeddings (how much it's expressed)
    - Position embeddings (order in sequence)
    """
    
    def __init__(self, config: DistilledTahoeConfig):
        super().__init__()
        self.config = config
        
        # Gene identity embeddings
        self.gene_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id
        )
        
        # Expression value embeddings (bins)
        if config.use_expression_embeddings:
            self.expression_embeddings = nn.Embedding(
                config.n_bins,
                config.hidden_size
            )
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )
        
        # Layer norm and dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Register position_ids as buffer
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False
        )
    
    def forward(
        self,
        gene_ids: torch.LongTensor,
        expression_bins: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        """
        Args:
            gene_ids: Gene token IDs (batch_size, seq_len)
            expression_bins: Expression bin indices (batch_size, seq_len)
            position_ids: Position indices (batch_size, seq_len)
        
        Returns:
            embeddings: Combined embeddings (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_length = gene_ids.shape
        
        # Gene embeddings
        embeddings = self.gene_embeddings(gene_ids)
        
        # Add expression embeddings
        if self.config.use_expression_embeddings and expression_bins is not None:
            expr_embeddings = self.expression_embeddings(expression_bins)
            embeddings = embeddings + expr_embeddings
        
        # Add position embeddings
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = embeddings + position_embeddings
        
        # Normalize and dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention layer"""
    
    def __init__(self, config: DistilledTahoeConfig):
        super().__init__()
        
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by "
                f"num_attention_heads ({config.num_attention_heads})"
            )
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Query, Key, Value projections
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # Output projection
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape for multi-head attention"""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
            output_attentions: Whether to return attention weights
        
        Returns:
            output: (batch_size, seq_len, hidden_size)
            attention_probs: Optional attention weights
        """
        batch_size, seq_length, _ = hidden_states.shape
        
        # Project to Q, K, V
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask
        if attention_mask is not None:
            # Convert 0/1 mask to -inf/0 mask
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (bs, 1, 1, seq_len)
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_scores.dtype).min
            attention_scores = attention_scores + attention_mask
        
        # Normalize to probabilities
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_shape)
        
        # Output projection with residual connection
        output = self.dense(context_layer)
        output = self.output_dropout(output)
        output = self.LayerNorm(output + hidden_states)
        
        outputs = (output, attention_probs) if output_attentions else (output,)
        return outputs


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, config: DistilledTahoeConfig):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation = nn.GELU()
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
        
        Returns:
            output: (batch_size, seq_len, hidden_size)
        """
        residual = hidden_states
        
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        output = self.LayerNorm(hidden_states + residual)
        return output


class TransformerLayer(nn.Module):
    """Single transformer layer"""
    
    def __init__(self, config: DistilledTahoeConfig):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)
        self.feed_forward = FeedForward(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
            output_attentions: Whether to return attention weights
        
        Returns:
            output: (batch_size, seq_len, hidden_size)
            attention_weights: Optional attention weights
        """
        # Self-attention
        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        hidden_states = attention_output[0]
        
        # Feed-forward
        hidden_states = self.feed_forward(hidden_states)
        
        outputs = (hidden_states,) + attention_output[1:]
        return outputs


class DistilledTahoeEncoder(nn.Module):
    """Stack of transformer layers"""
    
    def __init__(self, config: DistilledTahoeConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            TransformerLayer(config)
            for _ in range(config.num_hidden_layers)
        ])
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True
    ) -> Union[Tuple, BaseModelOutput]:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
            output_attentions: Return attention weights
            output_hidden_states: Return all hidden states
            return_dict: Return BaseModelOutput instead of tuple
        
        Returns:
            BaseModelOutput or tuple
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions
            )
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions]
                if v is not None
            )
        
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions
        )


class DistilledTahoeModel(PreTrainedModel):
    """
    Distilled Tahoe X1 Encoder - HuggingFace Compatible
    
    This is a student model that learns to produce embeddings similar to
    Tahoe X1 but with a simpler, more deployable architecture.
    
    Example:
        >>> from transformers import AutoModel
        >>> model = AutoModel.from_pretrained("yourusername/distilled-tahoe-70m")
        >>> 
        >>> # Tokenize single-cell data
        >>> inputs = tokenizer(adata)
        >>> 
        >>> # Get embeddings
        >>> outputs = model(**inputs)
        >>> embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
    """
    
    config_class = DistilledTahoeConfig
    base_model_prefix = "distilled_tahoe"
    
    def __init__(self, config: DistilledTahoeConfig):
        super().__init__(config)
        self.config = config
        
        # Embeddings
        self.embeddings = GeneExpressionEmbedding(config)
        
        # Encoder
        self.encoder = DistilledTahoeEncoder(config)
        
        # Initialize weights
        self.post_init()
    
    def get_input_embeddings(self):
        return self.embeddings.gene_embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings.gene_embeddings = value
    
    def forward(
        self,
        gene_ids: torch.LongTensor,
        expression_bins: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, BaseModelOutput]:
        """
        Args:
            gene_ids: Gene token IDs (batch_size, seq_len)
            expression_bins: Expression bin indices (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            position_ids: Position indices (batch_size, seq_len)
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return BaseModelOutput
        
        Returns:
            BaseModelOutput with:
                - last_hidden_state: (batch_size, seq_len, hidden_size)
                - pooler_output: (batch_size, hidden_size) - pooled representation
                - hidden_states: All layer hidden states (if requested)
                - attentions: All attention weights (if requested)
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (gene_ids != self.config.pad_token_id).long()
        
        # Get embeddings
        embedding_output = self.embeddings(
            gene_ids=gene_ids,
            expression_bins=expression_bins,
            position_ids=position_ids
        )
        
        # Encode
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
        sequence_output = encoder_outputs[0] if not return_dict else encoder_outputs.last_hidden_state
        
        # Pool output based on strategy
        pooled_output = self._pool_output(sequence_output, attention_mask)
        
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    
    def _pool_output(self, sequence_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Pool sequence output to get cell-level embedding.
        
        Args:
            sequence_output: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            pooled: (batch_size, hidden_size)
        """
        if self.config.pooling_strategy == "cls":
            # Use CLS token (first token)
            pooled = sequence_output[:, 0, :]
        
        elif self.config.pooling_strategy == "mean":
            # Mean pooling over non-masked tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
            sum_embeddings = torch.sum(sequence_output * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        
        elif self.config.pooling_strategy == "max":
            # Max pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
            sequence_output = sequence_output.clone()
            sequence_output[mask_expanded == 0] = -1e9
            pooled = torch.max(sequence_output, dim=1)[0]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.config.pooling_strategy}")
        
        return pooled
    
    def get_cell_embeddings(
        self,
        gene_ids: torch.LongTensor,
        expression_bins: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        """
        Convenience method to get cell-level embeddings.
        
        Args:
            gene_ids: Gene token IDs (batch_size, seq_len)
            expression_bins: Expression bin indices (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
        
        Returns:
            embeddings: Cell embeddings (batch_size, hidden_size)
        """
        outputs = self.forward(
            gene_ids=gene_ids,
            expression_bins=expression_bins,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        return outputs.pooler_output


# Register the model for AutoModel
from transformers import AutoConfig, AutoModel

AutoConfig.register("distilled_tahoe", DistilledTahoeConfig)
AutoModel.register(DistilledTahoeConfig, DistilledTahoeModel)
