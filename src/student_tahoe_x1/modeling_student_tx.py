import logging
from typing import Mapping, Optional, Dict, Any, Tuple, Union
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.transformer import _get_clones
from transformers import PreTrainedModel

from tahoe_x1.utils import download_file_from_s3_url # For GeneEncoder
from tahoe_x1.tokenizer import GeneVocab # Needed for GeneEncoder
from .configuration_student_tx import StudentTXConfig

log = logging.getLogger(__name__)

# --- Helper functions (re-used from previous llmfoundry removal) ---

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
    if norm_type == "layernorm" or norm_type == "low_precision_layernorm":
        return nn.LayerNorm(d_model, eps=eps)
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")

def _init_weights(module: nn.Module, n_layers: int, d_model: int, init_config: Dict[str, Any]):
    """Basic weight initialization function."""
    if init_config["name"] == "kaiming_normal_":
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if init_config.get("init_div_is_residual", True) and n_layers > 1:
                scale = 1.0 / n_layers
                nn.init.kaiming_normal_(module.weight, mode=init_config["fan_mode"], nonlinearity=init_config["init_nonlinearity"])
                module.weight.data.mul_(scale)
            else:
                nn.init.kaiming_normal_(module.weight, mode=init_config["fan_mode"], nonlinearity=init_config["init_nonlinearity"])
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_config["name"] == "xavier_uniform_":
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if init_config.get("init_div_is_residual", True) and n_layers > 1:
                scale = 1.0 / n_layers
                nn.init.xavier_uniform_(module.weight, gain=init_config["init_gain"])
                module.weight.data.mul_(scale)
            else:
                nn.init.xavier_uniform_(module.weight, gain=init_config["init_gain"])
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_config["name"] == "skip":
        pass
    else:
        log.warning(f"Unsupported init_config name: {init_config['name']}. Using default PyTorch initialization.")


class CustomMultiheadAttention(nn.Module):
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
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        if self.attn_impl == "flash":
            q = q.view(query.shape[0], -1, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(key.shape[0], -1, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(value.shape[0], -1, self.n_heads, self.head_dim).transpose(1, 2)
            
            if key_padding_mask is not None:
                if attn_mask is None:
                    attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
                else:
                    attn_mask = attn_mask | key_padding_mask.unsqueeze(1).unsqueeze(2)

            output = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attn_mask,
                dropout_p=self.attn_pdrop if self.training else 0.0,
                is_causal=is_causal,
            )
            output = output.transpose(1, 2).reshape(query.shape[0], -1, self.d_model)
        else:
            q = q.view(query.shape[0], query.shape[1], self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(key.shape[0], key.shape[1], self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(value.shape[0], value.shape[1], self.n_heads, self.head_dim).transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
                elif attn_mask.dim() == 3:
                    scores = scores + attn_mask.unsqueeze(1)
            
            if key_padding_mask is not None:
                scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            output = torch.matmul(attn_weights, v)
            output = output.transpose(1, 2).contiguous().view(query.shape[0], -1, self.d_model)

        output = self.out_proj(output)
        return output, None, None


# --- Re-implementations of blocks.py components without llmfoundry ---

class StudentTXBlock(nn.Module):
    def __init__(self, config: StudentTXConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.norm_scheme = config.norm_scheme
        
        attn_config = config.attn_config
        norm_config = config.norm_config

        self.self_attn = CustomMultiheadAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            attn_impl=attn_config.get("attn_impl", "torch"),
            attn_pdrop=attn_config.get("attn_pdrop", 0.0),
        )
        
        dim_feedforward = config.d_model * config.expansion_ratio 
        self.up_proj = nn.Linear(self.d_model, dim_feedforward)
        self.down_proj = nn.Linear(dim_feedforward, self.d_model)
        self.use_glu = config.use_glu
        if self.use_glu:
            self.gate_proj = nn.Linear(self.d_model, dim_feedforward)

        self.norm1 = _get_norm_layer(norm_config["norm_type"], self.d_model, eps=norm_config.get("eps", 1e-5))
        self.norm2 = _get_norm_layer(norm_config["norm_type"], self.d_model, eps=norm_config.get("eps", 1e-5))
        self.post_sa_dropout = nn.Dropout(attn_config.get("attn_pdrop", 0.0))
        self.post_ffn_dropout = nn.Dropout(config.cv_encoder_dropout)

        self.activation = _get_activation_fn(config.transformer_activation)

    def forward(
        self,
        hidden_states: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        
        if self.norm_scheme == "pre":
            x = hidden_states + self._sa_block(
                self.norm1(hidden_states),
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                hidden_states
                + self._sa_block(
                    hidden_states,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    is_causal=is_causal,
                ),
            )
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(
        self,
        hidden_states: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        x, _, _ = self.self_attn(
            hidden_states,
            hidden_states,
            hidden_states,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
        )
        return self.post_sa_dropout(x)

    def _ff_block(self, hidden_states: Tensor) -> Tensor:
        if self.use_glu:
            hidden_states = self.down_proj(self.activation(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        else:
            hidden_states = self.down_proj(self.activation(self.up_proj(hidden_states)))
        return self.post_ffn_dropout(hidden_states)


class StudentTXEncoder(nn.Module):
    def __init__(self, config: StudentTXConfig, encoder_layer: StudentTXBlock):
        super().__init__()
        self.layers = _get_clones(encoder_layer, config.n_layers)
        self.num_layers = config.n_layers
        self.use_norm = config.norm_scheme == "pre"
        self.use_attn_mask = config.attn_config.get("use_attn_mask", True)
        self.attn_impl = config.attn_config.get("attn_impl", "torch")

        if self.use_norm:
            norm_config = config.norm_config
            self.norm = _get_norm_layer(norm_config["norm_type"], config.d_model, eps=norm_config.get("eps", 1e-5))

    def forward(
        self,
        total_embs: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        gen_mask: Optional[Tensor] = None,
    ) -> Tensor:
        
        attn_mask = None
        is_causal = False

        if self.use_attn_mask:
            mask_dim = 2
            assert gen_mask.dtype == torch.bool and gen_mask.dim() == mask_dim
            _, S = gen_mask.shape
            device = gen_mask.device

            pcpt_cols = (~gen_mask).unsqueeze(1)
            eye = torch.eye(S, dtype=torch.bool, device=device)
            gen_diag = gen_mask.unsqueeze(2) & eye
            attn_mask_bool = pcpt_cols | gen_diag

            if self.attn_impl == "flash":
                attn_mask = ~attn_mask_bool
                if key_padding_mask is not None:
                     attn_mask = attn_mask | key_padding_mask.unsqueeze(1).unsqueeze(2)

            else:
                attn_mask = torch.zeros_like(
                    attn_mask_bool,
                    dtype=total_embs.dtype,
                    device=device,
                    requires_grad=False,
                ).masked_fill_(
                    ~attn_mask_bool,
                    torch.finfo(total_embs.dtype).min,
                )

        for mod in self.layers:
            total_embs = mod(
                total_embs,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
            )

        if self.use_norm:
            total_embs = self.norm(total_embs)

        return total_embs

class GeneEncoder(nn.Module):
    def __init__(self, config: StudentTXConfig):
        super().__init__()
        num_embeddings = config.vocab_size
        embedding_dim = config.d_model
        padding_idx = config.gene_encoder_config.get("padding_idx", None)
        use_norm = config.gene_encoder_config.get("use_norm", False)
        
        self.embedding = nn.Embedding(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
        )
        self.use_norm = use_norm
        
        self.extra_embeddings = nn.ModuleDict() 
        self.extra_norms = nn.ModuleDict()
        
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
        x = torch.cat(reps, dim=-1) if len(reps) > 1 else reps[0]
        x = self.project(x)
        if self.use_norm:
            x = self.enc_norm(x)
        return x


# ChemEncoder is removed as per user's request
# class ChemEncoder(nn.Module):
#     def __init__(self, config: StudentTXConfig):
#         super().__init__()
#         d_out = config.d_model
#         padding_idx = config.chem_encoder_padding_idx
#         activation_str = config.chem_encoder_activation
#         freeze = config.chem_encoder_freeze
#         drug_fps_path_cfg = config.drug_fps_path
#         num_drugs = config.num_drugs
#         fp_dim = config.fp_dim

#         drug_fps = None
#         if drug_fps_path_cfg is not None:
#             local_path = drug_fps_path_cfg["local"]
#             if Path(local_path).exists():
#                 drug_fps = torch.as_tensor(
#                     np.load(local_path),
#                     dtype=torch.float32,
#                 )
#                 embedding_dim = drug_fps.shape[1]
#             else:
#                 log.warning(f"Drug fingerprints not found at {local_path}. Initializing with zeros.")
#                 assert num_drugs is not None and fp_dim is not None
#                 embedding_dim = fp_dim
#                 drug_fps = torch.zeros((num_drugs, fp_dim), dtype=torch.float32)
#         else:
#             assert num_drugs is not None and fp_dim is not None
#             embedding_dim = fp_dim
#             drug_fps = torch.zeros((num_drugs, fp_dim), dtype=torch.float32)

#         self.embedding = nn.Embedding.from_pretrained(
#             drug_fps,
#             padding_idx=padding_idx,
#             freeze=freeze,
#         )
#         self.fc = nn.Linear(embedding_dim, d_out)
#         self.activation = _get_activation_fn(activation_str)
#         self.proj = nn.Linear(d_out, d_out)

#         self.use_norm = config.cv_encoder_use_norm
#         if self.use_norm:
#             self.norm = nn.LayerNorm(d_out)

#     def forward(self, x: Tensor) -> Tensor:
#         x = self.embedding(x)
#         x = self.activation(self.fc(x))
#         x = self.proj(x)

#         if self.use_norm:
#             x = self.norm(x)
#         return x


class ContinuousValueEncoder(nn.Module):
    def __init__(self, config: StudentTXConfig):
        super().__init__()
        d_model = config.d_model
        dropout = config.cv_encoder_dropout
        max_value = config.cv_encoder_max_value
        activation_str = config.cv_encoder_activation
        use_norm = config.cv_encoder_use_norm
        
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = _get_activation_fn(activation_str)
        self.linear2 = nn.Linear(d_model, d_model)
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(-1)
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        if self.use_norm:
            x = self.norm(x)
        return self.dropout(x)


class ExprDecoder(nn.Module):
    def __init__(self, config: StudentTXConfig):
        super().__init__()
        d_model = config.d_model
        n_outputs = config.expr_decoder_n_outputs
        n_layers = config.expr_decoder_n_layers
        activation_str = config.expr_decoder_activation
        
        d_in = d_model
        self.activation = _get_activation_fn(activation_str)
        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_in, d_model) for _ in range(n_layers)],
        )
        self.out_proj = nn.Linear(d_model, n_outputs)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        for layer in self.linear_layers:
            x = self.activation(layer(x))
        pred_value = self.out_proj(x)
        if pred_value.shape[-1] == 1:
            pred_value = pred_value.squeeze(-1)
        return {"pred": pred_value}


class MVCDecoder(nn.Module):
    def __init__(self, config: StudentTXConfig):
        super().__init__()
        d_model = config.d_model
        arch_style = config.mvc_arch_style
        query_activation_str = config.mvc_query_activation
        self.scaled_dot_product = config.mvc_scaled_dot_product
        
        if arch_style == "inner product":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = _get_activation_fn(query_activation_str)
            self.W = nn.Linear(d_model, d_model, bias=False)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")

        self.arch_style = arch_style

    def forward(
        self,
        cell_emb: Tensor,
        gene_embs: Tensor,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        if self.arch_style == "inner product":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            inner_product_dimension = query_vecs.shape[-1]
            cell_emb = cell_emb.unsqueeze(2)
            pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(2)
            if self.scaled_dot_product:
                pred_value = pred_value / torch.sqrt(
                    torch.tensor(inner_product_dimension, dtype=pred_value.dtype),
                )
            return {"pred": pred_value}
        else:
            raise ValueError(f"Unknown arch_style: {self.arch_style}")


# --- StudentTXModel implementation ---

class StudentTXModel(PreTrainedModel):
    config_class = StudentTXConfig
    base_model_prefix = "student_tx"

    def __init__(self, config: StudentTXConfig):
        super().__init__(config)
        self.config = config
        
        self.model_type = "Transformer"
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.pad_token_id = config.pad_token_id
        self.pad_value = config.pad_value
        self.cell_emb_style = config.cell_emb_style
        # Removed self.use_chem_token = config.use_chem_token
        self.keep_first_n_tokens = config.keep_first_n_tokens
        self.return_gene_embeddings = config.return_gene_embeddings

        self.gene_encoder = GeneEncoder(config)
        self.flag_encoder = nn.Embedding(2, self.d_model)

        self.expression_encoder = ContinuousValueEncoder(config)

        # Removed ChemEncoder instantiation
        # if self.use_chem_token:
        #     self.chem_encoder = ChemEncoder(config)

        encoder_layer = StudentTXBlock(config)
        self.transformer_encoder = StudentTXEncoder(config, encoder_layer)

        self.expression_decoder = ExprDecoder(config)
        
        if config.mvc_arch_style is not None:
            self.mvc_decoder = MVCDecoder(config)
        else:
            self.mvc_decoder = None

        if config.init_config and config.init_config["name"] != "skip":
            self.apply(lambda module: _init_weights(module, config.n_layers, config.d_model, config.init_config))

    def _get_cell_emb_from_layer(
        self,
        layer_output: Tensor,
        weights: Tensor = None,
    ) -> Tensor:
        if self.cell_emb_style == "cls":
            cell_emb = layer_output[:, 0, :]
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim=1)
        elif self.cell_emb_style == "w-pool":
            if weights is None:
                raise ValueError("weights is required when cell_emb_style is w-pool")
            if weights.dim() != 2:
                raise ValueError("weights should be 2D")
            cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
            cell_emb = F.normalize(cell_emb, p=2, dim=1)
        return cell_emb

    def _transformer_generate(
        self,
        genes: Tensor,
        values: Tensor,
        gen_masks: Tensor,
        key_padding_mask: Tensor,
        # Removed drug_ids from arguments
        # drug_ids: Optional[Tensor] = None,
    ) -> Tensor:
        token_embs = self.gene_encoder(genes)
        token_values = self.expression_encoder(values)
        token_values = token_values.masked_fill(gen_masks.unsqueeze(-1), 0.0)
        
        flag_input = torch.tensor(1, device=token_embs.device)
        flag = self.flag_encoder(flag_input).reshape(1, 1, -1)
        flag_embs = (gen_masks.unsqueeze(-1).to(token_embs.dtype) * flag)
        
        total_embs = token_embs + token_values + flag_embs

        # Removed ChemEncoder usage
        # if self.use_chem_token and drug_ids is not None:
        #     drug_embs = self.chem_encoder(drug_ids)
        #     total_embs[:, 1, :] = drug_embs

        self.cur_gene_token_embs = token_embs

        output = self.transformer_encoder(
            total_embs=total_embs,
            key_padding_mask=key_padding_mask,
            gen_mask=gen_masks,
        )
        return output

    def forward(
        self,
        genes: Optional[Tensor] = None,
        values: Optional[Tensor] = None,
        gen_masks: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        # Removed drug_ids from arguments
        # drug_ids: Optional[Tensor] = None,
        return_dict: Optional[bool] = None,
        skip_decoders: Optional[bool] = None,
    ) -> Union[Tuple, Mapping[str, Tensor]]:

        if genes is None or values is None or gen_masks is None:
            raise ValueError("`genes`, `values`, and `gen_masks` must be provided to the forward pass.")

        if skip_decoders is None:
            skip_decoders = (not self.training)

        if attention_mask is None:
            key_padding_mask = ~genes.eq(self.config.pad_token_id)
        else:
            key_padding_mask = attention_mask.bool()

        transformer_output = self._transformer_generate(
            genes,
            values,
            gen_masks,
            key_padding_mask,
            # Removed drug_ids from _transformer_generate call
            # drug_ids=drug_ids,
        )

        output = {}
        if not skip_decoders:
            decoder_output = self.expression_decoder(transformer_output)
            full_preds = decoder_output["pred"]
            output["expr_preds"] = full_preds

        cell_emb = self._get_cell_emb_from_layer(transformer_output)
        output["cell_emb"] = cell_emb
        
        if self.return_gene_embeddings:
            output["gene_ids"] = genes
            output["gene_emb"] = transformer_output

        if not skip_decoders and self.mvc_decoder is not None:
            if not hasattr(self, 'cur_gene_token_embs'):
                log.warning("cur_gene_token_embs not found, recalculating for MVC decoder. Ensure _transformer_generate is called.")
                self.cur_gene_token_embs = self.gene_encoder(genes)

            mvc_output = self.mvc_decoder(
                cell_emb,
                self.cur_gene_token_embs,
            )
            output["mvc_output"] = mvc_output["pred"]

        if not return_dict:
            return tuple(v for v in output.values())

        return output

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path], *model_args, **kwargs):
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
