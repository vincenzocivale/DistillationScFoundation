from transformers import PretrainedConfig
from typing import Optional, Dict, Any

class StudentTXConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`StudentTXModel`]. It is used to instantiate a Student-TX model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration
    to that of the `TXModel` architecture.

    Args:
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the Student-TX model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`StudentTXModel`].
        n_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        d_model (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and in the Transformer encoder.
        expansion_ratio (`int`, *optional*, defaults to 4):
            Expansion ratio for the feed-forward network in the Transformer encoder.
        norm_scheme (`str`, *optional*, defaults to "pre"):
            Normalization scheme to use in the Transformer blocks ("pre" or "post").
        transformer_activation (`str`, *optional*, defaults to "gelu"):
            Activation function for the Transformer encoder layers.
        cell_emb_style (`str`, *optional*, defaults to "cls"):
            Style for generating cell embeddings ("cls", "avg-pool", "w-pool").
        pad_token_id (`int`, *optional*, defaults to 0):
            The ID of the padding token.
        pad_value (`int`, *optional*, defaults to 0):
            The value used for padding expression values.
        n_input_bins (`int`, *optional*, defaults to 51):
            Number of bins for expression value binning.
        target_sum (`int`, *optional*, defaults to 10000):
            The target sum for log transformation of expression values.
        use_flash_attention (`bool`, *optional*, defaults to `False`):
            Whether to use Flash Attention. If `True`, `attn_impl` will be set to "flash".
        attn_config (`Dict[str, Any]`, *optional*, defaults to `{"attn_type": "grouped_query_attention", "attn_pdrop": 0.0, "attn_impl": "torch", "use_attn_mask": True, "qk_ln": False, "qk_gn": False, "clip_qkv": None, "softmax_scale": None}`):
            Configuration for the attention mechanism. Overrides `use_flash_attention` if `attn_impl` is specified.
        norm_config (`Dict[str, Any]`, *optional*, defaults to `{"norm_type": "low_precision_layernorm", "eps": 1e-5}`):
            Configuration for normalization layers.
        init_config (`Dict[str, Any]`, *optional*, defaults to `{"name": "kaiming_normal_", "fan_mode": "fan_in", "init_nonlinearity": "relu", "init_div_is_residual": True, "emb_init_std": None, "emb_init_uniform_lim": None, "init_std": None, "init_gain": 0.0}`):
            Configuration for parameter initialization.
        gene_encoder_config (`Dict[str, Any]`, *optional*, defaults to `{"use_norm": False}`):
            Configuration for the gene encoder.
        keep_first_n_tokens (`int`, *optional*, defaults to 1):
            Number of tokens to keep unchanged from sampling.
        return_gene_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model should return gene embeddings.
        use_glu (`bool`, *optional*, defaults to `False`):
            Whether to use Gated Linear Units in the FFN.
        # GeneEncoder specific
        gene_encoder_padding_idx (`Optional[int]`, *optional*, defaults to `None`):
            Padding index for the gene encoder.
        gene_encoder_use_norm (`bool`, *optional*, defaults to `False`):
            Whether to use normalization in the gene encoder.
        # ContinuousValueEncoder specific
        cv_encoder_dropout (`float`, *optional*, defaults to 0.1):
            Dropout rate for the continuous value encoder.
        cv_encoder_max_value (`int`, *optional*, defaults to 512):
            Maximum value for clipping in the continuous value encoder.
        cv_encoder_activation (`str`, *optional*, defaults to "relu"):
            Activation function for the continuous value encoder.
        cv_encoder_use_norm (`bool`, *optional*, defaults to `False`):
            Whether to use normalization in the continuous value encoder.
        # ExprDecoder specific
        expr_decoder_n_outputs (`int`, *optional*, defaults to 1):
            Number of outputs for the expression decoder.
        expr_decoder_n_layers (`int`, *optional*, defaults to 2):
            Number of layers in the expression decoder.
        expr_decoder_activation (`str`, *optional*, defaults to "leaky_relu"):
            Activation function for the expression decoder.
        # MVCDecoder specific
        mvc_arch_style (`Optional[str]`, *optional*, defaults to `None`):
            Architecture style for the MVC decoder ("inner product").
        mvc_query_activation (`Optional[str]`, *optional*, defaults to `None`):
            Activation function for query vectors in MVC decoder.
        mvc_scaled_dot_product (`bool`, *optional*, defaults to `False`):
            Whether to use scaled dot product in MVC decoder.
    """
    model_type = "student-tx"

    def __init__(
        self,
        vocab_size: int = 50257,
        n_layers: int = 12,
        n_heads: int = 8,
        d_model: int = 768,
        expansion_ratio: int = 4,
        norm_scheme: str = "pre",
        transformer_activation: str = "gelu",
        cell_emb_style: str = "cls",
        pad_token_id: int = 0,
        pad_value: int = 0,
        n_input_bins: int = 51,
        target_sum: int = 10000,
        use_flash_attention: bool = False,
        attn_config: Optional[Dict[str, Any]] = None,
        norm_config: Optional[Dict[str, Any]] = None,
        init_config: Optional[Dict[str, Any]] = None,
        gene_encoder_config: Optional[Dict[str, Any]] = None,
        keep_first_n_tokens: int = 1,
        return_gene_embeddings: bool = False,
        use_glu: bool = False,
        # GeneEncoder specific
        gene_encoder_padding_idx: Optional[int] = None,
        gene_encoder_use_norm: bool = False,
        # ContinuousValueEncoder specific
        cv_encoder_dropout: float = 0.1,
        cv_encoder_max_value: int = 512,
        cv_encoder_activation: str = "relu",
        cv_encoder_use_norm: bool = False,
        # ExprDecoder specific
        expr_decoder_n_outputs: int = 1,
        expr_decoder_n_layers: int = 2,
        expr_decoder_activation: str = "leaky_relu",
        # MVCDecoder specific
        mvc_arch_style: Optional[str] = None,
        mvc_query_activation: Optional[str] = None,
        mvc_scaled_dot_product: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.expansion_ratio = expansion_ratio
        self.norm_scheme = norm_scheme
        self.transformer_activation = transformer_activation
        self.cell_emb_style = cell_emb_style
        self.pad_token_id = pad_token_id
        self.pad_value = pad_value
        self.n_input_bins = n_input_bins
        self.target_sum = target_sum
        self.keep_first_n_tokens = keep_first_n_tokens
        self.return_gene_embeddings = return_gene_embeddings
        self.use_glu = use_glu

        if attn_config is None:
            self.attn_config = {
                "attn_type": "grouped_query_attention",
                "attn_pdrop": 0.0,
                "attn_impl": "torch",  # Default to torch
                "use_attn_mask": True,
                "qk_ln": False,
                "qk_gn": False,
                "clip_qkv": None,
                "softmax_scale": None,
            }
        else:
            self.attn_config = attn_config.copy() # Make a copy to avoid modifying the original dict

        # Override attn_impl if use_flash_attention is True, unless explicitly set in attn_config
        if use_flash_attention and self.attn_config.get("attn_impl", "torch") == "torch":
            self.attn_config["attn_impl"] = "flash"
            self.attn_config["use_attn_mask"] = False # Flash attention usually doesn't use explicit masks

        if norm_config is None:
            self.norm_config = {
                "norm_type": "low_precision_layernorm",
                "eps": 1e-5,
            }
        else:
            self.norm_config = norm_config

        if init_config is None:
            self.init_config = {
                "name": "kaiming_normal_",
                "fan_mode": "fan_in",
                "init_nonlinearity": "relu",
                "init_div_is_residual": True,
                "emb_init_std": None,
                "emb_init_uniform_lim": None,
                "init_std": None,
                "init_gain": 0.0,
            }
        else:
            self.init_config = init_config

        if gene_encoder_config is None:
            self.gene_encoder_config = {
                "use_norm": gene_encoder_use_norm,
                "padding_idx": gene_encoder_padding_idx,
            }
        else:
            self.gene_encoder_config = gene_encoder_config
            if "padding_idx" not in self.gene_encoder_config:
                self.gene_encoder_config["padding_idx"] = gene_encoder_padding_idx
            if "use_norm" not in self.gene_encoder_config:
                self.gene_encoder_config["use_norm"] = gene_encoder_use_norm


        self.cv_encoder_dropout = cv_encoder_dropout
        self.cv_encoder_max_value = cv_encoder_max_value
        self.cv_encoder_activation = cv_encoder_activation
        self.cv_encoder_use_norm = cv_encoder_use_norm

        # Removed ChemEncoder specific parameters
        # self.chem_encoder_padding_idx = chem_encoder_padding_idx
        # self.chem_encoder_activation = chem_encoder_activation
        # self.chem_encoder_freeze = chem_encoder_freeze
        # self.drug_fps_path = drug_fps_path
        # self.num_drugs = num_drugs
        # self.fp_dim = fp_dim

        self.expr_decoder_n_outputs = expr_decoder_n_outputs
        self.expr_decoder_n_layers = expr_decoder_n_layers
        self.expr_decoder_activation = expr_decoder_activation

        self.mvc_arch_style = mvc_arch_style
        self.mvc_query_activation = mvc_query_activation
        self.mvc_scaled_dot_product = mvc_scaled_dot_product

        super().__init__(pad_token_id=pad_token_id, **kwargs)