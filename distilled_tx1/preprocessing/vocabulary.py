"""
Gene Vocabulary Module

Handles gene vocabulary management for Tahoe X1-style tokenization.
Adapted from: https://github.com/tahoebio/tahoe-x1/tree/main/tahoe_x1/tokenizer
"""

import json
from typing import Dict, List, Optional, Set, Union
from pathlib import Path
import numpy as np
from anndata import AnnData


class GeneVocabulary:
    """
    Gene vocabulary for single-cell RNA-seq data.
    
    Maps between gene identifiers (ENSEMBL IDs, gene symbols) and token indices.
    Includes special tokens for padding, masking, and classification.
    """
    
    # Special token definitions
    PAD_TOKEN = "<pad>"
    MASK_TOKEN = "<mask>"
    CLS_TOKEN = "<cls>"
    SEP_TOKEN = "<sep>"
    UNK_TOKEN = "<unk>"
    DRUG_TOKEN = "<drug>"  # For perturbation experiments
    
    SPECIAL_TOKENS = [PAD_TOKEN, MASK_TOKEN, CLS_TOKEN, SEP_TOKEN, UNK_TOKEN, DRUG_TOKEN]
    
    def __init__(
        self,
        gene_to_idx: Optional[Dict[str, int]] = None,
        idx_to_gene: Optional[Dict[int, str]] = None
    ):
        """
        Initialize gene vocabulary.
        
        Args:
            gene_to_idx: Mapping from gene identifier to token index
            idx_to_gene: Mapping from token index to gene identifier
        """
        self.gene_to_idx = gene_to_idx or {}
        self.idx_to_gene = idx_to_gene or {}
        
        # Add special tokens if not present
        self._add_special_tokens()
        
    def _add_special_tokens(self):
        """Add special tokens to vocabulary if not present"""
        for token in self.SPECIAL_TOKENS:
            if token not in self.gene_to_idx:
                idx = len(self.gene_to_idx)
                self.gene_to_idx[token] = idx
                self.idx_to_gene[idx] = token
    
    @classmethod
    def from_json(cls, vocab_path: Union[str, Path]) -> "GeneVocabulary":
        """
        Load vocabulary from JSON file.
        
        Args:
            vocab_path: Path to vocabulary JSON file
            
        Returns:
            GeneVocabulary instance
        """
        with open(vocab_path, 'r') as f:
            gene_to_idx = json.load(f)
        
        idx_to_gene = {int(idx): gene for gene, idx in gene_to_idx.items()}
        
        return cls(gene_to_idx=gene_to_idx, idx_to_gene=idx_to_gene)
    
    @classmethod
    def from_adata(
        cls,
        adata: AnnData,
        gene_id_key: Optional[str] = "ensembl_id",
        min_cells: int = 0,
        min_counts: int = 0
    ) -> "GeneVocabulary":
        """
        Build vocabulary from AnnData object.
        
        Args:
            adata: AnnData object
            gene_id_key: Key in adata.var containing gene IDs (if None, uses var_names)
            min_cells: Minimum number of cells a gene must be expressed in
            min_counts: Minimum total counts for a gene
            
        Returns:
            GeneVocabulary instance
        """
        import scanpy as sc
        
        # Filter genes
        if min_cells > 0 or min_counts > 0:
            sc.pp.filter_genes(adata, min_cells=min_cells, min_counts=min_counts)
        
        # Get gene IDs
        if gene_id_key and gene_id_key in adata.var.columns:
            genes = adata.var[gene_id_key].tolist()
        else:
            genes = adata.var_names.tolist()
        
        # Create mappings (reserve indices for special tokens)
        n_special = len(cls.SPECIAL_TOKENS)
        gene_to_idx = {gene: idx + n_special for idx, gene in enumerate(genes)}
        idx_to_gene = {idx + n_special: gene for idx, gene in enumerate(genes)}
        
        return cls(gene_to_idx=gene_to_idx, idx_to_gene=idx_to_gene)
    
    @classmethod
    def from_tahoe_checkpoint(cls, checkpoint_dir: Union[str, Path]) -> "GeneVocabulary":
        """
        Load vocabulary from Tahoe X1 checkpoint.
        
        Args:
            checkpoint_dir: Path to Tahoe checkpoint directory
            
        Returns:
            GeneVocabulary instance
        """
        vocab_path = Path(checkpoint_dir) / "vocab.json"
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"Vocabulary file not found at {vocab_path}. "
                "Expected vocab.json in checkpoint directory."
            )
        
        return cls.from_json(vocab_path)
    
    def save(self, vocab_path: Union[str, Path]):
        """
        Save vocabulary to JSON file.
        
        Args:
            vocab_path: Output path for vocabulary file
        """
        with open(vocab_path, 'w') as f:
            json.dump(self.gene_to_idx, f, indent=2)
    
    def encode(self, gene_ids: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Convert gene IDs to token indices.
        
        Args:
            gene_ids: Gene ID or list of gene IDs
            
        Returns:
            Token index or list of token indices
        """
        if isinstance(gene_ids, str):
            return self.gene_to_idx.get(gene_ids, self.unk_token_id)
        
        return [self.gene_to_idx.get(g, self.unk_token_id) for g in gene_ids]
    
    def decode(self, token_ids: Union[int, List[int]]) -> Union[str, List[str]]:
        """
        Convert token indices to gene IDs.
        
        Args:
            token_ids: Token index or list of token indices
            
        Returns:
            Gene ID or list of gene IDs
        """
        if isinstance(token_ids, int):
            return self.idx_to_gene.get(token_ids, self.UNK_TOKEN)
        
        return [self.idx_to_gene.get(idx, self.UNK_TOKEN) for idx in token_ids]
    
    def filter_genes(self, gene_ids: List[str]) -> List[str]:
        """
        Filter gene list to only include genes in vocabulary.
        
        Args:
            gene_ids: List of gene IDs
            
        Returns:
            Filtered list of gene IDs
        """
        return [g for g in gene_ids if g in self.gene_to_idx]
    
    def get_coverage(self, gene_ids: List[str]) -> float:
        """
        Calculate vocabulary coverage for a list of genes.
        
        Args:
            gene_ids: List of gene IDs
            
        Returns:
            Fraction of genes in vocabulary (0-1)
        """
        if not gene_ids:
            return 0.0
        
        n_in_vocab = sum(1 for g in gene_ids if g in self.gene_to_idx)
        return n_in_vocab / len(gene_ids)
    
    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens"""
        return len(self.gene_to_idx)
    
    @property
    def n_genes(self) -> int:
        """Number of genes (excluding special tokens)"""
        return len(self.gene_to_idx) - len(self.SPECIAL_TOKENS)
    
    @property
    def pad_token_id(self) -> int:
        """Index of padding token"""
        return self.gene_to_idx[self.PAD_TOKEN]
    
    @property
    def mask_token_id(self) -> int:
        """Index of mask token"""
        return self.gene_to_idx[self.MASK_TOKEN]
    
    @property
    def cls_token_id(self) -> int:
        """Index of CLS token"""
        return self.gene_to_idx[self.CLS_TOKEN]
    
    @property
    def sep_token_id(self) -> int:
        """Index of SEP token"""
        return self.gene_to_idx[self.SEP_TOKEN]
    
    @property
    def unk_token_id(self) -> int:
        """Index of unknown token"""
        return self.gene_to_idx[self.UNK_TOKEN]
    
    @property
    def drug_token_id(self) -> int:
        """Index of drug token"""
        return self.gene_to_idx[self.DRUG_TOKEN]
    
    def __len__(self) -> int:
        """Return vocabulary size"""
        return self.vocab_size
    
    def __contains__(self, gene_id: str) -> bool:
        """Check if gene is in vocabulary"""
        return gene_id in self.gene_to_idx
    
    def __repr__(self) -> str:
        return f"GeneVocabulary(vocab_size={self.vocab_size}, n_genes={self.n_genes})"


def download_tahoe_vocab(model_size: str = "70m", cache_dir: Optional[Path] = None) -> Path:
    """
    Download Tahoe X1 vocabulary from HuggingFace or S3.
    
    Args:
        model_size: Model size ("70m", "1b", or "3b")
        cache_dir: Directory to cache vocabulary file
        
    Returns:
        Path to downloaded vocabulary file
    """
    from huggingface_hub import hf_hub_download
    
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "distilled-tx1"
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    vocab_path = cache_dir / f"tahoe_vocab_{model_size}.json"
    
    if vocab_path.exists():
        return vocab_path
    
    # Download from HuggingFace
    try:
        downloaded_path = hf_hub_download(
            repo_id="tahoebio/Tahoe-x1",
            filename=f"{model_size}/vocab.json",
            cache_dir=cache_dir
        )
        
        # Copy to standard location
        import shutil
        shutil.copy(downloaded_path, vocab_path)
        
        return vocab_path
    
    except Exception as e:
        raise RuntimeError(
            f"Failed to download Tahoe vocabulary: {e}\n"
            "Please download manually from https://huggingface.co/tahoebio/Tahoe-x1"
        )
