import anndata as ad
import numpy as np
import torch
from datasets import Dataset, Features, Value, Sequence
from typing import Optional, Dict, List, Tuple
from pathlib import Path

from .tokenization_student_tx import StudentTXTokenizer


def build_hf_dataset_from_h5ad(
    h5ad_path: str,
    tokenizer: StudentTXTokenizer,
    cls_token_id: int,
    pad_value: float,
    max_seq_len: Optional[int] = None,
    # Removed drug_to_id_mapping: Optional[Dict[str, int]] = None,
    # Removed drug_key: str = "drug_name",
    teacher_embedding_key: Optional[str] = None,
) -> Dataset:
    """
    Builds a HuggingFace Dataset from an H5AD file, preparing input for StudentTXModel
    and optionally including pre-generated teacher embeddings.

    Args:
        h5ad_path (str): Path to the H5AD file.
        tokenizer (StudentTXTokenizer): The tokenizer to convert gene names to IDs.
        cls_token_id (int): The ID of the <cls> token.
        pad_value (float): The expression value for padding.
        max_seq_len (Optional[int]): Maximum sequence length. If provided, sequences will be
                                      truncated/padded to this length.
        teacher_embedding_key (Optional[str]): If provided, loads teacher embeddings from
                                                `adata.obsm[teacher_embedding_key]` and includes them
                                                in the dataset.

    Returns:
        Dataset: A HuggingFace Dataset with 'genes', 'expressions', 'gen_masks',
                 and optionally 'teacher_embeddings'.
    """
    if not Path(h5ad_path).exists():
        raise FileNotFoundError(f"H5AD file not found at: {h5ad_path}")

    adata = ad.read_h5ad(h5ad_path, backed='r')

    gene_names = adata.var_names.tolist()

    all_genes_list = []
    all_expressions_list = []
    all_gen_masks_list = []
    # Removed all_drug_ids_list = [] if drug_to_id_mapping else None
    all_teacher_embeddings_list = [] if teacher_embedding_key else None

    for i in range(adata.n_obs):
        cell_data = adata[i, :]
        if hasattr(cell_data.X, 'toarray'):
            expressions = cell_data.X.toarray().flatten()
        else:
            expressions = cell_data.X.flatten()

        current_gene_names = gene_names
        current_expressions = expressions

        gene_ids = tokenizer.convert_tokens_to_ids(current_gene_names)
        gene_ids = torch.tensor(gene_ids, dtype=torch.long)
        expressions_tensor = torch.tensor(current_expressions, dtype=torch.float)

        gene_ids = torch.cat([torch.tensor([cls_token_id], dtype=torch.long), gene_ids])
        expressions_tensor = torch.cat([torch.tensor([pad_value], dtype=torch.float), expressions_tensor])
        gen_mask = torch.zeros_like(gene_ids, dtype=torch.bool)
        
        # Removed drug_ids handling
        # if drug_to_id_mapping is not None and drug_key in adata.obs:
        #     drug_name = str(adata.obs[drug_key].iloc[i])
        #     drug_id = drug_to_id_mapping.get(drug_name, tokenizer.unk_token_id)
        #     all_drug_ids_list.append(drug_id)

        # Load teacher embedding if key is provided
        if teacher_embedding_key and teacher_embedding_key in adata.obsm:
            if teacher_embedding_key not in adata.obsm:
                raise KeyError(f"Teacher embedding key '{teacher_embedding_key}' not found in adata.obsm.")
            teacher_emb = adata.obsm[teacher_embedding_key][i]
            all_teacher_embeddings_list.append(teacher_emb.astype(np.float32).tolist())

        if max_seq_len is not None:
            if len(gene_ids) > max_seq_len:
                gene_ids = gene_ids[:max_seq_len]
                expressions_tensor = expressions_tensor[:max_seq_len]
                gen_mask = gen_mask[:max_seq_len]
            elif len(gene_ids) < max_seq_len:
                pad_len = max_seq_len - len(gene_ids)
                gene_ids = torch.cat([gene_ids, torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)])
                expressions_tensor = torch.cat([expressions_tensor, torch.full((pad_len,), pad_value, dtype=torch.float)])
                gen_mask = torch.cat([gen_mask, torch.full((pad_len,), False, dtype=torch.bool)])

        all_genes_list.append(gene_ids.tolist())
        all_expressions_list.append(expressions_tensor.tolist())
        all_gen_masks_list.append(gen_mask.tolist())

    features_dict = {
        "genes": Sequence(feature=Value(dtype="int64")),
        "expressions": Sequence(feature=Value(dtype="float32")),
        "gen_masks": Sequence(feature=Value(dtype="bool")),
    }
    # Removed drug_ids from features_dict
    # if all_drug_ids_list is not None:
    #     features_dict["drug_ids"] = Value(dtype="int64")
    if all_teacher_embeddings_list is not None:
        embedding_dim = len(all_teacher_embeddings_list[0]) if all_teacher_embeddings_list else None
        if embedding_dim:
            features_dict["teacher_embeddings"] = Sequence(feature=Value(dtype="float32"), length=embedding_dim)


    hf_dataset_dict = {
        "genes": all_genes_list,
        "expressions": all_expressions_list,
        "gen_masks": all_gen_masks_list,
    }
    # Removed drug_ids from hf_dataset_dict
    # if all_drug_ids_list is not None:
    #     hf_dataset_dict["drug_ids"] = all_drug_ids_list
    if all_teacher_embeddings_list is not None:
        hf_dataset_dict["teacher_embeddings"] = all_teacher_embeddings_list

    return Dataset.from_dict(hf_dataset_dict, features=Features(features_dict))
