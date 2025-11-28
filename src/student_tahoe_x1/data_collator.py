import json
from typing import Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from transformers import DefaultDataCollator

# Assuming these are available and work as in original project
# from composer.utils import dist 
# from tahoe_x1.utils import download_file_from_s3_url 

from .configuration_student_tx import StudentTXConfig
from .tokenization_student_tx import StudentTXTokenizer


@torch.no_grad()
def log_transform(
    row: Union[np.ndarray, torch.Tensor],
    target_sum: int,
    eps: float = 1e-9,
) -> Union[np.ndarray, torch.Tensor]:
    dtype = row.dtype
    is_tensor = isinstance(row, torch.Tensor)
    if not is_tensor:
        row = torch.as_tensor(row)
    row = (row / (row.sum(axis=-1, keepdims=True) + eps)) * target_sum
    row = torch.log1p(row)
    if not is_tensor:
        return row.numpy().astype(dtype)
    return row


@torch.no_grad()
def binning(
    row: Union[np.ndarray, torch.Tensor],
    n_bins: int,
    right: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    dtype = row.dtype
    return_np = not (isinstance(row, torch.Tensor))
    if not isinstance(row, torch.Tensor):
        row = torch.as_tensor(row)
    GRADES = torch.linspace(0, 1, n_bins - 1, dtype=torch.float32, requires_grad=False)
    if row.min() <= 0:
        non_zero_ids = row.nonzero()
        non_zero_row = row[non_zero_ids]
        bins = torch.quantile(non_zero_row, GRADES)
        non_zero_digits = torch.bucketize(non_zero_row, bins, right=right)
        binned_row = torch.zeros_like(row, dtype=non_zero_digits.dtype)
        binned_row[non_zero_ids] = non_zero_digits
    else:
        bins = torch.quantile(row, GRADES)
        binned_row = torch.bucketize(row, bins, right=right)
    if return_np:
        binned_row = binned_row.astype(dtype)
    if not (right):
        binned_row = binned_row + 1
    return binned_row


class StudentTXDataCollator(DefaultDataCollator):
    """
    Data collator for StudentTXModel. It pads the sequences to the maximum length
    in the batch, and can mask gene expression values for MLM.
    """

    def __init__(
        self,
        config: StudentTXConfig,
        tokenizer: StudentTXTokenizer,
        do_mlm: bool = True,
        mlm_probability: float = 0.15,
        mask_value: int = -1,
        do_binning: bool = True,
        log_transform_expr: bool = False, # Renamed to avoid conflict with function
        target_sum: int = 10000,
        unexp_padding: bool = False, # Padding with unexpressed genes
        sampling: bool = True, # Sampling instead of truncation if length > max_length
        right_binning: bool = False,
        return_tensors: str = "pt",
    ):
        super().__init__(return_tensors=return_tensors)
        self.config = config
        self.tokenizer = tokenizer
        self.do_mlm = do_mlm
        self.mlm_probability = mlm_probability
        self.mask_value = mask_value
        self.do_binning = do_binning
        self.log_transform_expr = log_transform_expr
        self.target_sum = target_sum
        self.unexp_padding = unexp_padding
        self.sampling = sampling
        self.right_binning = right_binning

        # Parameters from config for padding and special tokens
        self.pad_token_id = self.tokenizer.pad_token_id
        self.pad_value = self.config.pad_value # Expression pad value
        self.keep_first_n_tokens = self.config.keep_first_n_tokens
        self.n_input_bins = self.config.n_input_bins # For binning

        # Pre-filter non_special gene_ids for unexp_padding
        gene_to_id = self.tokenizer.get_vocab()
        self.non_special_gene_ids = torch.tensor(
            [
                gene_id
                for gene_name, gene_id in gene_to_id.items()
                if not gene_name.startswith("<")
            ],
            dtype=torch.long
        )

        if self.do_binning and self.log_transform_expr:
            raise ValueError(
                "Only one of `do_binning` and `log_transform_expr` can be True.",
            )
        if self.unexp_padding and not (self.config.max_position_embeddings is not None):
             raise ValueError("`max_position_embeddings` should be set in config if `unexp_padding` is True.")
        
        # Ensure max_length is available if padding/sampling is active
        if self.sampling or self.unexp_padding:
             if self.config.max_position_embeddings is None:
                 raise ValueError("`config.max_position_embeddings` must be set if sampling or unexp_padding is True.")

    def __call__(
        self,
        examples: List[Dict[str, Union[List[int], List[float], List[bool], int, List[float]]]], # Added List[float] for teacher_embeddings
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            examples (:obj:`List[Dict[str, Union[List[int], List[float], List[bool], int, List[float]]]]`):
                A list of data dicts, each corresponding to one cell.
                Each dict is expected to have 'genes', 'expressions', 'gen_masks', and optionally 'drug_ids'
                as prepared by `build_hf_dataset_from_h5ad`, and optionally 'teacher_embeddings'.
        Returns:
            :obj:`Dict[str, torch.Tensor]`: a dict of tensors, ready for the model.
        """
        # Convert lists to tensors for easier processing
        teacher_embeddings_present = "teacher_embeddings" in examples[0] if examples else False # Check if examples list is not empty
        all_teacher_embeddings_list = []

        for example in examples:
            if isinstance(example["genes"], list):
                example["genes"] = torch.as_tensor(example["genes"], dtype=torch.long)
            if isinstance(example["expressions"], list):
                example["expressions"] = torch.as_tensor(example["expressions"], dtype=torch.float)
            if isinstance(example["gen_masks"], list):
                example["gen_masks"] = torch.as_tensor(example["gen_masks"], dtype=torch.bool)
            # Removed self.config.use_chem_token check and drug_ids handling
            # if self.config.use_chem_token and "drug_ids" in example:
            #     if not isinstance(example["drug_ids"], torch.Tensor):
            #         example["drug_ids"] = torch.as_tensor(example["drug_ids"], dtype=torch.long)
            # Handle teacher_embeddings
            if teacher_embeddings_present:
                all_teacher_embeddings_list.append(torch.as_tensor(example["teacher_embeddings"], dtype=torch.float))


        # Determine the maximum sequence length in the current batch if max_position_embeddings is not set
        # If max_position_embeddings is set, use it. Otherwise, use the max length in batch.
        if self.config.max_position_embeddings is None:
            max_batch_len = max(len(example["genes"]) for example in examples)
        else:
            max_batch_len = self.config.max_position_embeddings
        
        # Pad all examples to max_batch_len
        padded_genes = []
        masked_exprs = []
        expr_targets = []
        expr_raws = []
        gen_masks_out = []
        # Removed drug_ids_out initialization
        # drug_ids_out = [] if self.config.use_chem_token else None

        for example in examples:
            genes = example["genes"]
            expressions = example["expressions"]
            original_gen_mask = example["gen_masks"] # This is the initial generative mask (usually all False except special tokens)

            # --- Apply binning/log_transform ---
            raw_expressions = expressions.detach().clone()
            if self.do_binning:
                expressions[self.keep_first_n_tokens:] = binning(
                    row=expressions[self.keep_first_n_tokens:],
                    n_bins=self.n_input_bins,
                    right=self.right_binning,
                )
            elif self.log_transform_expr:
                expressions[self.keep_first_n_tokens:] = log_transform(
                    row=expressions[self.keep_first_n_tokens:],
                    target_sum=self.target_sum,
                )

            # --- Handle sequence length adjustments (sampling/truncation/padding) ---
            genes, expressions, raw_expressions, gen_mask_final = self._sample_or_truncate_plus_pad_all(
                genes,
                expressions,
                raw_expressions,
                original_gen_mask,
                max_length=max_batch_len,
            )
            
            # --- MLM Masking ---
            expr_target = expressions.detach().clone() # Target for MLM
            if self.do_mlm:
                # Create random mask only for the parts of the sequence that are not special tokens or padding
                mlm_mask = self._create_random_mlm_mask(expressions, gen_mask_final)
                masked_expr = expressions.masked_fill(mlm_mask, self.mask_value)
                # Combine original generative mask with mlm_mask for the model's gen_mask input
                gen_mask_final = gen_mask_final | mlm_mask
            else:
                masked_expr = expressions


            padded_genes.append(genes)
            masked_exprs.append(masked_expr)
            expr_targets.append(expr_target)
            expr_raws.append(raw_expressions)
            gen_masks_out.append(gen_mask_final) # Final generative mask including MLM

            # Removed drug_ids_out appending
            # if self.config.use_chem_token and drug_ids_out is not None:
            #     drug_ids_out.append(example.get("drug_ids", self.tokenizer.unk_token_id)) # Use unk_token_id if not found

        # Stack into batch tensors
        data_dict = {
            "genes": torch.stack(padded_genes, dim=0),
            "values": torch.stack(masked_exprs, dim=0), # Renamed to 'values' for StudentTXModel.forward
            "gen_masks": torch.stack(gen_masks_out, dim=0),
            "expr_target": torch.stack(expr_targets, dim=0), # For loss calculation
            "expr_raw": torch.stack(expr_raws, dim=0), # For potential metrics
        }
        # Removed drug_ids from data_dict
        # if self.config.use_chem_token and drug_ids_out is not None:
        #     data_dict["drug_ids"] = torch.stack(drug_ids_out, dim=0)
        # Add teacher_embeddings to data_dict
        if teacher_embeddings_present:
            data_dict["teacher_embeddings"] = torch.stack(all_teacher_embeddings_list, dim=0)


        # Add attention_mask (key_padding_mask)
        data_dict["attention_mask"] = ~data_dict["genes"].eq(self.pad_token_id)

        return data_dict

    def _create_random_mlm_mask(
        self,
        expressions: torch.Tensor,
        gen_mask_final: torch.Tensor, # This is the mask considering special tokens and padding
    ) -> torch.Tensor:
        """Generate a random mask for expressions based on mlm probability, excluding special and padded tokens."""
        device = expressions.device
        
        # Determine valid tokens for MLM masking (not special, not padded)
        # Assuming original_gen_mask indicates non-generative (perceptual) tokens initially
        # Tokens that are NOT pad and are NOT in the initial gen_mask (e.g. CLS)
        valid_for_mlm = ~expressions.eq(self.pad_value) & ~gen_mask_final

        # Exclude keep_first_n_tokens from MLM masking
        if self.keep_first_n_tokens > 0:
            valid_for_mlm[:self.keep_first_n_tokens] = False

        num_valid_tokens = valid_for_mlm.sum().item()
        num_to_mask = int(num_valid_tokens * self.mlm_probability)

        mlm_mask = torch.zeros_like(expressions, dtype=torch.bool, device=device)

        if num_to_mask > 0 and num_valid_tokens > 0:
            # Get indices of valid tokens
            valid_indices = torch.nonzero(valid_for_mlm, as_tuple=True)[0]
            # Randomly select num_to_mask indices from valid_indices
            indices_to_mask = valid_indices[torch.randperm(num_valid_tokens, device=device)[:num_to_mask]]
            mlm_mask[indices_to_mask] = True
            
        return mlm_mask


    def _sample_or_truncate_plus_pad_all(
        self,
        genes: torch.Tensor,
        expressions: torch.Tensor,
        raw_expressions: torch.Tensor,
        gen_mask: torch.Tensor,
        max_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        arrays = (genes, expressions, raw_expressions, gen_mask)
        assert all(array.shape[0] == arrays[0].shape[0] for array in arrays), "All arrays must have the same length."

        current_len = genes.shape[0]

        if current_len == max_length:
            return genes, expressions, raw_expressions, gen_mask
        
        if current_len > max_length:  # sample or truncate
            if self.sampling:
                return self._sample_all(*arrays, max_length=max_length)
            else:
                return tuple(array[:max_length] for array in arrays)
        else: # current_len < max_length: pad
            if self.unexp_padding:
                return self._pad_unexp_genes_all(*arrays, max_length=max_length)
            else:
                return self._pad_all(*arrays, max_length=max_length)

    def _sample_all(
        self,
        genes: torch.Tensor,
        expressions: torch.Tensor,
        raw_expressions: torch.Tensor,
        gen_mask: torch.Tensor,
        max_length: int,
    ) -> Tuple[torch.Tensor, ...]:
        device = genes.device
        
        if self.keep_first_n_tokens == 0:
            indices = torch.randperm(len(genes), device=device)[:max_length]
        else:
            _n = self.keep_first_n_tokens
            # Ensure we don't try to sample more than available for the non-kept part
            num_to_sample_from_rest = min(max_length - _n, len(genes) - _n)
            indices = torch.randperm(len(genes) - _n, device=device)[:num_to_sample_from_rest]
            indices = torch.cat([torch.arange(_n, device=device), indices + _n], dim=0)
            
            # If after combining, length is still less than max_length (due to num_to_sample_from_rest),
            # this would be an issue. But sampling implies current_len > max_length.
            # So the combined indices length should be max_length.
            # However, `_sample_or_truncate_plus_pad_all` should ensure `max_length` is respected.
            # If for some reason `indices` is shorter than `max_length`, we need to pad.
            if len(indices) < max_length:
                # This case should ideally not happen for sampling when current_len > max_length
                # but adding a safeguard.
                num_to_pad = max_length - len(indices)
                pad_indices = torch.full((num_to_pad,), len(genes)-1, dtype=torch.long, device=device) # Pad with last index
                indices = torch.cat([indices, pad_indices])

        return genes[indices], expressions[indices], raw_expressions[indices], gen_mask[indices]

    def _pad_all(
        self,
        genes: torch.Tensor,
        expressions: torch.Tensor,
        raw_expressions: torch.Tensor,
        gen_mask: torch.Tensor,
        max_length: int,
    ) -> Tuple[torch.Tensor, ...]:
        device = genes.device
        num_to_pad = max_length - genes.shape[0]

        padded_genes = torch.cat(
            [genes, torch.full((num_to_pad,), self.pad_token_id, dtype=torch.long, device=device)],
        )
        padded_expressions = torch.cat(
            [expressions, torch.full((num_to_pad,), self.pad_value, dtype=torch.float, device=device)],
        )
        padded_raw_expressions = torch.cat(
            [raw_expressions, torch.full((num_to_pad,), self.pad_value, dtype=torch.float, device=device)],
        )
        padded_gen_mask = torch.cat(
            [gen_mask, torch.full((num_to_pad,), False, dtype=torch.bool, device=device)],
        )
        return padded_genes, padded_expressions, padded_raw_expressions, padded_gen_mask


    def _pad_unexp_genes_all(
        self,
        genes: torch.Tensor,
        expressions: torch.Tensor,
        raw_expressions: torch.Tensor,
        gen_mask: torch.Tensor,
        max_length: int,
    ) -> Tuple[torch.Tensor, ...]:
        device = genes.device
        num_to_pad = max_length - genes.shape[0]

        # get list of all valid gene ids
        non_special_gene_ids = self.non_special_gene_ids.to(device)

        # filter out the expressed gene ids
        mask = ~torch.isin(non_special_gene_ids, genes)
        unexp_genes = non_special_gene_ids[mask]

        # randomly sample from unexpressed gene ids
        # Ensure we don't sample more than available unexpressed genes
        num_unexp_available = unexp_genes.shape[0]
        if num_unexp_available == 0: # Fallback if no unexpressed genes are available
            random_unexp_genes = torch.full((num_to_pad,), self.pad_token_id, dtype=torch.long, device=device)
        else:
            idx = torch.randperm(num_unexp_available, device=device)[:num_to_pad]
            random_unexp_genes = unexp_genes[idx]
            
            # If not enough unexpressed genes, fill the rest with pad_token_id
            if len(random_unexp_genes) < num_to_pad:
                fill_len = num_to_pad - len(random_unexp_genes)
                random_unexp_genes = torch.cat([random_unexp_genes, torch.full((fill_len,), self.pad_token_id, dtype=torch.long, device=device)])


        padded_genes = torch.cat(
            [genes, random_unexp_genes],
        )
        padded_expressions = torch.cat(
            [expressions, torch.zeros(num_to_pad, dtype=torch.float, device=device)],
        )
        padded_raw_expressions = torch.cat(
            [raw_expressions, torch.zeros(num_to_pad, dtype=torch.float, device=device)],
        )
        padded_gen_mask = torch.cat(
            [gen_mask, torch.full((num_to_pad,), False, dtype=torch.bool, device=device)],
        )
        return padded_genes, padded_expressions, padded_raw_expressions, padded_gen_mask