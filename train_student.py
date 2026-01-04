# %%
import numpy as np
import scanpy as sc
from pathlib import Path
import torch
from tqdm.auto import tqdm
import torch.nn
import os # Added os import for os.cpu_count()
from transformers.utils import is_flash_attn_2_available

from distilled_tx1.preprocessing.pipeline import TahoePreprocessor, PreprocessingConfig
from distilled_tx1.models.modeling_distilled_tahoe import DistilledTahoeModel, DistilledTahoeConfig
from distilled_tx1.training.distillation import train_distilled_model, DistillationDataset # Imported DistillationDataset

# %%
# --- Configuration ---
DATA_DIR = Path("/data/scClassificationDatasets/data_yuto/tahoe_x1_embeddings/70m")
REF_ADATA_PATH = "70m/data_yuto_with_clusters_chunk_001.h5ad"
VOCAB_PATH = "vocab.json"
OUTPUT_DIR = "./model_outputs/distilled_tahoe_optimized"
WANDB_PROJECT = "distilled-tahoe-x1-optimized"

# --- Preprocessing Config ---
preproc_config = PreprocessingConfig(
    seq_len=512,
    n_bins=51,
    normalize=True,
    target_sum=1e4,
    gene_sampling_strategy="topk",
    add_cls_token=True,
    gene_id_key="gene_id"
)

# --- Student Model Config ---
student_config = DistilledTahoeConfig(
    vocab_size=30000,  # Will be updated by preprocessor vocab
    n_bins=preproc_config.n_bins,
    hidden_size=512,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=1024,
    max_position_embeddings=preproc_config.seq_len,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    pooling_strategy="cls"
)

# Enable Flash Attention 2 if available for an extra speed boost
if torch.cuda.is_available() and is_flash_attn_2_available():
    print("Flash Attention 2 detected, enabling for training.")
    student_config.attn_implementation = "flash_attention_2"
else:
    print("Flash Attention 2 not available, using standard attention.")
    student_config.attn_implementation = "eager"

# --- Training Hyperparameters ---
training_args = {
    "num_epochs": 25,
    "batch_size": 16, # Can be larger with AMP
    "learning_rate": 5e-4,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "logging_steps": 100,
    "save_steps": 5000,
    "eval_split": 0.1,
    "use_wandb": True,
    "wandb_project": WANDB_PROJECT,
    "cosine_loss_weight": 1.0, # Add cosine similarity loss
    "num_workers": min(os.cpu_count() or 1, 8) # Use up to 8 CPU cores for data loading, handle None for os.cpu_count()
}

# %%
# --- Data Loading and Preprocessing ---
print("Initializing preprocessor...")
ref_adata = sc.read_h5ad(REF_ADATA_PATH)
preprocessor = TahoePreprocessor(
    config=preproc_config,
    tahoe_model_size="70m",
    vocab_path=VOCAB_PATH
)
student_config.vocab_size = preprocessor.vocab.vocab_size # Update vocab size

print(f"Loading and preprocessing data from {DATA_DIR}...")
all_gene_ids, all_expr_bins, all_attn_masks, all_teacher_embs = [], [], [], []

for h5ad_file in tqdm(sorted(list(DATA_DIR.glob("*.h5ad"))), desc="Processing files"):
    adata = sc.read_h5ad(h5ad_file)
    adata.var['gene_id'] = ref_adata.var['gene_id'].astype(str)
    
    processed = preprocessor.process_adata(adata, return_dict=True)
    
    all_gene_ids.append(processed["gene_ids"].numpy())
    all_expr_bins.append(processed["expression_bins"].numpy())
    all_attn_masks.append(processed["attention_mask"].numpy())
    all_teacher_embs.append(adata.obsm['Tx1-70m'])
    del adata

print("Concatenating data arrays...")
full_gene_ids = np.concatenate(all_gene_ids)
full_expression_bins = np.concatenate(all_expr_bins)
full_attention_masks = np.concatenate(all_attn_masks)
full_teacher_embeddings = np.concatenate(all_teacher_embs)

print("Creating PyTorch Dataset...")
full_dataset = DistillationDataset(
    gene_ids=full_gene_ids,
    expression_bins=full_expression_bins,
    attention_masks=full_attention_masks,
    teacher_embeddings=full_teacher_embeddings,
    labels=None
)
print(f"Dataset created with {len(full_dataset)} samples.")

# %%
# --- Start Training ---
print("Starting optimized training...")
try:
    trained_model = train_distilled_model(
        full_dataset=full_dataset,
        config=student_config,
        output_dir=OUTPUT_DIR,
        **training_args
    )
except Exception as e:
    print(f"An error occurred during training: {e}")
    import traceback
    traceback.print_exc()

# %%
print("Training script finished.")
