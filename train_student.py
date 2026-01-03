# %%
import numpy as np
import scanpy as sc
from pathlib import Path
import torch
from tqdm.auto import tqdm

from distilled_tx1.preprocessing.pipeline import TahoePreprocessor, PreprocessingConfig
from distilled_tx1.models.modeling_distilled_tahoe import DistilledTahoeModel, DistilledTahoeConfig
from distilled_tx1.training.distillation import train_distilled_model
from distilled_tx1.data.load_h5ad_folder import load_h5ad_folder_lazy

# %%
ref_adata = sc.read_h5ad("data_yuto_with_clusters_chunk_001.h5ad")

# %%

# %%
teacher_embeddings = np.array([])

# %%
config = PreprocessingConfig(
        seq_len=512,
        n_bins=51,
        normalize=False,
        target_sum=1e4,
        gene_sampling_strategy="topk",
        add_cls_token=True,
        gene_id_key="gene_id"  # or None to use var_names
    )
    
preprocessor = TahoePreprocessor(
    config=config,
    tahoe_model_size="70m",
    vocab_path="vocab.json"
)

# %%
gene_ids = np.array([])
expression_bins = np.array([])
attention_masks = np.array([])

# %%
for h5ad_file in tqdm(Path("70m").glob("*.h5ad")):
    adata = sc.read_h5ad(h5ad_file)
    adata.var['gene_id'] = ref_adata.var['gene_id']
    
    processed = preprocessor.process_adata(adata, return_dict=True)

    if gene_ids.size == 0:
        gene_ids = processed["gene_ids"].numpy()
    else:
        gene_ids = np.concatenate([gene_ids, processed["gene_ids"].numpy()])
    
    if expression_bins.size == 0:
        expression_bins = processed["expression_bins"].numpy()
    else:
        expression_bins = np.concatenate([expression_bins, processed["expression_bins"].numpy()])
    
    if attention_masks.size == 0:
        attention_masks = processed["attention_mask"].numpy()
    else:
        attention_masks = np.concatenate([attention_masks, processed["attention_mask"].numpy()])

    if teacher_embeddings.size == 0:
        teacher_embeddings = adata.obsm['Tx1-70m']
    else:
        teacher_embeddings = np.concatenate([teacher_embeddings, adata.obsm['Tx1-70m']])

    del adata

# %%
student_config = DistilledTahoeConfig(
        vocab_size=preprocessor.vocab.vocab_size,
        n_bins=config.n_bins,
        hidden_size=512,  # Match teacher embedding dimension
        num_hidden_layers=4,  # Smaller than Tahoe X1 (12-24 layers)
        num_attention_heads=4,
        intermediate_size=1024,
        max_position_embeddings=config.seq_len,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pooling_strategy="cls"  # or "mean"
    )

# %%
try:
    model = train_distilled_model(
            gene_ids=gene_ids,
            expression_bins=expression_bins,
            attention_masks=attention_masks,
            teacher_embeddings=teacher_embeddings,
            labels=None,  # Optional: add classification labels
            config=student_config,
            output_dir="./model_outputs/distilled_tahoe",
            num_epochs=5,
            batch_size=64,  # Adjust based on GPU memory
            learning_rate=5e-3,
            warmup_steps=1000,
            weight_decay=0.01,
            max_grad_norm=1.0,
            logging_steps=100,
            save_steps=5000,
            eval_split=0.1,
            use_wandb=True,  # Optional: log to W&B
            wandb_project="distilled-tahoe-x1",
        )
except Exception as e:
    print(e)
    


