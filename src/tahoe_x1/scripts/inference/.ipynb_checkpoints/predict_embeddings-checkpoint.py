# Copyright (C) Tahoe Therapeutics 2025. All rights reserved.

import logging
import sys
from typing import List

import numpy as np
import scanpy as sc
import torch
from composer import Trainer
from composer.core import Precision
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from tqdm.auto import tqdm

from tahoe_x1.model import ComposerTX
from tahoe_x1.utils.util import load_model, loader_from_adata

log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
    level=logging.INFO,
)


###############################################
#  FIX UNIVERSALE PER I TENSORI DI INDICI
###############################################

def move_batch_to_device(batch, device, fp_dtype):
    """
    - I tensori FLOAT → FP16/BF16
    - I tensori INT → LONG
    - Evita conversioni che rompono Embedding()
    """

    batch_fixed = {}

    for k, v in batch.items():
        if not hasattr(v, "to"):  # non è un tensore
            batch_fixed[k] = v
            continue

        # FLOAT → FP16/BF16
        if torch.is_floating_point(v):
            batch_fixed[k] = v.to(device, dtype=fp_dtype)

        # INT → LONG (per embedding)
        elif v.dtype in (torch.int32, torch.int64, torch.int16, torch.int8):
            batch_fixed[k] = v.to(device, dtype=torch.long)

        # BOOL, MASK, altri → solo device
        else:
            batch_fixed[k] = v.to(device)

    return batch_fixed


###############################################
#  MAIN FUNCTION
###############################################

def predict_embeddings(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = cfg.get("model_name", None)
    gene_id_key = cfg.data.gene_id_key
    return_gene_embeddings = cfg.predict.get("return_gene_embeddings", False)
    batch_size = cfg.predict.get("batch_size", 1)
    max_length = cfg.predict.get("seq_len_dataset", 2048)

    # Low-VRAM dataloader settings
    num_workers = cfg.predict.get("num_workers", 8)
    prefetch_factor = cfg.predict.get("prefetch_factor", 8)

    # memory optimization flags
    use_fp16 = cfg.predict.get("use_fp16", True)
    use_gradient_checkpointing = cfg.predict.get("use_gradient_checkpointing", False)
    clear_cache_every_n_batches = cfg.predict.get("clear_cache_every_n_batches", 1)

    adata_output_path = cfg.paths.get("adata_output", None)

    ###############################################
    # Load model (CPU first → avoids VRAM spike)
    ###############################################
    model_dir = cfg.paths.get("model_dir", None)
    if model_dir is not None:
        model, vocab, _, coll_cfg = load_model(
            model_dir,
            device="cpu",
            return_gene_embeddings=return_gene_embeddings,
        )
    else:
        hf_repo_id = cfg.paths.get("hf_repo_id")
        hf_model_size = cfg.paths.get("hf_model_size")
        model, vocab, _, coll_cfg = ComposerTX.from_hf(
            hf_repo_id,
            hf_model_size,
            return_gene_embeddings=return_gene_embeddings,
        )

    if use_gradient_checkpointing and hasattr(model.model, "gradient_checkpointing_enable"):
        model.model.gradient_checkpointing_enable()

    # FP16/BF16 precision setting
    if device.type == "cuda" and use_fp16:
        if torch.cuda.is_bf16_supported():
            model = model.to(dtype=torch.bfloat16)
            fp_dtype = torch.bfloat16
        else:
            model = model.to(dtype=torch.float16)
            fp_dtype = torch.float16
    else:
        fp_dtype = torch.float32

    model.to(device)
    model.eval()

    ###############################################
    # Load AnnData
    ###############################################

    adata = sc.read_h5ad(cfg.paths.adata_input)

    adata.var["id_in_vocab"] = [
        vocab[gene] if gene in vocab else -1 for gene in adata.var[gene_id_key]
    ]
    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    genes = adata.var[gene_id_key].tolist()
    gene_ids = np.array([vocab[g] for g in genes], dtype=int)

    loader = loader_from_adata(
        adata=adata,
        collator_cfg=coll_cfg,
        vocab=vocab,
        batch_size=batch_size,
        max_length=max_length,
        gene_ids=gene_ids,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    cell_embs = []
    if return_gene_embeddings:
        gene_embs = []
        gene_ids_list = []

    ###############################################
    # Inference loop (Low VRAM)
    ###############################################

    with torch.inference_mode():
        progress = tqdm(
            loader,
            desc="Inferenza modello",
            unit="batch",
            dynamic_ncols=True,
        )

        for i, batch in enumerate(progress):

            # FIX: conversione corretta dei tipi
            batch = move_batch_to_device(batch, device, fp_dtype)

            out = model(batch)

            cell_embs.append(out["cell_emb"].float().cpu())

            if return_gene_embeddings:
                gene_embs.append(out["gene_emb"].float().cpu())
                gene_ids_list.append(out["gene_ids"].cpu())

            del batch, out
            torch.cuda.empty_cache()

            if clear_cache_every_n_batches > 0 and (i + 1) % clear_cache_every_n_batches == 0:
                torch.cuda.empty_cache()

    ###############################################
    # Assemble output
    ###############################################

    cell_array = torch.cat(cell_embs, dim=0).numpy()
    cell_array = cell_array / np.linalg.norm(cell_array, axis=1, keepdims=True)

    adata.obsm[model_name] = cell_array

    if return_gene_embeddings:
        gene_embs = torch.cat(gene_embs)
        gene_ids_list = torch.cat(gene_ids_list)

        flat_ids = gene_ids_list.flatten()
        flat_embs = gene_embs.flatten(0, 1)

        valid = flat_ids != coll_cfg["pad_token_id"]
        flat_ids = flat_ids[valid]
        flat_embs = flat_embs[valid]

        sums = np.zeros((len(vocab), flat_embs.size(-1)), dtype=np.float32)
        counts = np.zeros((len(vocab), 1), dtype=np.float32)

        for idx, emb in zip(flat_ids.numpy(), flat_embs.numpy()):
            sums[idx] += emb
            counts[idx] += 1

        means = sums / np.maximum(counts, 1)
        gene_array = means[list(vocab.get_stoi().values())]

        adata.varm[model_name] = gene_array[gene_ids, :]

    if adata_output_path:
        adata.write_h5ad(adata_output_path)

    return adata


###############################################
# CLI
###############################################

if __name__ == "__main__":
    num_mand_args = 2
    if len(sys.argv) < num_mand_args:
        raise SystemExit("Usage: predict_embeddings.py <config.yaml> [--key=value ...]")

    cfg = om.load(sys.argv[1])

    cli_args = []
    for arg in sys.argv[num_mand_args:]:
        if arg.startswith("--"):
            cli_args.append(arg[2:])
        else:
            cli_args.append(arg)

    cli_cfg = om.from_cli(cli_args)
    cfg = om.merge(cfg, cli_cfg)

    om.resolve(cfg)
    predict_embeddings(cfg)
