"""
Knowledge Distillation Training Script

Train a student encoder to match Tahoe X1 teacher embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup
from typing import Dict, Optional, Tuple
import numpy as np
from pathlib import Path
from tqdm import tqdm
import wandb
import pkg_resources

from ..models.modeling_distilled_tahoe import DistilledTahoeModel, DistilledTahoeConfig
from ..preprocessing import TahoePreprocessor


class DistillationDataset(Dataset):
    """
    Dataset for knowledge distillation.
    
    Loads pre-computed teacher embeddings and corresponding tokenized inputs.
    """
    
    def __init__(
        self,
        gene_ids: np.ndarray,  # (n_cells, seq_len)
        expression_bins: np.ndarray,  # (n_cells, seq_len)
        attention_masks: np.ndarray,  # (n_cells, seq_len)
        teacher_embeddings: np.ndarray,  # (n_cells, embedding_dim)
        labels: Optional[np.ndarray] = None  # (n_cells,) - for optional classification
    ):
        self.gene_ids = torch.from_numpy(gene_ids).long()
        self.expression_bins = torch.from_numpy(expression_bins).long()
        self.attention_masks = torch.from_numpy(attention_masks).long()
        self.teacher_embeddings = torch.from_numpy(teacher_embeddings).float()
        
        if labels is not None:
            self.labels = torch.from_numpy(labels).long()
        else:
            self.labels = None
    
    def __len__(self) -> int:
        return len(self.gene_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "gene_ids": self.gene_ids[idx],
            "expression_bins": self.expression_bins[idx],
            "attention_mask": self.attention_masks[idx],
            "teacher_embeddings": self.teacher_embeddings[idx]
        }
        
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        
        return item


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation.
    
    Combines:
    - MSE loss between student and teacher embeddings
    - Optional classification loss (if labels provided)
    - Optional cosine similarity loss
    """
    
    def __init__(
        self,
        embedding_loss_weight: float = 1.0,
        classification_loss_weight: float = 0.0,
        cosine_loss_weight: float = 0.0,
        temperature: float = 1.0
    ):
        super().__init__()
        self.embedding_loss_weight = embedding_loss_weight
        self.classification_loss_weight = classification_loss_weight
        self.cosine_loss_weight = cosine_loss_weight
        self.temperature = temperature
        
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
    
    def forward(
        self,
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor,
        student_logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute distillation loss.
        
        Args:
            student_embeddings: Student model embeddings
            teacher_embeddings: Teacher model embeddings
            student_logits: Optional classification logits
            labels: Optional ground truth labels
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual loss components
        """
        loss_dict = {}
        
        # Embedding MSE loss
        embedding_loss = self.mse_loss(student_embeddings, teacher_embeddings)
        loss_dict["embedding_loss"] = embedding_loss.item()
        
        total_loss = self.embedding_loss_weight * embedding_loss
        
        # Cosine similarity loss
        if self.cosine_loss_weight > 0:
            # Maximize cosine similarity (minimize negative cosine)
            target = torch.ones(student_embeddings.size(0), device=student_embeddings.device)
            cosine_loss = self.cosine_loss(
                student_embeddings,
                teacher_embeddings,
                target
            )
            loss_dict["cosine_loss"] = cosine_loss.item()
            total_loss = total_loss + self.cosine_loss_weight * cosine_loss
        
        # Classification loss (optional)
        if self.classification_loss_weight > 0 and student_logits is not None and labels is not None:
            ce_loss = F.cross_entropy(student_logits, labels)
            loss_dict["classification_loss"] = ce_loss.item()
            total_loss = total_loss + self.classification_loss_weight * ce_loss
        
        loss_dict["total_loss"] = total_loss.item()
        
        return total_loss, loss_dict


def train_distilled_model(
    full_dataset: DistillationDataset,
    config: DistilledTahoeConfig,
    output_dir: str = "./distilled_model",
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 5e-5,
    warmup_steps: int = 500,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    logging_steps: int = 100,
    save_steps: int = 1000,
    eval_split: float = 0.1,
    use_wandb: bool = False,
    wandb_project: str = "distilled-tahoe",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers: int = 4,
    **loss_kwargs
) -> DistilledTahoeModel:
    """
    Train a distilled Tahoe encoder with optimizations.
    
    Args:
        full_dataset: The complete DistillationDataset.
        config: Model configuration.
        output_dir: Directory to save model checkpoints.
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: AdamW learning rate.
        warmup_steps: Linear warmup steps.
        weight_decay: Weight decay for AdamW.
        max_grad_norm: Gradient clipping threshold.
        logging_steps: Log every N steps.
        save_steps: Save checkpoint every N steps.
        eval_split: Fraction of data for validation.
        use_wandb: Log to Weights & Biases.
        wandb_project: W&B project name.
        device: "cuda" or "cpu".
        num_workers: Number of workers for DataLoader.
        **loss_kwargs: Arguments for DistillationLoss.
    
    Returns:
        The trained and optimized DistilledTahoeModel.
    """
    use_amp = device == "cuda"
    
    # Initialize W&B
    if use_wandb:
        wandb_config = {
            "num_epochs": num_epochs, "batch_size": batch_size, "learning_rate": learning_rate,
            "torch_version": torch.__version__, "attn_implementation": config.attn_implementation,
            **loss_kwargs,
        }
        wandb_config.update(config.to_dict())
        wandb.init(project=wandb_project, config=wandb_config)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split data
    n_samples = len(full_dataset)
    n_eval = int(n_samples * eval_split)
    n_train = n_samples - n_eval
    train_dataset, eval_dataset = random_split(full_dataset, [n_train, n_eval])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=batch_size * 2, shuffle=False,  # Larger batch for eval
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0
    )
    
    # Initialize model
    model = DistilledTahoeModel(config).to(device)
    
    # OPTIMIZATION: torch.compile() for PyTorch 2.x
    torch_version = pkg_resources.get_distribution("torch").version
    if torch_version.startswith("2.") and device == "cuda":
        print(f"PyTorch {torch_version} detected, compiling model...")
        model = torch.compile(model)
        if use_wandb: wandb.config.update({"torch_compile": True})
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    # Initialize loss and AMP GradScaler
    criterion = DistillationLoss(**loss_kwargs)
    scaler = GradScaler(enabled=use_amp)
    if use_wandb: wandb.config.update({"use_amp": use_amp})

    # Training loop
    global_step = 0
    best_eval_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            # OPTIMIZATION: Automatic Mixed Precision (AMP)
            with autocast(enabled=use_amp):
                outputs = model(
                    gene_ids=batch["gene_ids"],
                    expression_bins=batch["expression_bins"],
                    attention_mask=batch["attention_mask"],
                    return_dict=True
                )
                student_emb = outputs.pooler_output
                loss, loss_dict = criterion(
                    student_embeddings=student_emb,
                    teacher_embeddings=batch["teacher_embeddings"]
                )
            
            # Backward pass with GradScaler
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer) # Unscale gradients for clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True) # More efficient
            scheduler.step()
            
            # Logging
            train_losses.append(loss.item())
            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if global_step % logging_steps == 0 and use_wandb:
                wandb.log({f"train/{k}": v for k, v in loss_dict.items()}, step=global_step)
            
            if global_step % save_steps == 0:
                checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                # Handle compiled model saving
                save_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                save_model.save_pretrained(checkpoint_dir)
                print(f"\nSaved checkpoint to {checkpoint_dir}")
        
        # Evaluation
        model.eval()
        eval_losses, eval_cosine_sims = [], []
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                
                # AMP for evaluation
                with autocast(enabled=use_amp):
                    outputs = model(
                        gene_ids=batch["gene_ids"],
                        expression_bins=batch["expression_bins"],
                        attention_mask=batch["attention_mask"],
                        return_dict=True
                    )
                    student_emb = outputs.pooler_output
                    loss, loss_dict = criterion(
                        student_embeddings=student_emb,
                        teacher_embeddings=batch["teacher_embeddings"]
                    )
                
                eval_losses.append(loss.item())
                sim = F.cosine_similarity(student_emb, batch["teacher_embeddings"], dim=-1)
                eval_cosine_sims.extend(sim.cpu().numpy())

        avg_train_loss = np.mean(train_losses)
        avg_eval_loss = np.mean(eval_losses)
        avg_eval_cosine_sim = np.mean(eval_cosine_sims)
        
        print(f"Train Loss: {avg_train_loss:.4f} | Eval Loss: {avg_eval_loss:.4f} | Eval Cosine Sim: {avg_eval_cosine_sim:.4f}")
        
        if use_wandb:
            wandb.log({
                "train/epoch_loss": avg_train_loss, "eval/epoch_loss": avg_eval_loss,
                "eval/cosine_similarity": avg_eval_cosine_sim, "epoch": epoch
            }, step=global_step)
        
        # Save best model
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            # Handle compiled model saving
            save_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            save_model.save_pretrained(output_dir / "best_model")
            print(f"Saved best model (eval_loss: {best_eval_loss:.4f})")
    
    # Save final model
    save_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    save_model.save_pretrained(output_dir / "final_model")
    print(f"\nTraining complete! Final model saved to {output_dir / 'final_model'}")
    
    if use_wandb:
        wandb.finish()
    
    return model