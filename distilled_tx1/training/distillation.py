"""
Knowledge Distillation Training Script

Train a student encoder to match Tahoe X1 teacher embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from typing import Dict, Optional, Tuple
import numpy as np
from pathlib import Path
from tqdm import tqdm
import wandb

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
    gene_ids: np.ndarray,
    expression_bins: np.ndarray,
    attention_masks: np.ndarray,
    teacher_embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    config: Optional[DistilledTahoeConfig] = None,
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
    **loss_kwargs
) -> DistilledTahoeModel:
    """
    Train a distilled Tahoe encoder.
    
    Args:
        gene_ids: Tokenized gene IDs (n_cells, seq_len)
        expression_bins: Expression bins (n_cells, seq_len)
        attention_masks: Attention masks (n_cells, seq_len)
        teacher_embeddings: Pre-computed teacher embeddings (n_cells, embedding_dim)
        labels: Optional classification labels
        config: Model configuration
        output_dir: Directory to save model checkpoints
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        weight_decay: Weight decay
        max_grad_norm: Max gradient norm for clipping
        logging_steps: Log every N steps
        save_steps: Save checkpoint every N steps
        eval_split: Fraction of data for validation
        use_wandb: Whether to log to Weights & Biases
        wandb_project: W&B project name
        device: Device to train on
        **loss_kwargs: Additional arguments for DistillationLoss
    
    Returns:
        Trained DistilledTahoeModel
    """
    # Initialize W&B
    if use_wandb:
        wandb_config = {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            **loss_kwargs,
        }
        wandb_config.update(config.__dict__)
        wandb.init(project=wandb_project, config=wandb_config)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split data
    n_samples = len(gene_ids)
    n_eval = int(n_samples * eval_split)
    indices = np.random.permutation(n_samples)
    
    train_indices = indices[n_eval:]
    eval_indices = indices[:n_eval]
    
    # Create datasets
    train_dataset = DistillationDataset(
        gene_ids[train_indices],
        expression_bins[train_indices],
        attention_masks[train_indices],
        teacher_embeddings[train_indices],
        labels[train_indices] if labels is not None else None
    )
    
    eval_dataset = DistillationDataset(
        gene_ids[eval_indices],
        expression_bins[eval_indices],
        attention_masks[eval_indices],
        teacher_embeddings[eval_indices],
        labels[eval_indices] if labels is not None else None
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    if config is None:
        # Infer config from data
        vocab_size = int(gene_ids.max()) + 1
        n_bins = int(expression_bins.max()) + 1
        embedding_dim = teacher_embeddings.shape[1]
        
        config = DistilledTahoeConfig(
            vocab_size=vocab_size,
            n_bins=n_bins,
            hidden_size=embedding_dim,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=embedding_dim * 4
        )
    
    model = DistilledTahoeModel(config)
    model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Initialize loss
    criterion = DistillationLoss(**loss_kwargs)
    
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
            # Move to device
            gene_ids_batch = batch["gene_ids"].to(device)
            expression_bins_batch = batch["expression_bins"].to(device)
            attention_mask_batch = batch["attention_mask"].to(device)
            teacher_emb_batch = batch["teacher_embeddings"].to(device)
            
            # Forward pass
            outputs = model(
                gene_ids=gene_ids_batch,
                expression_bins=expression_bins_batch,
                attention_mask=attention_mask_batch,
                return_dict=True
            )
            
            student_emb = outputs.pooler_output
            
            # Compute loss
            loss, loss_dict = criterion(
                student_embeddings=student_emb,
                teacher_embeddings=teacher_emb_batch
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            # Logging
            train_losses.append(loss.item())
            global_step += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if global_step % logging_steps == 0:
                if use_wandb:
                    wandb.log({
                        f"train/{k}": v for k, v in loss_dict.items()
                    }, step=global_step)
            
            if global_step % save_steps == 0:
                checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                model.save_pretrained(checkpoint_dir)
                print(f"\nSaved checkpoint to {checkpoint_dir}")
        
        # Evaluation
        model.eval()
        eval_losses = []
        eval_cosine_sims = []
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                gene_ids_batch = batch["gene_ids"].to(device)
                expression_bins_batch = batch["expression_bins"].to(device)
                attention_mask_batch = batch["attention_mask"].to(device)
                teacher_emb_batch = batch["teacher_embeddings"].to(device)
                
                outputs = model(
                    gene_ids=gene_ids_batch,
                    expression_bins=expression_bins_batch,
                    attention_mask=attention_mask_batch,
                    return_dict=True
                )
                
                student_emb = outputs.pooler_output
                
                loss, loss_dict = criterion(
                    student_embeddings=student_emb,
                    teacher_embeddings=teacher_emb_batch
                )
                
                eval_losses.append(loss.item())

                # Calculate cosine similarity
                sim = F.cosine_similarity(student_emb, teacher_emb_batch, dim=-1)
                eval_cosine_sims.extend(sim.cpu().numpy())

        avg_train_loss = np.mean(train_losses)
        avg_eval_loss = np.mean(eval_losses)
        avg_eval_cosine_sim = np.mean(eval_cosine_sims)
        
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Eval Loss: {avg_eval_loss:.4f}")
        print(f"Eval Cosine Similarity: {avg_eval_cosine_sim:.4f}")
        
        if use_wandb:
            wandb.log({
                "train/epoch_loss": avg_train_loss,
                "eval/epoch_loss": avg_eval_loss,
                "eval/cosine_similarity": avg_eval_cosine_sim,
                "epoch": epoch
            }, step=global_step)
        
        # Save best model
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            model.save_pretrained(output_dir / "best_model")
            print(f"Saved best model (eval_loss: {best_eval_loss:.4f})")
    
    # Save final model
    model.save_pretrained(output_dir / "final_model")
    print(f"\nTraining complete! Final model saved to {output_dir / 'final_model'}")
    
    if use_wandb:
        wandb.finish()
    
    return model