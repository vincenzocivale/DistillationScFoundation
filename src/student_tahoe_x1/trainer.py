import torch
from torch import nn
import transformers
from transformers import TrainingArguments, Trainer, IntervalStrategy
from typing import Dict, Union, Optional, List, Tuple, Any
import evaluate
import numpy as np
import os
import pandas as pd
import anndata as ad
from pathlib import Path

from .configuration_student_tx import StudentTXConfig
from .modeling_student_tx import StudentTXModel
from .tokenization_student_tx import StudentTXTokenizer
from .data_builder import build_hf_dataset_from_h5ad
from .data_collator import StudentTXDataCollator
from tahoe_x1.tokenizer import GeneVocab as TeacherGeneVocab # Still needed for dummy vocab creation


class DistillationTrainer(Trainer):
    def __init__(self, distillation_loss_fn: nn.Module = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distillation_loss_fn = distillation_loss_fn

    def compute_loss(self, model: StudentTXModel, inputs: Dict[str, Union[torch.Tensor, Any]], return_outputs=False):
        # Student model forward pass
        student_outputs = model(
            genes=inputs["genes"],
            values=inputs["values"],
            gen_masks=inputs["gen_masks"],
            attention_mask=inputs["attention_mask"],
            # drug_ids=inputs.get("drug_ids", None), # Removed drug_ids
            return_dict=True,
            skip_decoders=True, # We only need cell embeddings for distillation
        )
        student_embeddings = student_outputs["cell_emb"]

        # Directly use teacher embeddings from the input batch
        teacher_embeddings = inputs["teacher_embeddings"]
        
        # Calculate distillation loss
        loss = self.distillation_loss_fn(student_embeddings, teacher_embeddings)

        return (loss, student_outputs) if return_outputs else loss


def train_distillation_model(
    student_config_path: str,
    teacher_vocab_path: str, # Path to teacher's vocab (used for student tokenizer)
    train_h5ad_path: str,
    validation_h5ad_path: str,
    output_dir: str,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    logging_steps: int = 100,
    save_steps: int = 500,
    eval_steps: int = 500,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    mlm_probability: float = 0.15,
    do_binning: bool = True,
    log_transform_expr: bool = False,
    max_seq_len: Optional[int] = None,
    teacher_embedding_key: str = "teacher_embeddings", # New argument to specify key in H5AD
    # Removed use_chem_token, drug_to_id_mapping, drug_key
):
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load Student Config, Tokenizer, and Model
    student_config = StudentTXConfig.from_json_file(student_config_path)
    
    student_tokenizer = StudentTXTokenizer(vocab_file=teacher_vocab_path) 
    
    # Update student_config with tokenizer info if needed
    student_config.pad_token_id = student_tokenizer.pad_token_id
    student_config.vocab_size = student_tokenizer.vocab_size
    # Removed student_config.use_chem_token = use_chem_token
    if max_seq_len:
        student_config.max_position_embeddings = max_seq_len
    
    student_model = StudentTXModel(student_config)

    # 2. Define Distillation Loss
    distillation_loss_fn = nn.MSELoss()

    # 3. Prepare Data
    train_dataset = build_hf_dataset_from_h5ad(
        h5ad_path=train_h5ad_path,
        tokenizer=student_tokenizer,
        cls_token_id=student_tokenizer.cls_token_id,
        pad_value=student_config.pad_value,
        max_seq_len=max_seq_len,
        teacher_embedding_key=teacher_embedding_key,
    )
    eval_dataset = build_hf_dataset_from_h5ad(
        h5ad_path=validation_h5ad_path,
        tokenizer=student_tokenizer,
        cls_token_id=student_tokenizer.cls_token_id,
        pad_value=student_config.pad_value,
        max_seq_len=max_seq_len,
        # Removed drug_to_id_mapping, drug_key
        teacher_embedding_key=teacher_embedding_key,
    )

    data_collator = StudentTXDataCollator(
        config=student_config,
        tokenizer=student_tokenizer,
        mlm_probability=mlm_probability,
        do_binning=do_binning,
        log_transform_expr=log_transform_expr,
        target_sum=student_config.target_sum,
        mask_value=student_config.pad_value,
    )

    # 4. Setup Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_strategy="steps", # Added
        eval_strategy="steps", # Re-added
        eval_steps=eval_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        report_to="wandb", # Can be changed to "wandb"
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # 5. Instantiate and Train Trainer
    trainer = DistillationTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        distillation_loss_fn=distillation_loss_fn,
    )

    trainer.train()

    # Save the final student model
    student_model.save_pretrained(os.path.join(output_dir, "student_model"))
    student_tokenizer.save_pretrained(os.path.join(output_dir, "student_tokenizer"))
