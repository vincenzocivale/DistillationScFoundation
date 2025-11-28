# trainer_hf.py

import os
import torch
import wandb
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from wandb import Table
from torch.utils.data import WeightedRandomSampler


# ================================================================
# 1) Class weights + WeightedSampler
# ================================================================

def compute_class_weights(dataset, label2id):
    labels = [label2id[y] for y in dataset["label"]]
    labels_tensor = torch.tensor(labels)

    class_counts = torch.bincount(labels_tensor)
    total = labels_tensor.shape[0]

    weights = total / (len(class_counts) * class_counts)
    weights = weights / weights.sum()

    return weights.float()


def build_weighted_sampler(dataset, label2id):
    """Restituisce WeightedRandomSampler per dataset sbilanciati."""
    labels = [label2id[y] for y in dataset["label"]]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts

    sample_weights = [class_weights[l] for l in labels]
    sample_weights = torch.DoubleTensor(sample_weights)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ================================================================
# 2) Confusion Matrix (tabellare)
# ================================================================

def log_confusion_matrix(name, true, pred, id2label):
    cm = confusion_matrix(true, pred)
    labels = [id2label[i] for i in range(len(id2label))]
    table = Table(columns=["true", "pred", "count"])

    for i, t_label in enumerate(labels):
        for j, p_label in enumerate(labels):
            table.add_data(t_label, p_label, int(cm[i][j]))

    wandb.log({f"confusion_matrix/{name}": table})


# ================================================================
# 3) ROC multi-class
# ================================================================

def log_multiclass_roc(name, true, probs, id2label):
    """Genera e logga curve ROC multi-class su W&B."""
    true = np.array(true)
    probs = np.array(probs)

    # One-hot
    n_classes = len(id2label)
    true_oh = np.zeros((len(true), n_classes))
    true_oh[np.arange(len(true)), true] = 1

    for c in range(n_classes):
        if true_oh[:, c].sum() == 0:
            continue  # Evita classi non presenti

        fpr, tpr, _ = roc_curve(true_oh[:, c], probs[:, c])
        roc_auc = auc(fpr, tpr)

        wandb.log({
            f"roc/{name}/{id2label[c]}": wandb.plot.line_series(
                xs=fpr.tolist(),
                ys=[tpr.tolist()],
                keys=[f"AUC={roc_auc:.4f}"],
                title=f"ROC - {id2label[c]}",
                xname="FPR",
            )
        })


# ================================================================
# 4) Custom Trainer con metriche + confusion + ROC
# ================================================================

class TrainerWithMetrics(Trainer):

    def __init__(self, id2label, label2id, **kwargs):
        super().__init__(**kwargs)
        self.id2label = id2label
        self.label2id = label2id

    def evaluate(self, eval_dataset=None, **kwargs):
        outputs = super().evaluate(eval_dataset, **kwargs)

        preds_logits = self.predict(eval_dataset).predictions
        preds = np.argmax(preds_logits, axis=1)
        labels = [self.label2id[l] for l in eval_dataset["label"]]

        # metrics
        outputs.update(self._compute_metrics(preds, labels))

        # confusion
        log_confusion_matrix("validation", labels, preds, self.id2label)

        # roc
        log_multiclass_roc("validation", labels, preds_logits, self.id2label)

        return outputs

    def predict(self, test_dataset, **kwargs):
        outputs = super().predict(test_dataset, **kwargs)

        preds_logits = outputs.predictions
        preds = np.argmax(preds_logits, axis=1)
        labels = [self.label2id[l] for l in test_dataset["label"]]

        # confusion
        log_confusion_matrix("test", labels, preds, self.id2label)
        log_multiclass_roc("test", labels, preds_logits, self.id2label)

        return outputs

    @staticmethod
    def _compute_metrics(preds, labels):
        """Metriche avanzate (macro/weighted)."""

        metrics = {
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
            "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
            "precision_weighted": precision_score(labels, preds, average="weighted", zero_division=0),
            "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
            "recall_weighted": recall_score(labels, preds, average="weighted", zero_division=0),
        }
        wandb.log(metrics)

        return metrics


# ================================================================
# 5) Main training function
# ================================================================

def train_classifier(
    model,
    dataset_dict,
    output_dir: str,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    project_name="cell_classifier",
    run_name="mlp_run",
    epochs=20,
    batch_size=64,
    lr=1e-4,
    patience=4,
    push_to_hub=False,
    hub_repo_id=None,
):
    wandb.init(project=project_name, name=run_name)

    # ===== Class weights =====
    class_weights = compute_class_weights(dataset_dict["train"], label2id)
    model.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(model.output_projection.weight.device))

    # ===== Weighted sampler =====
    train_sampler = build_weighted_sampler(dataset_dict["train"], label2id)

    args = TrainingArguments(
        output_dir=output_dir,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        save_total_limit=2,
        evalu_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_steps=20,
        report_to="wandb",
        run_name=run_name,
        fp16=torch.cuda.is_available(),
        push_to_hub=push_to_hub,
        hub_model_id=hub_repo_id,
    )

    trainer = TrainerWithMetrics(
        model=model,
        args=args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        id2label=id2label,
        label2id=label2id,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
        train_sampler=train_sampler,
    )

    trainer.train()

    # Evaluate test set
    trainer.predict(dataset_dict["test"])

    wandb.finish()

    return trainer
