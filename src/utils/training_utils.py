"""Training utilities: parameter counting, schedulers, loss computation."""

import math
import torch
import torch.nn as nn


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def get_scheduler(optimizer, scheduler_type="cosine", num_training_steps=1000,
                  warmup_steps=100):
    """Create a learning rate scheduler with warmup."""
    if scheduler_type == "cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_type == "linear":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return max(0.0, 1.0 - (step - warmup_steps) / max(1, num_training_steps - warmup_steps))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, num_training_steps // 3))

    return scheduler


def compute_loss(
    logits,
    labels,
    mask=None,
    aux_outputs=None,
    aux_loss_weight=0.1,
    velocity_targets=None,
    temporal_smoothness_weight=0.0,
    velocity_loss_weight=0.0,
):
    """Compute total loss including language modeling and auxiliary losses."""
    lm_loss = nn.functional.cross_entropy(
        logits.contiguous().view(-1, logits.size(-1)),
        labels.contiguous().view(-1),
        reduction="none",
    )

    if mask is not None:
        mask_flat = mask.contiguous().reshape(-1)
        lm_loss = (lm_loss * mask_flat).sum() / (mask_flat.sum() + 1e-8)
    else:
        lm_loss = lm_loss.mean()

    loss_dict = {"lm_loss": lm_loss.item()}
    total_loss = lm_loss

    if aux_outputs is not None:
        if "gate_values" in aux_outputs:
            gate_vals = aux_outputs["gate_values"]
            gate_diversity_loss = -torch.std(gate_vals)
            total_loss = total_loss + aux_loss_weight * gate_diversity_loss
            loss_dict["gate_diversity"] = gate_diversity_loss.item()

            if temporal_smoothness_weight > 0 and gate_vals.size(1) > 1:
                smooth = torch.abs(gate_vals[:, 1:] - gate_vals[:, :-1]).mean()
                total_loss = total_loss + temporal_smoothness_weight * smooth
                loss_dict["gate_temporal_smoothness"] = smooth.item()

        if "gain" in aux_outputs:
            gain = aux_outputs["gain"]
            gain_sparsity_loss = torch.abs(gain - 0.5).mean()
            total_loss = total_loss + aux_loss_weight * gain_sparsity_loss
            loss_dict["gain_sparsity"] = gain_sparsity_loss.item()

        if velocity_targets is not None and velocity_loss_weight > 0 and "pred_velocity" in aux_outputs:
            pred_velocity = aux_outputs["pred_velocity"]
            vel_loss = nn.functional.mse_loss(pred_velocity, velocity_targets)
            total_loss = total_loss + velocity_loss_weight * vel_loss
            loss_dict["velocity_loss"] = vel_loss.item()

    loss_dict["total_loss"] = total_loss.item()
    return total_loss, loss_dict
