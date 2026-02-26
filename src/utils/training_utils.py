"""Training utilities."""

import math
import torch
import torch.nn as nn


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def get_scheduler(optimizer, scheduler_type="cosine", num_training_steps=1000, warmup_steps=100):
    if scheduler_type == "cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if scheduler_type == "linear":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return max(0.0, 1.0 - (step - warmup_steps) / max(1, num_training_steps - warmup_steps))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, num_training_steps // 3))


def compute_loss(
    logits,
    labels,
    mask=None,
    aux_outputs=None,
    aux_loss_weight=0.1,
    velocity_targets=None,
    vector_targets=None,
    temporal_smoothness_weight=0.0,
    velocity_loss_weight=0.0,
    vector_loss_weight=0.0,
):
    lm_loss = nn.functional.cross_entropy(logits.contiguous().view(-1, logits.size(-1)), labels.contiguous().view(-1), reduction="none")
    if mask is not None:
        mf = mask.contiguous().reshape(-1)
        lm_loss = (lm_loss * mf).sum() / (mf.sum() + 1e-8)
    else:
        lm_loss = lm_loss.mean()

    total_loss = lm_loss
    loss_dict = {"lm_loss": lm_loss.item()}

    if aux_outputs is not None:
        gate_vals = aux_outputs.get("gate_values")
        if gate_vals is not None:
            gate_div = -torch.std(gate_vals)
            total_loss += aux_loss_weight * gate_div
            loss_dict["gate_diversity"] = gate_div.item()
            if temporal_smoothness_weight > 0 and gate_vals.size(1) > 1:
                smooth = torch.abs(gate_vals[:, 1:] - gate_vals[:, :-1]).mean()
                total_loss += temporal_smoothness_weight * smooth
                loss_dict["gate_temporal_smoothness"] = smooth.item()

        gain = aux_outputs.get("gain")
        if gain is not None:
            spars = torch.abs(gain - 0.5).mean()
            total_loss += aux_loss_weight * spars
            loss_dict["gain_sparsity"] = spars.item()

        if velocity_targets is not None and velocity_loss_weight > 0 and "pred_velocity" in aux_outputs:
            vel_loss = nn.functional.mse_loss(aux_outputs["pred_velocity"], velocity_targets)
            total_loss += velocity_loss_weight * vel_loss
            loss_dict["velocity_loss"] = vel_loss.item()

        if vector_targets is not None and vector_loss_weight > 0 and "pred_vector" in aux_outputs:
            pred_vec = aux_outputs["pred_vector"].mean(dim=1)
            vec_loss = nn.functional.mse_loss(pred_vec, vector_targets)
            total_loss += vector_loss_weight * vec_loss
            loss_dict["vector_loss"] = vec_loss.item()

    loss_dict["total_loss"] = total_loss.item()
    return total_loss, loss_dict
