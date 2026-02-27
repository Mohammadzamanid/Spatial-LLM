#!/usr/bin/env python3
"""Training script for the Spatial LLM."""

import argparse
import os
import time

import torch
import yaml
from tqdm import tqdm

from src.models.spatial_transformer import EmbodiedSpatiotemporalLLM
from src.utils.data_utils import SpatialReasoningDataset, create_spatial_dataloader
from src.utils.training_utils import count_parameters, get_scheduler, compute_loss


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _move_batch_to_device(batch, device):
    """Move all tensor fields in a batch dict to the given device."""
    out = {}
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            out[key] = val.to(device)
        else:
            out[key] = val
    return out


def train_epoch(model, dataloader, optimizer, scheduler, config, device, epoch):
    model.train()
    total_loss = 0.0
    step_count = 0
    grad_clip = config["training"].get("gradient_clip", 1.0)
    aux_weight = config["training"].get("aux_loss_weight", 0.0)
    log_every = config["logging"].get("log_every", 20)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for step, batch in enumerate(pbar):
        batch = _move_batch_to_device(batch, device)

        input_ids = batch["input_ids"]
        labels = batch["labels"]
        mask = batch.get("attention_mask")

        optimizer.zero_grad()

        logits, aux_outputs = model(
            input_ids,
            schema_types=batch.get("schema_types"),
            spatial_features=batch.get("spatial_features"),
            temporal_distances=batch.get("temporal_distances"),
            spatial_distances=batch.get("spatial_distances"),
            velocity=batch.get("velocity"),
            visual_features=batch.get("visual_features"),
            attention_mask=mask,
        )

        loss, loss_dict = compute_loss(
            logits, labels, mask=mask,
            aux_outputs=aux_outputs if aux_weight > 0 else None,
            aux_loss_weight=aux_weight,
            velocity_targets=batch.get("velocity"),
            vector_targets=batch.get("visual_features"),
            temporal_smoothness_weight=config["training"].get("temporal_smoothness_weight", 0.0),
            velocity_loss_weight=config["training"].get("velocity_loss_weight", 0.0),
            vector_loss_weight=config["training"].get("vector_loss_weight", 0.0),
        )

        # Skip NaN losses
        if torch.isnan(loss):
            print(f"  NaN loss at step {step}, skipping")
            optimizer.zero_grad()
            continue

        loss.backward()

        # Skip NaN gradients
        has_nan_grad = any(
            torch.isnan(p.grad).any()
            for p in model.parameters()
            if p.grad is not None
        )
        if has_nan_grad:
            print(f"  NaN gradient at step {step}, skipping")
            optimizer.zero_grad()
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss_dict["total_loss"]
        step_count += 1

        if step % log_every == 0:
            pbar.set_postfix(
                loss=f"{loss_dict['total_loss']:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

    return total_loss / max(step_count, 1)


@torch.no_grad()
def evaluate(model, dataloader, config, device):
    model.eval()
    total_loss = 0.0
    step_count = 0
    aux_weight = config["training"].get("aux_loss_weight", 0.0)

    for batch in dataloader:
        batch = _move_batch_to_device(batch, device)

        input_ids = batch["input_ids"]
        labels = batch["labels"]
        mask = batch.get("attention_mask")

        logits, aux_outputs = model(
            input_ids,
            schema_types=batch.get("schema_types"),
            spatial_features=batch.get("spatial_features"),
            temporal_distances=batch.get("temporal_distances"),
            spatial_distances=batch.get("spatial_distances"),
            velocity=batch.get("velocity"),
            visual_features=batch.get("visual_features"),
            attention_mask=mask,
        )

        loss, loss_dict = compute_loss(
            logits, labels, mask=mask,
            aux_outputs=aux_outputs if aux_weight > 0 else None,
            aux_loss_weight=aux_weight,
            velocity_targets=batch.get("velocity"),
            vector_targets=batch.get("visual_features"),
            temporal_smoothness_weight=config["training"].get("temporal_smoothness_weight", 0.0),
            velocity_loss_weight=config["training"].get("velocity_loss_weight", 0.0),
            vector_loss_weight=config["training"].get("vector_loss_weight", 0.0),
        )

        if not torch.isnan(loss):
            total_loss += loss_dict["total_loss"]
            step_count += 1

    return total_loss / max(step_count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Directories
    os.makedirs(config["logging"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["logging"]["log_dir"], exist_ok=True)

    # Model
    mc = config["model"]
    model = EmbodiedSpatiotemporalLLM(
        vocab_size=mc["vocab_size"],
        d_model=mc["d_model"],
        n_layers=mc["n_layers"],
        n_heads=mc["n_heads"],
        n_reference_frames=mc["n_reference_frames"],
        max_seq_len=mc["max_seq_len"],
        dropout=mc["dropout"],
        n_schema_types=mc.get("n_schema_types", 8),
        memory_slots=mc.get("memory_slots", 256),
        visual_dim=mc.get("visual_dim", 512),
    ).to(device)

    params = count_parameters(model)
    print(f"Parameters: {params['trainable']:,} trainable / {params['total']:,} total")

    # Data
    dc = config["data"]
    train_dataset = SpatialReasoningDataset(
        synthetic=dc.get("synthetic", True),
        max_seq_len=dc["max_seq_len"],
        num_samples=dc.get("num_samples", 1000),
        data_path=dc.get("train_path"),
        visual_dim=mc.get("visual_dim", 512),
    )
    val_dataset = SpatialReasoningDataset(
        synthetic=dc.get("synthetic", True),
        max_seq_len=dc["max_seq_len"],
        num_samples=dc.get("num_val_samples", 200),
        data_path=dc.get("val_path"),
        visual_dim=mc.get("visual_dim", 512),
    )

    tc = config["training"]
    train_loader = create_spatial_dataloader(
        train_dataset, batch_size=tc["batch_size"],
        shuffle=True, num_workers=dc.get("num_workers", 0),
    )
    val_loader = create_spatial_dataloader(
        val_dataset, batch_size=tc["batch_size"],
        shuffle=False, num_workers=dc.get("num_workers", 0),
    )

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"],
    )
    total_steps = len(train_loader) * tc["num_epochs"]
    scheduler = get_scheduler(
        optimizer, tc.get("scheduler_type", "cosine"),
        num_training_steps=total_steps, warmup_steps=tc.get("warmup_steps", 100),
    )

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    patience = tc.get("patience", 5)

    print(f"\nTraining for {tc['num_epochs']} epochs...")
    print(f"  Batch size: {tc['batch_size']}")
    print(f"  Learning rate: {tc['learning_rate']}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print()

    for epoch in range(tc["num_epochs"]):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, config, device, epoch)
        val_loss = evaluate(model, val_loader, config, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1}/{tc['num_epochs']} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_path = os.path.join(config["logging"]["checkpoint_dir"], "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": val_loss,
                    "config": config,
                },
                ckpt_path,
            )
            print(f"  Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping after {patience} epochs without improvement")
                break

        if config["logging"].get("save_every") and (epoch + 1) % config["logging"]["save_every"] == 0:
            ckpt_path = os.path.join(
                config["logging"]["checkpoint_dir"], f"checkpoint_epoch{epoch+1}.pt"
            )
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "loss": val_loss, "config": config},
                ckpt_path,
            )

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
