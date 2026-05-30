"""
src/training/trainer.py
Training entry point. Loads config, builds model, runs HuggingFace Trainer.
Usage: python -m src.training.trainer --config configs/train_config.yaml
"""

import argparse
import logging
import os
from functools import partial

import torch
import yaml
from transformers import (
    Trainer,
    TrainingArguments,
)

from ..data.loader import SpatialQADataset
from ..data.tokenizer import SpatialTokenizer
from ..models.llm_wrapper import SpatialLLM

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def collate_fn(batch: list[dict], spatial_tokenizer: SpatialTokenizer) -> dict:
    """Custom collator: tokenizes text + stacks coords and pixel_values."""
    tokenized = [
        spatial_tokenizer.encode_spatial(
            question=item["question"],
            lat=item["coords"][0].item(),
            lon=item["coords"][1].item(),
            answer=item["answer"],
        )
        for item in batch
    ]

    keys = list(tokenized[0].keys())
    result = {k: torch.stack([t[k] for t in tokenized]) for k in keys}

    result["coords"] = torch.stack([item["coords"] for item in batch])

    if "pixel_values" in batch[0]:
        result["pixel_values"] = torch.stack([item["pixel_values"] for item in batch])

    return result


def main(config_path: str):
    # Avoid interactive wandb login prompt in notebooks/Colab.
    # Training reports to wandb only if a project is set AND WANDB_API_KEY exists.
    os.environ.setdefault("WANDB_SILENT", "true")
    if not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_DISABLED"] = "true"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # ── Tokenizer ──────────────────────────────────────────────────────
    spatial_tok = SpatialTokenizer(
        model_name=cfg["model"]["base_llm"],
        max_length=cfg["data"]["max_text_length"],
    )

    # ── Datasets ───────────────────────────────────────────────────────
    train_ds = SpatialQADataset(
        jsonl_path=cfg["data"]["train_path"],
        tile_dir=cfg["data"]["tile_dir"],
        max_text_length=cfg["data"]["max_text_length"],
    )
    val_ds = SpatialQADataset(
        jsonl_path=cfg["data"]["val_path"],
        tile_dir=cfg["data"]["tile_dir"],
        max_text_length=cfg["data"]["max_text_length"],
    )

    collator = partial(collate_fn, spatial_tokenizer=spatial_tok)

    # ── Model ──────────────────────────────────────────────────────────
    model = SpatialLLM(
        base_llm=cfg["model"]["base_llm"],
        vit_model_name=cfg["model"]["vit_backbone"],
        coord_embed_dim=cfg["model"]["coord_embed_dim"],
        coord_num_freqs=cfg["model"]["coord_num_freqs"],
        fusion_num_heads=cfg["model"]["fusion_num_heads"],
        lora_r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_target_modules=cfg["lora"]["target_modules"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        load_in_4bit=cfg["model"].get("load_in_4bit", False),
        use_place_memory=cfg["model"].get("use_place_memory", True),
        use_predictive_coding=cfg["model"].get("use_predictive_coding", True),
        use_neuromodulation=cfg["model"].get("use_neuromodulation", True),
    )
    model.llm.print_trainable_parameters()

    # ── Memory safety for T4: gradient checkpointing on the LLM backbone ──
    # Re-computes activations during backward instead of storing them all,
    # cutting peak activation memory by ~60-70%. Essential to avoid OOM spikes.
    if hasattr(model.llm, "gradient_checkpointing_enable"):
        model.llm.gradient_checkpointing_enable()
        if hasattr(model.llm, "enable_input_require_grads"):
            model.llm.enable_input_require_grads()
        if hasattr(model.llm, "config"):
            model.llm.config.use_cache = False   # incompatible with checkpointing
        logger.info("Gradient checkpointing enabled (T4 memory safety)")

    # ── Training args ──────────────────────────────────────────────────
    t_cfg = cfg["training"]
    ta_kwargs = dict(
        output_dir=t_cfg["output_dir"],
        num_train_epochs=t_cfg["num_epochs"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=t_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=t_cfg["learning_rate"],
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        warmup_ratio=t_cfg["warmup_ratio"],
        weight_decay=t_cfg["weight_decay"],
        fp16=t_cfg["fp16"],
        logging_steps=t_cfg["logging_steps"],
        eval_steps=t_cfg["eval_steps"],
        save_steps=t_cfg["save_steps"],
        save_total_limit=t_cfg["save_total_limit"],
        optim=t_cfg.get("optim", "paged_adamw_8bit"),
        eval_strategy="steps",
        remove_unused_columns=False,
        report_to="wandb" if (cfg["wandb"]["project"] and os.environ.get("WANDB_API_KEY")) else "none",
        run_name=cfg["training"]["output_dir"].split("/")[-1],
    )
    # save_safetensors only exists in newer transformers. Qwen ties
    # embed_tokens<->lm_head, which safetensors refuses to serialize, so disable
    # it when available; older versions default to torch.save anyway.
    import inspect as _inspect
    if "save_safetensors" in _inspect.signature(TrainingArguments.__init__).parameters:
        ta_kwargs["save_safetensors"] = False
    training_args = TrainingArguments(**ta_kwargs)

    if cfg["wandb"]["project"] and os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_PROJECT"] = cfg["wandb"]["project"]
        if cfg["wandb"]["entity"]:
            os.environ["WANDB_ENTITY"] = cfg["wandb"]["entity"]

    # ── Trainer ────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    logger.info("Starting training...")
    trainer.train()
    trainer.save_model()
    logger.info(f"Model saved to {t_cfg['output_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to train_config.yaml")
    args = parser.parse_args()
    main(args.config)
