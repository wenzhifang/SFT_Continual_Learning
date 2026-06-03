"""Training loop, optimizer setup, and checkpoint saving."""

from __future__ import annotations

import json
import math
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import LORA_TARGET_MODULES
from data_preparation import build_dataloader
from model import build_model_and_tokenizer


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model,
    dataloader: DataLoader,
    optimizer: torch.optim.Adam,
    device: torch.device,
    epoch: int,
    log_every: int,
) -> float:
    model.train()
    total_loss = 0.0
    num_steps = 0

    progress = tqdm(dataloader, desc=f"epoch {epoch}", leave=True)
    for step, batch in enumerate(progress, start=1):
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        loss_value = float(loss.item())
        total_loss += loss_value
        num_steps += 1
        progress.set_postfix(loss=f"{loss_value:.4f}")

        if log_every > 0 and step % log_every == 0:
            print(f"[epoch {epoch} step {step}] loss={loss_value:.4f}")

    return total_loss / max(num_steps, 1)


def save_adapter(model, tokenizer, output_dir: Path, extra: dict[str, Any] | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    if extra is not None:
        (output_dir / "train_meta.json").write_text(json.dumps(extra, indent=2) + "\n", encoding="utf-8")


def run_training(args: Namespace) -> None:
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    train_path = data_dir / args.train_file
    if not train_path.exists():
        raise FileNotFoundError(f"Missing training file: {train_path}")

    output_dir = Path(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: CUDA not available; training on CPU will be very slow.")

    model, tokenizer = build_model_and_tokenizer(
        args.model_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bf16=args.bf16,
    )
    model.to(device)

    dataset, dataloader = build_dataloader(
        train_path,
        tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = math.ceil(len(dataset) / args.batch_size)
    print(
        f"Training samples: {len(dataset)} | batch_size: {args.batch_size} | "
        f"steps/epoch: {steps_per_epoch} | epochs: {args.epochs}"
    )

    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            log_every=args.log_every,
        )
        history.append({"epoch": epoch, "avg_loss": avg_loss})
        print(f"Epoch {epoch} finished | avg_loss={avg_loss:.4f}")

        if args.save_every_epoch:
            epoch_dir = output_dir / f"epoch_{epoch}"
            save_adapter(
                model,
                tokenizer,
                epoch_dir,
                extra={"epoch": epoch, "avg_loss": avg_loss, "args": vars(args)},
            )

    save_adapter(
        model,
        tokenizer,
        output_dir,
        extra={
            "task": "ARC-Challenge",
            "train_file": str(train_path),
            "num_samples": len(dataset),
            "history": history,
            "lora_target_modules": LORA_TARGET_MODULES,
            "args": vars(args),
        },
    )
    print(f"Saved LoRA adapter to {output_dir}")
