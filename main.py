#!/usr/bin/env python3
"""Entry point for ARC-Challenge LoRA SFT."""

from __future__ import annotations

import argparse

from config import DEFAULT_DATA_DIR, DEFAULT_MODEL, DEFAULT_OUTPUT_DIR
from train import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT on ARC-Challenge commonsense training JSON.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL, help="HF causal LM checkpoint.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing train.json.",
    )
    parser.add_argument("--train-file", type=str, default="train.json", help="Training JSON filename.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="LoRA adapter save dir.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay.")
    parser.add_argument("--max-length", type=int, default=1024, help="Max sequence length after tokenization.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes.")
    parser.add_argument("--log-every", type=int, default=10, help="Log training loss every N steps.")
    parser.add_argument("--save-every-epoch", action="store_true", help="Save adapter after each epoch.")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True, help="Train in bfloat16.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
