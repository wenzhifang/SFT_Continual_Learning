"""Load and tokenize ARC-Challenge SFT examples with left padding."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Dataset


def row_to_user_content(row: dict[str, Any]) -> str:
    instruction = str(row["instruction"])
    extra_input = str(row.get("input", "")).strip()
    if extra_input:
        return f"{instruction}\n{extra_input}"
    return instruction


@dataclass(frozen=True)
class EncodedSample:
    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]


class ARCChallengeSFTDataset(Dataset):
    def __init__(self, json_path: Path, tokenizer, max_length: int) -> None:
        rows = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"Expected a JSON list in {json_path}")
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> EncodedSample:
        row = self.rows[index]
        user_content = row_to_user_content(row)
        assistant_content = str(row["output"])

        prompt_messages = [{"role": "user", "content": user_content}]
        full_messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        full_ids = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        if len(full_ids) < len(prompt_ids):
            raise ValueError(
                f"Full sequence shorter than prompt for index {index}: "
                f"prompt_len={len(prompt_ids)}, full_len={len(full_ids)}"
            )

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        attention_mask = [1] * len(full_ids)

        return EncodedSample(
            input_ids=full_ids,
            labels=labels,
            attention_mask=attention_mask,
        )


def left_pad_collate(
    batch: list[EncodedSample],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    max_len = max(len(sample.input_ids) for sample in batch)

    input_ids_batch: list[list[int]] = []
    labels_batch: list[list[int]] = []
    attention_mask_batch: list[list[int]] = []

    for sample in batch:
        pad_len = max_len - len(sample.input_ids)
        input_ids_batch.append([pad_token_id] * pad_len + sample.input_ids)
        labels_batch.append([-100] * pad_len + sample.labels)
        attention_mask_batch.append([0] * pad_len + sample.attention_mask)

    return {
        "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
        "labels": torch.tensor(labels_batch, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
    }


def build_dataloader(
    train_path: Path,
    tokenizer,
    *,
    max_length: int,
    batch_size: int,
    num_workers: int,
) -> tuple[ARCChallengeSFTDataset, DataLoader]:
    dataset = ARCChallengeSFTDataset(train_path, tokenizer, max_length=max_length)
    collate_fn: Callable = lambda batch: left_pad_collate(batch, pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return dataset, dataloader
