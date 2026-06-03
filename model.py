"""Build causal LM with LoRA adapters and left-padded tokenizer."""

from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import LORA_TARGET_MODULES


def build_model_and_tokenizer(
    model_path: str,
    *,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    bf16: bool,
):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer
