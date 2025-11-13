"""Model builder: 4-bit QLoRA (PEFT) with optional LoRA injection."""
import logging
from typing import List, Tuple

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)
CACHE_DIR = ".cache/"


def build_model(cfg):
    model_name = cfg.model.get("init_checkpoint", cfg.model.name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=CACHE_DIR,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant_cfg,
    )

    if cfg.model.lora.enabled:
        lora_cfg = LoraConfig(
            r=cfg.model.lora.rank,
            lora_alpha=cfg.model.lora.alpha,
            lora_dropout=cfg.model.lora.dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        # Mark LoRA parameters for adaptive LR routines
        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.is_lora = True
    return model, tokenizer


def collect_lora_parameters(model) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    lora, base = [], []
    for p in model.parameters():
        (lora if getattr(p, "is_lora", False) else base).append(p)
    return lora, base