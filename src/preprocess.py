"""Dataset preprocessing & DataLoader utilities with trial-mode support."""
from typing import Tuple

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase

CACHE_DIR = ".cache/"


class GSM8KCollator(DataCollatorForLanguageModeling):
    """Causal-LM collator that also keeps the raw answer string for accuracy eval."""
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        # Ensure tokenizer has a padding token set for proper batch padding
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        super().__init__(tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8)

    def __call__(self, examples):
        answers = [ex.pop("answer") for ex in examples]
        batch = super().__call__(examples)
        batch["answers"] = answers
        return batch


def _tokenise(example, tokenizer: PreTrainedTokenizerBase, cfg):
    prompt = f"Question: {example['question']}\nAnswer: "
    full_text = prompt + example["answer"]
    # Do not pad here - let the collator handle padding during batching
    enc = tokenizer(full_text, truncation=True, max_length=cfg.dataset.max_length, padding=False)
    enc["labels"] = enc["input_ids"].copy()
    enc["answer"] = example["answer"]
    return enc


def build_dataloaders(cfg, tokenizer: PreTrainedTokenizerBase) -> Tuple[DataLoader, DataLoader]:
    """Returns train & validation DataLoaders; truncates dataset when cfg.mode=='trial'."""
    ds = load_dataset(cfg.dataset.name, cfg.dataset.config, cache_dir=CACHE_DIR)

    train_ds = ds["train"].map(lambda e: _tokenise(e, tokenizer, cfg), remove_columns=ds["train"].column_names)
    val_ds = ds["test"].map(lambda e: _tokenise(e, tokenizer, cfg), remove_columns=ds["test"].column_names)

    # Trial-mode: restrict dataset size to speed up CI validation
    if cfg.mode == "trial":
        subset_sz = min(len(train_ds), 8)  # at most 8 examples ≈ 1–2 batches
        train_ds = train_ds.select(list(range(subset_sz)))
        val_ds = val_ds.select(list(range(min(len(val_ds), 8))))
        train_short, val_short = True, True
    else:
        train_short = val_short = False

    collator = GSM8KCollator(tokenizer)
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collator)

    # Tag loaders so evaluation can early-exit if needed
    train_loader._trial_short = train_short  # type: ignore[attr-defined]
    val_loader._trial_short = val_short   # type: ignore[attr-defined]
    return train_loader, val_loader