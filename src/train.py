import logging
import math
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import optuna
import torch
from hydra.utils import to_absolute_path, get_original_cwd
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

import wandb  # mandatory logging backend

# Add parent directory to path to enable absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

# Local modules
from src.model import build_model, collect_lora_parameters
from src.preprocess import build_dataloaders

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger("train")

# Register Optuna resolver for OmegaConf to handle ${optuna:param_name} interpolations
# This resolver provides default values that will be overridden during Optuna sweep
def _optuna_resolver(param_name: str) -> float:
    """Default values for optuna hyperparameters before sweep."""
    defaults = {
        "lora_rank": 16,
        "batch_size": 32,
        "base_learning_rate": 1e-4,
        "basicalr.R_star": 0.3,
        "basicalr.update_interval": 32,
    }
    return defaults.get(param_name, 0.0)

OmegaConf.register_new_resolver("optuna", _optuna_resolver, replace=True)

###############################################################################
# BaSiCALR (proposed) & HiNoALR (baseline) helpers
###############################################################################

def _apply_basicalr(lora_param_groups: List[Dict], cfg, global_step: int, base_lr: float):
    """Bidirectional LR tuner – executed every cfg.basicalr.update_interval steps."""
    if not cfg.get("basicalr") or not cfg.basicalr.enabled:
        return
    if (global_step + 1) % cfg.basicalr.update_interval != 0:
        return

    kappa = cfg.basicalr.kappa
    R_star = cfg.basicalr.R_star
    clip_min = cfg.basicalr.lr_clip_min_factor * base_lr
    clip_max = cfg.basicalr.lr_clip_max_factor * base_lr

    for group in lora_param_groups:
        p = group["params"][0]
        state = group["opt_state"]
        if p.grad is None or "exp_avg" not in state:
            continue
        same_sign = (torch.sign(p.grad).mean() * torch.sign(state["exp_avg"].mean())) >= 0
        if same_sign:
            p.c_pos += 1
        else:
            p.c_neg += 1
        # Beta-posterior reliability & LR scaling ratio
        p_hat = (1.0 + p.c_pos.item()) / (2.0 + p.c_pos.item() + p.c_neg.item())
        R_t = 2.0 * abs(p_hat - 0.5)
        rho = math.exp(kappa * (R_t - R_star))
        new_lr = max(min(group["lr"] * rho, clip_max), clip_min)
        group["lr"] = new_lr
        p.lr = float(new_lr)  # for logging / analysis
        # Reset counters if near-neutral update
        if abs(rho - 1.0) < cfg.basicalr.reset_threshold:
            p.c_pos.zero_(); p.c_neg.zero_()


def _apply_hinoalr(lora_param_groups: List[Dict], cfg):
    """One-directional LR shrinker (baseline HiNoALR)."""
    if not cfg.get("hinoalr") or not cfg.hinoalr.enabled:
        return
    window = cfg.hinoalr.observe_window
    for group in lora_param_groups:
        p = group["params"][0]
        state = group["opt_state"]
        if p.grad is None or "exp_avg" not in state:
            continue
        same_sign = (torch.sign(p.grad).mean() * torch.sign(state["exp_avg"].mean())) >= 0
        if not hasattr(p, "_window"):
            p._window = []
        p._window.append(int(not same_sign))  # 1 if opposite sign
        if len(p._window) > window:
            p._window.pop(0)
        if (not getattr(p, "_shrunk", False)) and (sum(p._window) / len(p._window) > cfg.hinoalr.trigger_threshold):
            group["lr"] *= cfg.hinoalr.downscale_factor
            p.lr = float(group["lr"])
            p._shrunk = True

###############################################################################
# Logging helpers
###############################################################################

def _init_wandb(cfg):
    if cfg.wandb.mode == "disabled":
        return None
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        id=cfg.run_id,
        resume="allow",
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    print(f"[wandb] URL: {run.get_url()}")
    return run


def _evaluate(model, tokenizer, loader: DataLoader, device: torch.device, return_preds=False):
    import re
    model.eval()
    correct, total = 0, 0
    preds, gts = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            # Evaluation shortening in trial-mode – dataloader already truncated but double-check
            if getattr(loader, "_trial_short", False) and i >= 2:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=32)
            out_ids = generated[:, input_ids.size(1):]
            pred_texts = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
            gt_texts = batch["answers"]
            for p_txt, g_txt in zip(pred_texts, gt_texts):
                p_num = re.findall(r"(-?\d+(?:\.\d+)?)", p_txt.strip())
                g_num = re.findall(r"(-?\d+(?:\.\d+)?)", g_txt.strip())
                if p_num and g_num and p_num[-1] == g_num[-1]:
                    correct += 1
                total += 1
            preds.extend(pred_texts)
            gts.extend(gt_texts)
    acc = correct / max(total, 1)
    model.train()
    if return_preds:
        return acc, preds, gts
    return acc

###############################################################################
# Optimiser & scheduler builders
###############################################################################

def _build_optimizer(model, cfg):
    lora_params, base_params = collect_lora_parameters(model)
    param_groups = []
    for p in lora_params:
        param_groups.append({"params": [p], "lr": cfg.training.base_learning_rate})
    if base_params:
        param_groups.append({"params": base_params, "lr": cfg.training.base_learning_rate})

    optimizer = AdamW(
        param_groups,
        lr=cfg.training.base_learning_rate,
        betas=(cfg.training.beta1, cfg.training.beta2),
        eps=cfg.training.eps,
        weight_decay=cfg.training.weight_decay,
    )

    # Attach optimiser state pointer + per-adapter counters
    for group in optimizer.param_groups:
        for p in group["params"]:
            group["opt_state"] = optimizer.state[p]
            if getattr(p, "is_lora", False):
                p.c_pos = torch.zeros((), dtype=torch.int16)
                p.c_neg = torch.zeros((), dtype=torch.int16)
                p.lr = cfg.training.base_learning_rate
    return optimizer, param_groups

###############################################################################
# Core training run
###############################################################################

def _single_run(cfg) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = build_model(cfg)
    model.to(device)

    train_loader, dev_loader = build_dataloaders(cfg, tokenizer)
    optimizer, param_groups = _build_optimizer(model, cfg)

    total_steps = cfg.training.max_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.training.warmup_steps, total_steps)

    use_fp16 = cfg.training.mixed_precision == "fp16"
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16) if torch.cuda.is_available() else torch.amp.GradScaler('cpu', enabled=False)

    wandb_run = _init_wandb(cfg)

    base_lr = cfg.training.base_learning_rate
    grad_acc = cfg.training.get("gradient_accumulation_steps", 1)
    global_step, micro_step = 0, 0
    t0 = time.time()

    wall_hist, acc_hist = [], []

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(cfg.training.epochs):
        for batch_idx, batch in enumerate(train_loader):
            if cfg.mode == "trial" and global_step >= 2:
                break

            with torch.cuda.amp.autocast(enabled=use_fp16):
                outputs = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
                loss = outputs.loss / grad_acc
            scaler.scale(loss).backward()
            micro_step += 1

            if micro_step % grad_acc == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                lora_groups = [g for g in param_groups if len(g["params"]) == 1 and getattr(g["params"][0], "is_lora", False)]
                _apply_basicalr(lora_groups, cfg, global_step, base_lr)
                _apply_hinoalr(lora_groups, cfg)

                if wandb_run is not None:
                    wandb.log({
                        "train_loss": loss.item() * grad_acc,
                        "lr": scheduler.get_last_lr()[0],
                        "step": global_step,
                        "wallclock_sec": time.time() - t0,
                    }, step=global_step)

                # periodic validation
                if ((global_step + 1) % cfg.training.eval_interval_steps == 0) or (cfg.mode == "trial" and global_step == 0):
                    val_acc = _evaluate(model, tokenizer, dev_loader, device)
                    wall_hist.append(time.time() - t0)
                    acc_hist.append(val_acc)
                    if wandb_run is not None:
                        wandb.log({"val_acc": val_acc}, step=global_step)

                global_step += 1
                if global_step >= total_steps:
                    break
        if global_step >= total_steps or (cfg.mode == "trial" and global_step >= 2):
            break

    test_acc, preds, gts = _evaluate(model, tokenizer, dev_loader, device, return_preds=True)
    if len(acc_hist) >= 2:
        auc = float(np.trapz(acc_hist, wall_hist) / wall_hist[-1])
    elif acc_hist:
        auc = acc_hist[0]
    else:
        auc = test_acc

    lora_params, _ = collect_lora_parameters(model)
    up_scaled = sum(float(p.lr > base_lr) for p in lora_params)
    down_scaled = sum(float(p.lr < base_lr) for p in lora_params)
    norms = torch.tensor([p.data.norm().item() for p in lora_params])
    norm_cv = float(norms.std() / norms.mean()) if norms.mean() > 0 else 0.0

    metrics = {
        "test_acc": test_acc,
        "dev_accuracy_auc": auc,
        "up_scaled_pct": up_scaled / max(1, len(lora_params)) * 100,
        "down_scaled_pct": down_scaled / max(1, len(lora_params)) * 100,
        "adapter_norm_cv": norm_cv,
    }

    if wandb_run is not None:
        for k, v in metrics.items():
            wandb_run.summary[k] = v
        wandb_run.summary["predictions"] = preds
        wandb_run.summary["ground_truths"] = gts
        wandb_run.finish()

    return metrics

###############################################################################
# Optuna sweep (offline ‑ only best hyper-params logged)
###############################################################################

def _sample(space: Dict, trial: optuna.trial.Trial):
    params = {}
    for name, spec in space.items():
        ptype = spec["type"].lower()
        if ptype == "loguniform":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
        elif ptype == "uniform":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=False)
        elif ptype == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unsupported space type {ptype}")
    return params


def _run_optuna(cfg):
    base_cfg = deepcopy(cfg)

    def objective(trial: optuna.trial.Trial):
        sampled = _sample(base_cfg.optuna.search_space, trial)
        trial_cfg = deepcopy(base_cfg)
        for k, v in sampled.items():
            OmegaConf.update(trial_cfg, k, v, merge=True)
        trial_cfg.wandb.mode = "disabled"
        trial_cfg.training.epochs = 1
        trial_cfg.training.max_steps = min(trial_cfg.training.eval_interval_steps, 128)
        res = _single_run(trial_cfg)
        return res[base_cfg.optuna.metric_name]

    study = optuna.create_study(direction=cfg.optuna.direction)
    study.optimize(objective, n_trials=cfg.optuna.n_trials)
    logger.info("Optuna best value %.4f params %s", study.best_value, study.best_params)
    for k, v in study.best_params.items():
        OmegaConf.update(cfg, k, v, merge=True)

###############################################################################
# Hydra entry point
###############################################################################

@hydra.main(config_path="../config", config_name="config")
def main(cfg):
    # Compute the config path relative to this script file
    script_dir = Path(__file__).parent.resolve()
    run_yaml = script_dir.parent / "config" / "runs" / f"{cfg.run}.yaml"
    if not run_yaml.exists():
        raise FileNotFoundError(str(run_yaml))
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, OmegaConf.load(run_yaml))
    OmegaConf.set_struct(cfg, True)

    # Switches for trial / full modes
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.epochs = 1
        cfg.training.max_steps = 2
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError("mode must be 'trial' or 'full'")

    Path(to_absolute_path(cfg.results_dir)).mkdir(parents=True, exist_ok=True)

    if cfg.get("optuna") and cfg.optuna.n_trials:
        _run_optuna(cfg)

    final_metrics = _single_run(cfg)
    logger.info("Run %s finished – metrics: %s", cfg.run_id, final_metrics)


if __name__ == "__main__":
    main()