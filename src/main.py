import subprocess
import sys
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path, get_original_cwd
from omegaconf import OmegaConf

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

@hydra.main(config_path="../config", config_name="config")
def main(cfg):
    # Compute the config path relative to this script file
    script_dir = Path(__file__).parent.resolve()
    run_cfg = script_dir.parent / "config" / "runs" / f"{cfg.run}.yaml"
    if not run_cfg.exists():
        raise FileNotFoundError(str(run_cfg))

    # Build override list for the child Hydra process (src.train)
    overrides = [
        f"run={cfg.run}",
        f"results_dir={to_absolute_path(cfg.results_dir)}",
        f"mode={cfg.mode}",
    ]
    if cfg.mode == "trial":
        overrides += [
            "wandb.mode=disabled",
            "optuna.n_trials=0",
            "training.epochs=1",
            "training.max_steps=2",
        ]
    elif cfg.mode == "full":
        overrides += ["wandb.mode=online"]
    else:
        raise ValueError("mode must be 'trial' or 'full'")

    # Get the absolute path to train.py
    train_script = Path(__file__).parent / "train.py"
    cmd = [sys.executable, "-u", str(train_script)] + overrides
    print("[main] Launching training subprocess:\n  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()