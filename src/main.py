import subprocess
import sys
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path, get_original_cwd

@hydra.main(config_path="../config", config_name="config")
def main(cfg):
    run_cfg = Path(get_original_cwd()) / "config" / "runs" / f"{cfg.run}.yaml"
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