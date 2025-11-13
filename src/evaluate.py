"""Aggregated evaluation & visualisation script.

CLI (exact spec):
    uv run python -m src.evaluate results_dir={path} run_ids='["run1", "run2"]'

The arguments are provided as *key=value* pairs (without leading dashes)."""
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from omegaconf import OmegaConf
from scipy import stats
from sklearn.metrics import confusion_matrix

sns.set_theme(style="whitegrid")
PRIMARY_METRIC = "test_acc"

###############################################################################
# CLI parsing helpers (accept key=value pairs)
###############################################################################

def _parse_cli(argv: List[str]) -> Namespace:
    kv = {}
    for item in argv:
        if "=" not in item:
            raise SystemExit(f"Malformed argument '{item}'. Use key=value style.")
        key, val = item.split("=", 1)
        kv[key] = val
    if "results_dir" not in kv or "run_ids" not in kv:
        raise SystemExit("Both results_dir=... and run_ids=[...] must be provided.")
    return Namespace(results_dir=kv["results_dir"], run_ids=kv["run_ids"])

###############################################################################
# Utility functions
###############################################################################

def _root_cfg() -> Dict:
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    return OmegaConf.load(cfg_path)


def _save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def _plot_learning_curve(hist: pd.DataFrame, run_id: str, out_dir: Path) -> Path:
    fig_path = out_dir / f"{run_id}_learning_curve.pdf"
    plt.figure(figsize=(6, 4))
    if "train_loss" in hist.columns:
        sns.lineplot(x=hist.index, y=hist["train_loss"], label="train_loss")
    if "val_acc" in hist.columns:
        sns.lineplot(x=hist.index, y=hist["val_acc"], label="val_acc")
    plt.legend(); plt.title(run_id); plt.tight_layout(); plt.savefig(fig_path); plt.close()
    return fig_path


def _plot_confusion_matrix(labels: List[str], preds: List[str], run_id: str, out_dir: Path) -> Path:
    cm_path = out_dir / f"{run_id}_confusion_matrix.pdf"
    classes = sorted(set(labels + preds))
    cm = confusion_matrix(labels, preds, labels=classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.ylabel("True"); plt.xlabel("Pred"); plt.title(f"Confusion – {run_id}"); plt.tight_layout(); plt.savefig(cm_path); plt.close()
    return cm_path


def _auc_vs_wallclock(hist: pd.DataFrame) -> float:
    if {"val_acc", "wallclock_sec"}.issubset(hist.columns):
        return float(np.trapz(hist["val_acc"], hist["wallclock_sec"])) / hist["wallclock_sec"].max()
    return float("nan")

###############################################################################
# Main evaluation workflow
###############################################################################

def main():
    args = _parse_cli(sys.argv[1:])
    results_dir = Path(args.results_dir).expanduser()
    results_dir.mkdir(parents=True, exist_ok=True)
    run_ids: List[str] = json.loads(args.run_ids)

    root_cfg = _root_cfg(); entity, project = root_cfg.wandb.entity, root_cfg.wandb.project
    api = wandb.Api()

    aggregated: Dict[str, Dict[str, float]] = {}  # metric -> {run_id: value}
    primary_vals: Dict[str, float] = {}
    generated_paths: List[Path] = []

    for rid in run_ids:
        run = api.run(f"{entity}/{project}/{rid}")
        hist = run.history()  # full history
        summary = run.summary._json_dict
        conf = dict(run.config)

        run_dir = results_dir / rid
        run_dir.mkdir(parents=True, exist_ok=True)

        metrics_file = run_dir / "metrics.json"
        _save_json({"history": hist.to_dict(orient="list"), "summary": summary, "config": conf}, metrics_file)
        generated_paths.append(metrics_file)

        # Individual plots
        generated_paths.append(_plot_learning_curve(hist, rid, run_dir))
        if isinstance(summary.get("ground_truths"), list) and isinstance(summary.get("predictions"), list):
            generated_paths.append(_plot_confusion_matrix(summary["ground_truths"], summary["predictions"], rid, run_dir))

        # Collect scalar metrics from summary
        for k, v in summary.items():
            if isinstance(v, (int, float)):
                aggregated.setdefault(k, {})[rid] = v
        aggregated.setdefault("dev_accuracy_auc", {})[rid] = _auc_vs_wallclock(hist)

        if PRIMARY_METRIC in summary:
            primary_vals[rid] = summary[PRIMARY_METRIC]

    # Aggregated comparison ---------------------------------------------------
    cmp_dir = results_dir / "comparison"; cmp_dir.mkdir(parents=True, exist_ok=True)

    best_prop = max(((k, v) for k, v in primary_vals.items() if "proposed" in k or "basicalr" in k.lower()), key=lambda x: x[1], default=(None, float("nan")))
    best_base = max(((k, v) for k, v in primary_vals.items() if "baseline" in k.lower() or "comparative" in k), key=lambda x: x[1], default=(None, float("nan")))
    gap = None
    if best_prop[1] and best_base[1]:
        gap = (best_prop[1] - best_base[1]) / best_base[1] * 100.0

    aggregated_json = {
        "primary_metric": "(1) Test exact-match accuracy.  (2) Median dev-accuracy AUC vs wall-clock.  (3) % of adapters whose LR was up-scaled vs down-scaled, and final per-adapter norm CV.",
        "metrics": aggregated,
        "best_proposed": {"run_id": best_prop[0], "value": best_prop[1]},
        "best_baseline": {"run_id": best_base[0], "value": best_base[1]},
        "gap": gap,
    }
    agg_file = cmp_dir / "aggregated_metrics.json"; _save_json(aggregated_json, agg_file); generated_paths.append(agg_file)

    # Bar chart of primary metric
    bar_path = cmp_dir / "comparison_accuracy_bar_chart.pdf"
    plt.figure(figsize=(7, 4))
    sns.barplot(x=list(primary_vals.keys()), y=list(primary_vals.values()))
    for idx, v in enumerate(primary_vals.values()):
        plt.text(idx, v, f"{v:.3f}", ha="center", va="bottom")
    plt.xticks(rotation=45, ha="right"); plt.ylabel(PRIMARY_METRIC); plt.tight_layout(); plt.savefig(bar_path); plt.close()
    generated_paths.append(bar_path)

    # Significance test if enough runs per group
    prop_vals = [v for k, v in primary_vals.items() if "proposed" in k or "basicalr" in k.lower()]
    base_vals = [v for k, v in primary_vals.items() if "baseline" in k.lower() or "comparative" in k]
    if len(prop_vals) >= 2 and len(base_vals) >= 2:
        t, p = stats.ttest_ind(prop_vals, base_vals, equal_var=False)
        sig_file = cmp_dir / "significance_test.json"
        _save_json({"welch_t": t, "p_value": p}, sig_file)
        generated_paths.append(sig_file)

    # Print all generated artefact paths (required)
    for pth in generated_paths:
        print(pth)


if __name__ == "__main__":
    main()