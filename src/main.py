import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig


def _get_run_id(cfg: DictConfig) -> str:
    if "run" in cfg and cfg.run is not None:
        run_id = cfg.run.get("run_id")
        if run_id:
            return str(run_id)
    run_id = cfg.get("run_id")
    if run_id:
        return str(run_id)
    raise ValueError("run must be provided via CLI: run=<run_id>")


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    run_id = _get_run_id(cfg)
    if not cfg.get("results_dir"):
        raise ValueError("results_dir must be provided via CLI: results_dir=<path>")

    if cfg.mode not in {"trial", "full"}:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    overrides = [f"run={run_id}", f"results_dir={cfg.results_dir}", f"mode={cfg.mode}"]
    if cfg.mode == "trial":
        overrides.extend([
            "wandb.mode=disabled",
            "trial_max_batches=2",
            "trial_max_examples=2",
            "run.training.epochs=1",
        ])
        if "optuna" in cfg.run and cfg.run.optuna is not None:
            overrides.append("run.optuna.n_trials=0")
    elif cfg.mode == "full":
        overrides.append("wandb.mode=online")

    cmd = [sys.executable, "-u", "-m", "src.train"] + overrides
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
