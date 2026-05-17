"""Weights & Biases integration for PINN training."""

from __future__ import annotations

from typing import Any


class WandbLogger:
    def __init__(self, cfg: dict | None):
        self.cfg = cfg or {}
        self.enabled = bool(self.cfg.get("enabled", False))
        self.run = None

    def start(self, config: dict[str, Any], run_name: str | None = None):
        if not self.enabled:
            return
        import wandb

        self.run = wandb.init(
            project=self.cfg.get("project", "pinn-laser-heating"),
            entity=self.cfg.get("entity"),
            name=run_name or self.cfg.get("run_name"),
            tags=self.cfg.get("tags", []),
            config=config,
            mode=self.cfg.get("mode", "online"),
        )

    def log_losses(self, step: int, losses: list[float], phase: str = "train"):
        if not self.enabled or self.run is None:
            return
        import wandb

        payload = {f"{phase}/total": sum(losses)}
        for i, v in enumerate(losses):
            payload[f"{phase}/loss_{i}"] = v
        wandb.log(payload, step=step)

    def log_scalar(self, key: str, value: float, step: int | None = None):
        if not self.enabled or self.run is None:
            return
        import wandb

        wandb.log({key: value}, step=step)

    def log_artifact(self, path: str, name: str, type_: str = "model"):
        if not self.enabled or self.run is None:
            return
        import wandb

        art = wandb.Artifact(name, type=type_)
        art.add_file(path)
        self.run.log_artifact(art)

    def finish(self):
        if self.run is not None:
            self.run.finish()
            self.run = None
