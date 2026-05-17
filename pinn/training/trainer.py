"""Training loop with optional WandB logging and multi-phase optimizers."""

from __future__ import annotations

from pathlib import Path

import deepxde as dde
import numpy as np

from pinn.models.constraints import apply_hard_constraint
from pinn.models.networks import build_network
from pinn.physics.domain import feature_transform
from pinn.training.wandb_logger import WandbLogger


class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.wandb = WandbLogger(cfg.get("wandb"))
        self.output_dir = Path(cfg.get("output_dir", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_model(self, data, *, external_trainable_variables=None):
        cfg = self.cfg
        phys = cfg["physics"]
        net = build_network(cfg["network"]["architecture"], cfg["network"])
        net.apply_feature_transform(
            feature_transform(phys["Lx"], phys["Ly"], phys["Lz"], phys["T_end"])
        )
        apply_hard_constraint(
            net,
            cfg["network"].get("hard_constraint", "ramp_softplus"),
            T0=phys["T0"],
            pulse_sigma=phys["pulse_sigma"],
        )
        model = dde.Model(data, net)
        return model

    def train(
        self,
        model: dde.Model,
        *,
        external_trainable_variables=None,
    ) -> tuple[list, object]:
        train_cfg = self.cfg["training"]
        self.wandb.start(self.cfg, run_name=train_cfg.get("run_name"))

        all_history = []
        state = None
        global_step = 0

        phases = train_cfg.get("phases", [{"optimizer": "adam", "lr": 1e-3, "iterations": 10000}])
        loss_weights = train_cfg.get("loss_weights")

        for phase in phases:
            opt = phase["optimizer"]
            lr = phase.get("lr", 1e-3)
            iters = phase["iterations"]
            compile_kw = dict(
                external_trainable_variables=external_trainable_variables,
            )
            if loss_weights is not None:
                compile_kw["loss_weights"] = loss_weights

            if opt.lower() == "adam":
                model.compile("adam", lr=lr, **compile_kw)
            elif opt.lower() in ("l-bfgs", "lbfgs"):
                model.compile("L-BFGS", **compile_kw)
            else:
                raise ValueError(f"Unknown optimizer: {opt}")

            callbacks = []
            if external_trainable_variables is not None and phase.get("log_k_period"):
                callbacks.append(
                    dde.callbacks.VariableValue(
                        external_trainable_variables,
                        period=phase["log_k_period"],
                    )
                )

            class WandbCallback(dde.callbacks.Callback):
                def __init__(self, logger, phase_name, step_offset):
                    self.logger = logger
                    self.phase_name = phase_name
                    self.step_offset = step_offset

                def on_epoch_end(self):
                    if self.model.train_state.step % train_cfg.get("wandb_log_every", 100) != 0:
                        return
                    loss_train = self.model.train_state.loss_train
                    if loss_train:
                        last = loss_train[-1]
                        self.logger.log_losses(
                            self.step_offset + self.model.train_state.step,
                            list(last),
                            phase=self.phase_name,
                        )

            if self.wandb.enabled:
                callbacks.append(WandbCallback(self.wandb, opt, global_step))

            losshistory, state = model.train(
                iterations=iters,
                display_every=train_cfg.get("display_every", 500),
                callbacks=callbacks,
            )
            all_history.append(losshistory)
            global_step += iters

            if external_trainable_variables is not None:
                import torch

                k_val = float(torch.exp(external_trainable_variables).item())
                self.wandb.log_scalar(f"k_after_{opt}", k_val, step=global_step)

        save_name = train_cfg.get("checkpoint", "model")
        save_path = self.output_dir / save_name
        model.save(str(save_path))
        self.wandb.log_scalar("training/complete", 1.0)
        self.wandb.finish()

        return all_history, state
