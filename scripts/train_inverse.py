#!/usr/bin/env python3
"""Train PINN for the inverse problem (identify thermal conductivity k)."""

import argparse
import os
import sys

os.environ.setdefault("DDE_BACKEND", "pytorch")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import deepxde as dde  # noqa: E402
import torch  # noqa: E402

from pinn.config import load_config  # noqa: E402
from pinn.problems.inverse import build_inverse_problem  # noqa: E402
from pinn.training.trainer import Trainer  # noqa: E402
from pinn.visualization.plots import plot_training_results  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="PINN inverse problem")
    parser.add_argument(
        "--config",
        default=os.path.join(ROOT, "configs", "inverse_fnn.yaml"),
    )
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.get("seed") is not None:
        dde.config.set_random_seed(cfg["seed"])

    print(f"DeepXDE backend: {dde.backend.backend_name}")
    data, _, log_k = build_inverse_problem(cfg)
    trainer = Trainer(cfg)
    model = trainer.build_model(data)
    trainer.train(model, external_trainable_variables=log_k)

    k_identified = float(torch.exp(log_k).item())
    print(f"\nIdentified k = {k_identified:.4f} W/(m·K)  (log_k = {float(log_k.item()):.4f})")
    print(f"True k (reference) ≈ {cfg['physics'].get('k_true', 0.6)} W/(m·K)")

    if args.plot:
        plot_training_results(model, cfg, output_dir=cfg.get("output_dir", "outputs"))


if __name__ == "__main__":
    main()
