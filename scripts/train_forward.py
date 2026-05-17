#!/usr/bin/env python3
"""Train PINN for the forward problem (known thermal conductivity)."""

import argparse
import os
import sys

# PyTorch backend works for FNN; use tensorflow.compat.v1 for Fourier nets
os.environ.setdefault("DDE_BACKEND", "pytorch")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import deepxde as dde  # noqa: E402

from pinn.config import load_config  # noqa: E402
from pinn.problems.forward import build_forward_problem  # noqa: E402
from pinn.training.trainer import Trainer  # noqa: E402
from pinn.visualization.plots import plot_training_results  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="PINN forward problem")
    parser.add_argument(
        "--config",
        default=os.path.join(ROOT, "configs", "forward_fnn.yaml"),
    )
    parser.add_argument("--plot", action="store_true", help="Save diagnostic plots")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.get("seed") is not None:
        dde.config.set_random_seed(cfg["seed"])

    print(f"DeepXDE backend: {dde.backend.backend_name}")
    data, _ = build_forward_problem(cfg)
    trainer = Trainer(cfg)
    model = trainer.build_model(data)
    trainer.train(model)

    if args.plot:
        plot_training_results(model, cfg, output_dir=cfg.get("output_dir", "outputs"))


if __name__ == "__main__":
    main()
