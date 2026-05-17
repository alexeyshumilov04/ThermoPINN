"""Inverse problem: learn thermal conductivity k from temperature sensors."""

import deepxde as dde
import numpy as np

from pinn.data.load_synthetic import load_tt_reference, make_sensor_anchors
from pinn.physics.domain import build_geomtime
from pinn.physics.pde import make_pde


def build_inverse_problem(cfg: dict):
    """
  Estimate k via trainable log_k = dde.Variable.
  Uses multi-depth sensor anchors from synthetic data (no Neumann BC required).
  """
    phys = cfg["physics"]
    data_cfg = cfg["data"]

    log_k_init = np.log(phys.get("k_init", 0.5))
    log_k = dde.Variable(float(log_k_init))

    geomtime = build_geomtime(phys["Lx"], phys["Ly"], phys["Lz"], phys["T_end"])
    pde = make_pde(phys, trainable_k=log_k)

    path = data_cfg.get("reference_path")
    t_ref, T0, T1, T3 = load_tt_reference(path)
    depths = data_cfg.get("sensor_depths_m", [0.0, 1e-3, 3e-3])
    cols = [T0, T1, T3][: len(depths)]

    bcs = []
    for X, T in make_sensor_anchors(
        t_ref, cols, depths, n_anchors=data_cfg.get("n_anchors", 50)
    ):
        bcs.append(dde.icbc.PointSetBC(X, T))

    train = cfg["training"]
    data = dde.data.TimePDE(
        geomtime,
        pde,
        bcs,
        num_domain=train.get("num_domain", 5000),
        num_boundary=train.get("num_boundary", 0),
        num_initial=train.get("num_initial", 1000),
    )
    return data, geomtime, log_k
