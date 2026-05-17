"""Forward problem: known k, predict temperature field."""

import deepxde as dde
import numpy as np

from pinn.data.load_synthetic import load_tt_reference, make_sensor_anchors
from pinn.physics.domain import build_geomtime
from pinn.physics.pde import make_pde


def build_forward_problem(cfg: dict):
    """TimePDE with Neumann BC, optional sensor data, and fixed conductivity."""
    phys = cfg["physics"]
    data_cfg = cfg.get("data", {})
    bc_cfg = cfg.get("boundary", {})

    geomtime = build_geomtime(phys["Lx"], phys["Ly"], phys["Lz"], phys["T_end"])
    pde = make_pde(phys)

    bcs = []
    if bc_cfg.get("neumann", True):
        T_end = phys["T_end"]

        def on_spatial_boundary(x, on_boundary):
            return on_boundary and not (
                np.isclose(x[3], 0) or np.isclose(x[3], T_end)
            )

        bcs.append(
            dde.icbc.NeumannBC(geomtime, lambda x: 0.0, on_spatial_boundary)
        )

    if data_cfg.get("use_sensors", False):
        path = data_cfg.get("reference_path")
        t_ref, T0, T1, T3 = load_tt_reference(path)
        depths = data_cfg.get("sensor_depths_m", [0.0, 1e-3, 3e-3])
        cols = [T0, T1, T3][: len(depths)]
        for X, T in make_sensor_anchors(
            t_ref, cols, depths, n_anchors=data_cfg.get("n_anchors", 50)
        ):
            bcs.append(dde.icbc.PointSetBC(X, T))

    num_domain = cfg["training"].get("num_domain", 5000)
    num_boundary = cfg["training"].get("num_boundary", 0)
    num_initial = cfg["training"].get("num_initial", 1000)

    data = dde.data.TimePDE(
        geomtime,
        pde,
        bcs,
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
    )
    return data, geomtime
