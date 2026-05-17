"""Spatial–temporal domain for DeepXDE."""

import deepxde as dde
import numpy as np


def build_geomtime(Lx: float, Ly: float, Lz: float, T_end: float):
    space = dde.geometry.Cuboid(
        xmin=[-Lx / 2, -Ly / 2, 0], xmax=[Lx / 2, Ly / 2, Lz]
    )
    timedomain = dde.geometry.TimeDomain(0, T_end)
    return dde.geometry.GeometryXTime(space, timedomain)


def feature_transform(x, Lx: float, Ly: float, Lz: float, T_end: float):
    """Normalize coordinates to [0, 1]."""

    def _transform(x_in):
        x_n = (x_in[:, 0:1] + Lx / 2) / Lx
        y_n = (x_in[:, 1:2] + Ly / 2) / Ly
        z_n = x_in[:, 2:3] / Lz
        t_n = x_in[:, 3:4] / T_end
        return dde.backend.concat([x_n, y_n, z_n, t_n], axis=1)

    return _transform
