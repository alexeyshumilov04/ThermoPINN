"""Load temperature time series from the C++ ADI generator."""

from pathlib import Path

import numpy as np


def default_data_path() -> Path:
    return Path(__file__).resolve().parents[2] / "synthetic_data" / "data" / "out_Tt.txt"


def load_tt_reference(
    path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load out_Tt.txt columns: time, T(z=0), T(z=1mm), T(z=3mm) at beam centre.
    """
    path = Path(path) if path else default_data_path()
    ref = np.loadtxt(path, usecols=(0, 1, 2, 3))
    return ref[:, 0], ref[:, 1], ref[:, 2], ref[:, 3]


def make_sensor_anchors(
    t_ref: np.ndarray,
    T_cols: list[np.ndarray],
    depths_m: list[float],
    n_anchors: int = 50,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Subsample reference data into PointSetBC arrays (X, T) per depth.
    X shape (N, 4): x, y, z, t at beam centre.
    """
    step = max(1, len(t_ref) // n_anchors)
    t_anchor = t_ref[::step]
    anchors = []
    for z_m, T_col in zip(depths_m, T_cols):
        X = np.stack(
            [
                np.zeros_like(t_anchor),
                np.zeros_like(t_anchor),
                np.full_like(t_anchor, z_m),
                t_anchor,
            ],
            axis=-1,
        )
        T = T_col[::step].reshape(-1, 1)
        anchors.append((X, T))
    return anchors
