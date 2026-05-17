"""Post-training figures: T(t), residuals, spatial snapshots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pinn.data.load_synthetic import load_tt_reference
from pinn.physics.heat_source import peak_power_density


def plot_training_results(
    model,
    cfg: dict,
    *,
    output_dir: Path | str = "outputs",
    show: bool = False,
):
    phys = cfg["physics"]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    T_end = phys["T_end"]
    Lx, Ly, Lz = phys["Lx"], phys["Ly"], phys["Lz"]
    t_peak = phys["t_peak"]
    pulse_sigma = phys["pulse_sigma"]

    t_vals = np.linspace(0, T_end, 500)
    depths_m = [0.0, 1e-3, 3e-3]
    depth_labels = ["z = 0 mm", "z = 1 mm", "z = 3 mm"]
    colors = ["#d62728", "#ff7f0e", "#1f77b4"]

    pts = np.array([[0.0, 0.0, z, t] for z in depths_m for t in t_vals])
    T_all = model.predict(pts).ravel() - 273.15
    n = len(t_vals)
    T_by_depth = {z: T_all[i * n : (i + 1) * n] for i, z in enumerate(depths_m)}

    t_ref, T0, T1, T3 = load_tt_reference(cfg.get("data", {}).get("reference_path"))
    T_ref_1mm = T1

    fig, ax = plt.subplots(figsize=(12, 5))
    for z, label, color in zip(depths_m, depth_labels, colors):
        ax.plot(t_vals, T_by_depth[z], color=color, lw=2, label=f"PINN {label}")
    ax.plot(t_ref, T_ref_1mm - 273.15, "k--", lw=1.5, label="Reference z = 1 mm")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("Temperature at beam centre")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out / "Tt_depths.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    T_pinn = T_by_depth[1e-3]
    T_ref_i = np.interp(t_vals, t_ref, T_ref_1mm - 273.15)
    residual = T_pinn - T_ref_i
    rmse = float(np.sqrt(np.mean(residual**2)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(t_vals, T_pinn, "b-", lw=2, label="PINN")
    ax1.plot(t_ref, T_ref_1mm - 273.15, "k--", lw=1.5, label="Reference")
    ax1.set_title("PINN vs reference at z = 1 mm")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(t_vals, residual, "r-")
    ax2.axhline(0, color="k", ls="--")
    ax2.set_title(f"Residual (RMSE = {rmse:.3f} °C)")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out / "Tt_comparison.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    return {"rmse_1mm_C": rmse}
