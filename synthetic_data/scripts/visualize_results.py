#!/usr/bin/env python3
"""Visualize C++ ADI output (Txz plane and Tt time series at three depths)."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def load_txz(filename: Path):
    data = np.loadtxt(filename)
    x = data[:, 0] * 1e3
    z = data[:, 1] * 1e3
    T = data[:, 2]
    x_center = (x.max() + x.min()) / 2.0
    x = x - x_center
    return x, z, T


def load_tt(filename: Path):
    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]


def plot_txz(x, z, T, ax):
    x_unique = np.unique(x)
    z_unique = np.unique(z)
    X, Z = np.meshgrid(x_unique, z_unique)
    T_grid = griddata((x, z), T, (X, Z), method="linear")
    levels = np.linspace(T.min(), T.max(), 20)
    contourf = ax.contourf(X, Z, T_grid, levels=levels, cmap="hot")
    ax.contour(X, Z, T_grid, levels=levels[::2], colors="black", linewidths=0.5, alpha=0.3)
    plt.colorbar(contourf, ax=ax, label="Temperature [K]")
    ax.set_xlabel("X [mm] (beam centre at x=0)")
    ax.set_ylabel("Depth Z [mm]")
    ax.set_title("Temperature (XZ plane, Y=0)")
    ax.invert_yaxis()
    ax.set_ylim(4, 0)
    ax.axvline(0, color="cyan", linestyle="--", linewidth=2, alpha=0.7)


def plot_tt(t, T, ax, depth_mm, color="b"):
    T_C = T - 273.15
    ax.plot(t, T_C, color=color, linewidth=2.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title(f"Centre (0, 0, {depth_mm} mm)")
    ax.grid(True, alpha=0.3)
    ax.axhline(T_C[0], color="gray", linestyle="--", alpha=0.4)


def main():
    parser = argparse.ArgumentParser(description="Visualize synthetic ADI data")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "results_visualization.png",
    )
    args = parser.parse_args()

    x, z, T_xz = load_txz(args.data_dir / "out_Txz.txt")
    t, T_0, T_1, T_3 = load_tt(args.data_dir / "out_Tt.txt")

    fig = plt.figure(figsize=(16, 10))
    plot_txz(x, z, T_xz, plt.subplot(2, 2, 1))
    plot_tt(t, T_0, plt.subplot(2, 2, 2), 0, "red")
    plot_tt(t, T_1, plt.subplot(2, 2, 3), 1, "orange")
    plot_tt(t, T_3, plt.subplot(2, 2, 4), 3, "blue")
    fig.suptitle("3 Gaussian pulses — agar gel phantom", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
