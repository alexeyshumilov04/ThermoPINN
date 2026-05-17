#!/usr/bin/env python3
"""
Visualization script for agar gel heat diffusion results
- Txz: Temperature in xz-plane (depth profile)
- Tt: Temperature evolution over time
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.signal import find_peaks

# Set style
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def load_txz_data(filename='out_Txz.txt'):
    """Load xz-plane temperature data"""
    # Load with error handling for incomplete lines
    try:
        data = np.loadtxt(filename)
    except ValueError:
        # If there are incomplete lines, load with genfromtxt
        data = np.genfromtxt(filename, invalid_raise=False)
        # Remove rows with NaN
        data = data[~np.isnan(data).any(axis=1)]

    x = data[:, 0] * 1e3  # Convert to mm
    z = data[:, 1] * 1e3  # Convert to mm
    T = data[:, 2]

    # Shift coordinates so beam center is at (0,0,0)
    x_center = (x.max() + x.min()) / 2.0
    x = x - x_center  # Beam at x=0

    return x, z, T

def load_tt_data(filename='out_Tt.txt'):
    """Load temperature vs time data for three depths"""
    # Load with error handling for incomplete lines
    try:
        data = np.loadtxt(filename)
    except ValueError:
        # If there are incomplete lines, load with genfromtxt
        data = np.genfromtxt(filename, invalid_raise=False)
        # Remove rows with NaN
        data = data[~np.isnan(data).any(axis=1)]

    t = data[:, 0]
    T_0mm = data[:, 1]
    T_1mm = data[:, 2]
    T_3mm = data[:, 3]
    return t, T_0mm, T_1mm, T_3mm

def plot_txz(x, z, T, ax):
    """Plot Txz projection as 2D heatmap"""
    # Create regular grid for contour plot
    from scipy.interpolate import griddata

    # Get unique x and z values
    x_unique = np.unique(x)
    z_unique = np.unique(z)

    # Create meshgrid
    X, Z = np.meshgrid(x_unique, z_unique)

    # Interpolate T onto regular grid
    T_grid = griddata((x, z), T, (X, Z), method='linear')

    # Create contour plot
    levels = np.linspace(T.min(), T.max(), 20)
    contourf = ax.contourf(X, Z, T_grid, levels=levels, cmap='hot')
    contour = ax.contour(X, Z, T_grid, levels=levels[::2], colors='black',
                         linewidths=0.5, alpha=0.3)

    # Colorbar
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label('Temperature [K]', fontsize=11)

    # Labels and formatting
    ax.set_xlabel('X position [mm] (beam center at x=0)', fontsize=12)
    ax.set_ylabel('Depth Z [mm]', fontsize=12)
    ax.set_title('Temperature Distribution (XZ-plane at Y=0)', fontsize=13, fontweight='bold')
    ax.invert_yaxis()  # Depth increases downward
    ax.set_ylim(4, 0)  # Zoom to 0-4 mm depth range

    # Mark beam center
    ax.axvline(0, color='cyan', linestyle='--', linewidth=2, alpha=0.7, label='Beam center')
    ax.legend(loc='upper right', fontsize=9)

    # Add temperature annotation
    T_max = T.max()
    T_min = T.min()
    ax.text(0.02, 0.98, f'T_max = {T_max:.1f} K ({T_max-273.15:.1f}°C)\nT_min = {T_min:.1f} K',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    return contourf

def plot_tt(t, T, ax, depth_mm, color='b'):
    """Plot temperature evolution over time at one depth"""
    # Convert to Celsius
    T_C = T - 273.15

    # Main plot
    ax.plot(t, T_C, color=color, linewidth=2.5)

    # Labels
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel('Temperature [°C]', fontsize=11)
    ax.set_title(f'(0, 0, {depth_mm}mm)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add initial temperature line
    ax.axhline(T_C[0], color='gray', linestyle='--', alpha=0.4, linewidth=1)

def main():
    """Main visualization function"""
    print("=" * 60)
    print("  AGAR GEL HEAT DIFFUSION - Results Visualization")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    try:
        x, z, T_xz = load_txz_data()
        print(f"  ✓ Loaded Txz data: {len(x)} points")
    except Exception as e:
        print(f"  ✗ Error loading Txz: {e}")
        return

    try:
        t, T_0mm, T_1mm, T_3mm = load_tt_data()
        print(f"  ✓ Loaded Tt data: {len(t)} points at 3 depths")
    except Exception as e:
        print(f"  ✗ Error loading Tt: {e}")
        return

    # Create figure with 4 subplots (2x2)
    fig = plt.figure(figsize=(16, 10))

    # Subplot 1: Txz projection (top left)
    ax1 = plt.subplot(2, 2, 1)
    plot_txz(x, z, T_xz, ax1)

    # Subplot 2: Temperature at surface (0mm) - top right
    ax2 = plt.subplot(2, 2, 2)
    plot_tt(t, T_0mm, ax2, depth_mm=0, color='red')

    # Subplot 3: Temperature at 1mm depth - bottom left
    ax3 = plt.subplot(2, 2, 3)
    plot_tt(t, T_1mm, ax3, depth_mm=1, color='orange')

    # Subplot 4: Temperature at 3mm depth - bottom right
    ax4 = plt.subplot(2, 2, 4)
    plot_tt(t, T_3mm, ax4, depth_mm=3, color='blue')

    # Overall title
    fig.suptitle('3 Gaussian Pulses in Agar Gel - Temperature at Multiple Depths',
                 fontsize=15, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save figure
    output_file = 'results_visualization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"Simulation time: {t[-1]:.1f} s")
    print(f"Spatial domain (x): {x.min():.2f} to {x.max():.2f} mm (beam at x=0)")
    print(f"Depth domain (z): {z.min():.2f} to {z.max():.2f} mm")
    print(f"\nTemperature at three depths:")
    print(f"  Surface (0mm):  {T_0mm.min()-273.15:.1f} - {T_0mm.max()-273.15:.1f} °C  (ΔT = {T_0mm.max()-T_0mm[0]:.1f} K)")
    print(f"  1mm depth:      {T_1mm.min()-273.15:.1f} - {T_1mm.max()-273.15:.1f} °C  (ΔT = {T_1mm.max()-T_1mm[0]:.1f} K)")
    print(f"  3mm depth:      {T_3mm.min()-273.15:.1f} - {T_3mm.max()-273.15:.1f} °C  (ΔT = {T_3mm.max()-T_3mm[0]:.1f} K)")
    print("\nCoordinate system: Beam center at (0, 0, 0)")
    print("=" * 60)

    # Show plot
    plt.show()

if __name__ == '__main__':
    main()
