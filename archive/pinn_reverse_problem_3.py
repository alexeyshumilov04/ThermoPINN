
import os

os.environ['DDE_BACKEND'] = 'pytorch'

import deepxde as dde
print(f"DeepXDE backend: {dde.backend.backend_name}") # Должно напечатать используемый бэкенд (pytorch / 'tensorflow.compat.v1' / ...)

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

# Установка глобального сидаp
SEED = 42
dde.config.set_random_seed(SEED)

"""# Параметры"""

# Константы материала
rho_cp = 4e6          # Дж/(м³·К)
# k = 0.6               # Вт/(м·К)
alpha_absorption = 48.2      # 1/м
alpha = 1.5e-7
A = 0.4

T0 = 291.0          # начальная температура, K

# Геометрия
Lx = 2e-3
Ly = 2e-3
Lz = 5e-3   # глубина


# Параметры импульсов
r0 = 1e-3             # м
Ep = 0.1             # Дж
pulse_fwhm = 3.0      # с
pulse_sigma = pulse_fwhm / 2.355
t_peak = 3.0 * pulse_sigma
pulse_period = 20.0   # с
N_pulses = 1

# Время моделирования
T_end = 60.0

"""# Домен"""

# Пространственная область
space = dde.geometry.Cuboid(xmin=[-Lx/2, -Ly/2, 0], xmax=[Lx/2, Ly/2, Lz])
timedomain = dde.geometry.TimeDomain(0, T_end)
geomtime = dde.geometry.GeometryXTime(space, timedomain)

"""# PDE"""

# Пиковая плотность мощности (из энергии импульса)
Q0 = Ep / (np.pi * r0**2 * np.sqrt(2*np.pi) * pulse_sigma)

# Функция источника
def heat_source(x, y, z, t):
    """
    Возвращает Q(x,y,z,t)
    """
    # Поперечный множитель (FWHM)
    r2 = x**2 + y**2
    spatial = dde.backend.exp(-r2 * np.log(2) / r0**2)
    depth = alpha_absorption * dde.backend.exp(-alpha_absorption * z)

    Q_sum = 0.0
    for n in range(N_pulses):
        t_n = t_peak + n * pulse_period
        temp = dde.backend.exp(-(t - t_n)**2 / (2 * pulse_sigma**2))
        Q_sum += temp

    return A * Q0 * Q_sum * spatial * depth

# Parametrise k as exp(log_k) to guarantee k > 0 at all times.
# log_k is the actual trainable scalar; k(t) = exp(log_k) is always positive.
# True value: k ≈ 0.6  →  log_k ≈ ln(0.6) ≈ -0.511
log_k = dde.Variable(float(np.log(0.5)))   # initialise at k = 0.5

# Уравнение PDE
def pde(x, T):
    """
    x[:,0] = x, x[:,1] = y, x[:,2] = z, x[:,3] = t
    """
    k_pos = dde.backend.exp(log_k)          # always positive
    T_t  = dde.grad.jacobian(T, x, i=0, j=3)
    T_xx = dde.grad.hessian(T, x, i=0, j=0)
    T_yy = dde.grad.hessian(T, x, i=1, j=1)
    T_zz = dde.grad.hessian(T, x, i=2, j=2)
    Q_val = heat_source(x[:,0:1], x[:,1:2], x[:,2:3], x[:,3:4])
    return T_t - (k_pos / rho_cp) * (T_xx + T_yy + T_zz) - Q_val / rho_cp

"""# Граничные и начальные условия"""

# Формируем данные для подачи в Loss

# Загружаем эталон
# Columns: 0=time, 1=T(z=0 mm), 2=T(z=1 mm), 3=T(z=3 mm) — all at beam centre x=0,y=0
ref = np.loadtxt('out_Tt.txt', usecols=(0, 1, 2, 3))
t_ref      = ref[:, 0]   # time [s]
T_ref_0mm  = ref[:, 1]   # T at z = 0 mm [K]
T_ref_1mm  = ref[:, 2]   # T at z = 1 mm [K]
T_ref_3mm  = ref[:, 3]   # T at z = 3 mm [K]

# Subsample to ~50 anchor points per depth
step = max(1, len(t_ref) // 50)
t_anchor = t_ref[::step]

def _make_anchor(z_m, T_col):
    """Build (X, T) anchor arrays for a sensor at depth z_m."""
    X = np.stack([np.zeros_like(t_anchor),
                  np.zeros_like(t_anchor),
                  np.full_like(t_anchor, z_m),
                  t_anchor], axis=-1)
    T = T_col[::step].reshape(-1, 1)
    return X, T

X_anchor_0mm, T_anchor_0mm = _make_anchor(0.000, T_ref_0mm)
X_anchor_1mm, T_anchor_1mm = _make_anchor(0.001, T_ref_1mm)
X_anchor_3mm, T_anchor_3mm = _make_anchor(0.003, T_ref_3mm)

# Keep for backward-compat with diagnostics / visualisation
X_anchor = X_anchor_1mm
T_anchor = T_anchor_1mm

print(f'Sensor anchors: {len(t_anchor)} pts × 3 depths = {3*len(t_anchor)} total')

# Three PointSetBC – one per depth (all at beam centre)
data_bc_0mm = dde.icbc.PointSetBC(X_anchor_0mm, T_anchor_0mm)
data_bc_1mm = dde.icbc.PointSetBC(X_anchor_1mm, T_anchor_1mm)
data_bc_3mm = dde.icbc.PointSetBC(X_anchor_3mm, T_anchor_3mm)

data = dde.data.TimePDE(
    geomtime,
    pde,
    [data_bc_0mm, data_bc_1mm, data_bc_3mm],  # 3-depth sensors → k well-identified
    num_domain=5_000,
    num_boundary=0,
    num_initial=1_000,
)

"""# Нейросеть"""

# Input normalization: map each coordinate to [0, 1]
# x,y: [-Lx/2, Lx/2] -> [0,1]; z: [0, Lz] -> [0,1]; t: [0, T_end] -> [0,1]
def feature_transform(x):
    x_n = (x[:, 0:1] + Lx / 2) / Lx
    y_n = (x[:, 1:2] + Ly / 2) / Ly
    z_n = x[:, 2:3] / Lz
    t_n = x[:, 3:4] / T_end
    return dde.backend.concat([x_n, y_n, z_n, t_n], axis=1)

# Определение сети
# MsFFN not available in the PyTorch backend — use FNN instead
net = dde.nn.FNN(
    layer_sizes = [4] + [100] * 5 + [1],  # 4 входа, 5 скрытых слоев по 100 нейронов, 1 выход
    activation = "tanh",
    kernel_initializer = "Glorot normal",
    # regularization = ["l2", 0.001],
    )
net.apply_feature_transform(feature_transform)

# Hard constraint for initial condition: T(x,0) = T0
# softplus(y) > 0 always  →  T >= T0 everywhere (physics: only heat addition, no cooling).
# (1 - exp(-t/tau)) ramps from 0 at t=0 to ~1 at large t.
_tau = pulse_sigma
def output_transform(x, y):
    t = x[:, 3:4]
    return T0 + (1.0 - dde.backend.exp(-t / _tau)) * F.softplus(y)
net.apply_output_transform(output_transform)

# Модель
model = dde.Model(data, net)

# PDE loss ~ (Q/rho_cp)^2 ~ 2e-3 (K/s)^2
# Data loss ~ ΔT^2 ~ 0.01 K^2
# At convergence both terms are ~1e-7 → [1,1,1,1] is already balanced
model.compile("adam", lr=1e-3, external_trainable_variables=log_k, loss_weights=[1, 1, 1, 1])



"""# Обучение"""

# Загрузка ранее сохранённых весов сети
# model.restore("my_model.pt", verbose=True)

# Phase 1 — Adam: explore the loss landscape
variable = dde.callbacks.VariableValue(log_k, period=500)
losshistory, train_state = model.train(iterations=10_000, display_every=500, callbacks=[variable])

import torch as _torch
k_adam = float(_torch.exp(log_k).item())
print(f"\nAfter Adam:  k = {k_adam:.4f}  W/(m·K)  (log_k = {float(log_k.item()):.4f})")

# Phase 2 — L-BFGS: fine-tune near the optimum (more precise, no step-size noise)
model.compile("L-BFGS", external_trainable_variables=log_k, loss_weights=[1, 1, 1, 1])
losshistory2, train_state = model.train(callbacks=[variable])

k_identified = float(_torch.exp(log_k).item())
print(f"After L-BFGS: k = {k_identified:.4f}  W/(m·K)  (log_k = {float(log_k.item()):.4f})")

# Сохранение весов сети
model.save("my_model")

# Loss history: Adam + L-BFGS combined [pde, data_0mm, data_1mm, data_3mm]
all_train = list(losshistory.loss_train) + list(losshistory2.loss_train)
pde_losses   = [arr[0] for arr in all_train]
data_losses  = [arr[1] + arr[2] + arr[3] for arr in all_train]
total_losses = [p + d for p, d in zip(pde_losses, data_losses)]
steps = list(range(len(pde_losses)))
adam_steps = len(losshistory.loss_train)  # vertical line separating phases

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
ax1, ax2, ax3 = axes

ax1.plot(steps, pde_losses,  color='b')
ax1.axvline(adam_steps, color='k', linestyle='--', linewidth=0.8, label='Adam→L-BFGS')
ax1.set_xlabel('Step');  ax1.set_ylabel('PDE Loss')
ax1.set_title('PDE Loss');  ax1.set_yscale('log');  ax1.grid(True);  ax1.legend()

ax2.plot(steps, data_losses, color='purple')
ax2.axvline(adam_steps, color='k', linestyle='--', linewidth=0.8, label='Adam→L-BFGS')
ax2.set_xlabel('Step');  ax2.set_ylabel('Data Loss')
ax2.set_title('Data Loss (sensor fit)');  ax2.set_yscale('log');  ax2.grid(True);  ax2.legend()

ax3.plot(steps, total_losses, color='g')
ax3.set_xlabel('Step');  ax3.set_ylabel('Total Loss')
ax3.set_title('Total Loss (PDE + Data)');  ax3.set_yscale('log');  ax3.grid(True)

plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute all predictions needed for visualisation
# ─────────────────────────────────────────────────────────────────────────────

t_vals    = np.linspace(0, T_end, 500)
depths_m  = [0.0, 1e-3, 3e-3]
depth_labels = ['z = 0 mm', 'z = 1 mm', 'z = 3 mm']
line_colors  = ['#d62728', '#ff7f0e', '#1f77b4']  # red, orange, blue

# -- T(t) at beam centre for each depth --------------------------------------
pts_temporal = np.array(
    [[0.0, 0.0, z, t] for z in depths_m for t in t_vals]
)
T_temporal_C = model.predict(pts_temporal).ravel() - 273.15
n = len(t_vals)
T_by_depth = {z: T_temporal_C[i*n:(i+1)*n] for i, z in enumerate(depths_m)}

# -- 2-D spatial grid in the x-z plane (y=0) --------------------------------
nx, nz     = 60, 80
x_1d       = np.linspace(-Lx/2, Lx/2, nx)
z_1d       = np.linspace(0, Lz, nz)
XX, ZZ     = np.meshgrid(x_1d, z_1d)          # shape (nz, nx)

t_snaps    = [t_peak / 2, t_peak,
              t_peak + pulse_sigma,
              t_peak + 4 * pulse_sigma]
snap_labels = [f't = {t:.1f} s' for t in t_snaps]

spatial_maps = []
for t_s in t_snaps:
    pts = np.column_stack([
        XX.ravel(), np.zeros(nx * nz),
        ZZ.ravel(), np.full(nx * nz, t_s)
    ])
    T_map = model.predict(pts).reshape(nz, nx) - 273.15
    spatial_maps.append(T_map)

# -- Depth profiles T(z) at beam centre at multiple times -------------------
z_vals          = np.linspace(0, Lz, 200)
t_profile_times = [t_peak / 2, t_peak,
                   t_peak + pulse_sigma,
                   t_peak + 5 * pulse_sigma]

depth_profiles = []
for t_s in t_profile_times:
    pts = np.column_stack([
        np.zeros(200), np.zeros(200),
        z_vals, np.full(200, t_s)
    ])
    depth_profiles.append(model.predict(pts).ravel() - 273.15)

# Heat source temporal profile at beam centre (x=0, y=0) and surface (z=0) – numpy only
# Q(0,0,0,t) = A * Q0 * alpha_absorption * sum_n exp(-(t-t_n)^2 / (2*sigma^2))
Q_surf_t = np.zeros_like(t_vals)
for _n in range(N_pulses):
    _t_n = t_peak + _n * pulse_period
    Q_surf_t += np.exp(-(t_vals - _t_n)**2 / (2 * pulse_sigma**2))
Q_surf_t *= A * Q0 * alpha_absorption   # W/m³

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1  –  T(t) at beam centre: PINN vs reference + sensor anchors
#              + heat source envelope on twin axis
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

for z, label, color in zip(depths_m, depth_labels, line_colors):
    ax.plot(t_vals, T_by_depth[z], color=color, lw=2, label=f'PINN  {label}')

# Reference curve at z = 1 mm (column 1 of out_Tt.txt)
ax.plot(t_ref, T_ref_1mm - 273.15, 'k--', lw=1.5,
        alpha=0.8, label='Reference  z = 1 mm')

# Anchor (sensor) data points
ax.scatter(t_anchor, T_anchor_1mm.ravel() - 273.15,
           s=25, color='black', zorder=5, label='Sensor anchors')

ax.set_xlabel('Time  [s]', fontsize=12)
ax.set_ylabel('Temperature  [°C]', fontsize=12)
ax.set_title('Temperature evolution at beam centre  (x = 0, y = 0)', fontsize=13)
ax.minorticks_on()
ax.grid(which='both', alpha=0.3)

# Twin axis – heat source envelope
ax_q = ax.twinx()
ax_q.fill_between(t_vals, Q_surf_t * 1e-6, alpha=0.15, color='goldenrod')
ax_q.plot(t_vals, Q_surf_t * 1e-6, color='goldenrod', lw=1.5,
          ls='--', label='Q  (z = 0, r = 0)')
ax_q.set_ylabel('Heat source  Q  [MW/m³]', fontsize=11, color='goldenrod')
ax_q.tick_params(axis='y', labelcolor='goldenrod')
ax_q.set_ylim(bottom=0)

# Merge legends from both axes
handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax_q.get_legend_handles_labels()
ax.legend(handles1 + handles2, labels1 + labels2, fontsize=10)

plt.tight_layout()
plt.savefig('Tt_depths.png', dpi=150)
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2  –  PINN vs Reference at z = 1 mm  +  residual
# ─────────────────────────────────────────────────────────────────────────────
T_pinn_1mm   = T_by_depth[1e-3]
T_ref_interp = np.interp(t_vals, t_ref, T_ref_1mm - 273.15)
residual     = T_pinn_1mm - T_ref_interp
rmse         = float(np.sqrt(np.mean(residual ** 2)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(t_vals, T_pinn_1mm, 'b-',  lw=2,   label='PINN')
ax1.plot(t_ref,  T_ref_1mm - 273.15, 'k--', lw=1.5, label='Reference')
ax1.scatter(t_anchor, T_anchor_1mm.ravel() - 273.15,
            s=30, color='red', zorder=5, label='Reference points (not used in training)')
ax1.set_xlabel('Time  [s]');  ax1.set_ylabel('Temperature  [°C]')
ax1.set_title('PINN vs Reference at z = 1 mm')
ax1.legend();  ax1.grid(alpha=0.3)

ax2.plot(t_vals, residual, 'r-', lw=1.5)
ax2.axhline(0, color='k', lw=0.8, ls='--')
ax2.fill_between(t_vals, residual, 0, alpha=0.15, color='red')
ax2.set_xlabel('Time  [s]');  ax2.set_ylabel('ΔT  [°C]')
ax2.set_title(f'Residual  (PINN − Reference)    RMSE = {rmse:.3f} °C')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('Tt_comparison.png', dpi=150)
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3  –  2-D temperature maps in the x-z plane at key time snapshots
# ─────────────────────────────────────────────────────────────────────────────
vmin = min(m.min() for m in spatial_maps)
vmax = max(m.max() for m in spatial_maps)

fig, axes = plt.subplots(1, len(t_snaps), figsize=(16, 4),
                         constrained_layout=True)
for ax, T_map, label in zip(axes, spatial_maps, snap_labels):
    im = ax.pcolormesh(x_1d * 1e3, z_1d * 1e3, T_map,
                       cmap='hot', vmin=vmin, vmax=vmax, shading='auto')
    ax.set_xlabel('x  [mm]');  ax.set_ylabel('z  [mm]')
    ax.set_title(label)
    ax.invert_yaxis()          # surface (z = 0) at top
    plt.colorbar(im, ax=ax, label='T  [°C]')

fig.suptitle('Temperature field in x-z plane  (y = 0)', fontsize=13)
plt.savefig('T_xz_snapshots.png', dpi=150)
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4  –  Depth profiles T(z) at beam centre at multiple times
# ─────────────────────────────────────────────────────────────────────────────
cmap_p = plt.cm.plasma
fig, ax = plt.subplots(figsize=(8, 5))

for i, (t_s, T_z) in enumerate(zip(t_profile_times, depth_profiles)):
    ax.plot(z_vals * 1e3, T_z,
            color=cmap_p(i / len(t_profile_times)),
            lw=2, label=f't = {t_s:.1f} s')

# Mark the sensor depth (z = 1 mm)
ax.axvline(1.0, color='gray', ls=':', lw=1, label='Sensor depth  z = 1 mm')

ax.set_xlabel('Depth z  [mm]', fontsize=12)
ax.set_ylabel('Temperature  [°C]', fontsize=12)
ax.set_title('Depth profiles at beam centre  (x = 0, y = 0)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('T_depth_profiles.png', dpi=150)
plt.show()

