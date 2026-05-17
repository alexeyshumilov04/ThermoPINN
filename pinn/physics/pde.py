"""Heat equation residual for PINN."""

import deepxde as dde

from .heat_source import heat_source


def make_pde(physics_cfg: dict, *, trainable_k=None):
    """
    Build PDE residual. If trainable_k is a dde.Variable (log_k), k is learned (inverse problem).
    Otherwise uses fixed thermal diffusivity alpha = k / rho_cp.
    """
    rho_cp = physics_cfg["rho_cp"]
    k_fixed = physics_cfg.get("k")
    alpha_absorption = physics_cfg["alpha_absorption"]
    A = physics_cfg["A"]
    r0 = physics_cfg["r0"]
    Ep = physics_cfg["Ep"]
    pulse_sigma = physics_cfg["pulse_sigma"]
    t_peak = physics_cfg["t_peak"]
    pulse_period = physics_cfg["pulse_period"]
    N_pulses = physics_cfg["N_pulses"]

    from .heat_source import peak_power_density

    Q0 = peak_power_density(Ep, r0, pulse_sigma)

    def pde(x, T):
        if trainable_k is not None:
            k_pos = dde.backend.exp(trainable_k)
            diff = k_pos / rho_cp
        else:
            diff = k_fixed / rho_cp

        T_t = dde.grad.jacobian(T, x, i=0, j=3)
        T_xx = dde.grad.hessian(T, x, i=0, j=0)
        T_yy = dde.grad.hessian(T, x, i=1, j=1)
        T_zz = dde.grad.hessian(T, x, i=2, j=2)
        Q_val = heat_source(
            x[:, 0:1],
            x[:, 1:2],
            x[:, 2:3],
            x[:, 3:4],
            Q0=Q0,
            r0=r0,
            pulse_sigma=pulse_sigma,
            t_peak=t_peak,
            pulse_period=pulse_period,
            N_pulses=N_pulses,
            A=A,
            alpha_absorption=alpha_absorption,
        )
        return T_t - diff * (T_xx + T_yy + T_zz) - Q_val / rho_cp

    return pde
