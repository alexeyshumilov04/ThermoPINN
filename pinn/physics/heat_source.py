"""Beer–Lambert volumetric laser heat source."""

import numpy as np


def peak_power_density(Ep: float, r0: float, pulse_sigma: float) -> float:
    return Ep / (np.pi * r0**2 * np.sqrt(2 * np.pi) * pulse_sigma)


def heat_source(x, y, z, t, *, Q0, r0, pulse_sigma, t_peak, pulse_period, N_pulses, A, alpha_absorption):
    """Q(x,y,z,t) [W/m³] — sum of Gaussian pulses in time, Gaussian in r, Beer–Lambert in z."""
    import deepxde as dde

    r2 = x**2 + y**2
    spatial = dde.backend.exp(-r2 * np.log(2) / r0**2)
    depth = alpha_absorption * dde.backend.exp(-alpha_absorption * z)

    Q_sum = 0.0
    for n in range(N_pulses):
        t_n = t_peak + n * pulse_period
        Q_sum += dde.backend.exp(-(t - t_n) ** 2 / (2 * pulse_sigma**2))

    return A * Q0 * Q_sum * spatial * depth
