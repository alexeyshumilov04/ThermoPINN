"""Hard constraints on network output (initial condition, positivity)."""

import deepxde as dde


def apply_hard_constraint(net, constraint: str, *, T0: float, pulse_sigma: float):
    """
    constraint:
      - 'ramp_softplus': T = T0 + (1 - exp(-t/tau)) * softplus(y)  [default in notebooks]
      - 'linear_ramp':   T = T0 + t * y
      - 'none':          no transform
    """
    if constraint == "none":
        return

    tau = pulse_sigma

    if constraint == "ramp_softplus":
        import torch.nn.functional as F

        def output_transform(x, y):
            t = x[:, 3:4]
            return T0 + (1.0 - dde.backend.exp(-t / tau)) * F.softplus(y)

    elif constraint == "linear_ramp":

        def output_transform(x, y):
            t = x[:, 3:4]
            return T0 + t * y

    else:
        raise ValueError(f"Unknown hard constraint: {constraint}")

    net.apply_output_transform(output_transform)
