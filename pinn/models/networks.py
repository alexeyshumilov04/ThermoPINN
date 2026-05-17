"""Neural network architectures for the temperature field."""

import deepxde as dde


def build_network(arch: str, net_cfg: dict):
    """
    arch: 'fnn' | 'msffn' | 'stmsffn'
    """
    layers = net_cfg.get("layer_sizes", [4, 100, 100, 100, 100, 100, 1])
    activation = net_cfg.get("activation", "tanh")
    kernel_init = net_cfg.get("kernel_initializer", "Glorot normal")
    regularization = net_cfg.get("regularization")

    common = dict(
        layer_sizes=layers,
        activation=activation,
        kernel_initializer=kernel_init,
    )
    if regularization:
        common["regularization"] = regularization

    arch = arch.lower()
    if arch == "fnn":
        return dde.nn.FNN(**common)
    if arch == "msffn":
        return dde.nn.MsFFN(sigmas=net_cfg.get("sigmas", [0.16, 4]), **common)
    if arch == "stmsffn":
        return dde.nn.STMsFFN(
            sigmas_x=net_cfg.get("sigmas_x", [0]),
            sigmas_t=net_cfg.get("sigmas_t", [0.16, 4]),
            **common,
        )
    raise ValueError(f"Unknown architecture: {arch}")
