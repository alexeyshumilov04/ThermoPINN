# PINN for Laser Heating Modelling 

Physics-informed neural networks (PINNs) for predicting the **temperature field** in an **agar gel phantom** heated by a **pulsed Gaussian laser** (980 nm, Beer–Lambert absorption). The repository contains:

- a **C++ ADI finite-difference** synthetic data generator;
- **DeepXDE** PINNs for the **forward** problem (predict \(T\)) and **inverse** problem (identify thermal conductivity \(k\));
- optional **Weights & Biases** experiment logging.


## Repository structure

```
├── synthetic_data/          # C++ ADI solver + reference outputs
│   ├── src/                 # solver sources
│   ├── data/                # synthetic data
│   ├── scripts/             # visualize_results.py
│   └── Makefile
├── pinn/                    # Python package
│   ├── data/                # Load C++ outputs
│   ├── physics/             # Domain, heat source, PDE
│   ├── models/              # FNN / MsFFN / STMsFFN + hard constraints
│   ├── problems/            # Forward & inverse problems
│   ├── training/            # Trainer + WandB
│   └── visualization/
├── configs/                 # YAML experiment configs
├── scripts/                 # CLI entry points
```

## Installation

**Requirements:** Python 3.10+, C++ compiler with OpenMP (for synthetic data).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
```

**DeepXDE backend**

- **FNN (default):** PyTorch — set `DDE_BACKEND=pytorch` (default in scripts).
- **MsFFN / STMsFFN:** use TensorFlow 1.x compatibility mode:

```bash
set DDE_BACKEND=tensorflow.compat.v1   # Windows CMD
export DDE_BACKEND=tensorflow.compat.v1  # Linux/macOS
```

**WandB (optional)**

```bash
wandb login
```

Disable in any config: `wandb.enabled: false`.

## Generate synthetic data (C++)

```bash
cd synthetic_data
make
cd build
../build/heat_diffusion_agar
```

Interactive prompts: scan speed `v` [mm/s], pulse period [s], energy per pulse [J].  
Outputs are written to `synthetic_data/data/` (`out_Tt.txt`, `out_Txz.txt`, …).

Visualize reference fields:

```bash
python synthetic_data/scripts/visualize_results.py
```

## Forward problem (predict temperature)

Known k = 0.6 W/(m*K). PINN solves the heat equation with Neumann boundaries (and optional sensor points).

```bash
python scripts/train_forward.py --config configs/forward_fnn.yaml --plot
```

Fourier-feature network (TensorFlow backend):

```bash
set DDE_BACKEND=tensorflow.compat.v1
python scripts/train_forward.py --config configs/forward_msffn.yaml
```

## Inverse problem (identify conductivity)

Unknown \(k\); parametrized as `log_k` with \(k = exp(log_k) > 0\). Training uses temperature anchors at **z = 0, 1, 3 mm** from `out_Tt.txt`.

```bash
python scripts/train_inverse.py --config configs/inverse_fnn.yaml --plot
```

After training, the script prints the identified \(k\) and saves the model under `outputs/inverse_fnn/`.

## Configuration

YAML files in `configs/` control physics, network architecture, hard constraints, collocation counts, and training phases (Adam / L-BFGS).

| Key | Description |
|-----|-------------|
| `network.architecture` | `fnn`, `msffn`, `stmsffn` |
| `network.hard_constraint` | `ramp_softplus`, `linear_ramp`, `none` |
| `training.phases` | List of `{optimizer, lr, iterations}` |
| `wandb.enabled` | Turn experiment tracking on/off |

## Hard constraints

Implemented in `pinn/models/constraints.py`:

- **`ramp_softplus`** — \(T = T_0 + (1 - e^{-t/\tau})\,\mathrm{softplus}(y)\): enforces \(T \ge T_0\) and \(T(\cdot, t{=}0) = T_0\).
- **`linear_ramp`** — \(T = T_0 + t\,y\): exact IC at \(t=0\).
- **`none`** — raw network output.

## Citation & license

Academic research project. 


