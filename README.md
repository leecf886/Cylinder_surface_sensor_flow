# Cylinder Surface Sensor Flow (cylinder-flow)

Physics-informed deep learning and POD pipeline for 2D cylinder surface sensor flow reconstruction.

## Pretrained Checkpoints & Assets

To enable lightweight evaluation and quickstart verification, selected pre-trained model weights and mesh geometry are bundled directly in the repository under [`examples/`](examples/) (~6.26 MiB total):

| Model / Asset | File Path | Size | Description |
| :--- | :--- | :--- | :--- |
| **Fixed-Layout MLP** | `examples/checkpoints/mlp/best_model.pt` | ~0.14 MiB | Uniform 6-sensor layout (`[0, 10, 20, 30, 40, 50]`), predicts $Re$ & phase $\phi$ |
| **Dynamic DeepSets** | `examples/checkpoints/deepsets/best_model.pt` | ~0.34 MiB | Permutation-invariant set encoder for arbitrary sensor counts |
| **pPINN Model** | `examples/checkpoints/ppinn/best_ppinn.pt` | ~0.32 MiB | Parametric PINN for full velocity-pressure field reconstruction $(u, v, p)$ |
| **POD Baseline** | `examples/checkpoints/pod/pod_model.npz` | ~4.62 MiB | Linear POD basis regression using 60 candidate sensors |
| **Mesh & Geometry** | `examples/data/mesh.h5` | ~0.75 MiB | 2D cylinder boundary and candidate sensor coordinates ($N_{\text{cand}} = 60$) |

Each checkpoint folder includes its accompanying configuration JSON with normalization parameters, model architectures, and sensor layout definitions. Integrity SHA-256 checksums are tracked in [`examples/artifacts.json`](examples/artifacts.json). Full unsteady flow datasets (`ReXXX.h5`) remain external; see [`docs/data_format.md`](docs/data_format.md) for data schemas and coordinate specifications.

## Installation Guide

### Prerequisites
- Operating System: Linux or Windows
- Python >= 3.10
- Conda (Miniconda or Anaconda, recommended) or standard `venv`
- CUDA-compatible GPU (recommended) or CPU

### 1. Clone Repository
```bash
git clone https://github.com/your-username/cylinder_surface_sensor_flow.git
cd cylinder_surface_sensor_flow
```

### 2. Set Up Python Environment

#### Option A: Using Conda `environment.yml` (Recommended)
```bash
conda env create -f environment.yml
conda activate surface_cylinder_flow
```

#### Option B: Manual Conda Installation
```bash
conda create -n surface_cylinder_flow python=3.10 -y
conda activate surface_cylinder_flow

# Install PyTorch with CUDA (e.g. CUDA 12.4):
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install package in editable mode:
pip install -e .
```

#### Option C: Standard Python Virtual Environment (`venv`, without Conda)
```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# 3. Install PyTorch with CUDA (e.g. CUDA 12.4) or CPU-only:
pip install torch --index-url https://download.pytorch.org/whl/cu124
# For CPU-only: pip install torch

# 4. Install package and dependencies in editable mode:
pip install -e .
```

### 3. Verify Installation
Verify that the package is correctly installed and the CLI entry point is functional:
```bash
python -m cylinder_flow --help
```
