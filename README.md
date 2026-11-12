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

## Usage Examples

### 1. Training Fixed-Layout MLP
```python
from cylinder_flow.training import EncoderTrainingConfig, train_sensor_mlp

config = EncoderTrainingConfig(
    num_epochs=100,
    batch_size=512,
    learning_rate=1e-3,
    seed=42,
)
model, history = train_sensor_mlp(
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    sensor_indices=[0, 10, 20, 30, 40, 50],
    output_dir="runs/mlp_uniform6",
)
```

### 2. Training Dynamic DeepSets Encoder
The dynamic DeepSets encoder accepts variable-count runtime sensor token sets with permutation invariance.
Formal training evaluates the frozen 60-candidate validation panel (`seed=400042`, budgets `[6, 12, 24, 48]`, 8 layouts each) for checkpoint selection:

```python
from cylinder_flow.data.sampling import (
    BalancedRandomSubsetSampler,
    create_dynamic_sensor_data_loader,
)
from cylinder_flow.training import DeepSetsTrainer, DeepSetsTrainingConfig, train_sensor_deepsets

# Configure dynamic token batch sampling with budget balancing & augmentation
sampler = BalancedRandomSubsetSampler(
    candidate_count=60,
    sensor_counts=[6, 12, 24, 48],
    seed=100045,
)
train_loader = create_dynamic_sensor_data_loader(
    dataset=train_dataset,
    batch_size=512,
    shuffle=True,
    layout_sampler=sampler,
    token_dropout_prob=0.05,
    sensor_noise_rel_std=0.01,
)

# Formal training using frozen 60-candidate panel for best epoch selection
config = DeepSetsTrainingConfig(
    num_epochs=200,
    batch_size=512,
    learning_rate=1e-3,
    seed=42,
    schema_version=2,  # Schema v2 native ReLU training
    training_sensor_counts=(6, 12, 24, 48),
    token_dropout_prob=0.05,
    sensor_noise_rel_std=0.01,
)

best_model, history = train_sensor_deepsets(
    train_loader=train_loader,
    val_loader=val_dataset,  # Evaluated over frozen panel
    config=config,
    output_dir="runs/deepsets_dynamic",
)
```

> [!NOTE]
> **Activation & Schema Compatibility**:
> - **Inference on Bundled Checkpoint**: The bundled legacy asset (`examples/checkpoints/deepsets/best_model.pt`) was trained with GELU activations under **Schema v1**. `load_deepsets_checkpoint` automatically detects and verifies schema 1 contracts without weight reinterpretation.
> - **New Model Training**: Current baseline and newly trained models use native ReLU activations under **Schema v2** (`activation='relu'`). Checkpoint serialization enforces schema 2 contracts and verifies internal ReLU activations.

