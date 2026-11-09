# Cylinder Surface Sensor Flow (cylinder-flow)

Physics-informed deep learning and POD pipeline for 2D cylinder surface sensor flow reconstruction.

## Reproduction Assets

Selected MLP, DeepSets, pPINN, and POD checkpoints, their metadata, and the matching mesh are included in [examples/](examples/README.md) (about 6.26 MiB). Full HDF5 flow cases remain external. Model-loading and inference examples will be added as the package migration progresses.

## Installation Guide

### Prerequisites
- Operating System: Linux or Windows
- Conda (Miniconda or Anaconda)
- CUDA-compatible GPU (recommended) or CPU
- Python >= 3.10

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

#### Option B: Manual Installation
```bash
conda create -n surface_cylinder_flow python=3.10 -y
conda activate surface_cylinder_flow

# Install PyTorch with CUDA (e.g. CUDA 12.4):
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install package in editable mode:
pip install -e .
```

### 3. Verify Installation
Verify that the package is correctly installed and the CLI entry point is functional:
```bash
python -m cylinder_flow --help
```
