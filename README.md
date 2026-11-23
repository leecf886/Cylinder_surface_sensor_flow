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
| **WP-A Sample Metrics** | `examples/results/wp_a_sample_metrics_long.csv` | ~23 KiB | Synthetic validation-$Re$ (`450, 800`) 23-field long table (`LONG_TABLE_FIELDS`) |
| **WP-A Sample Layouts** | `examples/results/wp_a_sample_layouts.json` | ~1.7 KiB | Sample 12-sensor layouts on the 60-location candidate pool |
| **WP-D Sample Results** | `examples/results/wp_d_sample_results_long.csv` | ~11 KiB | Synthetic validation-$Re$ (`450, 800`) WP-D long table across $\lambda_{\text{phys}} \in \{0, 1, 2\}$ |

Each checkpoint folder includes its accompanying configuration JSON with normalization parameters, model architectures, and sensor layout definitions. Integrity SHA-256 checksums for checkpoints, geometry, and sample results are tracked in [`examples/artifacts.json`](examples/artifacts.json). Full unsteady flow datasets (`ReXXX.h5`) remain external; see [`docs/data_format.md`](docs/data_format.md) for data schemas and coordinate specifications.

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

### 3. Training Parametric PINN (pPINN)
Physics-informed collaborative training fitting supervised flow field data while enforcing the non-dimensional 2D incompressible Navier-Stokes momentum and continuity equations:

```python
from cylinder_flow.training import PPINNTrainingConfig, train_ppinn

config = PPINNTrainingConfig(
    train_re=[200, 225, 275, 300, 350, 400, 500, 550, 600, 700, 750, 850, 900, 1000],
    val_re=[450, 800],
    test_re=[250, 650],
    batch_size=2048,
    n_collocation=4096,
    num_epochs=200,
    learning_rate=1e-3,
    lambda_data=1.0,
    lambda_phys=1.0,
    pressure_scale_mode="rho_uinf2",
    checkpoint_selection_metric="data_mse",
    seed=42,
)

best_model, history = train_ppinn(
    train_cases=train_cases,
    val_cases=val_cases,
    mesh=mesh,
    config=config,
    output_dir="runs/ppinn_default",
)
```

### 4. Architecture Ablation Encoders & Baselines (Phase 8d)
The pipeline supports 8 immutable architecture ablation variants under independent checkpoint schema `dynamic_arch_ablation_schema_v1` (7 neural variants + 1 closed-form Fourier-K2 Ridge baseline `arch_fourier_k2_ridge`):

```python
from cylinder_flow.artifacts import load_ablation_checkpoint, save_ablation_checkpoint
from cylinder_flow.inference import predict_re_phase_full, reconstruct_field_closed_loop

# 1. Load trained ablation package with verified contract & physical provenance
package = load_ablation_checkpoint("path/to/ablation_checkpoint.pt")
print(f"Variant: {package.arch_variant}, Schema: {package.schema_name}")

# 2. Direct sensor inference (supports DynamicSensorBatch or flattened arrays)
prediction = predict_re_phase_full(model=package, sensor_inputs=sensor_inputs)
print(f"Predicted Re: {prediction.re}, Phase: {prediction.phase}")

# 3. Fixed flattened layout adapter
adapter = package.as_fixed_adapter()  # takes [B, 3*K] flattened sensor observations

# 4. Closed-loop flow field reconstruction with pPINN
field_pred, enc_pred = reconstruct_field_closed_loop(
    mlp=package,
    ppinn=ppinn_package,
    mesh_xy=mesh.roi_xy,
    sensor_inputs=sensor_inputs,
)
```

> [!NOTE]
> **Activation & Schema Compatibility**:
> - **Inference on Bundled Checkpoint**: The bundled legacy asset (`examples/checkpoints/deepsets/best_model.pt`) was trained with GELU activations under **Schema v1**. `load_deepsets_checkpoint` automatically detects and verifies schema 1 contracts without weight reinterpretation.
> - **New Model Training**: Current baseline and newly trained models use native ReLU activations under **Schema v2** (`activation='relu'`). Checkpoint serialization enforces schema 2 contracts and verifies internal ReLU activations.
> - **Architecture Ablation Checkpoints**: Use independent schema `dynamic_arch_ablation_schema_v1` loaded via `load_ablation_checkpoint` / `save_ablation_checkpoint` returning an `AblationPackage`.

### 5. Unified CLI & Work Package Workflows

Inspect all available subcommands and flags via the unified CLI:

```bash
python -m cylinder_flow --help
```

Each Work Package (`wp-a`, `wp-b`, `wp-c`, `wp-d`) defaults to the safe `preflight` stage and enforces disjoint Reynolds splits and locked-test guards (`Re=250, 650` remain blocked unless `--allow-test` is passed with an approved analysis lock):

```bash
# Safe preflight & dry-run checks with example protocol templates
python -m cylinder_flow wp-a --config configs/experiments/wp_a/protocol.example.json --stages preflight --dry-run
python -m cylinder_flow wp-b --config configs/experiments/wp_b/protocol.example.json --stages preflight --mode dynamic --dry-run
python -m cylinder_flow wp-b --config configs/experiments/wp_b/arch_ablation_protocol.example.json --stages preflight --mode arch_ablation --dry-run
python -m cylinder_flow wp-c --config configs/experiments/wp_c/protocol.example.json --stages preflight --dry-run
python -m cylinder_flow wp-d --config configs/experiments/wp_d/protocol.example.json --stages preflight --dry-run

# Single-stage pipeline subcommands (dry-run examples)
python -m cylinder_flow train-encoder --encoder deepsets --output-dir runs/encoder_smoke --epochs 1 --seed 42 --dry-run
python -m cylinder_flow train-ppinn --output-dir runs/ppinn_smoke --epochs 1 --seed 42 --dry-run
python -m cylinder_flow eval --mode closed_loop --ppinn-ckpt examples/checkpoints/ppinn/best_ppinn.pt --encoder-ckpt examples/checkpoints/deepsets/best_model.pt --re-list 450,800 --dry-run
```

On Windows PowerShell, thin wrappers under `scripts/windows/` forward arguments directly to `python -m cylinder_flow` with a configurable `-PythonExe` (defaulting to `python`):

```powershell
.\scripts\windows\run_wp_a.ps1 -Stages preflight -DryRun
.\scripts\windows\run_wp_b.ps1 -Stages preflight -Mode dynamic -DryRun
.\scripts\windows\run_wp_c.ps1 --config configs/experiments/wp_c/protocol.example.json --stages preflight --dry-run
.\scripts\windows\run_wp_d.ps1 -Stages preflight -DryRun
```

### 6. Quick Aggregation, Statistics & Plotting on Bundled Example Results

The lightweight validation-$Re$ sample assets in `examples/results/` allow running aggregation, paired statistics, and headless plotting out of the box without external HDF5 datasets:

```python
import json, math, tempfile
from pathlib import Path
import numpy as np
from cylinder_flow.analysis import (
    load_results_long,
    plot_sensor_layouts,
    plot_wp_d_results,
    run_comparison,
)

# 1. Paired statistical comparison on 10 paired seeds (val Re 450, 800)
wp_a_rows = load_results_long("examples/results/wp_a_sample_metrics_long.csv")
stat = run_comparison(
    wp_a_rows,
    {
        "comparison_id": "sample_amortized_vs_uniform",
        "strategy_a": "amortized_chain_greedy",
        "strategy_b": "uniform",
        "evaluator": "ppinn_closed_loop",
        "metric": "overall_rel_l2_gauge_sameden",
        "granularity": "per_cycle",
        "nsensor": 12,
        "split": "val",
        "paired_unit": "seed",
        "confirmatory": True,
        "holm_family": "demo",
    },
    n_bootstrap=200,
)
print(stat.mean_improvement, stat.wilcoxon_p_raw, stat.wilcoxon_p_status)

# 2. Headless layout and per-Re reconstruction plotting into a temporary directory
layouts = json.loads(Path("examples/results/wp_a_sample_layouts.json").read_text(encoding="utf-8"))["layouts"]
wp_d_rows = load_results_long("examples/results/wp_d_sample_results_long.csv")
theta = np.linspace(0.0, 2.0 * math.pi, 60, endpoint=False)

with tempfile.TemporaryDirectory() as out_dir:
    plot_sensor_layouts(layouts, theta=theta, output_dir=Path(out_dir) / "layouts", no_png=True)
    plot_wp_d_results(wp_d_rows, [], Path(out_dir) / "wp_d", no_png=True)
```

## Documentation Map

- [`docs/architecture.md`](docs/architecture.md) — Package layers, module dependency graph, 4-tier path resolution priority, and scientific boundary guards.
- [`docs/data_format.md`](docs/data_format.md) — Coordinate system, HDF5 schemas (`mesh.h5`, `ReXXX.h5`), non-dimensionalization, and normalization protocols.
- [`docs/metrics.md`](docs/metrics.md) — Relative $L_2$, Welford/Chan gauge-consistent pressure $L_2$, Strouhal guard, cycle-level raw SSE accumulation, spatial metrics, 23-field long-table schema, and paired statistical testing rules.

