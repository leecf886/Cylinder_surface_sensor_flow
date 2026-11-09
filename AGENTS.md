# AGENTS.md — Cylinder Surface Sensor Flow

## One-Line Summary
Physics-informed pipeline: Fluent H5 -> preprocess -> Encoder (sensor->Re,phase) -> pPINN (x,y,Re,phase->u,v,p). POD baseline for comparison.

## Environment & Execution Rules

### Active Environment (Local Workstation Context)
> **Note**: The following paths provide local workstation context. When preparing for public open-source release, local drive letters will be pruned.
- **Conda Environment**: `surface_cylinder_flow`
- **Local Python Interpreter**: `D:\anaconda_envs\surface_cylinder_flow\python.exe`
- **Local Data Root**: `E:/2d_cylinder_flow_data`
- **Default Processed Data**: `E:/2d_cylinder_flow_data/processed_data_base_60`
- **Checkpoints & Runs Root**: `E:/2d_cylinder_flow_data/runs`

### Execution Guideline
- Prefer executing via active conda environment:
  `conda run -n surface_cylinder_flow python <script.py> <args>`
  or explicit interpreter:
  `D:\anaconda_envs\surface_cylinder_flow\python.exe <script.py> <args>`
- NEVER use bare `python` or `python3` if multiple system interpreters exist.

### Path Resolution
- NEVER hardcode absolute paths (`E:/...` or `D:/...`) in `src/cylinder_flow/`.
- All paths must resolve through `cylinder_flow.config.get_path_config()`, CLI `--data-dir` arguments, or `configs/paths.json`.

---

## Core Scientific & Architecture Rules (Mandatory)

### 1. Reynolds Number Split Integrity
- Train, validation, and test splits must remain strictly disjoint.
- Validation Re (e.g., 450, 800) is used exclusively for model checkpoint and sensor layout selection.
- Test Re (250, 650) is locked: **never open or evaluate test Re during model training or layout search**.

### 2. Pressure Gauge Invariance
- In incompressible flow, pressure is determined up to an arbitrary constant.
- For pressure error calculation, always use the Welford/Chan gauge-consistent same-denominator relative L2 metric. Raw non-gauge pressure comparison is physically flawed.

### 3. Strouhal Number Relation Guard
- When computing shedding frequency St from Re, use `compute_st_from_re` with physical coefficients calibrated to U_inf/D. Non-positive or out-of-range St ([0.01, 1.0]) must raise warnings.

### 4. Paired Seeds for Deterministic Comparisons
- To isolate sensor layout effects from training noise, layout evaluations must use paired training seeds across compared strategies.
- Seeds and fingerprints must be recorded in run metadata.

### 5. Vortex Shedding Cycle Alignment
- Segment snapshots into complete shedding cycles based on circular phase wrap ($[0, 2\pi)$).
- Accumulate raw SSE sums across complete cycles rather than averaging snapshot relative errors.

---

## Code Quality & Verification
- Test commands: run `pytest` or execute minimal smoke tests (`tests/smoke/`).
- Do not run long full-matrix training runs unless explicitly requested.
