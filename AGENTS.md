# AGENTS.md — Cylinder Surface Sensor Flow

## One-Line Summary
Physics-informed pipeline: Fluent H5 -> preprocess -> Encoder (sensor->Re,phase) -> pPINN (x,y,Re,phase->u,v,p). POD baseline for comparison.

## Environment & Execution Rules

### Python Environment
- **Recommended Conda Environment**: `surface_cylinder_flow` (see `environment.yml`)
- Prefer executing via the active virtual/conda environment, e.g.:
  `conda run -n surface_cylinder_flow python <script.py> <args>`
  or an explicit environment interpreter path.
- Avoid invoking an unintended system `python` when multiple interpreters are installed.

### Path Resolution
- NEVER hardcode machine-specific absolute paths in `src/cylinder_flow/`.
- All paths must resolve through `cylinder_flow.config.get_path_config()`, CLI `--data-dir` arguments, `CYLINDER_DATA_ROOT`, or local `configs/paths.json`.

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
