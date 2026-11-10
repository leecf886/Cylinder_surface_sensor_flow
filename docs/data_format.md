# Data Format & Conventions

This document specifies the exact dataset schema, HDF5 internal group paths, physical coordinates, sensor representations, non-dimensionalization rules, and model-specific normalization protocols used across `cylinder_flow`.

---

## 1. Physical Domain & Coordinate System

- **Coordinate System**: 2D Cartesian $(x, y)$ centered at the cylinder's cross-sectional center:
  - $(0, 0)$ is the geometric center of the circular cylinder.
  - $+x$: Streamwise direction (free-stream flow moves from $-x$ to $+x$).
  - $+y$: Cross-flow / transverse direction.
- **Cylinder Geometry**:
  - Diameter: $D = 0.01\ \text{m}$ (radius $R = 0.005\ \text{m}$).
  - Cylinder wall boundary: $x^2 + y^2 = R^2 = (D/2)^2$.
- **Region of Interest (ROI)**:
  - Bounding box: $x \in [-0.01, 0.07]\ \text{m}$, $y \in [-0.02, 0.02]\ \text{m}$.
  - Fluid domain excludes the cylinder interior ($r < R$).
- **Surface Candidate Sensors**:
  - $N_{\text{cand}} = 60$ candidate sensor locations placed uniformly around the cylinder wall circumference.
  - Angular position $\theta \in [0, 2\pi)$:
    $$\theta_k = \frac{2\pi k}{60}, \quad k = 0, 1, \dots, 59$$
  - $\theta = 0$: Cylinder trailing edge / wake stagnation point $(+R, 0)$.
  - $\theta = \pi$: Cylinder leading edge / upstream stagnation point $(-R, 0)$.
  - Angle increases counter-clockwise.

---

## 2. HDF5 Storage Specifications

All datasets are structured in standard HDF5 format, categorized into mesh geometry (`mesh.h5`) and unsteady flow cases (`ReXXX.h5`). Datasets are organized into hierarchical HDF5 groups.

### 2.1 Mesh & Sensor Geometry (`mesh.h5`)

The mesh file contains the ROI discretization, full mesh topology, and candidate sensor coordinates.

| Group & Dataset Path | Type | Shape | Description & Units |
| :--- | :--- | :--- | :--- |
| **`/mesh/roi_xy`** | `float64` | `[N_roi, 2]` | Physical coordinates $[x, y]$ of ROI cell centers (m) |
| **`/mesh/roi_xy_nd`** | `float64` | `[N_roi, 2]` | Non-dimensional coordinates $[x/D, y/D]$ centered at cylinder |
| `/mesh/cell_xy_full` | `float64` | `[N_cell_full, 2]` | All cell center coordinates in CFD domain (m) |
| `/mesh/node_xy_full` | `float64` | `[N_node_full, 2]` | All mesh grid vertices / nodes (m) |
| `/mesh/roi_cell_ids` | `int32` | `[N_roi]` | 0-based cell indices mapping into `cell_xy_full` |
| `/mesh/roi_mask` | `int32` | `[N_cell_full]` | Boolean mask (`1` for ROI cells, `0` outside) |
| **`/sensors/sensor_theta`** | `float64` | `[N_cand]` | Azimuthal angles of candidate surface sensors in radians $[0, 2\pi)$ |
| `/sensors/sensor_xy_nominal` | `float64` | `[N_cand, 2]` | Theoretical $(x, y)$ coordinates on cylinder boundary $r = R$ (m) |
| `/sensors/sensor_xy_actual` | `float64` | `[N_cand, 2]` | Nearest cell centroid coordinates where sensors sample CFD data (m) |
| `/sensors/sensor_cell_id` | `int32` | `[N_cand]` | Cell IDs corresponding to `sensor_xy_actual` |
| `/sensors/sensor_distance` | `float64` | `[N_cand]` | Distance between nominal and actual cell center (m) |
| **`/geometry/cylinder_diameter`** | `float64` | Scalar | Cylinder characteristic diameter $D = 0.01\ \text{m}$ |
| **`/geometry/cylinder_radius`** | `float64` | Scalar | Cylinder radius $R = 0.005\ \text{m}$ |
| **`/geometry/cylinder_center`** | `float64` | `[2]` | Center coordinates $[0.0, 0.0]$ |
| **`/config/roi_bounds`** | `float64` | `[4]` | Bounding box $[x_{\min}, x_{\max}, y_{\min}, y_{\max}] = [-0.01, 0.07, -0.02, 0.02]$ |
| `/config/n_candidate_sensor` | `int32` | Scalar | Number of candidate locations ($60$) |
| `/config/nu` | `float64` | Scalar | Kinematic viscosity $\nu = 10^{-6}\ \text{m}^2/\text{s}$ |

---

### 2.2 Unsteady Flow Case Files (`ReXXX.h5`)

Each case file corresponds to a specific Reynolds number (e.g., `Re200.h5`, `Re450.h5`).

| Group & Dataset Path | Type | Shape | Description & Units |
| :--- | :--- | :--- | :--- |
| **`/sensors/sensor_uvp`** | `float32` / `float64` | `[Nt, N_cand, 3]` | Surface candidate sensor readings: channels $[u, v, p]$ in physical units $(\text{m/s}, \text{m/s}, \text{Pa})$ |
| `/sensors/sensor_input_flat` | `float32` / `float64` | `[Nt, N_cand * 3]` | Pre-flattened sensor readings: $[u_0, v_0, p_0, u_1, v_1, p_1, \dots]$ |
| **`/snapshots/phase`** | `float64` | `[Nt]` | Instantaneous shedding phase $\phi \in [0, 2\pi)$ (rad) |
| `/snapshots/phase_sin` | `float64` | `[Nt]` | $\sin(\phi)$ harmonic projection |
| `/snapshots/phase_cos` | `float64` | `[Nt]` | $\cos(\phi)$ harmonic projection |
| **`/snapshots/periodic_mask`** | `int32` / `bool` | `[Nt]` | Validity mask: `1` for fully developed periodic shedding, `0` for initial transient |
| `/snapshots/time` | `float64` | `[Nt]` | Physical timestamps (s) |
| `/snapshots/Cl` | `float64` | `[Nt]` | Lift coefficient $C_L(t)$ time history |
| **`/field/uvp`** | `float32` / `float64` | `[Nt, N_roi, 3]` | Full-field ground truth $[u, v, p]$ in physical units $(\text{m/s}, \text{m/s}, \text{Pa})$ |
| `/field/u`, `/field/v`, `/field/p` | `float32` / `float64` | `[Nt, N_roi]` | Separate velocity and pressure channels |
| **`/meta/Re`** | `float64` | Scalar | Reynolds number $Re = U_\infty D / \nu$ |
| **`/meta/U_inf`** | `float64` | Scalar | Free-stream velocity $U_\infty$ (m/s) |
| **`/meta/St`** | `float64` | Scalar | Vortex shedding Strouhal number $St = f D / U_\infty$ |
| `/meta/D` | `float64` | Scalar | Cylinder diameter $D = 0.01\ \text{m}$ |
| `/meta/frequency_f` | `float64` | Scalar | Vortex shedding frequency $f$ (Hz) |
| `/meta/period_T` | `float64` | Scalar | Vortex shedding period $T = 1/f$ (s) |
| `/meta/omega` | `float64` | Scalar | Circular frequency $\omega = 2\pi f$ (rad/s) |
| `/meta/Nt` | `int32` | Scalar | Number of stored time snapshots |
| `/meta/Nroi` | `int32` | Scalar | Number of grid points in the ROI ($N_{\text{roi}} = 20560$) |

---

## 3. Model Normalization & Scaling Conventions

Different models in the pipeline operate with distinct input representations and normalizers. **It is critical not to confound raw physical inputs with non-dimensionalized representations.**

### 3.1 Fixed-Layout MLP (`SensorMLP`)

- **Input Data**: **Raw physical surface sensor values** $(u, v, p)$ in $(\text{m/s}, \text{m/s}, \text{Pa})$ from candidate sensor indices $[s_0, s_1, \dots, s_{K-1}]$, flattened into a 1D vector of length $3K$:
  $$\mathbf{x}_{\text{raw}} = [u_{s_0}, v_{s_0}, p_{s_0}, u_{s_1}, v_{s_1}, p_{s_1}, \dots, u_{s_{K-1}}, v_{s_{K-1}}, p_{s_{K-1}}] \in \mathbb{R}^{3K}$$
- **Normalization**: Standard $z$-score normalizer directly applied to the raw physical vector:
  $$\mathbf{x}_{\text{norm}} = \frac{\mathbf{x}_{\text{raw}} - \boldsymbol{\mu}_{\text{raw}}}{\boldsymbol{\sigma}_{\text{raw}}}$$
  The parameters $\boldsymbol{\mu}_{\text{raw}}$ and $\boldsymbol{\sigma}_{\text{raw}}$ (saved in `checkpoint_meta.json`) are fitted strictly on the training Re cases in their physical dimensions (e.g. pressure $\mu \sim -2.6\ \text{Pa}, \sigma \sim 2.4\ \text{Pa}$; velocity $\mu \sim 10^{-4}\ \text{m/s}, \sigma \sim 10^{-3}\ \text{m/s}$).
- **Target Output**: Predicts normalized Reynolds number and harmonic phase projections:
  $$\mathbf{y}_{\text{pred}} = [Re_{\text{norm}}, \sin\phi, \cos\phi]$$
  Physical $Re$ is restored via $Re = Re_{\text{norm}} \cdot \sigma_{Re} + \mu_{Re}$. Phase angle is restored via $\phi = \operatorname{wrap\_phase}(\operatorname{atan2}(\sin\phi, \cos\phi))$.
- > [!IMPORTANT]
  > **Do not pre-scale inputs by $U_\infty$ or $\rho U_\infty^2$ for the Fixed MLP.** Legacy checkpoint weights directly expect raw physical sensor readings normalized by the stored statistics.

### 3.2 Dynamic Set Encoder (`DeepSetsSensorEncoder`)

- **Input Data**: Unordered set of $K$ sensor tokens:
  $$\mathbf{x}_k = [u_k^*, v_k^*, p_k^*, \cos\theta_k, \sin\theta_k] \in \mathbb{R}^5$$
- **Normalization**: Sensor readings $(u, v, p)$ are normalized using channel-wise statistics $(\mu_c, \sigma_c)$ across all candidate sensors on the training set, combined with direct positional encodings $(\cos\theta, \sin\theta)$.

### 3.3 Parametric PINN (`PPINNModel`)

- **Input Space**: Non-dimensional spatial coordinates and normalized flow parameters:
  $$\mathbf{z} = [x^*, y^*, Re_{\text{norm}}, \sin\phi, \cos\phi], \quad \text{where } x^* = \frac{x}{D}, \ y^* = \frac{y}{D}$$
- **Output Space**: Non-dimensional velocity and pressure fields:
  $$\mathbf{s}^* = [u^*, v^*, p^*]$$
- **Physical Field Restoration**:
  $$u = u^* \cdot U_\infty, \quad v = v^* \cdot U_\infty, \quad p = p^* \cdot (\rho U_\infty^2) \quad (\text{mode } \texttt{rho\_uinf2})$$
  Here $\rho = 1000\ \text{kg/m}^3$ and $U_\infty$ is the case free-stream velocity.

### 3.4 POD Baseline (`SurfacePOD`)

- Operates as a direct linear mapping from active surface sensors to POD modal coefficients of the ROI flow field.

---

## 4. Vortex Shedding Phase & Circular Topology

1. **Circular Phase Angle**:
   $\phi \in [0, 2\pi)$ tracks the dynamic limit-cycle state within each vortex shedding period $T = 1/f$.
2. **Phase Difference**:
   Because phase is circular, Euclidean distance is improper. The signed shortest circular difference between two phases $\phi_1, \phi_2$ is:
   $$\Delta\phi = \operatorname{atan2}(\sin(\phi_1 - \phi_2), \cos(\phi_1 - \phi_2)) \in [-\pi, \pi]$$
   The absolute angular error is $|\Delta\phi|$.
3. **Phase Wrap**:
   $$\operatorname{wrap\_phase}(\psi) = \psi \pmod{2\pi}$$
