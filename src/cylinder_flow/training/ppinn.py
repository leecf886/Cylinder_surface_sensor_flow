"""Parametric Physics-Informed Neural Network (pPINN) training orchestration.

This module implements the physics+data collaborative training loop for pPINN:
    (x, y, Re, phase) -> (u*, v*, p*)
matching the legacy parametric PINN training protocol with strict physical guarantees:
  1. Scientific Guardrail 1: Strictly disjoint train/val partitions, test Reynolds
     numbers {250, 650} and configured test IDs remain unopened and locked.
  2. Physically Consistent Pressure Scaling: Requires pressure_scale_mode="rho_uinf2",
     matching the non-dimensional momentum equations.
  3. Supervised Sampling Draw Order: Grouped Re-case sampling strictly preserving the
     exact historical NumPy RNG index draw order and float64->float32 cast sequence.
  4. Automatic Differentiation: create_graph=True PDE residual backpropagation
     via cylinder_flow.physics.residuals.
  5. Validation Selection: Best checkpoint selection on validation Re partition only,
     restoring best model weights upon completion.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import json
import logging
import math
from pathlib import Path
import random
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn as nn

from cylinder_flow.artifacts.checkpoints import save_ppinn_checkpoint
from cylinder_flow.evaluation.cycles import wrap_phase
from cylinder_flow.evaluation.field import PressureGaugeAccum
from cylinder_flow.models.ppinn import PPINNModel, build_re_norm_stats
from cylinder_flow.physics.collocation import (
    CollocationConfig,
    ReSTContext,
    build_re_st_context,
    sample_collocation_points,
)
from cylinder_flow.physics.residuals import compute_pde_residuals, mse_zero
from cylinder_flow.physics.scaling import (
    DEFAULT_COEF_A,
    DEFAULT_COEF_B,
    scale_field_to_nondimensional,
)

if TYPE_CHECKING:
    from cylinder_flow.data.types import MeshData

logger = logging.getLogger(__name__)

LOCKED_TEST_RE: frozenset[int] = frozenset({250, 650})
SUPPORTED_SELECTION_METRICS: Dict[str, str] = {
    "data_mse": "lower_better",
    "overall_rel_l2_gauge_sameden": "lower_better",
    "p_rel_l2_gauge_sameden": "lower_better",
    "overall_rel_l2": "lower_better",
}


def derive_ppinn_sampling_seeds(base_seed: int) -> Dict[str, int]:
    """Derive deterministic seeds matching legacy train_ppinn_parametric protocol.

    Parameters:
        base_seed: Master integer random seed.

    Returns:
        Dict with keys:
          - 'model_init_seed': base_seed
          - 'supervised_seed': base_seed + 101
          - 'collocation_seed': base_seed + 202
    """
    seed = int(base_seed)
    return {
        "model_init_seed": seed,
        "supervised_seed": seed + 101,
        "collocation_seed": seed + 202,
    }


def set_training_seed(seed: int) -> None:
    """Set deterministic random seeds across Python, NumPy, and PyTorch runtimes.

    Parameters:
        seed: Integer random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _extract_case_re(case: Any) -> int:
    """Extract and validate integer Reynolds number from a case object.

    Raises:
        ValueError: If 're' and 're_value' attributes are missing or non-positive/non-finite.
    """
    val = getattr(case, "re", None)
    if val is None:
        val = getattr(case, "re_value", None)
    if val is None:
        raise ValueError(f"Case object {case!r} is missing required 're' or 're_value' attribute")
    try:
        re_float = float(val)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Re value {val!r} on case {case!r}") from exc
    if math.isnan(re_float) or math.isinf(re_float) or re_float <= 0.0:
        raise ValueError(f"Re value must be a positive finite number, got {val!r}")
    return int(round(re_float))


def validate_ppinn_splits(
    train_re: Sequence[Union[int, float]],
    val_re: Sequence[Union[int, float]],
    test_re: Sequence[Union[int, float]] = (250, 650),
    train_cases: Optional[Sequence[Any]] = None,
    val_cases: Optional[Sequence[Any]] = None,
) -> Tuple[List[int], List[int], List[int]]:
    """Validate Reynolds number partition integrity under Scientific Guardrail 1.

    Enforces:
      - Strictly disjoint train and validation partitions
      - Locked test Re {250, 650} and configured test IDs never appear in train or val
      - Passed case objects strictly match partition declarations
      - Rejects actual cases outside their declared partitions before accessing other fields
      - Missing Re raises ValueError immediately, never defaulting to 0
      - Allow intentional subsets while returning actual used case IDs truthfully

    Raises:
        ValueError: If partitions overlap, are empty, or cases fall outside declared partitions.
        PermissionError: If locked test Re appears in train or validation sets.
    """
    train_set = {int(round(float(x))) for x in train_re}
    val_set = {int(round(float(x))) for x in val_re}
    test_set = {int(round(float(x))) for x in test_re} | LOCKED_TEST_RE

    if not train_set:
        raise ValueError("train_re cannot be empty")
    if not val_set:
        raise ValueError("val_re cannot be empty")

    overlap_tv = train_set & val_set
    if overlap_tv:
        raise ValueError(
            f"train_re and val_re must be strictly disjoint, overlapping Re: {sorted(overlap_tv)}"
        )

    leaked_train = train_set & test_set
    if leaked_train:
        raise PermissionError(
            f"Scientific Guardrail 1 Violation: train_re contains test Re {sorted(leaked_train)}"
        )

    leaked_val = val_set & test_set
    if leaked_val:
        raise PermissionError(
            f"Scientific Guardrail 1 Violation: val_re contains test Re {sorted(leaked_val)}"
        )

    used_train_re = sorted(train_set)
    case_train_set: Optional[set[int]] = None
    if train_cases is not None:
        if len(train_cases) == 0:
            raise ValueError("train_cases cannot be empty")
        case_train_res = [_extract_case_re(c) for c in train_cases]
        case_train_set = set(case_train_res)

        leaked_case_train = case_train_set & test_set
        if leaked_case_train:
            raise PermissionError(
                f"Scientific Guardrail 1 Violation: train_cases contains test Re {sorted(leaked_case_train)}"
            )

        outside_train = case_train_set - train_set
        if outside_train:
            raise ValueError(
                f"Actual train_cases contains Re outside declared train partition: {sorted(outside_train)}"
            )

        used_train_re = sorted(case_train_set)

    used_val_re = sorted(val_set)
    if val_cases is not None:
        if len(val_cases) == 0:
            raise ValueError("val_cases cannot be empty")
        case_val_res = [_extract_case_re(c) for c in val_cases]
        case_val_set = set(case_val_res)

        leaked_case_val = case_val_set & test_set
        if leaked_case_val:
            raise PermissionError(
                f"Scientific Guardrail 1 Violation: val_cases contains test Re {sorted(leaked_case_val)}"
            )

        outside_val = case_val_set - val_set
        if outside_val:
            raise ValueError(
                f"Actual val_cases contains Re outside declared val partition: {sorted(outside_val)}"
            )

        if case_train_set is not None:
            overlap_cases = case_train_set & case_val_set
            if overlap_cases:
                raise ValueError(
                    f"train_cases and val_cases must be strictly disjoint, overlapping Re: {sorted(overlap_cases)}"
                )

        used_val_re = sorted(case_val_set)

    return used_train_re, used_val_re, sorted(test_set)


@dataclass
class PPINNTrainingConfig:
    """Hyperparameters and execution configuration for pPINN training.

    Attributes:
        train_re: Sequence of training Reynolds numbers.
        val_re: Sequence of validation Reynolds numbers used for checkpoint selection.
        test_re: Sequence of locked test Reynolds numbers.
        D: Characteristic cylinder diameter in meters (default 0.01).
        U_inf: Reference free-stream velocity in m/s (default 1.0).
        rho: Fluid density in kg/m^3 (default 1000.0).
        pressure_scale_mode: Pressure non-dimensionalization mode (must be 'rho_uinf2').
        hidden_dims: MLP hidden layer dimensions (default 6x128).
        activation: Activation function identifier (default 'tanh').
        batch_size: Supervised point batch size (default 2048).
        num_epochs: Total number of training epochs (default 200).
        learning_rate: Adam optimizer learning rate (default 1e-3).
        weight_decay: Adam optimizer L2 weight decay (default 0.0).
        lambda_data: Weight for supervised data MSE loss term (default 1.0).
        lambda_phys: Weight for Navier-Stokes PDE residual loss term (default 1.0).
        lambda_periodic: Weight for periodic loss (documented no-op, default 0.1).
        n_collocation: Number of collocation points sampled per batch/epoch (default 4096).
        collocation_sampling_method: Strategy ('latin_hypercube', 'uniform_random', 'sobol').
        collocation_re_mode: Re sampling mode ('continuous_random', 'discrete_train_re_random').
        exclude_solid_collocation: Whether to reject points inside cylinder disk.
        collocation_domain: Spatial domain ('roi_bbox', 'roi_bbox_fluid').
        resample_collocation_every_epoch: Resample collocation points at every epoch.
        resample_collocation_every_batch: Resample collocation points at every batch.
        use_periodic_loss: Legacy flag retained for compatibility (documented no-op).
        use_st_formula: Whether Strouhal number St uses empirical formula.
        frequency_re_coef_a: Intercept for Strouhal formula f = a + b * Re.
        frequency_re_coef_b: Slope for Strouhal formula f = a + b * Re.
        use_param_noise: Whether to add Gaussian noise to Re and phase before float32 cast.
        re_noise_std: Standard deviation of parameter noise added to Re.
        phi_noise_std: Standard deviation of parameter noise added to phase.
        re_norm_method: Normalization method for Reynolds numbers ('minmax' or 'standard').
        steps_per_epoch: Explicit gradient steps per epoch (0 = ceil(total_points / batch_size)).
        eval_chunk_size: Spatial chunk size during validation evaluation (default 16384).
        seed: Master random seed for reproducibility.
        device: PyTorch device ('auto', 'cpu', 'cuda').
        re_fold_id: Fold identifier string (default 'default').
        defer_test_evaluation: Lock indicating test partition is not evaluated during training.
        checkpoint_selection_metric: Metric for best model selection ('data_mse', etc.).
        smoke_not_for_formal: Audit flag indicating lightweight smoke run.
        ppinn_repeat_id: Repetition index for statistical trials.
    """

    train_re: Sequence[int] = (200, 225, 275, 300, 350, 400, 500, 550, 600, 700, 750, 850, 900, 1000)
    val_re: Sequence[int] = (450, 800)
    test_re: Sequence[int] = (250, 650)

    D: float = 0.01
    U_inf: float = 1.0
    rho: float = 1000.0
    pressure_scale_mode: str = "rho_uinf2"

    hidden_dims: Sequence[int] = (128, 128, 128, 128, 128, 128)
    activation: str = "tanh"
    batch_size: int = 2048
    num_epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 0.0

    lambda_data: float = 1.0
    lambda_phys: float = 1.0
    lambda_periodic: float = 0.1

    n_collocation: int = 4096
    collocation_sampling_method: str = "latin_hypercube"
    collocation_re_mode: str = "continuous_random"
    exclude_solid_collocation: bool = False
    collocation_domain: str = "roi_bbox"

    resample_collocation_every_epoch: bool = False
    resample_collocation_every_batch: bool = False

    use_periodic_loss: bool = False
    use_st_formula: bool = False
    frequency_re_coef_a: float = DEFAULT_COEF_A
    frequency_re_coef_b: float = DEFAULT_COEF_B

    use_param_noise: bool = False
    re_noise_std: float = 0.0
    phi_noise_std: float = 0.0

    re_norm_method: str = "minmax"
    steps_per_epoch: int = 0
    eval_chunk_size: int = 16384

    seed: int = 42
    device: str = "auto"
    re_fold_id: str = "default"
    defer_test_evaluation: bool = True
    checkpoint_selection_metric: str = "data_mse"
    smoke_not_for_formal: bool = False
    ppinn_repeat_id: int = 0

    def __post_init__(self) -> None:
        self.pressure_scale_mode = str(self.pressure_scale_mode).lower().strip()
        if self.pressure_scale_mode != "rho_uinf2":
            raise ValueError(
                f"Physically consistent Navier-Stokes momentum residuals require "
                f"pressure_scale_mode='rho_uinf2', got {self.pressure_scale_mode!r}. "
                f"Unsupported modes are rejected."
            )

        self.checkpoint_selection_metric = str(self.checkpoint_selection_metric).lower().strip()
        if self.checkpoint_selection_metric not in SUPPORTED_SELECTION_METRICS:
            raise ValueError(
                f"Unsupported checkpoint_selection_metric: {self.checkpoint_selection_metric!r}. "
                f"Supported metrics: {sorted(SUPPORTED_SELECTION_METRICS.keys())}"
            )

        self.collocation_sampling_method = str(self.collocation_sampling_method).lower().strip()
        self.collocation_re_mode = str(self.collocation_re_mode).lower().strip()
        self.collocation_domain = str(self.collocation_domain).lower().strip()
        self.re_norm_method = str(self.re_norm_method).lower().strip()
        self.device = str(self.device).lower().strip()

        valid_sampling_methods = {"uniform_random", "latin_hypercube", "sobol"}
        if self.collocation_sampling_method not in valid_sampling_methods:
            raise ValueError(
                f"Unsupported collocation_sampling_method: {self.collocation_sampling_method!r}. "
                f"Must be one of: {sorted(valid_sampling_methods)}"
            )

        valid_re_modes = {"continuous_random", "discrete_train_re_random"}
        if self.collocation_re_mode not in valid_re_modes:
            raise ValueError(
                f"Unsupported collocation_re_mode: {self.collocation_re_mode!r}. "
                f"Must be one of: {sorted(valid_re_modes)}"
            )

        if self.re_norm_method not in {"minmax", "standard"}:
            raise ValueError(f"re_norm_method must be 'minmax' or 'standard', got {self.re_norm_method!r}")

        if self.collocation_domain not in {"roi_bbox", "roi_bbox_fluid"}:
            raise ValueError(
                f"collocation_domain must be 'roi_bbox' or 'roi_bbox_fluid', got {self.collocation_domain!r}"
            )

        if self.exclude_solid_collocation and self.collocation_domain != "roi_bbox_fluid":
            self.collocation_domain = "roi_bbox_fluid"

        if self.D <= 0.0:
            raise ValueError(f"Characteristic diameter D must be positive, got {self.D}")
        if self.U_inf <= 0.0:
            raise ValueError(f"Free-stream velocity U_inf must be positive, got {self.U_inf}")
        if self.rho <= 0.0:
            raise ValueError(f"Fluid density rho must be positive, got {self.rho}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        if self.n_collocation <= 0:
            raise ValueError(f"n_collocation must be positive, got {self.n_collocation}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.lambda_data < 0.0 or self.lambda_phys < 0.0 or self.lambda_periodic < 0.0:
            raise ValueError("Loss weight multipliers lambda must be non-negative")
        if len(self.hidden_dims) == 0:
            raise ValueError("hidden_dims cannot be empty")
        if not self.defer_test_evaluation:
            raise ValueError(
                "Scientific Guardrail 1 Violation: PPINNTrainingConfig only supports defer_test_evaluation=True. "
                "Evaluating test partitions during training is strictly prohibited."
            )

        self.train_re = tuple(int(x) for x in self.train_re)
        self.val_re = tuple(int(x) for x in self.val_re)
        self.test_re = tuple(int(x) for x in self.test_re)
        self.hidden_dims = tuple(int(d) for d in self.hidden_dims)

        validate_ppinn_splits(self.train_re, self.val_re, self.test_re)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a dictionary."""
        return {
            "train_re": list(self.train_re),
            "val_re": list(self.val_re),
            "test_re": list(self.test_re),
            "D": float(self.D),
            "U_inf": float(self.U_inf),
            "rho": float(self.rho),
            "pressure_scale_mode": str(self.pressure_scale_mode),
            "hidden_dims": list(self.hidden_dims),
            "activation": str(self.activation),
            "batch_size": int(self.batch_size),
            "num_epochs": int(self.num_epochs),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "lambda_data": float(self.lambda_data),
            "lambda_phys": float(self.lambda_phys),
            "lambda_periodic": float(self.lambda_periodic),
            "n_collocation": int(self.n_collocation),
            "collocation_sampling_method": str(self.collocation_sampling_method),
            "collocation_re_mode": str(self.collocation_re_mode),
            "exclude_solid_collocation": bool(self.exclude_solid_collocation),
            "collocation_domain": str(self.collocation_domain),
            "resample_collocation_every_epoch": bool(self.resample_collocation_every_epoch),
            "resample_collocation_every_batch": bool(self.resample_collocation_every_batch),
            "use_periodic_loss": bool(self.use_periodic_loss),
            "use_st_formula": bool(self.use_st_formula),
            "frequency_re_coef_a": float(self.frequency_re_coef_a),
            "frequency_re_coef_b": float(self.frequency_re_coef_b),
            "use_param_noise": bool(self.use_param_noise),
            "re_noise_std": float(self.re_noise_std),
            "phi_noise_std": float(self.phi_noise_std),
            "re_norm_method": str(self.re_norm_method),
            "steps_per_epoch": int(self.steps_per_epoch),
            "eval_chunk_size": int(self.eval_chunk_size),
            "seed": int(self.seed),
            "device": str(self.device),
            "re_fold_id": str(self.re_fold_id),
            "defer_test_evaluation": bool(self.defer_test_evaluation),
            "checkpoint_selection_metric": str(self.checkpoint_selection_metric),
            "smoke_not_for_formal": bool(self.smoke_not_for_formal),
            "ppinn_repeat_id": int(self.ppinn_repeat_id),
        }


@dataclass
class PPINNTrainHistory:
    """Historical records of training losses and validation metrics across epochs.

    Attributes:
        epochs: 1-indexed epoch sequence.
        train_loss_total: Weighted sum of data and physics losses.
        train_loss_data: Supervised data MSE loss.
        train_loss_phys: Navier-Stokes momentum and continuity MSE loss.
        train_loss_periodic: Periodic loss term (documented no-op = 0.0).
        val_data_mse: Mean squared error on validation flow fields.
        val_u_rel_l2: Relative L2 error for stream-wise velocity u.
        val_v_rel_l2: Relative L2 error for cross-stream velocity v.
        val_p_rel_l2: Canonical gauge-consistent relative L2 for pressure p.
        val_overall_rel_l2: Mean of (u, v, p_gauge) relative L2 errors.
        val_overall_rel_l2_gauge_sameden: Explicit gauge-consistent same-denominator relative L2.
        wall_time_seconds: Elapsed time per epoch in seconds.
        best_epoch: Epoch that achieved the best selection score.
        best_metric_value: Best validation metric score achieved.
        best_metric_name: Name of metric used for checkpoint selection.
    """

    epochs: List[int] = field(default_factory=list)
    train_loss_total: List[float] = field(default_factory=list)
    train_loss_data: List[float] = field(default_factory=list)
    train_loss_phys: List[float] = field(default_factory=list)
    train_loss_periodic: List[float] = field(default_factory=list)
    val_data_mse: List[float] = field(default_factory=list)
    val_u_rel_l2: List[float] = field(default_factory=list)
    val_v_rel_l2: List[float] = field(default_factory=list)
    val_p_rel_l2: List[float] = field(default_factory=list)
    val_overall_rel_l2: List[float] = field(default_factory=list)
    val_overall_rel_l2_gauge_sameden: List[float] = field(default_factory=list)
    wall_time_seconds: List[float] = field(default_factory=list)
    best_epoch: Optional[int] = None
    best_metric_value: Optional[float] = None
    best_metric_name: str = "data_mse"
    seed_bundle: Optional[Dict[str, int]] = None
    collocation_fingerprint: Optional[str] = None
    collocation_sampling_stats: Optional[Dict[str, Any]] = None
    collocation_mode: Optional[str] = None

    def add_epoch_record(
        self,
        epoch: int,
        train_metrics: Mapping[str, float],
        val_metrics: Mapping[str, float],
        wall_time: float = 0.0,
    ) -> None:
        """Append metric values for a completed epoch."""
        self.epochs.append(int(epoch))
        self.train_loss_total.append(float(train_metrics.get("loss_total", 0.0)))
        self.train_loss_data.append(float(train_metrics.get("loss_data", 0.0)))
        self.train_loss_phys.append(float(train_metrics.get("loss_phys", 0.0)))
        self.train_loss_periodic.append(float(train_metrics.get("loss_periodic", 0.0)))

        self.val_data_mse.append(float(val_metrics.get("data_mse", float("nan"))))
        self.val_u_rel_l2.append(float(val_metrics.get("u_rel_l2", float("nan"))))
        self.val_v_rel_l2.append(float(val_metrics.get("v_rel_l2", float("nan"))))
        self.val_p_rel_l2.append(float(val_metrics.get("p_rel_l2", float("nan"))))
        self.val_overall_rel_l2.append(float(val_metrics.get("overall_rel_l2", float("nan"))))
        self.val_overall_rel_l2_gauge_sameden.append(
            float(val_metrics.get("overall_rel_l2_gauge_sameden", float("nan")))
        )
        self.wall_time_seconds.append(float(wall_time))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize train history to dictionary."""
        d: Dict[str, Any] = {
            "epochs": list(self.epochs),
            "train_loss_total": list(self.train_loss_total),
            "train_loss_data": list(self.train_loss_data),
            "train_loss_phys": list(self.train_loss_phys),
            "train_loss_periodic": list(self.train_loss_periodic),
            "val_data_mse": list(self.val_data_mse),
            "val_u_rel_l2": list(self.val_u_rel_l2),
            "val_v_rel_l2": list(self.val_v_rel_l2),
            "val_p_rel_l2": list(self.val_p_rel_l2),
            "val_overall_rel_l2": list(self.val_overall_rel_l2),
            "val_overall_rel_l2_gauge_sameden": list(self.val_overall_rel_l2_gauge_sameden),
            "wall_time_seconds": list(self.wall_time_seconds),
            "best_epoch": self.best_epoch,
            "best_metric_value": self.best_metric_value,
            "best_metric_name": self.best_metric_name,
        }
        if self.seed_bundle is not None:
            d["seed_bundle"] = dict(self.seed_bundle)
            d["derived_seeds"] = dict(self.seed_bundle)
        if self.collocation_mode is not None:
            d["collocation_mode"] = str(self.collocation_mode)
        if self.collocation_fingerprint is not None:
            d["collocation_fingerprint"] = str(self.collocation_fingerprint)
        if self.collocation_sampling_stats is not None:
            d["collocation_sampling_stats"] = dict(self.collocation_sampling_stats)
        return d


@dataclass
class _PreparedCase:
    """Internal normalized flow case ready for high-throughput batch sampling."""

    re_value: float
    st_value: float
    d: float
    u_inf: float
    phase: np.ndarray  # [Nt], wrapped to [0, 2pi)
    roi_xy: np.ndarray  # [Nroi, 2], meters
    uvp_star: np.ndarray  # [Nt, Nroi, 3], non-dimensional float32

    @property
    def nt(self) -> int:
        return int(self.phase.shape[0])

    @property
    def nroi(self) -> int:
        return int(self.roi_xy.shape[0])

    @property
    def n_points(self) -> int:
        return self.nt * self.nroi


def _prepare_case(case: Any, mesh: MeshData, cfg: PPINNTrainingConfig) -> _PreparedCase:
    """Extract physical metadata and non-dimensional field from a case."""
    re_val = float(_extract_case_re(case))
    st_val = float(getattr(case, "st", getattr(case, "st_value", 0.0)))
    d_val = float(getattr(case, "d", getattr(case, "D", cfg.D)))
    if d_val <= 0.0:
        raise ValueError(f"Case with Re={re_val} has non-positive cylinder diameter D={d_val}")
    uinf_val = float(getattr(case, "u_inf", cfg.U_inf))
    if uinf_val <= 0.0:
        raise ValueError(f"Case with Re={re_val} has non-positive free-stream velocity U_inf={uinf_val}")

    phase_arr = np.asarray(getattr(case, "phase"), dtype=np.float64)
    phase_wrapped = wrap_phase(phase_arr)

    if hasattr(case, "roi_xy") and case.roi_xy is not None:
        roi_xy = np.asarray(case.roi_xy, dtype=np.float64)
    else:
        roi_xy = np.asarray(mesh.roi_xy, dtype=np.float64)

    if hasattr(case, "uvp_star") and case.uvp_star is not None:
        uvp_star = np.asarray(case.uvp_star, dtype=np.float32)
    elif hasattr(case, "get_uvp_star"):
        uvp_star = case.get_uvp_star(rho=cfg.rho, mode=cfg.pressure_scale_mode)
        if uvp_star is None:
            raise ValueError(f"Case with Re={re_val} returned None for get_uvp_star()")
    elif hasattr(case, "field_uvp") and case.field_uvp is not None:
        uvp_star = scale_field_to_nondimensional(
            uvp=case.field_uvp,
            u_inf=uinf_val,
            pressure_scale_mode=cfg.pressure_scale_mode,
            rho=cfg.rho,
        ).astype(np.float32)
    else:
        raise ValueError(f"Case with Re={re_val} has neither field_uvp nor uvp_star")

    return _PreparedCase(
        re_value=re_val,
        st_value=st_val,
        d=d_val,
        u_inf=uinf_val,
        phase=phase_wrapped,
        roi_xy=roi_xy,
        uvp_star=uvp_star,
    )


class SupervisedBatchSampler:
    """Grouped supervised point batch sampler for pPINN training.

    Preserves exact legacy grouped draw order:
      1. case_idx = rng.choice(num_cases, size=batch_size, p=case_prob)
      2. For each unique cid in case_idx:
           t_idx = rng.integers(0, case.nt, size=n)
           i_idx = rng.integers(0, case.nroi, size=n)
      3. perm = rng.permutation(batch_size)
    """

    def __init__(self, cases: Sequence[_PreparedCase], rng: np.random.Generator) -> None:
        self.cases = list(cases)
        self.rng = rng

        n_points = np.asarray([c.n_points for c in self.cases], dtype=np.float64)
        if np.any(n_points <= 0):
            raise ValueError("All cases must contain at least one supervised point")
        self.case_prob = n_points / np.sum(n_points)

    @property
    def total_points(self) -> int:
        """Total number of space-time supervised points across all cases."""
        return int(sum(c.n_points for c in self.cases))

    def _sample_index_groups(
        self,
        batch_size: int,
    ) -> Tuple[List[Tuple[int, np.ndarray, np.ndarray]], np.ndarray]:
        """Draw exact grouped indices matching legacy order."""
        case_idx = self.rng.choice(len(self.cases), size=batch_size, p=self.case_prob)
        groups: List[Tuple[int, np.ndarray, np.ndarray]] = []
        for cid in np.unique(case_idx):
            mask = (case_idx == cid)
            n = int(np.sum(mask))
            c = self.cases[int(cid)]
            t_idx = self.rng.integers(0, c.nt, size=n)
            i_idx = self.rng.integers(0, c.nroi, size=n)
            groups.append((int(cid), t_idx, i_idx))
        return groups, self.rng.permutation(batch_size)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a batch of coordinates, operating states, and true field values."""
        groups, perm = self._sample_index_groups(batch_size)

        x = np.empty((batch_size, 1), dtype=np.float64)
        y = np.empty((batch_size, 1), dtype=np.float64)
        re = np.empty((batch_size, 1), dtype=np.float64)
        phi = np.empty((batch_size, 1), dtype=np.float64)
        uvp = np.empty((batch_size, 3), dtype=np.float32)

        cursor = 0
        for cid, t_idx, i_idx in groups:
            n = int(t_idx.size)
            c = self.cases[int(cid)]

            sl = slice(cursor, cursor + n)
            x[sl, 0] = c.roi_xy[i_idx, 0]
            y[sl, 0] = c.roi_xy[i_idx, 1]
            re[sl, 0] = c.re_value
            phi[sl, 0] = c.phase[t_idx]
            uvp[sl, :] = c.uvp_star[t_idx, i_idx, :]
            cursor += n

        return {
            "x": x[perm],
            "y": y[perm],
            "re": re[perm],
            "phi": wrap_phase(phi[perm]),
            "uvp": uvp[perm],
        }


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_sampler: SupervisedBatchSampler,
    steps_per_epoch: int,
    cfg: PPINNTrainingConfig,
    mesh: MeshData,
    st_ctx: ReSTContext,
    train_re_values: np.ndarray,
    supervised_rng: np.random.Generator,
    collocation_rng: np.random.Generator,
    device: torch.device,
    collocation_cache_global: Optional[Dict[str, np.ndarray]] = None,
    collocation_cache_epoch: Optional[Dict[str, np.ndarray]] = None,
    col_cfg: Optional[CollocationConfig] = None,
    roi_bounds: Optional[Tuple[float, float, float, float]] = None,
) -> Dict[str, float]:
    """Execute one training epoch of pPINN with physics+data backpropagation."""
    model.train()

    sum_total = 0.0
    sum_data = 0.0
    sum_phys = 0.0
    sum_periodic = 0.0

    # Residuals must use model.D consistent with original residual formulation
    model_d = getattr(model, "D", None)
    if model_d is not None:
        D_residual = float(model_d)
    else:
        D_residual = float(getattr(mesh, "cylinder_diameter", getattr(mesh, "D_mesh", cfg.D)))

    for _ in range(steps_per_epoch):
        sup = train_sampler.sample(cfg.batch_size)

        x_sup_np = sup["x"]
        y_sup_np = sup["y"]
        re_sup_np = sup["re"]
        phi_sup_np = sup["phi"]
        uvp_true_np = sup["uvp"]

        # Parameter noise applied in float64 before float32 cast
        if cfg.use_param_noise:
            if cfg.re_noise_std > 0.0:
                re_sup_np = re_sup_np + supervised_rng.normal(0.0, cfg.re_noise_std, size=re_sup_np.shape)
            if cfg.phi_noise_std > 0.0:
                phi_sup_np = wrap_phase(
                    phi_sup_np + supervised_rng.normal(0.0, cfg.phi_noise_std, size=phi_sup_np.shape)
                )

        x_sup = torch.from_numpy(x_sup_np.astype(np.float32, copy=False)).to(device)
        y_sup = torch.from_numpy(y_sup_np.astype(np.float32, copy=False)).to(device)
        re_sup = torch.from_numpy(re_sup_np.astype(np.float32, copy=False)).to(device)
        phi_sup = torch.from_numpy(phi_sup_np.astype(np.float32, copy=False)).to(device)
        uvp_true = torch.from_numpy(uvp_true_np.astype(np.float32, copy=False)).to(device)

        uvp_pred = model(x_sup, y_sup, re_sup, phi_sup)
        loss_data = torch.mean((uvp_pred - uvp_true) ** 2)

        # Collocation sampling and Navier-Stokes PDE residuals
        if cfg.lambda_phys > 0.0:
            if cfg.resample_collocation_every_batch:
                col, _ = sample_collocation_points(
                    n_points=cfg.n_collocation,
                    roi_bounds=roi_bounds,  # type: ignore[arg-type]
                    train_re_values=train_re_values,
                    cfg=col_cfg,
                    st_ctx=st_ctx,
                    rng=collocation_rng,
                    mesh=mesh,
                )
            elif cfg.resample_collocation_every_epoch:
                assert collocation_cache_epoch is not None
                col = collocation_cache_epoch
            else:
                assert collocation_cache_global is not None
                col = collocation_cache_global

            x_col = torch.from_numpy(col["x"].astype(np.float32, copy=False)).to(device)
            y_col = torch.from_numpy(col["y"].astype(np.float32, copy=False)).to(device)
            re_col = torch.from_numpy(col["re"].astype(np.float32, copy=False)).to(device)
            phi_col = torch.from_numpy(col["phi"].astype(np.float32, copy=False)).to(device)
            st_col = torch.from_numpy(col["st"].astype(np.float32, copy=False)).to(device)

            r_u, r_v, r_c = compute_pde_residuals(model, x_col, y_col, re_col, phi_col, st_col, D=D_residual)
            loss_phys = mse_zero(r_u) + mse_zero(r_v) + mse_zero(r_c)
        else:
            loss_phys = torch.zeros((), dtype=torch.float32, device=device)

        # Periodicity is enforced exactly via sin(phi)/cos(phi) features in PPINNModel;
        # explicit periodic loss term is a documented no-op
        loss_periodic = torch.zeros((), dtype=torch.float32, device=device)

        loss_total = (
            cfg.lambda_data * loss_data
            + cfg.lambda_phys * loss_phys
            + cfg.lambda_periodic * loss_periodic
        )

        optimizer.zero_grad(set_to_none=True)
        loss_total.backward()
        optimizer.step()

        sum_total += float(loss_total.item())
        sum_data += float(loss_data.item())
        sum_phys += float(loss_phys.item())
        sum_periodic += float(loss_periodic.item())

    denom = float(max(steps_per_epoch, 1))
    return {
        "loss_total": sum_total / denom,
        "loss_data": sum_data / denom,
        "loss_phys": sum_phys / denom,
        "loss_periodic": sum_periodic / denom,
    }


def evaluate_ppinn_validation(
    model: nn.Module,
    val_cases: Sequence[_PreparedCase],
    device: torch.device,
    chunk_size: int = 16384,
) -> Dict[str, float]:
    """Evaluate pPINN on validation partition using Welford/Chan gauge accumulation."""
    model.eval()

    u_err2 = v_err2 = 0.0
    u_ref2 = v_ref2 = 0.0
    uv_err2_total = 0.0
    data_err2_sum = 0.0
    n_total = 0
    p_gauge_total = PressureGaugeAccum()

    with torch.no_grad():
        for case in val_cases:
            x_full = case.roi_xy[:, 0]
            y_full = case.roi_xy[:, 1]
            nroi = case.nroi

            for t in range(case.nt):
                re_t = float(case.re_value)
                phi_t = float(case.phase[t])
                uvp_true_t = case.uvp_star[t]  # [Nroi, 3]

                n_e = 0
                mean_e_w = 0.0
                m2_e_w = 0.0
                n_p = 0
                mean_p_w = 0.0
                m2_p_w = 0.0

                for st in range(0, nroi, chunk_size):
                    ed = min(st + chunk_size, nroi)
                    n = ed - st

                    x_t = torch.from_numpy(x_full[st:ed].reshape(n, 1).astype(np.float32)).to(device)
                    y_t = torch.from_numpy(y_full[st:ed].reshape(n, 1).astype(np.float32)).to(device)
                    re_t_tensor = torch.full((n, 1), re_t, dtype=torch.float32, device=device)
                    phi_t_tensor = torch.full((n, 1), phi_t, dtype=torch.float32, device=device)

                    pred = model(x_t, y_t, re_t_tensor, phi_t_tensor).cpu().numpy().astype(np.float64)
                    true = uvp_true_t[st:ed, :].astype(np.float64)

                    diff = pred - true
                    data_err2_sum += float(np.sum(diff ** 2))
                    n_total += int(n)

                    u_err2 += float(np.sum((pred[:, 0] - true[:, 0]) ** 2))
                    v_err2 += float(np.sum((pred[:, 1] - true[:, 1]) ** 2))
                    u_ref2 += float(np.sum(true[:, 0] ** 2))
                    v_ref2 += float(np.sum(true[:, 1] ** 2))

                    p_pred = pred[:, 2]
                    p_true = true[:, 2]
                    p_diff = p_pred - p_true

                    mean_e_c = float(np.mean(p_diff))
                    m2_e_c = float(np.sum((p_diff - mean_e_c) ** 2))
                    mean_p_c = float(np.mean(p_true))
                    m2_p_c = float(np.sum((p_true - mean_p_c) ** 2))

                    n_e, mean_e_w, m2_e_w = PressureGaugeAccum.chan_merge(
                        n_e, mean_e_w, m2_e_w, n, mean_e_c, m2_e_c
                    )
                    n_p, mean_p_w, m2_p_w = PressureGaugeAccum.chan_merge(
                        n_p, mean_p_w, m2_p_w, n, mean_p_c, m2_p_c
                    )

                p_gauge_total.add_snapshot_welford(mean_e_w, m2_e_w, mean_p_w, m2_p_w, n_e)

    eps = 1e-12
    u_rel = math.sqrt(u_err2 / (u_ref2 + eps))
    v_rel = math.sqrt(v_err2 / (v_ref2 + eps))
    pgm = p_gauge_total.metrics(eps=eps)
    p_rel_raw = pgm["p_rel_l2_raw"]
    p_rel_gsd = pgm["p_rel_l2_gauge_sameden"]
    p_rel_gfd = pgm["p_rel_l2_gauge_fluctden"]
    uv_err2 = u_err2 + v_err2

    total_dof = max(n_total * 3, 1)
    data_mse_raw = data_err2_sum / total_dof
    data_mse_gauge = (uv_err2 + p_gauge_total.gauge_err_sse) / total_dof

    return {
        "u_rel_l2": float(u_rel),
        "v_rel_l2": float(v_rel),
        "p_rel_l2": float(p_rel_gsd),
        "p_rel_l2_raw": float(p_rel_raw),
        "p_rel_l2_gauge_sameden": float(p_rel_gsd),
        "p_rel_l2_gauge_fluctden": float(p_rel_gfd),
        "p_gauge_offset_mean_abs": float(pgm["p_gauge_offset_mean_abs"]),
        "overall_rel_l2": float((u_rel + v_rel + p_rel_gsd) / 3.0),
        "overall_rel_l2_raw": float((u_rel + v_rel + p_rel_raw) / 3.0),
        "overall_rel_l2_gauge_sameden": float((u_rel + v_rel + p_rel_gsd) / 3.0),
        "overall_rel_l2_gauge_fluctden": float((u_rel + v_rel + p_rel_gfd) / 3.0),
        "data_mse": float(data_mse_raw),
        "data_mse_raw": float(data_mse_raw),
        "data_mse_gauge": float(data_mse_gauge),
        "n_points": int(n_total),
    }


def _write_selection_manifest(
    path: Path,
    cfg: PPINNTrainingConfig,
    best_epoch: int,
    best_metric_value: float,
    train_re_ids: Sequence[int],
    val_re_ids: Sequence[int],
    val_summary: Optional[Mapping[str, float]] = None,
    seed_bundle: Optional[Mapping[str, int]] = None,
    collocation_fingerprint: Optional[str] = None,
    collocation_sampling_stats: Optional[Mapping[str, Any]] = None,
    collocation_mode: Optional[str] = None,
) -> None:
    """Write selection manifest capturing training metadata and validation score."""
    direction = SUPPORTED_SELECTION_METRICS[str(cfg.checkpoint_selection_metric)]
    manifest: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "selection_policy": "validation_only_during_training",
        "selection_data_domain": "val_re_only",
        "re_fold_id": str(cfg.re_fold_id),
        "seed": int(cfg.seed),
        "train_re": [int(x) for x in train_re_ids],
        "val_re": [int(x) for x in val_re_ids],
        "test_re": [int(x) for x in cfg.test_re],
        "declared_train_re": [int(x) for x in cfg.train_re],
        "declared_val_re": [int(x) for x in cfg.val_re],
        "actual_train_re": [int(x) for x in train_re_ids],
        "actual_val_re": [int(x) for x in val_re_ids],
        "selection_metric": str(cfg.checkpoint_selection_metric),
        "metric_direction": direction,
        "best_epoch": int(best_epoch),
        "best_metric_value": float(best_metric_value),
        "selection_value": float(best_metric_value),
        "test_metrics_used_for_selection": False,
        "defer_test_evaluation": True,
        "smoke_not_for_formal": bool(cfg.smoke_not_for_formal),
        "ppinn_repeat_id": int(cfg.ppinn_repeat_id),
    }
    if seed_bundle is not None:
        manifest["seed_bundle"] = dict(seed_bundle)
        manifest["derived_seeds"] = dict(seed_bundle)
    if collocation_mode is not None:
        manifest["collocation_mode"] = str(collocation_mode)
    if collocation_fingerprint is not None:
        manifest["collocation_fingerprint"] = str(collocation_fingerprint)
    elif seed_bundle is not None:
        manifest["collocation_seed_provenance"] = int(seed_bundle["collocation_seed"])
    if collocation_sampling_stats is not None:
        manifest["collocation_sampling_stats"] = dict(collocation_sampling_stats)
    if val_summary is not None:
        manifest["validation_summary"] = {k: float(v) for k, v in val_summary.items()}

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


class PPINNTrainer:
    """Trainer orchestrating physics-informed neural network (pPINN) training.

    Coordinates:
      - Train/val Re split validation and strict test isolation
      - Supervised field ground-truth sampling with exact grouped draw order
      - Collocation point sampling inside ROI mesh bounds (fixed, per-epoch, per-batch)
      - Non-dimensional Navier-Stokes PDE residuals with create_graph=True autograd
      - Optional parameter noise applied in float64 before float32 device conversion
      - Periodic evaluation and best checkpoint selection (data_mse, gauge-consistent L2)
      - Restoring best model weights upon completion
      - Checkpoint saving and selection manifest creation
    """

    def __init__(
        self,
        config: Optional[PPINNTrainingConfig] = None,
        model: Optional[PPINNModel] = None,
        output_dir: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.cfg = config or PPINNTrainingConfig()
        self.model = model
        self.output_dir = Path(output_dir) if output_dir is not None else None

        if device is not None:
            self.device = torch.device(device) if isinstance(device, str) else device
        else:
            if self.cfg.device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            elif self.cfg.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available; using CPU")
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(self.cfg.device)

    def fit(
        self,
        train_cases: Sequence[Any],
        val_cases: Sequence[Any],
        mesh: MeshData,
        model: Optional[PPINNModel] = None,
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> Tuple[PPINNModel, PPINNTrainHistory]:
        """Fit pPINN on training cases and select best model on validation cases.

        Parameters:
            train_cases: Sequence of flow case datasets for training Re.
            val_cases: Sequence of flow case datasets for validation Re.
            mesh: MeshData containing ROI spatial geometry and cylinder dimensions.
            model: Optional pre-instantiated PPINNModel. If None, instantiates one.
            output_dir: Target directory for checkpoints and selection manifest.

        Returns:
            Tuple of (best_model, train_history), where best_model has weights
            restored to the best validation epoch.
        """
        if "test_cases" in kwargs and kwargs["test_cases"]:
            raise PermissionError(
                "Scientific Guardrail 1 Violation: test_cases must not be passed to pPINN trainer"
            )

        out_dir = Path(output_dir) if output_dir is not None else self.output_dir
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)

        # 0. Validate physical geometry consistency before accessing fields or training
        D_mesh = float(getattr(mesh, "cylinder_diameter", getattr(mesh, "D_mesh", self.cfg.D)))
        if abs(D_mesh - self.cfg.D) > 1e-6:
            raise ValueError(
                f"Incompatible geometry: mesh cylinder diameter ({D_mesh}) does not match "
                f"config D ({self.cfg.D})"
            )

        active_model = model or self.model
        if active_model is not None and hasattr(active_model, "D") and active_model.D is not None:
            model_d = float(active_model.D)
            if abs(model_d - D_mesh) > 1e-6:
                raise ValueError(
                    f"Incompatible geometry: prebuilt model D={model_d} does not match "
                    f"mesh/config diameter D={D_mesh}"
                )

        for case in train_cases:
            c_d = getattr(case, "d", getattr(case, "D", None))
            if c_d is not None and abs(float(c_d) - D_mesh) > 1e-6:
                re_c = getattr(case, "re", getattr(case, "re_value", "?"))
                raise ValueError(
                    f"Incompatible geometry: train case (Re={re_c}) diameter ({float(c_d)}) "
                    f"does not match mesh/config diameter ({D_mesh})"
                )
        for case in val_cases:
            c_d = getattr(case, "d", getattr(case, "D", None))
            if c_d is not None and abs(float(c_d) - D_mesh) > 1e-6:
                re_c = getattr(case, "re", getattr(case, "re_value", "?"))
                raise ValueError(
                    f"Incompatible geometry: val case (Re={re_c}) diameter ({float(c_d)}) "
                    f"does not match mesh/config diameter ({D_mesh})"
                )

        # 1. Validate partition IDs before field normalization or point sampling
        train_re_ids, val_re_ids, test_re_ids = validate_ppinn_splits(
            train_re=self.cfg.train_re,
            val_re=self.cfg.val_re,
            test_re=self.cfg.test_re,
            train_cases=train_cases,
            val_cases=val_cases,
        )

        # 2. Derive deterministic seeds matching legacy protocol
        seed_bundle = derive_ppinn_sampling_seeds(self.cfg.seed)
        set_training_seed(seed_bundle["model_init_seed"])
        supervised_rng = np.random.default_rng(seed_bundle["supervised_seed"])
        collocation_rng = np.random.default_rng(seed_bundle["collocation_seed"])

        # 3. Prepare cases and non-dimensionalize ground truth fields
        prepared_train = [_prepare_case(c, mesh, self.cfg) for c in train_cases]
        prepared_val = [_prepare_case(c, mesh, self.cfg) for c in val_cases]

        # 4. Resolve cylinder diameter and Re normalization statistics
        D_ref = D_mesh
        train_re_vals = [c.re_value for c in prepared_train]
        re_center, re_scale = build_re_norm_stats(train_re_vals, method=self.cfg.re_norm_method)

        # 5. Model instantiation
        if active_model is None:
            active_model = PPINNModel(
                hidden_dims=self.cfg.hidden_dims,
                D=D_ref,
                re_norm_method=self.cfg.re_norm_method,
                re_mean=re_center,
                re_scale=re_scale,
                activation=self.cfg.activation,
            )
        active_model.to(self.device)

        # 6. Physical context and collocation configuration
        col_cfg = CollocationConfig(
            n_collocation=self.cfg.n_collocation,
            collocation_sampling_method=self.cfg.collocation_sampling_method,
            collocation_re_mode=self.cfg.collocation_re_mode,
            exclude_solid_collocation=self.cfg.exclude_solid_collocation,
            collocation_domain=self.cfg.collocation_domain,
            use_st_formula=self.cfg.use_st_formula,
            frequency_re_coef_a=self.cfg.frequency_re_coef_a,
            frequency_re_coef_b=self.cfg.frequency_re_coef_b,
            D=D_ref,
            U_inf=self.cfg.U_inf,
        )
        train_st_ctx = build_re_st_context(prepared_train, cfg=col_cfg)

        if mesh.roi_bounds is not None:
            roi_bounds = mesh.roi_bounds
        else:
            xmin = float(np.min(mesh.roi_xy[:, 0]))
            xmax = float(np.max(mesh.roi_xy[:, 0]))
            ymin = float(np.min(mesh.roi_xy[:, 1]))
            ymax = float(np.max(mesh.roi_xy[:, 1]))
            roi_bounds = (xmin, xmax, ymin, ymax)

        # Fixed global collocation design (default)
        fixed_collocation = (
            not self.cfg.resample_collocation_every_batch
            and not self.cfg.resample_collocation_every_epoch
        )
        collocation_cache_global: Optional[Dict[str, np.ndarray]] = None
        collocation_sampling_stats: Dict[str, Any] = {}
        if fixed_collocation and self.cfg.lambda_phys > 0.0:
            collocation_cache_global, collocation_sampling_stats = sample_collocation_points(
                n_points=self.cfg.n_collocation,
                roi_bounds=roi_bounds,
                train_re_values=np.asarray(train_re_vals, dtype=np.float64),
                cfg=col_cfg,
                st_ctx=train_st_ctx,
                rng=collocation_rng,
                mesh=mesh,
            )

        if fixed_collocation:
            collocation_mode = "fixed"
            col_fingerprint = collocation_sampling_stats.get("fingerprint")
        elif self.cfg.resample_collocation_every_batch:
            collocation_mode = "resample_every_batch"
            col_fingerprint = None
        else:
            collocation_mode = "resample_every_epoch"
            col_fingerprint = None

        # 7. Optimizer and batch sampler
        optimizer = torch.optim.Adam(
            active_model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        train_sampler = SupervisedBatchSampler(prepared_train, rng=supervised_rng)

        if self.cfg.steps_per_epoch > 0:
            steps_per_epoch = int(self.cfg.steps_per_epoch)
        else:
            steps_per_epoch = max(1, int(math.ceil(train_sampler.total_points / self.cfg.batch_size)))

        # 8. Training loop
        history = PPINNTrainHistory(
            best_metric_name=self.cfg.checkpoint_selection_metric,
            seed_bundle=seed_bundle,
            collocation_fingerprint=col_fingerprint,
            collocation_sampling_stats=collocation_sampling_stats if fixed_collocation else None,
            collocation_mode=collocation_mode,
        )
        best_metric_value = float("inf")
        best_epoch = -1
        best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        best_val_summary: Optional[Dict[str, float]] = None

        for epoch in range(1, self.cfg.num_epochs + 1):
            t0 = time.time()

            collocation_cache_epoch: Optional[Dict[str, np.ndarray]] = None
            if not self.cfg.resample_collocation_every_batch and self.cfg.resample_collocation_every_epoch:
                collocation_cache_epoch, _ = sample_collocation_points(
                    n_points=self.cfg.n_collocation,
                    roi_bounds=roi_bounds,
                    train_re_values=np.asarray(train_re_vals, dtype=np.float64),
                    cfg=col_cfg,
                    st_ctx=train_st_ctx,
                    rng=collocation_rng,
                    mesh=mesh,
                )

            train_out = train_one_epoch(
                model=active_model,
                optimizer=optimizer,
                train_sampler=train_sampler,
                steps_per_epoch=steps_per_epoch,
                cfg=self.cfg,
                mesh=mesh,
                st_ctx=train_st_ctx,
                train_re_values=np.asarray(train_re_vals, dtype=np.float64),
                supervised_rng=supervised_rng,
                collocation_rng=collocation_rng,
                device=self.device,
                collocation_cache_global=collocation_cache_global,
                collocation_cache_epoch=collocation_cache_epoch,
                col_cfg=col_cfg,
                roi_bounds=roi_bounds,
            )

            val_summary = evaluate_ppinn_validation(
                model=active_model,
                val_cases=prepared_val,
                device=self.device,
                chunk_size=self.cfg.eval_chunk_size,
            )

            wall_time = time.time() - t0
            history.add_epoch_record(
                epoch=epoch,
                train_metrics=train_out,
                val_metrics=val_summary,
                wall_time=wall_time,
            )

            # Checkpoint selection on validation score
            score = float(val_summary[self.cfg.checkpoint_selection_metric])
            if score < best_metric_value:
                best_metric_value = score
                best_epoch = epoch
                best_state_dict = copy.deepcopy(active_model.state_dict())
                best_val_summary = dict(val_summary)

                re_split_payload = {
                    "train_re": train_re_ids,
                    "val_re": val_re_ids,
                    "test_re": test_re_ids,
                    "actual_train_re": train_re_ids,
                    "actual_val_re": val_re_ids,
                    "declared_train_re": list(self.cfg.train_re),
                    "declared_val_re": list(self.cfg.val_re),
                }
                extra_meta: Dict[str, Any] = {
                    "seed_bundle": dict(seed_bundle),
                    "collocation_mode": collocation_mode,
                }
                if col_fingerprint is not None:
                    extra_meta["collocation_fingerprint"] = col_fingerprint
                else:
                    extra_meta["collocation_seed_provenance"] = int(seed_bundle["collocation_seed"])

                if out_dir is not None:
                    save_ppinn_checkpoint(
                        path=out_dir / "best_ppinn.pt",
                        model=active_model,
                        config=self.cfg,
                        epoch=best_epoch,
                        selection_metric=self.cfg.checkpoint_selection_metric,
                        selection_value=best_metric_value,
                        re_split=re_split_payload,
                        re_fold_id=self.cfg.re_fold_id,
                        extra_metadata=extra_meta,
                        save_config_json=True,
                    )
                    _write_selection_manifest(
                        path=out_dir / "ppinn_selection_manifest.json",
                        cfg=self.cfg,
                        best_epoch=best_epoch,
                        best_metric_value=best_metric_value,
                        train_re_ids=train_re_ids,
                        val_re_ids=val_re_ids,
                        val_summary=best_val_summary,
                        seed_bundle=seed_bundle,
                        collocation_fingerprint=col_fingerprint,
                        collocation_sampling_stats=collocation_sampling_stats if fixed_collocation else None,
                        collocation_mode=collocation_mode,
                    )

            if out_dir is not None:
                re_split_payload = {
                    "train_re": train_re_ids,
                    "val_re": val_re_ids,
                    "test_re": test_re_ids,
                    "actual_train_re": train_re_ids,
                    "actual_val_re": val_re_ids,
                    "declared_train_re": list(self.cfg.train_re),
                    "declared_val_re": list(self.cfg.val_re),
                }
                extra_meta = {
                    "seed_bundle": dict(seed_bundle),
                    "collocation_mode": collocation_mode,
                }
                if col_fingerprint is not None:
                    extra_meta["collocation_fingerprint"] = col_fingerprint
                else:
                    extra_meta["collocation_seed_provenance"] = int(seed_bundle["collocation_seed"])

                save_ppinn_checkpoint(
                    path=out_dir / "last_ppinn.pt",
                    model=active_model,
                    config=self.cfg,
                    epoch=epoch,
                    selection_metric=self.cfg.checkpoint_selection_metric,
                    selection_value=score,
                    re_split=re_split_payload,
                    re_fold_id=self.cfg.re_fold_id,
                    extra_metadata=extra_meta,
                    save_config_json=False,
                )

        history.best_epoch = best_epoch
        history.best_metric_value = best_metric_value

        # 9. Restore best returned model weights
        if best_state_dict is not None:
            active_model.load_state_dict(best_state_dict)

        # 10. Persist complete config and train history
        if out_dir is not None:
            cfg_path = out_dir / "config.json"
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(self.cfg.to_dict(), f, indent=2, default=str)

            hist_path = out_dir / "history.json"
            with open(hist_path, "w", encoding="utf-8") as f:
                json.dump(history.to_dict(), f, indent=2, default=str)

        return active_model, history


def train_ppinn(
    train_cases: Sequence[Any],
    val_cases: Sequence[Any],
    mesh: MeshData,
    config: Optional[PPINNTrainingConfig] = None,
    output_dir: Optional[Union[str, Path]] = None,
    model: Optional[PPINNModel] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[PPINNModel, PPINNTrainHistory]:
    """Functional convenience entrypoint to train a parametric PINN (pPINN).

    Parameters:
        train_cases: Training Re flow case datasets.
        val_cases: Validation Re flow case datasets for selection.
        mesh: MeshData geometry containing ROI coordinates and cylinder radius.
        config: Training configuration (defaults to PPINNTrainingConfig()).
        output_dir: Optional destination directory for checkpoints and manifests.
        model: Optional pre-constructed PPINNModel instance.
        device: PyTorch execution device override.

    Returns:
        Tuple of (best_model, train_history).
    """
    trainer = PPINNTrainer(config=config, model=model, output_dir=output_dir, device=device)
    return trainer.fit(
        train_cases=train_cases,
        val_cases=val_cases,
        mesh=mesh,
        model=model,
        output_dir=output_dir,
    )


# Backward-compatible alias
train_parametric_pinn = train_ppinn

__all__ = [
    "LOCKED_TEST_RE",
    "PPINNTrainHistory",
    "PPINNTrainer",
    "PPINNTrainingConfig",
    "SUPPORTED_SELECTION_METRICS",
    "SupervisedBatchSampler",
    "derive_ppinn_sampling_seeds",
    "evaluate_ppinn_validation",
    "train_one_epoch",
    "train_parametric_pinn",
    "train_ppinn",
    "validate_ppinn_splits",
]
