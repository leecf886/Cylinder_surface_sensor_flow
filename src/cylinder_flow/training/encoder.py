"""Training loop, evaluation, and optimizer routines for fixed-layout MLP encoders."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import logging
import math
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cylinder_flow.artifacts.checkpoints import (
    load_deepsets_checkpoint,
    save_deepsets_checkpoint,
    save_mlp_checkpoint,
)
from cylinder_flow.data.normalization import Normalizer
from cylinder_flow.data.sampling import (
    BalancedRandomSubsetSampler,
    DynamicBatchCollator,
    DynamicSensorDataset,
    create_dynamic_sensor_data_loader,
    reset_dynamic_loader,
)
from cylinder_flow.data.splits import ReSplit
from cylinder_flow.evaluation.regression import (
    compute_circular_metrics,
    compute_regression_metrics,
)
from cylinder_flow.models.deepsets import (
    DynamicSensorBatch,
    DynamicSetSensorEncoder,
    SetSensorEncoder,
    candidate_position_fingerprint,
    is_dynamic_token_regressor,
    positions_from_theta,
    validate_sensor_indices,
)
from cylinder_flow.models.mlp import SensorMLP
from cylinder_flow.training.dynamic_validation import (
    DYNAMIC_VALIDATION_LAYOUTS_PER_BUDGET,
    DYNAMIC_VALIDATION_PANEL_BUDGETS,
    DYNAMIC_VALIDATION_PANEL_CANDIDATE_COUNT,
    DYNAMIC_VALIDATION_PANEL_SEED,
    build_dynamic_validation_panel,
    dynamic_validation_history_values,
    evaluate_dynamic_validation_panel,
)

logger = logging.getLogger(__name__)
TWO_PI = 2.0 * math.pi


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


def _extract_dataset_re_ids(dataset: Any) -> Optional[set[int]]:
    """Extract the set of unique integer Reynolds numbers present in a dataset.

    Supports SensorDataset, PyTorch TensorDataset, Subset wrappers, mappings, and 5-tuple containers.
    Returns None if reliable re_id metadata cannot be determined.
    """
    if dataset is None:
        return None

    if isinstance(dataset, Mapping):
        return set(int(k) for k in dataset.keys())

    # 1. Direct re_id attribute
    if hasattr(dataset, "re_id"):
        re_id_attr = getattr(dataset, "re_id")
        if isinstance(re_id_attr, torch.Tensor):
            return set(int(r) for r in re_id_attr.unique().tolist())
        if isinstance(re_id_attr, np.ndarray):
            return set(int(r) for r in np.unique(re_id_attr).tolist())
        if isinstance(re_id_attr, (list, tuple, set)):
            return set(int(r) for r in re_id_attr)

    # 2. PyTorch ConcatDataset wrapper
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        all_re: set[int] = set()
        for sub in dataset.datasets:
            sub_re = _extract_dataset_re_ids(sub)
            if sub_re is None:
                return None
            all_re.update(sub_re)
        return all_re

    # 3. PyTorch Subset wrapper
    if isinstance(dataset, torch.utils.data.Subset):
        underlying = dataset.dataset
        indices = list(dataset.indices)
        if hasattr(underlying, "re_id"):
            re_id_attr = getattr(underlying, "re_id")
            if isinstance(re_id_attr, torch.Tensor):
                return set(int(re_id_attr[i].item()) for i in indices)
            if isinstance(re_id_attr, np.ndarray):
                return set(int(re_id_attr[i]) for i in indices)
            if isinstance(re_id_attr, (list, tuple)):
                return set(int(re_id_attr[i]) for i in indices)
        if hasattr(underlying, "tensors") and isinstance(underlying.tensors, (tuple, list)) and len(underlying.tensors) >= 5:
            re_tensor = underlying.tensors[4]
            if isinstance(re_tensor, torch.Tensor):
                return set(int(re_tensor[i].item()) for i in indices)
            if isinstance(re_tensor, np.ndarray):
                return set(int(re_tensor[i]) for i in indices)
        if hasattr(underlying, "__getitem__"):
            try:
                return set(int(underlying[i][4]) for i in indices)
            except Exception:
                pass
        return _extract_dataset_re_ids(underlying)

    # 4. Direct TensorDataset inspection via tensors attribute
    if hasattr(dataset, "tensors") and isinstance(dataset.tensors, (tuple, list)):
        if len(dataset.tensors) >= 5:
            re_tensor = dataset.tensors[4]
            if isinstance(re_tensor, torch.Tensor):
                return set(int(r) for r in torch.unique(re_tensor).tolist())
            if isinstance(re_tensor, np.ndarray):
                return set(int(r) for r in np.unique(re_tensor).tolist())
            if isinstance(re_tensor, (list, tuple)):
                return set(int(r) for r in set(re_tensor))

    # 5. Direct complete inspection across ALL samples if indexable with 5-tuple
    if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        try:
            n = len(dataset)
            if n > 0:
                first = dataset[0]
                if isinstance(first, (tuple, list)) and len(first) >= 5:
                    return set(int(dataset[i][4]) for i in range(n))
        except Exception:
            pass

    return None


def _collect_leaf_datasets(dataset: Any) -> List[Any]:
    """Recursively unwrap Subsets and ConcatDatasets to collect all constituent leaf datasets."""
    if dataset is None:
        return []
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        leaves: List[Any] = []
        for sub in dataset.datasets:
            leaves.extend(_collect_leaf_datasets(sub))
        return leaves
    if isinstance(dataset, torch.utils.data.Subset):
        return _collect_leaf_datasets(getattr(dataset, "dataset", None))
    return [dataset]


def _get_dataset_attr(dataset: Any, attr_name: str) -> Any:
    """Retrieve an attribute from dataset, recursively unpacking Subset and ConcatDataset wrappers."""
    if dataset is None:
        return None
    if hasattr(dataset, attr_name):
        return getattr(dataset, attr_name)
    if isinstance(dataset, torch.utils.data.Subset):
        return _get_dataset_attr(getattr(dataset, "dataset", None), attr_name)
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        leaves = _collect_leaf_datasets(dataset)
        for leaf in leaves:
            val = getattr(leaf, attr_name, None)
            if val is not None:
                return val
    return None


def _normalizers_are_consistent(
    norm_a: Any,
    norm_b: Any,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> Tuple[bool, str]:
    """Verify that two Normalizer instances have matching methods and fitted statistics.

    Returns:
        Tuple `(is_consistent, reason_if_inconsistent)`.
    """
    if norm_a is norm_b:
        return True, ""
    if norm_a is None and norm_b is None:
        return True, ""
    if norm_a is None or norm_b is None:
        return False, "One normalizer is None while the other is not"

    method_a = str(getattr(norm_a, "method", "")).strip().lower()
    method_b = str(getattr(norm_b, "method", "")).strip().lower()
    if method_a != method_b:
        return False, f"Method mismatch: {method_a!r} vs {method_b!r}"

    fitted_a = bool(getattr(norm_a, "fitted", False))
    fitted_b = bool(getattr(norm_b, "fitted", False))
    if fitted_a != fitted_b:
        return False, f"Fitted status mismatch: {fitted_a} vs {fitted_b}"

    if not fitted_a:
        return True, ""

    if method_a == "none":
        return True, ""

    if method_a == "standard":
        mean_a = getattr(norm_a, "mean", None)
        mean_b = getattr(norm_b, "mean", None)
        if mean_a is None or mean_b is None:
            return False, "Missing mean in standard normalizer"
        arr_mean_a = np.asarray(mean_a, dtype=np.float64)
        arr_mean_b = np.asarray(mean_b, dtype=np.float64)
        if arr_mean_a.shape != arr_mean_b.shape:
            return False, f"Mean shape mismatch: {arr_mean_a.shape} vs {arr_mean_b.shape}"
        if not np.allclose(arr_mean_a, arr_mean_b, rtol=rtol, atol=atol):
            return False, f"Mean mismatch: {arr_mean_a.squeeze()} vs {arr_mean_b.squeeze()}"

        std_a = getattr(norm_a, "std", None)
        std_b = getattr(norm_b, "std", None)
        if std_a is None or std_b is None:
            return False, "Missing std in standard normalizer"
        arr_std_a = np.asarray(std_a, dtype=np.float64)
        arr_std_b = np.asarray(std_b, dtype=np.float64)
        if arr_std_a.shape != arr_std_b.shape:
            return False, f"Std shape mismatch: {arr_std_a.shape} vs {arr_std_b.shape}"
        if not np.allclose(arr_std_a, arr_std_b, rtol=rtol, atol=atol):
            return False, f"Std mismatch: {arr_std_a.squeeze()} vs {arr_std_b.squeeze()}"

    elif method_a == "minmax":
        min_a = getattr(norm_a, "min", None)
        min_b = getattr(norm_b, "min", None)
        if min_a is None or min_b is None:
            return False, "Missing min in minmax normalizer"
        arr_min_a = np.asarray(min_a, dtype=np.float64)
        arr_min_b = np.asarray(min_b, dtype=np.float64)
        if arr_min_a.shape != arr_min_b.shape:
            return False, f"Min shape mismatch: {arr_min_a.shape} vs {arr_min_b.shape}"
        if not np.allclose(arr_min_a, arr_min_b, rtol=rtol, atol=atol):
            return False, f"Min mismatch: {arr_min_a.squeeze()} vs {arr_min_b.squeeze()}"

        max_a = getattr(norm_a, "max", None)
        max_b = getattr(norm_b, "max", None)
        if max_a is None or max_b is None:
            return False, "Missing max in minmax normalizer"
        arr_max_a = np.asarray(max_a, dtype=np.float64)
        arr_max_b = np.asarray(max_b, dtype=np.float64)
        if arr_max_a.shape != arr_max_b.shape:
            return False, f"Max shape mismatch: {arr_max_a.shape} vs {arr_max_b.shape}"
        if not np.allclose(arr_max_a, arr_max_b, rtol=rtol, atol=atol):
            return False, f"Max mismatch: {arr_max_a.squeeze()} vs {arr_max_b.squeeze()}"

    return True, ""


@dataclass
class EncoderTrainingConfig:
    """Hyperparameters and configuration for sensor MLP training.

    Attributes:
        learning_rate: Initial Adam learning rate (default 1e-3).
        weight_decay: L2 regularization coefficient (default 0.0).
        batch_size: Mini-batch size (default 512).
        num_epochs: Total number of training epochs (default 200).
        lambda_re: Loss weighting factor for Reynolds number regression (default 1.0).
        lambda_phase: Loss weighting factor for circular phase sin/cos regression (default 1.0).
        seed: Random seed for training reproducibility (default 42).
        device: Target compute device ('auto', 'cuda', 'cpu').
        patience: Optional early-stopping patience (number of epochs without improvement).
        min_delta: Minimum loss decrease required to qualify as improvement (default 0.0,
            strictly matching legacy monotonic decrease checkpoint selection).
        save_best: Whether to automatically serialize best model checkpoint (default True).
        checkpoint_name: Checkpoint filename when saving (default 'best_model.pt').
        verbose: Whether to log training progress at each epoch (default False).
    """

    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 512
    num_epochs: int = 200
    lambda_re: float = 1.0
    lambda_phase: float = 1.0
    seed: int = 42
    device: str = "auto"
    patience: Optional[int] = None
    min_delta: float = 0.0
    save_best: bool = True
    checkpoint_name: str = "best_model.pt"
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        if self.lambda_re < 0.0 or self.lambda_phase < 0.0:
            raise ValueError("lambda_re and lambda_phase must be non-negative")
        if self.patience is not None and self.patience <= 0:
            raise ValueError("patience must be a positive integer if specified")


@dataclass
class TrainHistory:
    """Historical records of training and validation metrics across epochs.

    Attributes:
        epochs: List of epoch numbers (1-indexed).
        train_loss_total: Overall combined training loss per epoch.
        train_loss_re: Training MSE loss for Re component per epoch.
        train_loss_phase: Training MSE loss for sin/cos phase component per epoch.
        val_loss_total: Combined validation loss per epoch (NaN if no validation loader).
        val_loss_re: Validation Re loss per epoch.
        val_loss_phase: Validation phase loss per epoch.
        val_re_mae: Mean Absolute Error of physical Re on validation set.
        val_re_rmse: Root Mean Squared Error of physical Re on validation set.
        val_phase_cmae: Circular MAE of phase (radians) on validation set.
        val_phase_crmse: Circular RMSE of phase (radians) on validation set.
        epoch_wall_time_seconds: Elapsed time per epoch in seconds.
        best_epoch: Epoch index that achieved the best selection loss.
        best_selection_loss: Value of best loss achieved.
    """

    epochs: List[int] = field(default_factory=list)
    train_loss_total: List[float] = field(default_factory=list)
    train_loss_re: List[float] = field(default_factory=list)
    train_loss_phase: List[float] = field(default_factory=list)
    val_loss_total: List[float] = field(default_factory=list)
    val_loss_re: List[float] = field(default_factory=list)
    val_loss_phase: List[float] = field(default_factory=list)
    val_re_mae: List[float] = field(default_factory=list)
    val_re_rmse: List[float] = field(default_factory=list)
    val_phase_cmae: List[float] = field(default_factory=list)
    val_phase_crmse: List[float] = field(default_factory=list)
    epoch_wall_time_seconds: List[float] = field(default_factory=list)
    best_epoch: Optional[int] = None
    best_selection_loss: Optional[float] = None

    def add_epoch_record(
        self,
        epoch: int,
        train_metrics: Mapping[str, float],
        val_metrics: Optional[Mapping[str, Any]] = None,
        wall_time: float = 0.0,
    ) -> None:
        """Append metric values for a single completed epoch."""
        self.epochs.append(int(epoch))
        self.train_loss_total.append(float(train_metrics["loss_total"]))
        self.train_loss_re.append(float(train_metrics["loss_re"]))
        self.train_loss_phase.append(float(train_metrics["loss_phase"]))
        self.epoch_wall_time_seconds.append(float(wall_time))

        if val_metrics is not None:
            self.val_loss_total.append(float(val_metrics["loss_total"]))
            self.val_loss_re.append(float(val_metrics["loss_re"]))
            self.val_loss_phase.append(float(val_metrics["loss_phase"]))
            self.val_re_mae.append(float(val_metrics.get("re_mae", float("nan"))))
            self.val_re_rmse.append(float(val_metrics.get("re_rmse", float("nan"))))
            self.val_phase_cmae.append(float(val_metrics.get("phase_cmae", float("nan"))))
            self.val_phase_crmse.append(float(val_metrics.get("phase_crmse", float("nan"))))
        else:
            self.val_loss_total.append(float("nan"))
            self.val_loss_re.append(float("nan"))
            self.val_loss_phase.append(float("nan"))
            self.val_re_mae.append(float("nan"))
            self.val_re_rmse.append(float("nan"))
            self.val_phase_cmae.append(float("nan"))
            self.val_phase_crmse.append(float("nan"))

    def to_dict(self) -> Dict[str, Any]:
        """Convert history into a dictionary representation."""
        return asdict(self)

    def records(self) -> List[Dict[str, float]]:
        """Return list of row dictionaries for tabular output / CSV logging."""
        out: List[Dict[str, float]] = []
        for i, ep in enumerate(self.epochs):
            out.append(
                {
                    "epoch": float(ep),
                    "train_loss_total": self.train_loss_total[i],
                    "train_loss_re": self.train_loss_re[i],
                    "train_loss_phase": self.train_loss_phase[i],
                    "val_loss_total": self.val_loss_total[i],
                    "val_loss_re": self.val_loss_re[i],
                    "val_loss_phase": self.val_loss_phase[i],
                    "val_re_mae": self.val_re_mae[i],
                    "val_re_rmse": self.val_re_rmse[i],
                    "val_phase_cmae": self.val_phase_cmae[i],
                    "val_phase_crmse": self.val_phase_crmse[i],
                    "epoch_wall_time_seconds": self.epoch_wall_time_seconds[i],
                }
            )
        return out


class MLPTrainer:
    """Trainer for fixed-layout SensorMLP models.

    Coordinates mini-batch training, loss evaluation, physical validation metrics,
    and checkpoint saving while maintaining scientific isolation between partitions.
    """

    def __init__(self, config: Optional[EncoderTrainingConfig] = None) -> None:
        """Initialize MLPTrainer with configuration.

        Parameters:
            config: EncoderTrainingConfig instance. If None, default settings are used.
        """
        self.config = config or EncoderTrainingConfig()

    def resolve_device(self, device_str: Optional[str] = None) -> torch.device:
        """Resolve device string to torch.device object."""
        dev_str = (device_str or self.config.device).strip().lower()
        if dev_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(dev_str)

    def train_one_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> Dict[str, float]:
        """Execute one complete training epoch over the provided DataLoader.

        Parameters:
            model: SensorMLP model in training mode.
            loader: PyTorch DataLoader yielding `(x, y, y_re_raw, phase_true, re_id)`.
            optimizer: PyTorch Optimizer (e.g. Adam).
            device: Active computation device.

        Returns:
            Dictionary with keys: `{'loss_total', 'loss_re', 'loss_phase'}`.
        """
        model.train()
        mse = nn.MSELoss()

        sum_total = 0.0
        sum_re = 0.0
        sum_phase = 0.0
        n_seen = 0

        for batch in loader:
            x, y, _, _, _ = batch
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)  # [B, 3]

            loss_re = mse(pred[:, 0], y[:, 0])
            loss_phase = mse(pred[:, 1:3], y[:, 1:3])
            loss = self.config.lambda_re * loss_re + self.config.lambda_phase * loss_phase

            loss.backward()
            optimizer.step()

            bs = int(x.shape[0])
            n_seen += bs
            sum_total += float(loss.item()) * bs
            sum_re += float(loss_re.item()) * bs
            sum_phase += float(loss_phase.item()) * bs

        count = max(n_seen, 1)
        if n_seen == 0:
            raise ValueError(
                "Training DataLoader processed 0 samples (n_seen=0). "
                f"Check DataLoader batch_size={getattr(loader, 'batch_size', None)}, "
                f"drop_last={getattr(loader, 'drop_last', False)}, and dataset size. "
                "Cannot execute training on an empty batch sequence."
            )
        return {
            "loss_total": sum_total / count,
            "loss_re": sum_re / count,
            "loss_phase": sum_phase / count,
        }

    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        re_normalizer: Optional[Normalizer],
        device: torch.device,
    ) -> Dict[str, Any]:
        """Evaluate model performance and compute regression and circular phase metrics.

        Parameters:
            model: SensorMLP model.
            loader: PyTorch DataLoader yielding `(x, y, y_re_raw, phase_true, re_id)`.
            re_normalizer: Fitted Normalizer used to un-normalize Re predictions back
                to physical Reynolds numbers. If None or unfitted, raw predictions are kept.
            device: Active computation device.

        Returns:
            Dictionary containing losses, physical regression metrics (MAE, RMSE),
            circular phase metrics (cMAE, cRMSE), and accumulated prediction arrays.
        """
        model.eval()
        mse = nn.MSELoss()

        sum_total = 0.0
        sum_re = 0.0
        sum_phase = 0.0
        n_seen = 0

        re_true_all: List[np.ndarray] = []
        re_pred_all: List[np.ndarray] = []
        phase_true_all: List[np.ndarray] = []
        phase_pred_all: List[np.ndarray] = []
        re_id_all: List[np.ndarray] = []

        with torch.no_grad():
            for batch in loader:
                x, y, re_true_raw, phase_true, re_id = batch
                x = x.to(device)
                y = y.to(device)

                pred = model(x)  # [B, 3]

                loss_re = mse(pred[:, 0], y[:, 0])
                loss_phase = mse(pred[:, 1:3], y[:, 1:3])
                loss = self.config.lambda_re * loss_re + self.config.lambda_phase * loss_phase

                bs = int(x.shape[0])
                n_seen += bs
                sum_total += float(loss.item()) * bs
                sum_re += float(loss_re.item()) * bs
                sum_phase += float(loss_phase.item()) * bs

                # Un-normalize Re
                re_pred_norm_np = pred[:, 0].detach().cpu().numpy().astype(np.float64)
                if re_normalizer is not None and re_normalizer.fitted:
                    re_pred_raw_np = re_normalizer.inverse_transform(re_pred_norm_np)
                else:
                    re_pred_raw_np = re_pred_norm_np

                # Reconstruct phase from sin/cos via atan2 mapped to [0, 2*pi)
                sin_pred_np = pred[:, 1].detach().cpu().numpy().astype(np.float64)
                cos_pred_np = pred[:, 2].detach().cpu().numpy().astype(np.float64)
                phase_pred_np = np.mod(np.arctan2(sin_pred_np, cos_pred_np), TWO_PI)

                re_true_all.append(re_true_raw.detach().cpu().numpy().astype(np.float64))
                re_pred_all.append(re_pred_raw_np)
                phase_true_all.append(phase_true.detach().cpu().numpy().astype(np.float64))
                phase_pred_all.append(phase_pred_np)
                re_id_all.append(re_id.detach().cpu().numpy().astype(np.int64))

        count = max(n_seen, 1)
        if n_seen == 0:
            raise ValueError(
                "Validation DataLoader processed 0 samples (n_seen=0). "
                f"Check DataLoader batch_size={getattr(loader, 'batch_size', None)}, "
                f"drop_last={getattr(loader, 'drop_last', False)}, and dataset size. "
                "Cannot compute validation loss for checkpoint selection on an empty validation sequence."
            )
        re_true_arr = np.concatenate(re_true_all, axis=0) if re_true_all else np.empty((0,))
        re_pred_arr = np.concatenate(re_pred_all, axis=0) if re_pred_all else np.empty((0,))
        phase_true_arr = np.concatenate(phase_true_all, axis=0) if phase_true_all else np.empty((0,))
        phase_pred_arr = np.concatenate(phase_pred_all, axis=0) if phase_pred_all else np.empty((0,))
        re_id_arr = np.concatenate(re_id_all, axis=0) if re_id_all else np.empty((0,))

        re_metrics = (
            compute_regression_metrics(re_true_arr, re_pred_arr)
            if re_true_arr.size > 0
            else {"mae": float("nan"), "rmse": float("nan")}
        )
        phase_metrics = (
            compute_circular_metrics(phase_true_arr, phase_pred_arr)
            if phase_true_arr.size > 0
            else {"circular_mae": float("nan"), "circular_rmse": float("nan")}
        )

        return {
            "loss_total": sum_total / count,
            "loss_re": sum_re / count,
            "loss_phase": sum_phase / count,
            "re_mae": float(re_metrics["mae"]),
            "re_rmse": float(re_metrics["rmse"]),
            "phase_cmae": float(phase_metrics["circular_mae"]),
            "phase_crmse": float(phase_metrics["circular_rmse"]),
            "re_true": re_true_arr,
            "re_pred": re_pred_arr,
            "phase_true": phase_true_arr,
            "phase_pred": phase_pred_arr,
            "re_id": re_id_arr,
        }

    def fit(
        self,
        model: SensorMLP,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        x_normalizer: Optional[Normalizer] = None,
        re_normalizer: Optional[Normalizer] = None,
        sensor_indices: Optional[Sequence[int]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        model_variant: str = "main",
        train_repeat_id: int = 0,
        strategy: str = "unknown",
        selection_seed: Optional[int] = None,
        layout_id: Optional[str] = None,
        re_fold_id: Optional[str] = None,
        re_split: Optional[Union[ReSplit, Mapping[str, Sequence[int]]]] = None,
        candidate_theta_fingerprint: Optional[str] = None,
        extra_metadata: Optional[Mapping[str, Any]] = None,
        reseed_model: bool = False,
    ) -> Tuple[SensorMLP, TrainHistory]:
        """Execute full training loop with checkpoint selection and validation tracking.

        Parameters:
            model: SensorMLP instance to train.
            train_loader: Training set DataLoader.
            val_loader: Optional validation set DataLoader for checkpoint selection.
            x_normalizer: Pre-fitted normalizer for sensor features (for checkpoint metadata).
            re_normalizer: Pre-fitted normalizer for Re (for validation evaluation & checkpoint).
            sensor_indices: Sequence of sensor indices used by this model layout.
            output_dir: Optional directory where best checkpoint and logs are written.
            optimizer: Optional pre-configured Optimizer. If None, Adam is constructed.
            model_variant: Variant label for provenance tracking (default 'main').
            train_repeat_id: Training run repetition index (default 0).
            strategy: Layout selection strategy (e.g. 'uniform', 'greedy').
            selection_seed: Seed used during layout selection.
            layout_id: Hash or unique identifier for the layout.
            re_fold_id: Fold identifier string.
            re_split: Mapping or ReSplit defining train/val/test partitions.
            candidate_theta_fingerprint: Fingerprint of full candidate array.
            extra_metadata: Extra arbitrary metadata dictionary for checkpoint.
            reseed_model: If True, calls model.reset_parameters(seed=config.seed) to ensure
                model weights are strictly initialized by config.seed.

        Returns:
            Tuple `(best_model, train_history)`.
        """
        train_ds = getattr(train_loader, "dataset", None)
        val_ds = getattr(val_loader, "dataset", None) if val_loader is not None else None

        # Check DataLoader batch viability upfront
        if hasattr(train_loader, "__len__"):
            try:
                if len(train_loader) == 0:
                    raise ValueError(
                        f"Training DataLoader is empty (yields 0 batches). "
                        f"Check DataLoader batch_size={getattr(train_loader, 'batch_size', None)}, "
                        f"drop_last={getattr(train_loader, 'drop_last', False)}, and dataset size. "
                        "Training set must produce at least one batch."
                    )
            except TypeError:
                pass

        if val_loader is not None and hasattr(val_loader, "__len__"):
            try:
                if len(val_loader) == 0:
                    raise ValueError(
                        f"Validation DataLoader is empty (yields 0 batches). "
                        f"Check DataLoader batch_size={getattr(val_loader, 'batch_size', None)}, "
                        f"drop_last={getattr(val_loader, 'drop_last', False)}, and dataset size. "
                        "Cannot compute validation loss for checkpoint selection on an empty validation batch sequence."
                    )
            except TypeError:
                pass

        # 1. Extract and validate train and validation Re disjointness
        train_re_set = _extract_dataset_re_ids(train_ds)
        val_re_set = _extract_dataset_re_ids(val_ds) if val_ds is not None else set()

        if train_re_set is not None and val_re_set is not None and len(val_re_set) > 0:
            overlap = train_re_set & val_re_set
            if overlap:
                raise ValueError(
                    f"Train and validation splits must be strictly disjoint, but found overlapping Re: {sorted(overlap)}"
                )

        # 2. Validate Re partition contracts and locked test Re integrity
        default_locked = {250, 650}
        if train_re_set and (train_re_set & default_locked):
            raise PermissionError(
                f"Test Re={sorted(train_re_set & default_locked)} is strictly locked and cannot appear in training split!"
            )
        if val_re_set and (val_re_set & default_locked):
            raise PermissionError(
                f"Test Re={sorted(val_re_set & default_locked)} is strictly locked and cannot appear in validation split!"
            )

        # Resolve partition split: fallback to default fold from configuration if omitted
        effective_split = re_split
        if effective_split is None:
            try:
                from cylinder_flow.data.splits import select_re_fold
                effective_split = select_re_fold("default")
            except Exception:
                effective_split = ReSplit(
                    train_re=[200, 225, 275, 300, 350, 400, 500, 550, 600, 700, 750, 850, 900, 1000],
                    val_re=[450, 800],
                    test_re=[250, 650],
                    fold_id="default",
                )
        elif isinstance(effective_split, Mapping):
            train_part = [int(x) for x in effective_split.get("train", effective_split.get("train_re", []))]
            val_part = [int(x) for x in effective_split.get("val", effective_split.get("val_re", []))]
            test_part = [int(x) for x in effective_split.get("test", effective_split.get("test_re", []))]
            fold_id = str(effective_split.get("fold_id", "custom"))
            effective_split = ReSplit(
                train_re=train_part,
                val_re=val_part,
                test_re=test_part,
                fold_id=fold_id,
            )

        if effective_split is not None:
            if train_re_set is None:
                raise ValueError(
                    "Cannot verify Re partition compliance: train_loader dataset does not provide re_id metadata."
                )
            if val_loader is not None and val_re_set is None:
                raise ValueError(
                    "Cannot verify Re partition compliance: val_loader dataset does not provide re_id metadata."
                )

            if isinstance(effective_split, ReSplit):
                effective_split.validate_disjoint(strict=True)
                test_re_set = set(int(r) for r in effective_split.test_re)
                train_decl = set(int(r) for r in effective_split.train_re)
                val_decl = set(int(r) for r in effective_split.val_re)
            else:
                test_re_set = set(int(r) for r in effective_split.get("test", effective_split.get("test_re", [])))
                train_decl = set(int(r) for r in effective_split.get("train", effective_split.get("train_re", [])))
                val_decl = set(int(r) for r in effective_split.get("val", effective_split.get("val_re", [])))
                ov = (train_decl & val_decl) | (train_decl & test_re_set) | (val_decl & test_re_set)
                if ov:
                    raise ValueError(f"Declared re_split has overlapping partitions: {sorted(ov)}")

            # Enforce that train_loader samples strictly belong to train_decl
            non_train = train_re_set - train_decl
            if non_train:
                if non_train & test_re_set:
                    raise PermissionError(
                        f"Test Re={sorted(non_train & test_re_set)} is strictly locked and cannot appear in training split!"
                    )
                raise ValueError(
                    f"train_loader contains Re={sorted(non_train)} which are not in declared train_re {sorted(train_decl)}! "
                    "Check for swapped loaders or partition leakage."
                )

            # Enforce that val_loader samples strictly belong to val_decl
            if val_re_set:
                non_val = val_re_set - val_decl
                if non_val:
                    if non_val & test_re_set:
                        raise PermissionError(
                            f"Test Re={sorted(non_val & test_re_set)} is strictly locked and cannot appear in validation split!"
                        )
                    raise ValueError(
                        f"val_loader contains Re={sorted(non_val)} which are not in declared val_re {sorted(val_decl)}! "
                        "Check for swapped loaders or partition leakage."
                    )

        # 3. Inherit or validate metadata against training dataset (including composite/ConcatDataset)
        train_leaves = _collect_leaf_datasets(train_ds) if train_ds is not None else []
        for idx, leaf in enumerate(train_leaves):
            leaf_indices = getattr(leaf, "sensor_indices", None)
            leaf_x_norm = getattr(leaf, "x_normalizer", None)
            leaf_re_norm = getattr(leaf, "re_normalizer", None)
            if len(train_leaves) > 1 and (
                leaf_indices is None or leaf_x_norm is None or leaf_re_norm is None
            ):
                raise ValueError(
                    f"Training dataset component {idx} ({type(leaf).__name__}) must provide "
                    "sensor_indices, x_normalizer, and re_normalizer metadata."
                )

            if leaf_indices is not None:
                leaf_indices_list = list(leaf_indices)
                if sensor_indices is None:
                    sensor_indices = leaf_indices_list
                elif list(sensor_indices) != leaf_indices_list:
                    raise ValueError(
                        f"Explicit sensor_indices={list(sensor_indices)} conflicts with dataset "
                        f"sensor_indices={leaf_indices_list} (component {idx}: {type(leaf).__name__})! "
                        "Refusing to train or save checkpoint with mismatched layouts."
                    )

            if leaf_x_norm is not None:
                if x_normalizer is None:
                    x_normalizer = leaf_x_norm
                else:
                    ok, reason = _normalizers_are_consistent(x_normalizer, leaf_x_norm)
                    if not ok:
                        raise ValueError(
                            f"Explicit x_normalizer conflicts with training dataset x_normalizer ({reason})! "
                            f"Dataset statistics take precedence; inconsistent explicit normalizers are rejected."
                        )

            if leaf_re_norm is not None:
                if re_normalizer is None:
                    re_normalizer = leaf_re_norm
                else:
                    ok, reason = _normalizers_are_consistent(re_normalizer, leaf_re_norm)
                    if not ok:
                        raise ValueError(
                            f"Explicit re_normalizer conflicts with training dataset re_normalizer ({reason})! "
                            f"Dataset statistics take precedence; inconsistent explicit normalizers are rejected."
                        )

        # 4. Validate validation dataset layout and normalizer consistency against training configuration
        if val_ds is not None:
            val_leaves = _collect_leaf_datasets(val_ds)
            target_indices = sensor_indices
            target_x_norm = x_normalizer
            target_re_norm = re_normalizer

            for idx, leaf in enumerate(val_leaves):
                leaf_indices = getattr(leaf, "sensor_indices", None)
                if target_indices is not None:
                    if leaf_indices is None:
                        raise ValueError(
                            f"Validation dataset component {idx} ({type(leaf).__name__}) does not provide "
                            f"sensor_indices metadata! Refusing to evaluate or select checkpoint with unverifiable validation layout."
                        )
                    if list(leaf_indices) != list(target_indices):
                        raise ValueError(
                            f"Validation dataset component {idx} ({type(leaf).__name__}) sensor_indices={list(leaf_indices)} "
                            f"conflicts with training sensor_indices={list(target_indices)}! "
                            f"Mismatched sensor layouts across validation subsets are forbidden."
                        )

                leaf_x_norm = getattr(leaf, "x_normalizer", None)
                if target_x_norm is not None:
                    if leaf_x_norm is None:
                        raise ValueError(
                            f"Validation dataset component {idx} ({type(leaf).__name__}) does not provide "
                            f"x_normalizer metadata! Refusing to evaluate or select checkpoint with unverifiable validation scaling."
                        )
                    ok, reason = _normalizers_are_consistent(leaf_x_norm, target_x_norm)
                    if not ok:
                        raise ValueError(
                            f"Validation dataset component {idx} ({type(leaf).__name__}) x_normalizer conflicts with "
                            f"training x_normalizer ({reason})! All validation components must share identical sensor normalization statistics."
                        )

                leaf_re_norm = getattr(leaf, "re_normalizer", None)
                if target_re_norm is not None:
                    if leaf_re_norm is None:
                        raise ValueError(
                            f"Validation dataset component {idx} ({type(leaf).__name__}) does not provide "
                            f"re_normalizer metadata! Refusing to evaluate or select checkpoint with unverifiable validation scaling."
                        )
                    ok, reason = _normalizers_are_consistent(leaf_re_norm, target_re_norm)
                    if not ok:
                        raise ValueError(
                            f"Validation dataset component {idx} ({type(leaf).__name__}) re_normalizer conflicts with "
                            f"training re_normalizer ({reason})! All validation components must share identical Re normalization statistics."
                        )

        out_path: Optional[Path] = Path(output_dir) if output_dir is not None else None
        if out_path is not None:
            out_path.mkdir(parents=True, exist_ok=True)

        # 4. Strict checkpoint metadata guard: refuse to save corrupt/arbitrary fallbacks
        if out_path is not None and self.config.save_best:
            if sensor_indices is None or len(sensor_indices) == 0:
                raise ValueError(
                    "Cannot save MLP checkpoint: sensor_indices is missing and could not be inferred from dataset. "
                    "Saving with arbitrary fallback indices is forbidden to prevent downstream layout misalignment."
                )
            if x_normalizer is None or not getattr(x_normalizer, "fitted", False):
                raise ValueError(
                    "Cannot save MLP checkpoint: x_normalizer is missing or unfitted. "
                    "Saving with identity/unfitted normalization is forbidden to prevent downstream scaling errors."
                )
            if re_normalizer is None or not getattr(re_normalizer, "fitted", False):
                raise ValueError(
                    "Cannot save MLP checkpoint: re_normalizer is missing or unfitted. "
                    "Saving with identity/unfitted normalization is forbidden to prevent downstream scaling errors."
                )

        if reseed_model:
            model.reset_parameters(seed=self.config.seed)

        # 5. Reset DataLoader generator seeds if present to guarantee identical batch sequence on reuse
        if hasattr(train_loader, "generator") and train_loader.generator is not None:
            train_loader.generator.manual_seed(int(self.config.seed))
        if val_loader is not None and hasattr(val_loader, "generator") and val_loader.generator is not None:
            val_loader.generator.manual_seed(int(self.config.seed))

        device = self.resolve_device(self.config.device)
        model = model.to(device)

        if optimizer is None:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

        history = TrainHistory()
        best_selection_loss = float("inf")
        best_epoch = -1
        best_state_dict = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0

        x_norm = x_normalizer or Normalizer(method="none", fitted=True)
        re_norm = re_normalizer or Normalizer(method="none", fitted=True)
        active_sensor_indices = (
            list(sensor_indices) if sensor_indices is not None else list(range(model.input_dim // 3))
        )

        for epoch in range(1, self.config.num_epochs + 1):
            t0 = time.perf_counter()
            train_out = self.train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
            )
            wall_time = float(time.perf_counter() - t0)

            val_out: Optional[Dict[str, Any]] = None
            if val_loader is not None:
                val_out = self.evaluate(
                    model=model,
                    loader=val_loader,
                    re_normalizer=re_norm,
                    device=device,
                )

            history.add_epoch_record(
                epoch=epoch,
                train_metrics=train_out,
                val_metrics=val_out,
                wall_time=wall_time,
            )

            # Checkpoint selection is based on validation total loss, or train total loss if no val
            selection_loss = (
                float(val_out["loss_total"]) if val_out is not None else float(train_out["loss_total"])
            )

            if selection_loss < (best_selection_loss - self.config.min_delta):
                best_selection_loss = selection_loss
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0

                if out_path is not None and self.config.save_best:
                    ckpt_file = out_path / self.config.checkpoint_name
                    save_mlp_checkpoint(
                        path=ckpt_file,
                        model=model,
                        x_normalizer=x_norm,
                        re_normalizer=re_norm,
                        sensor_indices=active_sensor_indices,
                        epoch=epoch,
                        train_seed=self.config.seed,
                        model_variant=model_variant,
                        train_repeat_id=train_repeat_id,
                        strategy=strategy,
                        selection_seed=selection_seed,
                        layout_id=layout_id,
                        re_fold_id=re_fold_id,
                        re_split=effective_split,
                        candidate_theta_fingerprint=candidate_theta_fingerprint,
                        training_protocol=asdict(self.config),
                        extra_metadata=extra_metadata,
                    )
            else:
                epochs_without_improvement += 1

            if self.config.verbose:
                val_str = (
                    f"val_loss={val_out['loss_total']:.6f} re_RMSE={val_out['re_rmse']:.3f} "
                    f"phase_cRMSE={val_out['phase_crmse']:.4f}"
                    if val_out is not None
                    else "val=None"
                )
                logger.info(
                    "Epoch %d/%d | train_loss=%.6f | %s | %.2fs",
                    epoch,
                    self.config.num_epochs,
                    train_out["loss_total"],
                    val_str,
                    wall_time,
                )

            if (
                self.config.patience is not None
                and epochs_without_improvement >= self.config.patience
            ):
                logger.info(
                    "Early stopping triggered at epoch %d (best epoch was %d, best loss %.6f)",
                    epoch,
                    best_epoch,
                    best_selection_loss,
                )
                break

        # Re-load the best checkpoint weights into model before returning
        model.load_state_dict(best_state_dict)
        history.best_epoch = best_epoch
        history.best_selection_loss = best_selection_loss

        return model, history


def train_sensor_mlp(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    config: Optional[EncoderTrainingConfig] = None,
    hidden_dims: Sequence[int] = (128, 128, 128),
    activation: str = "relu",
    sensor_indices: Optional[Sequence[int]] = None,
    x_normalizer: Optional[Normalizer] = None,
    re_normalizer: Optional[Normalizer] = None,
    re_split: Optional[Union[ReSplit, Mapping[str, Sequence[int]]]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    model_variant: str = "main",
    train_repeat_id: int = 0,
    strategy: str = "unknown",
    selection_seed: Optional[int] = None,
    layout_id: Optional[str] = None,
    re_fold_id: Optional[str] = None,
    candidate_theta_fingerprint: Optional[str] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Tuple[SensorMLP, TrainHistory]:
    """End-to-end training entrypoint with strict seed-controlled initialization and dataset inheritance.

    Guarantees:
    1. Global seeds set before model construction, ensuring model weights are deterministic.
    2. Inherits sensor indices and normalizers from train_loader.dataset when omitted.
    3. Re split partition disjointness and test-lock defense checked before execution.
    4. Best epoch selection uses zero-delta strict monotonic decrease (legacy parity).

    Parameters:
        train_loader: DataLoader for training partition.
        val_loader: Optional DataLoader for validation partition.
        config: Training hyperparameters (EncoderTrainingConfig).
        hidden_dims: MLP hidden layer dimensions.
        activation: Activation function name (e.g. 'relu', 'tanh').
        sensor_indices: Sensor indices. If None, inferred from dataset.
        x_normalizer: Input normalizer. If None, inferred from dataset.
        re_normalizer: Re normalizer. If None, inferred from dataset.
        re_split: Split metadata or ReSplit contract object.
        output_dir: Directory where checkpoints and metrics are saved.
        model_variant: Variant label for checkpoint provenance.
        train_repeat_id: Repetition index.
        strategy: Layout strategy name.
        selection_seed: Seed used during layout search.
        layout_id: Hash or unique ID for layout.
        re_fold_id: Reynolds split fold identifier.
        candidate_theta_fingerprint: Candidate angle fingerprint.
        extra_metadata: Additional dictionary written to checkpoint.

    Returns:
        Tuple `(best_model, train_history)`.
    """
    cfg = config or EncoderTrainingConfig()
    set_training_seed(cfg.seed)

    train_ds = getattr(train_loader, "dataset", None)
    if sensor_indices is None and train_ds is not None:
        sensor_indices = _get_dataset_attr(train_ds, "sensor_indices")
    if x_normalizer is None and train_ds is not None:
        x_normalizer = _get_dataset_attr(train_ds, "x_normalizer")
    if re_normalizer is None and train_ds is not None:
        re_normalizer = _get_dataset_attr(train_ds, "re_normalizer")

    if sensor_indices is not None and len(sensor_indices) > 0:
        input_dim = 3 * len(sensor_indices)
    elif train_ds is not None and hasattr(train_ds, "x"):
        input_dim = int(train_ds.x.shape[1])
    else:
        sample_batch = next(iter(train_loader))
        input_dim = int(sample_batch[0].shape[1])

    model = SensorMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=3,
        activation=activation,
        seed=cfg.seed,
    )

    trainer = MLPTrainer(config=cfg)
    return trainer.fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        x_normalizer=x_normalizer,
        re_normalizer=re_normalizer,
        sensor_indices=sensor_indices,
        output_dir=output_dir,
        model_variant=model_variant,
        train_repeat_id=train_repeat_id,
        strategy=strategy,
        selection_seed=selection_seed,
        layout_id=layout_id,
        re_fold_id=re_fold_id,
        re_split=re_split,
        candidate_theta_fingerprint=candidate_theta_fingerprint,
        extra_metadata=extra_metadata,
        reseed_model=False,
    )


AUTHORITATIVE_DEEPSETS_METADATA_KEYS = {
    "epoch",
    "model_state_dict",
    "input_dim",
    "hidden_dims",
    "sensor_indices",
    "sensor_channels",
    "default_sensor_indices",
    "schema_version",
    "encoder",
    "set_embed_dim",
    "set_phi_hidden",
    "set_rho_hidden",
    "pool",
    "set_pool",
    "position_encoding",
    "count_feature",
    "candidate_count",
    "candidate_positions",
    "candidate_position_fingerprint",
    "candidate_position_fingerprint_method",
    "channel_order",
    "chan_mean",
    "chan_std",
    "channel_mean",
    "channel_std",
    "channel_stats_source",
    "channel_stats_re_ids",
    "training_sensor_counts",
    "dynamic_contract_fingerprint",
    "set_activation",
    "x_normalizer",
    "re_normalizer",
    "train_seed",
    "seed",
    "output_dim",
    "nsensor",
    "selection_metric",
    "selection_metric_name",
    "selection_metric_value",
    "best_selection_loss",
    "selection_score",
    "smoke_not_for_formal",
    "declared_train_re",
    "declared_val_re",
    "declared_test_re",
    "defer_test_evaluation",
    "locked_test_opened",
    "collator_snapshot",
    "sampler_snapshot",
    "validation_score",
    "validation_panel_summary",
    "validation_metrics",
    "training_protocol",
    "token_dropout_seed",
    "sensor_noise_seed",
    "layout_sampler_seed",
    "actual_augmentation_seeds",
    "sample_distribution",
    "batch_distribution",
    "effective_count_distribution",
    "dynamic_validation_panel",
    "dynamic_validation_panel_evaluation",
    "validation_evaluation",
}


def _unwrap_to_dynamic_dataset(val_data: Any) -> Any:
    """Unwrap a DataLoader or Subset to expose underlying DynamicSensorDataset or mapping."""
    if isinstance(val_data, DataLoader):
        val_data = val_data.dataset
    if isinstance(val_data, torch.utils.data.Subset):
        underlying = val_data.dataset
        if isinstance(underlying, DynamicSensorDataset):
            indices = np.asarray(val_data.indices)
            return DynamicSensorDataset(
                values=underlying.values[indices],
                positions=underlying.positions,
                y_re_norm=underlying.y[indices, 0],
                y_sin=underlying.y[indices, 1],
                y_cos=underlying.y[indices, 2],
                y_re_raw=underlying.y_re_raw[indices],
                phase_true=underlying.phase_true[indices],
                re_id=underlying.re_id[indices],
                candidate_positions=underlying.candidate_positions,
                chan_mean=underlying.chan_mean,
                chan_std=underlying.chan_std,
                re_normalizer=underlying.re_normalizer,
                stats_train_re=underlying.stats_train_re,
            )
    return val_data


@dataclass
class DeepSetsTrainingConfig:
    """Hyperparameters and configuration for DeepSets sensor encoder training.

    Attributes:
        learning_rate: Initial Adam learning rate (default 1e-3).
        weight_decay: L2 regularization coefficient (default 0.0).
        batch_size: Mini-batch size (default 512).
        num_epochs: Total number of training epochs (default 200).
        lambda_re: Loss weighting factor for Reynolds number regression (default 1.0).
        lambda_phase: Loss weighting factor for circular phase sin/cos regression (default 1.0).
        seed: Random seed for training reproducibility (default 42).
        device: Target compute device ('auto', 'cuda', 'cpu').
        patience: Optional early-stopping patience (number of epochs without improvement).
        min_delta: Minimum loss decrease required to qualify as improvement (default 0.0,
            strictly matching legacy monotonic decrease checkpoint selection).
        save_best: Whether to automatically serialize best model checkpoint (default True).
        checkpoint_name: Checkpoint filename when saving (default 'best_model.pt').
        verbose: Whether to log training progress at each epoch (default False).
        layout_sampling: Layout sampling strategy ('random-subset' or 'fixed', default 'random-subset').
        training_sensor_counts: Sensor counts used during training (default (6, 12, 24, 48)).
        token_dropout_prob: Per-token dropout probability in [0, 1) (default 0.0).
        token_dropout_seed: Optional seed for token dropout RNG.
        sensor_noise_rel_std: Relative sensor noise std (default 0.0).
        sensor_noise_seed: Optional seed for sensor noise RNG.
        layout_sampler_seed: Optional seed for layout sampler RNG.
        schema_version: Checkpoint schema version (default 2, strictly ReLU).
        smoke_not_for_formal: Explicit flag marking lightweight/smoke training mode.
            Formal mode (False) strictly enforces validation data and frozen 60-candidate panel.
        chunk_size: Snapshot chunk size for dynamic validation evaluation (default 8192).
        validation_mode: Validation evaluation mode ('frozen_panel' or 'standard', default 'frozen_panel').
    """

    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 512
    num_epochs: int = 200
    lambda_re: float = 1.0
    lambda_phase: float = 1.0
    seed: int = 42
    device: str = "auto"
    patience: Optional[int] = None
    min_delta: float = 0.0
    save_best: bool = True
    checkpoint_name: str = "best_model.pt"
    verbose: bool = False
    layout_sampling: str = "random-subset"
    training_sensor_counts: Sequence[int] = (6, 12, 24, 48)
    token_dropout_prob: float = 0.0
    token_dropout_seed: Optional[int] = None
    sensor_noise_rel_std: float = 0.0
    sensor_noise_seed: Optional[int] = None
    layout_sampler_seed: Optional[int] = None
    schema_version: int = 2
    smoke_not_for_formal: bool = False
    chunk_size: int = 8192
    validation_mode: str = "frozen_panel"

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        if self.lambda_re < 0.0 or self.lambda_phase < 0.0:
            raise ValueError("lambda_re and lambda_phase must be non-negative")
        if self.patience is not None and self.patience <= 0:
            raise ValueError("patience must be a positive integer if specified")
        if not (0.0 <= self.token_dropout_prob < 1.0):
            raise ValueError(f"token_dropout_prob must be in [0, 1), got {self.token_dropout_prob}")
        if self.sensor_noise_rel_std < 0.0:
            raise ValueError(f"sensor_noise_rel_std must be non-negative, got {self.sensor_noise_rel_std}")
        if self.schema_version not in (1, 2):
            raise ValueError(f"schema_version must be 1 or 2, got {self.schema_version}")
        self.training_sensor_counts = tuple(sorted({int(x) for x in self.training_sensor_counts}))
        if not self.training_sensor_counts or any(c <= 0 for c in self.training_sensor_counts):
            raise ValueError("training_sensor_counts must contain positive integers")


@dataclass
class DeepSetsTrainHistory(TrainHistory):
    """Historical records of DeepSets training with frozen panel validation tracking."""

    val_equal_budget_selection_score: List[float] = field(default_factory=list)
    val_n6_loss_total: List[float] = field(default_factory=list)
    val_n6_re_rmse: List[float] = field(default_factory=list)
    val_n6_phase_crmse: List[float] = field(default_factory=list)
    val_n12_loss_total: List[float] = field(default_factory=list)
    val_n12_re_rmse: List[float] = field(default_factory=list)
    val_n12_phase_crmse: List[float] = field(default_factory=list)
    val_n24_loss_total: List[float] = field(default_factory=list)
    val_n24_re_rmse: List[float] = field(default_factory=list)
    val_n24_phase_crmse: List[float] = field(default_factory=list)
    val_n48_loss_total: List[float] = field(default_factory=list)
    val_n48_re_rmse: List[float] = field(default_factory=list)
    val_n48_phase_crmse: List[float] = field(default_factory=list)

    def add_epoch_record(
        self,
        epoch: int,
        train_metrics: Mapping[str, float],
        val_metrics: Optional[Mapping[str, Any]] = None,
        dynamic_metrics: Optional[Mapping[str, float]] = None,
        wall_time: float = 0.0,
    ) -> None:
        """Append metric values for a single completed epoch."""
        super().add_epoch_record(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            wall_time=wall_time,
        )
        if dynamic_metrics is not None:
            score = float(dynamic_metrics["val_equal_budget_selection_score"])
            self.val_equal_budget_selection_score.append(score)
            if len(self.val_loss_total) == len(self.val_equal_budget_selection_score):
                self.val_loss_total[-1] = score
            for nsensor in (6, 12, 24, 48):
                lt = float(dynamic_metrics.get(f"val_n{nsensor}_loss_total", float("nan")))
                re_err = float(dynamic_metrics.get(f"val_n{nsensor}_re_rmse", float("nan")))
                ph_err = float(dynamic_metrics.get(f"val_n{nsensor}_phase_crmse", float("nan")))
                getattr(self, f"val_n{nsensor}_loss_total").append(lt)
                getattr(self, f"val_n{nsensor}_re_rmse").append(re_err)
                getattr(self, f"val_n{nsensor}_phase_crmse").append(ph_err)
        else:
            self.val_equal_budget_selection_score.append(float("nan"))
            for nsensor in (6, 12, 24, 48):
                getattr(self, f"val_n{nsensor}_loss_total").append(float("nan"))
                getattr(self, f"val_n{nsensor}_re_rmse").append(float("nan"))
                getattr(self, f"val_n{nsensor}_phase_crmse").append(float("nan"))

    def records(self) -> List[Dict[str, float]]:
        """Return list of row dictionaries including dynamic panel validation metrics."""
        base_records = super().records()
        for i, row in enumerate(base_records):
            row["val_equal_budget_selection_score"] = self.val_equal_budget_selection_score[i]
            for nsensor in (6, 12, 24, 48):
                row[f"val_n{nsensor}_loss_total"] = getattr(self, f"val_n{nsensor}_loss_total")[i]
                row[f"val_n{nsensor}_re_rmse"] = getattr(self, f"val_n{nsensor}_re_rmse")[i]
                row[f"val_n{nsensor}_phase_crmse"] = getattr(self, f"val_n{nsensor}_phase_crmse")[i]
        return base_records


class DeepSetsTrainer:
    """Trainer for DeepSets sensor encoders with dynamic subset sampling and frozen panel validation.

    Supports both dynamic token encoders (:class:`DynamicSetSensorEncoder`) and fixed-layout
    set encoders (:class:`SetSensorEncoder`). Coordinates mini-batch training, frozen validation
    panel evaluation (seed 400042, budgets [6, 12, 24, 48]), strict monotonic checkpoint
    selection (min_delta=0.0), and state restoration.
    """

    def __init__(self, config: Optional[DeepSetsTrainingConfig] = None) -> None:
        self.config = config or DeepSetsTrainingConfig()

    def resolve_device(self, device_str: Optional[str] = None) -> torch.device:
        """Resolve device string to torch.device object."""
        dev_str = (device_str or self.config.device).strip().lower()
        if dev_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(dev_str)

    def train_one_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> Dict[str, float]:
        """Execute one complete training epoch over the provided DataLoader.

        Supports dynamic 8-tuple batches, fixed 5-tuple batches, and DynamicSensorBatch containers.
        """
        model.train()
        mse = nn.MSELoss()

        sum_total = 0.0
        sum_re = 0.0
        sum_phase = 0.0
        n_seen = 0

        for batch in loader:
            if isinstance(batch, (tuple, list)):
                if len(batch) == 8:
                    values, positions, mask, count, y, _, _, _ = batch
                    values = values.to(device)
                    positions = positions.to(device)
                    mask = mask.to(device)
                    count = count.to(device)
                    y = y.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    if hasattr(model, "forward_tokens"):
                        pred = model.forward_tokens(values, positions, mask, count)
                    elif isinstance(model, SetSensorEncoder):
                        x_flat = values.reshape(values.shape[0], -1)
                        pred = model(x_flat)
                    else:
                        pred = model(values, positions, mask, count)
                elif len(batch) == 5:
                    x, y, _, _, _ = batch
                    x = x.to(device)
                    y = y.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    pred = model(x)
                else:
                    raise ValueError(f"Unexpected batch tuple length: {len(batch)}")
            elif isinstance(batch, DynamicSensorBatch):
                y = batch.y.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(batch)
            else:
                raise TypeError(f"Unsupported batch type in DeepSets training: {type(batch)}")

            loss_re = mse(pred[:, 0], y[:, 0])
            loss_phase = mse(pred[:, 1:3], y[:, 1:3])
            loss = self.config.lambda_re * loss_re + self.config.lambda_phase * loss_phase

            loss.backward()
            optimizer.step()

            bs = int(pred.shape[0])
            n_seen += bs
            sum_total += float(loss.item()) * bs
            sum_re += float(loss_re.item()) * bs
            sum_phase += float(loss_phase.item()) * bs

        count_seen = max(n_seen, 1)
        if n_seen == 0:
            raise ValueError(
                "Training DataLoader processed 0 samples (n_seen=0). "
                f"Check DataLoader batch_size={getattr(loader, 'batch_size', None)}, "
                f"drop_last={getattr(loader, 'drop_last', False)}, and dataset size. "
                "Cannot execute training on an empty batch sequence."
            )
        return {
            "loss_total": sum_total / count_seen,
            "loss_re": sum_re / count_seen,
            "loss_phase": sum_phase / count_seen,
        }

    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        re_normalizer: Optional[Normalizer],
        device: torch.device,
    ) -> Dict[str, Any]:
        """Evaluate model performance on a DataLoader (standard evaluation mode)."""
        model.eval()
        mse = nn.MSELoss()

        sum_total = 0.0
        sum_re = 0.0
        sum_phase = 0.0
        n_seen = 0

        re_true_all: List[np.ndarray] = []
        re_pred_all: List[np.ndarray] = []
        phase_true_all: List[np.ndarray] = []
        phase_pred_all: List[np.ndarray] = []
        re_id_all: List[np.ndarray] = []

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (tuple, list)):
                    if len(batch) == 8:
                        values, positions, mask, count, y, re_true_raw, phase_true, re_id = batch
                        values = values.to(device)
                        positions = positions.to(device)
                        mask = mask.to(device)
                        count = count.to(device)
                        y = y.to(device)
                        if hasattr(model, "forward_tokens"):
                            pred = model.forward_tokens(values, positions, mask, count)
                        elif isinstance(model, SetSensorEncoder):
                            x_flat = values.reshape(values.shape[0], -1)
                            pred = model(x_flat)
                        else:
                            pred = model(values, positions, mask, count)
                    elif len(batch) == 5:
                        x, y, re_true_raw, phase_true, re_id = batch
                        x = x.to(device)
                        y = y.to(device)
                        pred = model(x)
                    else:
                        raise ValueError(f"Unexpected batch tuple length: {len(batch)}")
                elif isinstance(batch, DynamicSensorBatch):
                    pred = model(batch)
                    y = batch.y.to(device)
                    re_true_raw = getattr(batch, "re_raw", None)
                    phase_true = getattr(batch, "phase_true", None)
                    re_id = getattr(batch, "re_id", None)
                else:
                    raise TypeError(f"Unsupported batch type in evaluation: {type(batch)}")

                loss_re = mse(pred[:, 0], y[:, 0])
                loss_phase = mse(pred[:, 1:3], y[:, 1:3])
                loss = self.config.lambda_re * loss_re + self.config.lambda_phase * loss_phase

                bs = int(pred.shape[0])
                n_seen += bs
                sum_total += float(loss.item()) * bs
                sum_re += float(loss_re.item()) * bs
                sum_phase += float(loss_phase.item()) * bs

                # Un-normalize Re
                re_pred_norm_np = pred[:, 0].detach().cpu().numpy().astype(np.float64)
                if re_normalizer is not None and getattr(re_normalizer, "fitted", False):
                    re_pred_raw_np = re_normalizer.inverse_transform(re_pred_norm_np)
                else:
                    re_pred_raw_np = re_pred_norm_np

                sin_pred_np = pred[:, 1].detach().cpu().numpy().astype(np.float64)
                cos_pred_np = pred[:, 2].detach().cpu().numpy().astype(np.float64)
                phase_pred_np = np.mod(np.arctan2(sin_pred_np, cos_pred_np), TWO_PI)

                if re_true_raw is not None:
                    if isinstance(re_true_raw, torch.Tensor):
                        re_true_all.append(re_true_raw.detach().cpu().numpy().astype(np.float64))
                    else:
                        re_true_all.append(np.asarray(re_true_raw, dtype=np.float64))
                re_pred_all.append(re_pred_raw_np)

                if phase_true is not None:
                    if isinstance(phase_true, torch.Tensor):
                        phase_true_all.append(phase_true.detach().cpu().numpy().astype(np.float64))
                    else:
                        phase_true_all.append(np.asarray(phase_true, dtype=np.float64))
                phase_pred_all.append(phase_pred_np)

                if re_id is not None:
                    if isinstance(re_id, torch.Tensor):
                        re_id_all.append(re_id.detach().cpu().numpy().astype(np.int64))
                    else:
                        re_id_all.append(np.asarray(re_id, dtype=np.int64))

        count_seen = max(n_seen, 1)
        if n_seen == 0:
            raise ValueError(
                "Validation DataLoader processed 0 samples (n_seen=0). "
                f"Check DataLoader batch_size={getattr(loader, 'batch_size', None)}, "
                f"drop_last={getattr(loader, 'drop_last', False)}, and dataset size. "
                "Cannot compute validation loss for checkpoint selection on an empty validation sequence."
            )

        re_true_arr = np.concatenate(re_true_all, axis=0) if re_true_all else np.empty((0,))
        re_pred_arr = np.concatenate(re_pred_all, axis=0) if re_pred_all else np.empty((0,))
        phase_true_arr = np.concatenate(phase_true_all, axis=0) if phase_true_all else np.empty((0,))
        phase_pred_arr = np.concatenate(phase_pred_all, axis=0) if phase_pred_all else np.empty((0,))
        re_id_arr = np.concatenate(re_id_all, axis=0) if re_id_all else np.empty((0,), dtype=np.int64)

        if len(re_true_arr) > 0:
            re_metrics = compute_regression_metrics(re_true_arr, re_pred_arr)
            phase_metrics = compute_circular_metrics(phase_true_arr, phase_pred_arr)
        else:
            re_metrics = {"mae": float("nan"), "rmse": float("nan")}
            phase_metrics = {"circular_mae": float("nan"), "circular_rmse": float("nan")}

        return {
            "loss_total": sum_total / count_seen,
            "loss_re": sum_re / count_seen,
            "loss_phase": sum_phase / count_seen,
            "re_mae": float(re_metrics["mae"]),
            "re_rmse": float(re_metrics["rmse"]),
            "phase_cmae": float(phase_metrics["circular_mae"]),
            "phase_crmse": float(phase_metrics["circular_rmse"]),
            "re_true": re_true_arr,
            "re_pred": re_pred_arr,
            "phase_true": phase_true_arr,
            "phase_pred": phase_pred_arr,
            "re_id": re_id_arr,
        }

    def fit(
        self,
        model: Union[DynamicSetSensorEncoder, SetSensorEncoder, nn.Module],
        train_loader: DataLoader,
        val_loader: Optional[Union[DataLoader, DynamicSensorDataset, Mapping[int, Any]]] = None,
        candidate_positions: Optional[Union[Sequence[Sequence[float]], np.ndarray]] = None,
        re_normalizer: Optional[Normalizer] = None,
        training_sensor_counts: Optional[Sequence[int]] = None,
        default_sensor_indices: Optional[Sequence[int]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        re_split: Optional[Union[ReSplit, Mapping[str, Sequence[int]]]] = None,
        extra_metadata: Optional[Mapping[str, Any]] = None,
        reseed_model: bool = False,
    ) -> Tuple[Union[DynamicSetSensorEncoder, SetSensorEncoder, nn.Module], DeepSetsTrainHistory]:
        """Execute full training loop with checkpoint selection and validation tracking.

        Parameters:
            model: DynamicSetSensorEncoder or SetSensorEncoder instance to train.
            train_loader: Training set DataLoader.
            val_loader: Optional validation DataLoader, DynamicSensorDataset, or mapping.
            candidate_positions: Dense coordinates [Ncand, 2] for candidate sensors.
            re_normalizer: Fitted Normalizer for Reynolds number.
            training_sensor_counts: Sensor counts used during training.
            default_sensor_indices: Default 0-based sensor indices for default evaluation.
            output_dir: Directory where checkpoints and metrics are written.
            optimizer: Optional pre-configured Optimizer (default Adam).
            re_split: Mapping or ReSplit defining train/val/test partitions.
            extra_metadata: Extra arbitrary metadata dictionary for checkpoint.
            reseed_model: If True, calls model.reset_parameters(seed=config.seed).

        Returns:
            Tuple `(best_model, train_history)`.
        """
        train_ds = getattr(train_loader, "dataset", None)
        val_ds = _unwrap_to_dynamic_dataset(val_loader) if val_loader is not None else None

        # 1. Authoritative extra_metadata check
        if extra_metadata is not None:
            conflicts = set(extra_metadata.keys()) & AUTHORITATIVE_DEEPSETS_METADATA_KEYS
            if conflicts:
                raise ValueError(
                    f"extra_metadata cannot overwrite authoritative checkpoint keys: {sorted(conflicts)}"
                )

        # 2. Check DataLoader batch viability upfront
        if hasattr(train_loader, "__len__"):
            try:
                if len(train_loader) == 0:
                    raise ValueError(
                        f"Training DataLoader is empty (yields 0 batches). "
                        f"Check DataLoader batch_size={getattr(train_loader, 'batch_size', None)}, "
                        f"drop_last={getattr(train_loader, 'drop_last', False)}, and dataset size. "
                        "Training set must produce at least one batch."
                    )
            except TypeError:
                pass

        if val_loader is not None and hasattr(val_loader, "__len__"):
            try:
                if len(val_loader) == 0:
                    raise ValueError(
                        f"Validation DataLoader is empty (yields 0 batches). "
                        f"Check DataLoader batch_size={getattr(val_loader, 'batch_size', None)}, "
                        f"drop_last={getattr(val_loader, 'drop_last', False)}, and dataset size. "
                        "Cannot compute validation loss for checkpoint selection on an empty validation batch sequence."
                    )
            except TypeError:
                pass

        if train_ds is not None and hasattr(train_ds, "__len__") and len(train_ds) == 0:
            raise ValueError("Training dataset contains 0 samples; cannot train on empty dataset.")
        if val_ds is not None and hasattr(val_ds, "__len__") and len(val_ds) == 0:
            raise ValueError("Validation dataset contains 0 samples; cannot validate on empty dataset.")

        # 2. Stateful collator enforcement: requires num_workers=0
        collator = getattr(train_loader, "collate_fn", None)
        if isinstance(collator, DynamicBatchCollator) and getattr(train_loader, "num_workers", 0) != 0:
            raise ValueError(
                f"DynamicBatchCollator is stateful and requires num_workers=0 (got num_workers={train_loader.num_workers})"
            )

        # 3. Disjointness and locked test Re verification
        train_re_set = _extract_dataset_re_ids(train_ds)
        val_re_set = _extract_dataset_re_ids(val_ds) if val_ds is not None else set()

        if train_re_set is not None and val_re_set is not None and len(val_re_set) > 0:
            overlap = train_re_set & val_re_set
            if overlap:
                raise ValueError(
                    f"Train and validation splits must be strictly disjoint, but found overlapping Re: {sorted(overlap)}"
                )

        default_locked = {250, 650}
        if train_re_set and (train_re_set & default_locked):
            raise PermissionError(
                f"Test Re={sorted(train_re_set & default_locked)} is strictly locked and cannot appear in training split!"
            )
        if val_re_set and (val_re_set & default_locked):
            raise PermissionError(
                f"Test Re={sorted(val_re_set & default_locked)} is strictly locked and cannot appear in validation split!"
            )

        effective_split = re_split
        if effective_split is None:
            try:
                from cylinder_flow.data.splits import select_re_fold
                effective_split = select_re_fold("default")
            except Exception:
                effective_split = ReSplit(
                    train_re=[200, 225, 275, 300, 350, 400, 500, 550, 600, 700, 750, 850, 900, 1000],
                    val_re=[450, 800],
                    test_re=[250, 650],
                    fold_id="default",
                )
        elif isinstance(effective_split, Mapping):
            train_part = [int(x) for x in effective_split.get("train", effective_split.get("train_re", []))]
            val_part = [int(x) for x in effective_split.get("val", effective_split.get("val_re", []))]
            test_part = [int(x) for x in effective_split.get("test", effective_split.get("test_re", []))]
            fold_id = str(effective_split.get("fold_id", "custom"))
            effective_split = ReSplit(
                train_re=train_part,
                val_re=val_part,
                test_re=test_part,
                fold_id=fold_id,
            )

        if effective_split is not None:
            if train_re_set is None:
                raise ValueError(
                    "Cannot verify Re partition compliance: train_loader dataset does not provide re_id metadata."
                )
            if val_loader is not None and val_re_set is None:
                raise ValueError(
                    "Cannot verify Re partition compliance: val_loader dataset does not provide re_id metadata."
                )

            if isinstance(effective_split, ReSplit):
                effective_split.validate_disjoint(strict=True)
                test_re_set = set(int(r) for r in effective_split.test_re)
                train_decl = set(int(r) for r in effective_split.train_re)
                val_decl = set(int(r) for r in effective_split.val_re)
            else:
                test_re_set = set(int(r) for r in effective_split.get("test", effective_split.get("test_re", [])))
                train_decl = set(int(r) for r in effective_split.get("train", effective_split.get("train_re", [])))
                val_decl = set(int(r) for r in effective_split.get("val", effective_split.get("val_re", [])))
                ov = (train_decl & val_decl) | (train_decl & test_re_set) | (val_decl & test_re_set)
                if ov:
                    raise ValueError(f"Declared re_split has overlapping partitions: {sorted(ov)}")

            non_train = train_re_set - train_decl
            if non_train:
                if non_train & test_re_set:
                    raise PermissionError(
                        f"Test Re={sorted(non_train & test_re_set)} is strictly locked and cannot appear in training split!"
                    )
                raise ValueError(
                    f"train_loader contains Re={sorted(non_train)} which are not in declared train_re {sorted(train_decl)}! "
                    "Check for swapped loaders or partition leakage."
                )

            if val_re_set:
                non_val = val_re_set - val_decl
                if non_val:
                    if non_val & test_re_set:
                        raise PermissionError(
                            f"Test Re={sorted(non_val & test_re_set)} is strictly locked and cannot appear in validation split!"
                        )
                    raise ValueError(
                        f"val_loader contains Re={sorted(non_val)} which are not in declared val_re {sorted(val_decl)}! "
                        "Check for swapped loaders or partition leakage."
                    )
        else:
            test_re_set = default_locked
            train_decl = train_re_set or set()
            val_decl = val_re_set or set()

        # 4. Leaf datasets collection & metadata verification (Subset/ConcatDataset unwrap)
        train_leaves = _collect_leaf_datasets(train_ds) if train_ds is not None else []

        leaf_cand_pos: Optional[np.ndarray] = None
        leaf_cm: Optional[np.ndarray] = None
        leaf_cs: Optional[np.ndarray] = None
        leaf_re_norm: Optional[Normalizer] = None
        leaf_stats_train_re: Optional[List[int]] = None
        leaf_channel_stats_source: Optional[str] = None

        for idx, leaf in enumerate(train_leaves):
            # Candidate positions
            pos_attr = getattr(leaf, "candidate_positions", getattr(leaf, "positions", None))
            if pos_attr is not None:
                pos_arr = np.asarray(pos_attr, dtype=np.float64)
                if leaf_cand_pos is None:
                    leaf_cand_pos = pos_arr
                else:
                    if leaf_cand_pos.shape != pos_arr.shape or not np.allclose(leaf_cand_pos, pos_arr, atol=1e-5):
                        raise ValueError(
                            f"Training dataset component {idx} ({type(leaf).__name__}) candidate_positions "
                            "conflicts with earlier component candidate_positions."
                        )

            # chan_mean and chan_std
            cm_attr = getattr(leaf, "chan_mean", None)
            cs_attr = getattr(leaf, "chan_std", None)
            if cm_attr is not None and cs_attr is not None:
                cm_arr = np.asarray(cm_attr, dtype=np.float64).reshape(-1)
                cs_arr = np.asarray(cs_attr, dtype=np.float64).reshape(-1)
                if leaf_cm is None:
                    leaf_cm = cm_arr
                    leaf_cs = cs_arr
                else:
                    if not np.allclose(leaf_cm, cm_arr, atol=1e-5) or not np.allclose(leaf_cs, cs_arr, atol=1e-5):
                        raise ValueError(
                            f"Training dataset component {idx} ({type(leaf).__name__}) chan_mean/chan_std "
                            "conflicts with earlier component channel statistics."
                        )

            # re_normalizer
            rn_attr = getattr(leaf, "re_normalizer", None)
            if rn_attr is not None:
                if leaf_re_norm is None:
                    leaf_re_norm = rn_attr
                else:
                    ok, reason = _normalizers_are_consistent(leaf_re_norm, rn_attr)
                    if not ok:
                        raise ValueError(
                            f"Training dataset component {idx} ({type(leaf).__name__}) re_normalizer "
                            f"conflicts with earlier component re_normalizer: {reason}"
                        )

            # stats_train_re provenance and anti-leakage guards
            str_attr = getattr(leaf, "stats_train_re", None)
            if str_attr is not None:
                str_set = {int(r) for r in str_attr}
                val_leak = str_set & val_decl
                if val_leak:
                    raise ValueError(
                        f"Training dataset component {idx} channel statistics were fitted using validation/test Re ({sorted(val_leak)})! "
                        "Fitting normalizers on val/test is forbidden."
                    )
                test_leak = str_set & test_re_set
                if test_leak:
                    raise ValueError(
                        f"Training dataset component {idx} channel statistics were fitted using validation/test Re ({sorted(test_leak)})! "
                        "Fitting normalizers on val/test is forbidden."
                    )
                undecl = str_set - train_decl
                if undecl:
                    raise ValueError(
                        f"Training dataset component {idx} channel statistics contain undeclared Re ({sorted(undecl)})! "
                        f"Expected stats within declared train_re {sorted(train_decl)}."
                    )
                if leaf_stats_train_re is None:
                    leaf_stats_train_re = sorted(str_set)
                else:
                    if set(leaf_stats_train_re) != str_set:
                        raise ValueError(
                            f"Training dataset component {idx} ({type(leaf).__name__}) stats_train_re "
                            f"{sorted(str_set)} conflicts with earlier component {sorted(leaf_stats_train_re)}."
                        )

            src_attr = getattr(leaf, "channel_stats_source", None)
            if src_attr is not None and leaf_channel_stats_source is None:
                leaf_channel_stats_source = str(src_attr)

        # Candidate positions resolution & validation
        if candidate_positions is not None:
            cand_pos_arr = np.asarray(candidate_positions, dtype=np.float64)
            if leaf_cand_pos is not None:
                if cand_pos_arr.shape != leaf_cand_pos.shape or not np.allclose(cand_pos_arr, leaf_cand_pos, atol=1e-5):
                    raise ValueError(
                        "Explicit candidate_positions conflicts with dataset candidate_positions! "
                        "Dataset positions take precedence; inconsistent explicit candidate_positions are rejected."
                    )
        elif leaf_cand_pos is not None:
            cand_pos_arr = leaf_cand_pos
        elif hasattr(model, "candidate_positions"):
            cand_pos_arr = np.asarray(getattr(model, "candidate_positions"), dtype=np.float64)
        else:
            cand_pos_arr = None

        if cand_pos_arr is not None:
            if cand_pos_arr.ndim != 2 or cand_pos_arr.shape[1] != 2:
                raise ValueError(f"candidate_positions must have shape [Ncand, 2], got {cand_pos_arr.shape}")
            c_count = int(cand_pos_arr.shape[0])
        else:
            c_count = getattr(model, "candidate_count", getattr(train_ds, "candidate_count", 60))

        # Re normalizer resolution & validation
        if re_normalizer is not None:
            re_norm = re_normalizer
            if leaf_re_norm is not None:
                ok, reason = _normalizers_are_consistent(re_norm, leaf_re_norm)
                if not ok:
                    raise ValueError(
                        f"Explicit re_normalizer conflicts with training dataset re_normalizer ({reason})! "
                        "Dataset statistics take precedence; inconsistent explicit normalizers are rejected."
                    )
        elif leaf_re_norm is not None:
            re_norm = leaf_re_norm
        else:
            re_norm = None

        if re_norm is None or not getattr(re_norm, "fitted", False):
            raise ValueError("re_normalizer is required and must be fitted before training.")

        # Channel stats and model buffer alignment
        if leaf_cm is not None and leaf_cs is not None:
            if hasattr(model, "chan_mean") and hasattr(model, "chan_std"):
                if torch.all(model.chan_mean == 0) and torch.all(model.chan_std == 1):
                    model.chan_mean.copy_(torch.from_numpy(leaf_cm.astype(np.float32)).reshape(1, 1, 3))
                    model.chan_std.copy_(torch.from_numpy(leaf_cs.astype(np.float32)).reshape(1, 1, 3))
                else:
                    mod_cm = model.chan_mean.detach().cpu().reshape(-1).numpy()
                    mod_cs = model.chan_std.detach().cpu().reshape(-1).numpy()
                    if not np.allclose(mod_cm, leaf_cm, atol=1e-5):
                        raise ValueError("Provided model.chan_mean buffer conflicts with training dataset chan_mean")
                    if not np.allclose(mod_cs, leaf_cs, atol=1e-5):
                        raise ValueError("Provided model.chan_std buffer conflicts with training dataset chan_std")

        stats_train_re = leaf_stats_train_re

        # Validation dataset inspection
        if val_ds is not None:
            val_leaves = _collect_leaf_datasets(val_ds)
            for idx, leaf in enumerate(val_leaves):
                if getattr(leaf, "fit_normalizers", False):
                    raise ValueError(f"Validation dataset component {idx} attempted to fit normalizers; forbidden.")
                leaf_stats_re = getattr(leaf, "stats_train_re", None)
                if leaf_stats_re is not None:
                    leaf_leakage = {int(r) for r in leaf_stats_re} & (val_decl | test_re_set)
                    if leaf_leakage:
                        raise ValueError(f"Validation dataset component {idx} stats contain val/test Re: {sorted(leaf_leakage)}")
                leaf_re_norm_val = getattr(leaf, "re_normalizer", None)
                if leaf_re_norm_val is not None:
                    ok, reason = _normalizers_are_consistent(leaf_re_norm_val, re_norm)
                    if not ok:
                        raise ValueError(f"Validation dataset component {idx} re_normalizer conflicts with training: {reason}")
                leaf_pos_val = getattr(leaf, "candidate_positions", getattr(leaf, "positions", None))
                if leaf_pos_val is not None and cand_pos_arr is not None:
                    arr_val = np.asarray(leaf_pos_val, dtype=np.float64)
                    if arr_val.shape != cand_pos_arr.shape or not np.allclose(arr_val, cand_pos_arr, atol=1e-5):
                        raise ValueError(f"Validation dataset component {idx} candidate_positions conflicts with training candidate_positions")

        # Formal mode contracts
        is_dynamic = is_dynamic_token_regressor(model) or isinstance(model, DynamicSetSensorEncoder)
        if is_dynamic:
            if hasattr(model, "candidate_count") and model.candidate_count != c_count:
                raise ValueError(
                    f"candidate_positions count ({c_count}) does not match model.candidate_count ({model.candidate_count})"
                )

        if not self.config.smoke_not_for_formal:
            if not is_dynamic or isinstance(model, SetSensorEncoder):
                raise ValueError(
                    "Formal DeepSets training requires a dynamic token encoder (DynamicSetSensorEncoder); "
                    "SetSensorEncoder is lightweight smoke only. Set smoke_not_for_formal=True."
                )
            if self.config.layout_sampling != "random-subset":
                raise ValueError(
                    f"Formal DeepSets training requires layout_sampling='random-subset', got {self.config.layout_sampling!r}. "
                    "Set smoke_not_for_formal=True for fixed layout training."
                )
            if val_loader is None:
                raise ValueError(
                    "Formal dynamic DeepSets training requires validation data for frozen panel checkpoint selection. "
                    "If running in a deliberately marked lightweight smoke mode without frozen panel validation, "
                    "set smoke_not_for_formal=True."
                )
            if c_count != DYNAMIC_VALIDATION_PANEL_CANDIDATE_COUNT:
                raise ValueError(
                    f"The frozen WP-A validation panel requires the complete {DYNAMIC_VALIDATION_PANEL_CANDIDATE_COUNT}-candidate pool; "
                    f"got candidate_count={c_count}. If running in a deliberately marked lightweight smoke mode, set smoke_not_for_formal=True."
                )
            if stats_train_re is None:
                raise ValueError(
                    "Formal dynamic DeepSets training requires verified full pool train stats provenance (stats_train_re); "
                    "cannot verify channel statistics provenance."
                )
            if isinstance(collator, DynamicBatchCollator):
                if collator.layout_sampler is None:
                    raise ValueError(
                        "Formal dynamic DeepSets training requires a configured layout_sampler (BalancedRandomSubsetSampler) "
                        "matching training_sensor_counts; got collator.layout_sampler=None."
                    )
                expected_budgets = tuple(sorted({int(x) for x in (training_sensor_counts or self.config.training_sensor_counts)}))
                if tuple(collator.layout_sampler.sensor_counts) != expected_budgets:
                    raise ValueError(
                        f"train_loader layout_sampler budgets {collator.layout_sampler.sensor_counts} do not match "
                        f"training_sensor_counts {expected_budgets}."
                    )

        # 5. Reproducible reset of loaders with legacy independent default seeds
        if self.config.layout_sampler_seed is not None:
            ls_seed = int(self.config.layout_sampler_seed)
        elif isinstance(collator, DynamicBatchCollator) and collator.layout_sampler is not None and getattr(collator.layout_sampler, "seed", None) not in (None, 0, self.config.seed):
            ls_seed = int(collator.layout_sampler.seed)
        else:
            ls_seed = int(self.config.seed + 100003)

        if self.config.token_dropout_seed is not None:
            td_seed = int(self.config.token_dropout_seed)
        elif isinstance(collator, DynamicBatchCollator) and collator.token_dropout_seed not in (0, self.config.seed):
            td_seed = int(collator.token_dropout_seed)
        else:
            td_seed = int(self.config.seed + 200003)

        if self.config.sensor_noise_seed is not None:
            sn_seed = int(self.config.sensor_noise_seed)
        elif isinstance(collator, DynamicBatchCollator) and collator.sensor_noise_seed not in (0, self.config.seed):
            sn_seed = int(collator.sensor_noise_seed)
        else:
            sn_seed = int(self.config.seed + 300007)

        if hasattr(train_loader, "generator") and train_loader.generator is not None:
            train_loader.generator.manual_seed(int(self.config.seed))

        if hasattr(train_loader, "reset") and callable(train_loader.reset):
            train_loader.reset(
                seed=self.config.seed,
                token_dropout_seed=td_seed,
                sensor_noise_seed=sn_seed,
                layout_sampler_seed=ls_seed,
            )
        else:
            reset_dynamic_loader(
                train_loader,
                seed=self.config.seed,
                token_dropout_seed=td_seed,
                sensor_noise_seed=sn_seed,
                layout_sampler_seed=ls_seed,
            )

        if val_loader is not None and isinstance(val_loader, DataLoader):
            if hasattr(val_loader, "generator") and val_loader.generator is not None:
                val_loader.generator.manual_seed(int(self.config.seed))

        if reseed_model and hasattr(model, "reset_parameters"):
            model.reset_parameters(seed=self.config.seed)

        device = self.resolve_device(self.config.device)
        model = model.to(device)

        if optimizer is None:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

        history = DeepSetsTrainHistory()
        best_selection_loss = float("inf")
        best_epoch = -1
        best_state_dict = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0

        use_frozen_panel = (
            is_dynamic
            and (not self.config.smoke_not_for_formal or self.config.validation_mode == "frozen_panel")
            and c_count == DYNAMIC_VALIDATION_PANEL_CANDIDATE_COUNT
            and val_ds is not None
        )

        is_smoke = bool(
            self.config.smoke_not_for_formal
            or not is_dynamic
            or val_loader is None
            or not use_frozen_panel
            or self.config.layout_sampling == "fixed"
        )

        if use_frozen_panel:
            val_panel = build_dynamic_validation_panel(candidate_count=DYNAMIC_VALIDATION_PANEL_CANDIDATE_COUNT)
        else:
            val_panel = None

        out_path: Optional[Path] = Path(output_dir) if output_dir is not None else None
        if out_path is not None:
            out_path.mkdir(parents=True, exist_ok=True)

        val_loader_for_eval = val_loader if isinstance(val_loader, DataLoader) else (
            DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False) if val_ds is not None else None
        )

        latest_panel_out: Optional[Dict[str, object]] = None
        latest_val_out: Optional[Dict[str, Any]] = None

        for epoch in range(1, self.config.num_epochs + 1):
            t0 = time.perf_counter()
            train_out = self.train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
            )
            wall_time = float(time.perf_counter() - t0)

            if use_frozen_panel:
                panel_out = evaluate_dynamic_validation_panel(
                    model=model,
                    val_data=val_ds,
                    panel=val_panel,
                    candidate_positions=cand_pos_arr,
                    re_normalizer=re_norm,
                    device=device,
                    lambda_re=self.config.lambda_re,
                    lambda_phase=self.config.lambda_phase,
                    chunk_size=self.config.chunk_size,
                    split_name="val",
                    declared_val_re=sorted(val_decl),
                    re_split=effective_split,
                )
                latest_panel_out = panel_out
                selection_loss = float(panel_out["val_equal_budget_selection_score"])
                selection_metric = "val_equal_budget_selection_score"
                dyn_vals = dynamic_validation_history_values(panel_out)
                history.add_epoch_record(
                    epoch=epoch,
                    train_metrics=train_out,
                    val_metrics=None,
                    dynamic_metrics=dyn_vals,
                    wall_time=wall_time,
                )
            else:
                if val_loader_for_eval is not None:
                    val_out = self.evaluate(
                        model=model,
                        loader=val_loader_for_eval,
                        re_normalizer=re_norm,
                        device=device,
                    )
                    latest_val_out = val_out
                    selection_loss = float(val_out["loss_total"])
                    selection_metric = "val_loss_total"
                    history.add_epoch_record(
                        epoch=epoch,
                        train_metrics=train_out,
                        val_metrics=val_out,
                        dynamic_metrics=None,
                        wall_time=wall_time,
                    )
                else:
                    selection_loss = float(train_out["loss_total"])
                    selection_metric = "train_loss_total"
                    history.add_epoch_record(
                        epoch=epoch,
                        train_metrics=train_out,
                        val_metrics=None,
                        dynamic_metrics=None,
                        wall_time=wall_time,
                    )

            # Strict monotonic decrease checkpoint selection
            if selection_loss < (best_selection_loss - self.config.min_delta):
                best_selection_loss = selection_loss
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())
                best_panel_out = copy.deepcopy(latest_panel_out) if latest_panel_out is not None else None
                best_val_out = copy.deepcopy(latest_val_out) if latest_val_out is not None else None
                epochs_without_improvement = 0

                if out_path is not None and self.config.save_best:
                    ckpt_file = out_path / self.config.checkpoint_name
                    if isinstance(model, SetSensorEncoder) or self.config.layout_sampling == "fixed":
                        train_counts = [int(getattr(model, "n_sensors", c_count))]
                    elif isinstance(collator, DynamicBatchCollator) and collator.layout_sampler is not None:
                        train_counts = list(collator.layout_sampler.sensor_counts)
                    elif training_sensor_counts is not None:
                        train_counts = list(sorted({int(x) for x in training_sensor_counts}))
                    else:
                        train_counts = list(self.config.training_sensor_counts)

                    def_indices = (
                        list(default_sensor_indices)
                        if default_sensor_indices is not None
                        else list(range(c_count))
                    )
                    stats_re_ids = (
                        sorted(stats_train_re)
                        if stats_train_re is not None
                        else []
                    )

                    collator_snap = None
                    if isinstance(getattr(train_loader, "collate_fn", None), DynamicBatchCollator):
                        collator_snap = train_loader.collate_fn.snapshot()

                    sampler_snap = None
                    if hasattr(train_loader, "layout_sampler") and getattr(train_loader, "layout_sampler", None) is not None:
                        sampler_snap = train_loader.layout_sampler.snapshot()

                    sample_dist = collator_snap.get("sample_count", {}) if collator_snap is not None else {}
                    batch_dist = collator_snap.get("batch_count", {}) if collator_snap is not None else {}

                    actual_aug_seeds = {
                        "token_dropout_seed": td_seed,
                        "sensor_noise_seed": sn_seed,
                        "layout_sampler_seed": ls_seed,
                    }

                    checkpoint_extra: Dict[str, Any] = {
                        "selection_metric": selection_metric,
                        "selection_metric_name": selection_metric,
                        "selection_metric_value": float(best_selection_loss),
                        "best_selection_loss": float(best_selection_loss),
                        "selection_score": float(best_selection_loss),
                        "smoke_not_for_formal": is_smoke,
                        "declared_train_re": sorted(train_decl),
                        "declared_val_re": sorted(val_decl),
                        "declared_test_re": sorted(test_re_set),
                        "defer_test_evaluation": True,
                        "locked_test_opened": False,
                        "training_protocol": asdict(self.config),
                        "token_dropout_seed": td_seed,
                        "sensor_noise_seed": sn_seed,
                        "layout_sampler_seed": ls_seed,
                        "actual_augmentation_seeds": actual_aug_seeds,
                        "sample_distribution": sample_dist,
                        "batch_distribution": batch_dist,
                    }
                    if collator_snap is not None:
                        checkpoint_extra["collator_snapshot"] = collator_snap
                    if sampler_snap is not None:
                        checkpoint_extra["sampler_snapshot"] = sampler_snap
                    if use_frozen_panel and best_panel_out is not None:
                        checkpoint_extra["dynamic_validation_panel"] = val_panel
                        checkpoint_extra["dynamic_validation_panel_evaluation"] = best_panel_out
                        checkpoint_extra["validation_panel_summary"] = {
                            "panel_seed": int(best_panel_out.get("panel_seed", DYNAMIC_VALIDATION_PANEL_SEED)),
                            "budgets": list(best_panel_out.get("budgets", DYNAMIC_VALIDATION_PANEL_BUDGETS)),
                            "panel_size_per_budget": int(best_panel_out.get("panel_size_per_budget", DYNAMIC_VALIDATION_LAYOUTS_PER_BUDGET)),
                            "selection_score": float(best_selection_loss),
                        }
                        checkpoint_extra["validation_metrics"] = dynamic_validation_history_values(best_panel_out)
                    elif best_val_out is not None:
                        checkpoint_extra["dynamic_validation_panel"] = None
                        checkpoint_extra["dynamic_validation_panel_evaluation"] = None
                        checkpoint_extra["validation_evaluation"] = {k: v for k, v in best_val_out.items() if not isinstance(v, np.ndarray)}
                        checkpoint_extra["validation_metrics"] = {
                            "val_loss_total": float(best_val_out["loss_total"]),
                            "val_loss_re": float(best_val_out["loss_re"]),
                            "val_loss_phase": float(best_val_out["loss_phase"]),
                            "re_rmse": float(best_val_out.get("re_rmse", float("nan"))),
                            "phase_crmse": float(best_val_out.get("phase_crmse", float("nan"))),
                        }
                    else:
                        checkpoint_extra["dynamic_validation_panel"] = None
                        checkpoint_extra["dynamic_validation_panel_evaluation"] = None

                    if extra_metadata is not None:
                        merged_extra = dict(extra_metadata)
                        merged_extra.update(checkpoint_extra)
                        checkpoint_extra = merged_extra

                    if isinstance(model, DynamicSetSensorEncoder):
                        if stats_train_re is None:
                            raise ValueError(
                                "Cannot serialize DynamicSetSensorEncoder checkpoint: channel statistics "
                                "provenance (stats_train_re) is unknown; refusing to fabricate declared train IDs."
                            )
                        cm = model.chan_mean.detach().cpu().numpy().reshape(-1)
                        cs = model.chan_std.detach().cpu().numpy().reshape(-1)
                        save_deepsets_checkpoint(
                            path=ckpt_file,
                            model=model,
                            candidate_positions=cand_pos_arr,
                            training_sensor_counts=train_counts,
                            chan_mean=cm,
                            chan_std=cs,
                            channel_stats_re_ids=stats_re_ids,
                            re_normalizer=re_norm,
                            default_sensor_indices=def_indices,
                            channel_stats_source=leaf_channel_stats_source or getattr(train_ds, "channel_stats_source", "train_split_full_candidate_pool"),
                            channel_order=["u", "v", "p"],
                            x_normalizer=Normalizer(method="none", fitted=True),
                            schema_version=self.config.schema_version,
                            epoch=epoch,
                            train_seed=self.config.seed,
                            extra_metadata=checkpoint_extra,
                        )
                    elif isinstance(model, SetSensorEncoder):
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                                "model_type": "SetSensorEncoder",
                                "n_sensors": int(getattr(model, "n_sensors", c_count)),
                                "embed_dim": int(model.embed_dim),
                                "pool": str(model.pool),
                                "activation": str(model.activation),
                                "chan_mean": model.chan_mean.detach().cpu().numpy().tolist(),
                                "chan_std": model.chan_std.detach().cpu().numpy().tolist(),
                                "re_normalizer": re_norm.to_dict(),
                                "train_seed": self.config.seed,
                                "training_sensor_counts": train_counts,
                                "smoke_not_for_formal": True,
                                **checkpoint_extra,
                            },
                            ckpt_file,
                        )
            else:
                epochs_without_improvement += 1

            if self.config.verbose:
                val_disp = f"val_score={selection_loss:.6f}" if val_loader is not None else "val=None"
                logger.info(
                    "DeepSets Epoch %d/%d | train_loss=%.6f | %s | %.2fs",
                    epoch,
                    self.config.num_epochs,
                    train_out["loss_total"],
                    val_disp,
                    wall_time,
                )

            if (
                self.config.patience is not None
                and epochs_without_improvement >= self.config.patience
            ):
                logger.info(
                    "Early stopping triggered at epoch %d (best epoch %d, best loss %.6f)",
                    epoch,
                    best_epoch,
                    best_selection_loss,
                )
                break

        # Re-load best checkpoint weights into model before returning
        model.load_state_dict(best_state_dict)
        history.best_epoch = best_epoch
        history.best_selection_loss = best_selection_loss

        return model, history


def train_sensor_deepsets(
    train_loader: DataLoader,
    val_loader: Optional[Union[DataLoader, DynamicSensorDataset, Mapping[int, Any]]] = None,
    config: Optional[DeepSetsTrainingConfig] = None,
    candidate_positions: Optional[Union[Sequence[Sequence[float]], np.ndarray]] = None,
    candidate_count: Optional[int] = None,
    embed_dim: int = 128,
    phi_hidden: Sequence[int] = (128, 128),
    rho_hidden: Sequence[int] = (128, 128),
    pool: str = "mean+max",
    count_feature: str = "n_over_candidate_count",
    position_encoding: str = "cos_sin_theta",
    activation: str = "relu",
    chan_mean: Optional[Union[Sequence[float], np.ndarray]] = None,
    chan_std: Optional[Union[Sequence[float], np.ndarray]] = None,
    training_sensor_counts: Optional[Sequence[int]] = None,
    default_sensor_indices: Optional[Sequence[int]] = None,
    re_normalizer: Optional[Normalizer] = None,
    re_split: Optional[Union[ReSplit, Mapping[str, Sequence[int]]]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
    model: Optional[Union[DynamicSetSensorEncoder, SetSensorEncoder, nn.Module]] = None,
) -> Tuple[Union[DynamicSetSensorEncoder, SetSensorEncoder, nn.Module], DeepSetsTrainHistory]:
    """End-to-end DeepSets training entrypoint with strict seed-controlled initialization.

    Guarantees:
    1. Global seeds set before model construction, ensuring model weights are deterministic.
    2. Inherits candidate positions, channel stats, and normalizers from dataset when omitted.
    3. Re split partition disjointness and test-lock defense checked before execution.
    4. Frozen 60-candidate validation panel (seed 400042) used for best model selection in formal mode.
    5. Monotonic decrease selection (min_delta=0.0) with best weights restored upon return.

    Parameters:
        train_loader: DataLoader for training partition.
        val_loader: Optional DataLoader, DynamicSensorDataset, or ReCase mapping for validation.
        config: Training hyperparameters (DeepSetsTrainingConfig).
        candidate_positions: Candidate sensor coordinates [Ncand, 2].
        candidate_count: Total candidate pool size.
        embed_dim: Token embedding dimension.
        phi_hidden: Hidden layer sizes for per-sensor network phi.
        rho_hidden: Hidden layer sizes for aggregator network rho.
        pool: Aggregation mode ('mean', 'sum', 'max', 'mean+max').
        count_feature: Count feature mode ('none' or 'n_over_candidate_count').
        position_encoding: Position encoding mode ('cos_sin_theta' or 'none').
        activation: Non-linear activation ('relu' for schema 2, 'gelu' for schema 1).
        chan_mean: Sensor channel mean values [u, v, p].
        chan_std: Sensor channel standard deviation values [u, v, p].
        training_sensor_counts: Sensor counts used during training.
        default_sensor_indices: Sensor indices for default layout.
        re_normalizer: Fitted Reynolds number normalizer.
        re_split: Reynolds split metadata or ReSplit contract.
        output_dir: Directory where checkpoints and metrics are saved.
        extra_metadata: Additional dictionary written to checkpoint.
        model: Optional pre-constructed model instance. If None, DynamicSetSensorEncoder is created.

    Returns:
        Tuple `(best_model, train_history)`.
    """
    cfg = config or DeepSetsTrainingConfig()
    set_training_seed(cfg.seed)

    train_ds = getattr(train_loader, "dataset", None)
    train_leaves = _collect_leaf_datasets(train_ds) if train_ds is not None else []

    cand_pos = candidate_positions
    if cand_pos is None:
        for leaf in train_leaves:
            lp = getattr(leaf, "candidate_positions", getattr(leaf, "positions", None))
            if lp is not None:
                cand_pos = lp
                break

    if candidate_count is not None:
        c_count = int(candidate_count)
    elif cand_pos is not None:
        c_count = int(np.asarray(cand_pos).shape[0])
    elif train_ds is not None and hasattr(train_ds, "candidate_count"):
        c_count = int(train_ds.candidate_count)
    else:
        c_count = 60

    cm = chan_mean
    if cm is None:
        for leaf in train_leaves:
            lcm = getattr(leaf, "chan_mean", None)
            if lcm is not None:
                cm = lcm
                break

    cs = chan_std
    if cs is None:
        for leaf in train_leaves:
            lcs = getattr(leaf, "chan_std", None)
            if lcs is not None:
                cs = lcs
                break

    re_norm = re_normalizer
    if re_norm is None:
        for leaf in train_leaves:
            lrn = getattr(leaf, "re_normalizer", None)
            if lrn is not None:
                re_norm = lrn
                break

    # HIGH-LEVEL entrypoint configures actual collator / layout_sampler to honor cfg
    collator = getattr(train_loader, "collate_fn", None)
    if isinstance(collator, DynamicBatchCollator):
        if cfg.layout_sampling == "random-subset":
            if collator.layout_sampler is None:
                effective_counts = (
                    tuple(sorted({int(x) for x in training_sensor_counts}))
                    if training_sensor_counts is not None
                    else tuple(cfg.training_sensor_counts)
                )
                ls_seed = cfg.layout_sampler_seed if cfg.layout_sampler_seed is not None else (cfg.seed + 100003)
                sampler = BalancedRandomSubsetSampler(
                    candidate_count=c_count,
                    sensor_counts=effective_counts,
                    seed=ls_seed,
                )
                collator.layout_sampler = sampler
                if hasattr(train_loader, "layout_sampler"):
                    train_loader.layout_sampler = sampler
        if cfg.token_dropout_prob > 0.0 and collator.token_dropout_prob == 0.0:
            collator.token_dropout_prob = float(cfg.token_dropout_prob)
        if cfg.sensor_noise_rel_std > 0.0 and collator.sensor_noise_rel_std == 0.0:
            collator.sensor_noise_rel_std = float(cfg.sensor_noise_rel_std)
        if cfg.token_dropout_seed is not None:
            collator.token_dropout_seed = int(cfg.token_dropout_seed)
        elif collator.token_dropout_seed == 0:
            collator.token_dropout_seed = int(cfg.seed + 200003)
        if cfg.sensor_noise_seed is not None:
            collator.sensor_noise_seed = int(cfg.sensor_noise_seed)
        elif collator.sensor_noise_seed == 0:
            collator.sensor_noise_seed = int(cfg.seed + 300007)

    if model is None:
        model = DynamicSetSensorEncoder(
            candidate_count=c_count,
            embed_dim=embed_dim,
            phi_hidden=phi_hidden,
            rho_hidden=rho_hidden,
            out_dim=3,
            pool=pool,
            activation=activation,
            count_feature=count_feature,
            position_encoding=position_encoding,
            chan_mean=cm,
            chan_std=cs,
        )

    trainer = DeepSetsTrainer(config=cfg)
    return trainer.fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        candidate_positions=cand_pos,
        re_normalizer=re_norm,
        training_sensor_counts=training_sensor_counts,
        default_sensor_indices=default_sensor_indices,
        output_dir=output_dir,
        re_split=re_split,
        extra_metadata=extra_metadata,
        reseed_model=False,
    )


