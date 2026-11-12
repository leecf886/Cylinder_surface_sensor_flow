"""Sensor signals to predicted (Re, phase) inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from cylinder_flow.artifacts.checkpoints import DeepSetsPackage, LegacyMLPPackage
from cylinder_flow.data.normalization import Normalizer
from cylinder_flow.evaluation.cycles import wrap_phase
from cylinder_flow.models.deepsets import (
    DynamicSensorBatch,
    DynamicTokenRegressor,
    FixedFlattenedAdapter,
    SetSensorEncoder,
    is_dynamic_token_regressor,
    validate_sensor_indices,
    verify_positions_in_candidate_pool,
)
from cylinder_flow.models.mlp import SensorMLP

if TYPE_CHECKING:
    from cylinder_flow.data.types import ReCaseData


def _normalizers_equal(n1: Any, n2: Any) -> bool:
    """Check if two normalizers have identical methods, fitting state, and statistics."""
    if n1 is n2:
        return True
    if not isinstance(n1, Normalizer) or not isinstance(n2, Normalizer):
        return False
    if n1.method != n2.method or n1.fitted != n2.fitted:
        return False
    for attr in ("mean", "std", "min", "max"):
        v1 = getattr(n1, attr)
        v2 = getattr(n2, attr)
        if (v1 is None) != (v2 is None):
            return False
        if v1 is not None and v2 is not None:
            if not np.allclose(v1, v2, rtol=1e-7, atol=1e-7):
                return False
    return True


@dataclass
class EncoderPrediction:
    """Structured container for sensor encoder inference outputs.

    Attributes:
        re: Predicted physical Reynolds number [N] or scalar float.
        phase: Predicted circular phase in canonical range [0, 2*pi) [N] or scalar float.
        phase_sin: Model raw output for sin(phase) [N] or scalar float.
        phase_cos: Model raw output for cos(phase) [N] or scalar float.
    """

    re: Union[np.ndarray, float]
    phase: Union[np.ndarray, float]
    phase_sin: Union[np.ndarray, float]
    phase_cos: Union[np.ndarray, float]


def predict_re_phase(
    model: Union[
        SensorMLP,
        DynamicTokenRegressor,
        SetSensorEncoder,
        FixedFlattenedAdapter,
        DeepSetsPackage,
        LegacyMLPPackage,
        torch.nn.Module,
    ],
    sensor_inputs: Union[np.ndarray, torch.Tensor, DynamicSensorBatch],
    x_normalizer: Optional[Normalizer] = None,
    re_normalizer: Optional[Normalizer] = None,
    device: Union[str, torch.device] = "cpu",
    batch_size: Optional[int] = None,
    positions: Optional[Union[Sequence[Sequence[float]], np.ndarray, torch.Tensor]] = None,
) -> Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
    """Predict physical Re and shedding phase from flattened sensor observations or dynamic batches.

    Standard scheme (Scheme A): Returns exclusively (re_pred, phase_pred) tuple.
    For full outputs including sin/cos components, use `predict_re_phase_full`.

    Parameters:
        model: Trained encoder (SensorMLP, DeepSets, or package container).
        sensor_inputs: Raw sensor array of shape [Nt, 3*K] / [3*K] or DynamicSensorBatch.
        x_normalizer: Optional fitted Normalizer for sensor features (for MLP).
        re_normalizer: Optional fitted Normalizer for Reynolds number.
        device: PyTorch device ('cpu', 'cuda', etc.).
        batch_size: Optional batch size for chunked inference.
        positions: Optional sensor coordinates [K, 2] if passing flattened data to dynamic encoder.

    Returns:
        Tuple of (re_pred, phase_pred).
        If input is batch (2D/3D), returns two 1D numpy arrays of shape [Nt].
        If input is single sample, returns two scalar floats.
    """
    pred = predict_re_phase_full(
        model=model,
        sensor_inputs=sensor_inputs,
        x_normalizer=x_normalizer,
        re_normalizer=re_normalizer,
        device=device,
        batch_size=batch_size,
        positions=positions,
    )
    return pred.re, pred.phase


def predict_re_phase_full(
    model: Union[
        SensorMLP,
        DynamicTokenRegressor,
        SetSensorEncoder,
        FixedFlattenedAdapter,
        DeepSetsPackage,
        LegacyMLPPackage,
        torch.nn.Module,
    ],
    sensor_inputs: Union[np.ndarray, torch.Tensor, DynamicSensorBatch],
    x_normalizer: Optional[Normalizer] = None,
    re_normalizer: Optional[Normalizer] = None,
    device: Union[str, torch.device] = "cpu",
    batch_size: Optional[int] = None,
    positions: Optional[Union[Sequence[Sequence[float]], np.ndarray, torch.Tensor]] = None,
) -> EncoderPrediction:
    """Full sensor inference returning Re, Phase, sin(phase), and cos(phase).

    Parameters:
        model: Trained encoder (SensorMLP, DeepSets, or package container).
        sensor_inputs: Raw sensor array of shape [Nt, 3*K] / [3*K] or DynamicSensorBatch.
        x_normalizer: Optional fitted Normalizer for sensor features (for MLP).
        re_normalizer: Optional fitted Normalizer for Reynolds number.
        device: PyTorch device ('cpu', 'cuda', etc.).
        batch_size: Optional batch size for chunked inference.
        positions: Optional sensor coordinates [K, 2] if passing flattened data to dynamic encoder.

    Returns:
        EncoderPrediction containing re, phase, phase_sin, and phase_cos.
    """
    torch_device = torch.device(device)

    # 1. Resolve package wrappers
    default_positions = positions
    if isinstance(model, LegacyMLPPackage):
        model_obj: torch.nn.Module = model.model
        if x_normalizer is not None and not _normalizers_equal(x_normalizer, model.x_normalizer):
            raise ValueError("Explicit x_normalizer conflicts with LegacyMLPPackage x_normalizer")
        if re_normalizer is not None and not _normalizers_equal(re_normalizer, model.re_normalizer):
            raise ValueError("Explicit re_normalizer conflicts with LegacyMLPPackage re_normalizer")
        x_norm = model.x_normalizer
        re_norm = model.re_normalizer
    elif isinstance(model, DeepSetsPackage):
        model_obj = model.model
        if x_normalizer is not None and not _normalizers_equal(x_normalizer, model.x_normalizer):
            raise ValueError("Explicit x_normalizer conflicts with DeepSetsPackage x_normalizer")
        if re_normalizer is not None and not _normalizers_equal(re_normalizer, model.re_normalizer):
            raise ValueError("Explicit re_normalizer conflicts with DeepSetsPackage re_normalizer")
        x_norm = model.x_normalizer
        re_norm = model.re_normalizer
        if default_positions is None and model.default_sensor_indices:
            default_positions = model.candidate_positions[model.default_sensor_indices]
        elif default_positions is not None:
            verify_positions_in_candidate_pool(default_positions, model.candidate_positions)
    elif isinstance(model, torch.nn.Module):
        model_obj = model
        x_norm = x_normalizer
        re_norm = re_normalizer
        if isinstance(model_obj, SensorMLP):
            if x_norm is None or not getattr(x_norm, "fitted", False):
                raise ValueError("Bare SensorMLP requires an explicit fitted x_normalizer")
            if re_norm is None or not getattr(re_norm, "fitted", False):
                raise ValueError("Bare SensorMLP requires an explicit fitted re_normalizer")
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")

    model_obj.to(torch_device)
    model_obj.eval()

    is_dynamic = is_dynamic_token_regressor(model_obj)
    is_single = False

    # 2. Case A: DynamicSensorBatch input
    if isinstance(sensor_inputs, DynamicSensorBatch):
        if not is_dynamic:
            raise TypeError(
                f"DynamicSensorBatch input requires dynamic token encoder, got {type(model_obj)}"
            )

        if isinstance(model, DeepSetsPackage):
            # Verify valid unmasked token positions correspond to package candidate pool (masked padding ignored)
            m_arr = (
                sensor_inputs.mask.detach().cpu().numpy()
                if isinstance(sensor_inputs.mask, torch.Tensor)
                else np.asarray(sensor_inputs.mask)
            )
            p_arr = (
                sensor_inputs.pos.detach().cpu().numpy()
                if isinstance(sensor_inputs.pos, torch.Tensor)
                else np.asarray(sensor_inputs.pos)
            )
            valid_p = p_arr[m_arr]
            if valid_p.size > 0:
                verify_positions_in_candidate_pool(valid_p, model.candidate_positions)

        val_arr = np.asarray(sensor_inputs.values) if isinstance(sensor_inputs.values, np.ndarray) else sensor_inputs.values
        if val_arr.ndim == 2:
            is_single = True
            batch_data = sensor_inputs[None]
        else:
            is_single = False
            batch_data = sensor_inputs

        n_samples = batch_data.batch_size
        preds_list: list[np.ndarray] = []

        with torch.no_grad():
            if batch_size is not None and batch_size > 0 and n_samples > batch_size:
                for st in range(0, n_samples, batch_size):
                    ed = min(st + batch_size, n_samples)
                    sub = batch_data[st:ed]
                    v, p, m, c = sub.to_tensors(device=torch_device)
                    out_t = model_obj.forward_tokens(v, p, m, c)
                    preds_list.append(out_t.detach().cpu().numpy())
                pred_all = np.concatenate(preds_list, axis=0).astype(np.float64)
            else:
                v, p, m, c = batch_data.to_tensors(device=torch_device)
                out_t = model_obj.forward_tokens(v, p, m, c)
                pred_all = out_t.detach().cpu().numpy().astype(np.float64)

    # 3. Case B: Flattened array input
    else:
        if isinstance(sensor_inputs, torch.Tensor):
            x_raw = sensor_inputs.detach().cpu().numpy()
        else:
            x_raw = np.asarray(sensor_inputs, dtype=np.float64)

        if x_raw.ndim == 1:
            is_single = True
            x_raw = x_raw[None, :]
        elif x_raw.ndim == 2:
            is_single = False
        else:
            raise ValueError(f"sensor_inputs must have shape [Nt, 3*K] or [3*K], got shape {x_raw.shape}")

        if is_dynamic:
            # Dynamic token regressor given flattened input: unpack with positions
            if default_positions is None:
                raise ValueError(
                    "DynamicTokenRegressor received flattened sensor_inputs without positions. "
                    "Provide explicit positions or pass a DeepSetsPackage."
                )
            pos_arr = np.asarray(default_positions, dtype=np.float64)
            if pos_arr.ndim != 2 or pos_arr.shape[1] != 2:
                raise ValueError(f"positions must have shape [K, 2], got {pos_arr.shape}")
            k = pos_arr.shape[0]
            if x_raw.shape[1] != 3 * k:
                raise ValueError(
                    f"sensor_inputs feature dim ({x_raw.shape[1]}) does not match "
                    f"3 * len(positions) ({3 * k})"
                )

            nt = x_raw.shape[0]
            val_3d = x_raw.reshape(nt, k, 3).astype(np.float32)
            pos_3d = np.broadcast_to(pos_arr[None, :, :], (nt, k, 2)).astype(np.float32).copy()
            mask_2d = np.ones((nt, k), dtype=bool)
            count_2d = np.full((nt, 1), float(k), dtype=np.float32)
            batch_dyn = DynamicSensorBatch(values=val_3d, pos=pos_3d, mask=mask_2d, count=count_2d)

            preds_list = []
            with torch.no_grad():
                if batch_size is not None and batch_size > 0 and nt > batch_size:
                    for st in range(0, nt, batch_size):
                        ed = min(st + batch_size, nt)
                        v, p, m, c = batch_dyn[st:ed].to_tensors(device=torch_device)
                        out_t = model_obj.forward_tokens(v, p, m, c)
                        preds_list.append(out_t.detach().cpu().numpy())
                    pred_all = np.concatenate(preds_list, axis=0).astype(np.float64)
                else:
                    v, p, m, c = batch_dyn.to_tensors(device=torch_device)
                    out_t = model_obj.forward_tokens(v, p, m, c)
                    pred_all = out_t.detach().cpu().numpy().astype(np.float64)

        elif isinstance(model_obj, (SetSensorEncoder, FixedFlattenedAdapter)):
            # Fixed SetSensorEncoder or FixedFlattenedAdapter: accepts [B, 3*K] directly (unnormalized)
            expected_dim = int(getattr(model_obj, "n_sensors", 0)) * 3
            if expected_dim > 0 and x_raw.shape[1] != expected_dim:
                raise ValueError(
                    f"sensor_inputs feature dim ({x_raw.shape[1]}) does not match "
                    f"model expected dim ({expected_dim})"
                )

            nt = x_raw.shape[0]
            preds_list = []
            with torch.no_grad():
                if batch_size is not None and batch_size > 0 and nt > batch_size:
                    for st in range(0, nt, batch_size):
                        ed = min(st + batch_size, nt)
                        b_t = torch.from_numpy(x_raw[st:ed].astype(np.float32)).to(torch_device)
                        out_t = model_obj(b_t)
                        preds_list.append(out_t.detach().cpu().numpy())
                    pred_all = np.concatenate(preds_list, axis=0).astype(np.float64)
                else:
                    in_t = torch.from_numpy(x_raw.astype(np.float32)).to(torch_device)
                    out_t = model_obj(in_t)
                    pred_all = out_t.detach().cpu().numpy().astype(np.float64)

        else:
            # Standard SensorMLP: requires x_normalizer
            expected_dim = int(getattr(model_obj, "input_dim", x_raw.shape[1]))
            if x_raw.shape[1] != expected_dim:
                raise ValueError(
                    f"sensor_inputs feature dim ({x_raw.shape[1]}) does not match "
                    f"model.input_dim ({expected_dim})"
                )

            if x_norm is not None:
                x_norm_data = x_norm.transform(x_raw).astype(np.float32)
            else:
                x_norm_data = x_raw.astype(np.float32)

            nt = x_norm_data.shape[0]
            preds_list = []
            with torch.no_grad():
                if batch_size is not None and batch_size > 0 and nt > batch_size:
                    for st in range(0, nt, batch_size):
                        ed = min(st + batch_size, nt)
                        b_t = torch.from_numpy(x_norm_data[st:ed]).to(torch_device)
                        out_t = model_obj(b_t)
                        preds_list.append(out_t.detach().cpu().numpy())
                    pred_all = np.concatenate(preds_list, axis=0).astype(np.float64)
                else:
                    in_t = torch.from_numpy(x_norm_data).to(torch_device)
                    out_t = model_obj(in_t)
                    pred_all = out_t.detach().cpu().numpy().astype(np.float64)

    # 4. Invert Re normalization and wrap circular phase
    re_pred_norm = pred_all[:, 0]
    sin_pred = pred_all[:, 1]
    cos_pred = pred_all[:, 2]

    if re_norm is not None:
        re_pred = re_norm.inverse_transform(re_pred_norm).astype(np.float64)
    else:
        re_pred = re_pred_norm.astype(np.float64)

    phi_raw = np.arctan2(sin_pred, cos_pred)
    phi_pred = wrap_phase(phi_raw).astype(np.float64)

    if is_single:
        return EncoderPrediction(
            re=float(re_pred[0]),
            phase=float(phi_pred[0]),
            phase_sin=float(sin_pred[0]),
            phase_cos=float(cos_pred[0]),
        )

    return EncoderPrediction(
        re=re_pred,
        phase=phi_pred,
        phase_sin=sin_pred,
        phase_cos=cos_pred,
    )


def predict_re_case(
    package: Union[LegacyMLPPackage, DeepSetsPackage],
    case: ReCaseData,
    device: Union[str, torch.device] = "cpu",
    batch_size: Optional[int] = None,
    *,
    sensor_indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience helper: extract sensors from ReCaseData and predict (Re, phase).

    Parameters:
        package: Loaded LegacyMLPPackage or DeepSetsPackage.
        case: ReCaseData instance.
        device: PyTorch device ('cpu', 'cuda').
        batch_size: Optional chunked batch size.
        sensor_indices: Optional subset of candidate sensor indices. Keyword-only.

    Returns:
        Tuple of (re_pred, phase_pred) each of shape [Nt].
    """
    if isinstance(package, LegacyMLPPackage):
        if sensor_indices is not None:
            val_idx = validate_sensor_indices(sensor_indices, candidate_count=case.ncand)
            if list(val_idx) != list(package.sensor_indices):
                raise ValueError(
                    f"Legacy MLP fixed layout does not support changing sensor layout or order. "
                    f"Expected {package.sensor_indices}, got {list(val_idx)}"
                )
            idx = val_idx
        else:
            idx = package.sensor_indices

        sensor_inputs = case.get_sensors(idx, flatten=True)
        return predict_re_phase(
            model=package,
            sensor_inputs=sensor_inputs,
            device=device,
            batch_size=batch_size,
        )  # type: ignore[return-value]

    if isinstance(package, DeepSetsPackage):
        if case.ncand != package.candidate_count:
            raise ValueError(
                f"Case candidate count ({case.ncand}) does not match package pool count "
                f"({package.candidate_count})"
            )
        raw_idx = package.default_sensor_indices if sensor_indices is None else sensor_indices
        idx = validate_sensor_indices(raw_idx, candidate_count=package.candidate_count)

        values = case.get_sensors(idx, flatten=False)  # [Nt, K, 3]
        nt = values.shape[0]
        k = len(idx)
        pos = package.candidate_positions[idx]
        pos_batch = np.broadcast_to(pos[None, :, :], (nt, k, 2)).astype(np.float32).copy()
        mask = np.ones((nt, k), dtype=bool)
        count = np.full((nt, 1), float(k), dtype=np.float32)
        batch = DynamicSensorBatch(values=values.astype(np.float32), pos=pos_batch, mask=mask, count=count)
        return predict_re_phase(
            model=package,
            sensor_inputs=batch,
            device=device,
            batch_size=batch_size,
        )  # type: ignore[return-value]

    raise TypeError(f"package must be LegacyMLPPackage or DeepSetsPackage, got {type(package)}")
