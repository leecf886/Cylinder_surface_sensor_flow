"""Sensor signals to predicted (Re, phase) inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np
import torch

from cylinder_flow.data.normalization import Normalizer
from cylinder_flow.evaluation.cycles import wrap_phase
from cylinder_flow.models.mlp import SensorMLP

if TYPE_CHECKING:
    from cylinder_flow.artifacts.checkpoints import LegacyMLPPackage
    from cylinder_flow.data.types import ReCaseData


@dataclass
class EncoderPrediction:
    """Structured container for sensor encoder inference outputs.

    Attributes:
        re: Predicted physical Reynolds number [N].
        phase: Predicted circular phase in canonical range [0, 2*pi) [N].
        phase_sin: Model raw output for sin(phase) [N].
        phase_cos: Model raw output for cos(phase) [N].
    """

    re: Union[np.ndarray, float]
    phase: Union[np.ndarray, float]
    phase_sin: Union[np.ndarray, float]
    phase_cos: Union[np.ndarray, float]


def predict_re_phase(
    model: SensorMLP,
    sensor_inputs: Union[np.ndarray, torch.Tensor],
    x_normalizer: Normalizer,
    re_normalizer: Normalizer,
    device: Union[str, torch.device] = "cpu",
    batch_size: Optional[int] = None,
) -> Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
    """Predict physical Re and shedding phase from flattened sensor observations.

    Standard scheme (Scheme A): Returns exclusively (re_pred, phase_pred) tuple.
    For full outputs including sin/cos components, use `predict_re_phase_full`.

    Parameters:
        model: Trained SensorMLP model.
        sensor_inputs: Raw (unnormalized) sensor array of shape [Nt, 3*K] or [3*K].
        x_normalizer: Fitted Normalizer for sensor features.
        re_normalizer: Fitted Normalizer for Reynolds number.
        device: PyTorch device ('cpu', 'cuda', etc.).
        batch_size: Optional batch size for chunked inference.

    Returns:
        Tuple of (re_pred, phase_pred).
        If input is 2D [Nt, 3*K], returns two 1D numpy arrays of shape [Nt].
        If input is 1D [3*K], returns two scalar floats.
    """
    pred = predict_re_phase_full(
        model=model,
        sensor_inputs=sensor_inputs,
        x_normalizer=x_normalizer,
        re_normalizer=re_normalizer,
        device=device,
        batch_size=batch_size,
    )
    return pred.re, pred.phase


def predict_re_phase_full(
    model: SensorMLP,
    sensor_inputs: Union[np.ndarray, torch.Tensor],
    x_normalizer: Normalizer,
    re_normalizer: Normalizer,
    device: Union[str, torch.device] = "cpu",
    batch_size: Optional[int] = None,
) -> EncoderPrediction:
    """Full sensor inference returning Re, Phase, sin(phase), and cos(phase).

    Parameters:
        model: Trained SensorMLP model.
        sensor_inputs: Raw (unnormalized) sensor array of shape [Nt, 3*K] or [3*K].
        x_normalizer: Fitted Normalizer for sensor features.
        re_normalizer: Fitted Normalizer for Reynolds number.
        device: PyTorch device ('cpu', 'cuda', etc.).
        batch_size: Optional batch size for chunked inference.

    Returns:
        EncoderPrediction containing re, phase, phase_sin, and phase_cos.
    """
    if isinstance(sensor_inputs, torch.Tensor):
        x_raw = sensor_inputs.detach().cpu().numpy()
    else:
        x_raw = np.asarray(sensor_inputs, dtype=np.float64)

    is_single = x_raw.ndim == 1
    if is_single:
        x_raw = x_raw[None, :]

    if x_raw.ndim != 2:
        raise ValueError(
            f"sensor_inputs must have shape [Nt, 3*K] or [3*K], got shape {x_raw.shape}"
        )
    if x_raw.shape[1] != model.input_dim:
        raise ValueError(
            f"sensor_inputs feature dim ({x_raw.shape[1]}) does not match "
            f"model.input_dim ({model.input_dim})"
        )

    # 1. Normalize sensor inputs with x_normalizer
    x_norm = x_normalizer.transform(x_raw).astype(np.float32)

    # 2. Forward pass through SensorMLP
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()

    n_samples = x_norm.shape[0]
    preds_list = []

    with torch.no_grad():
        if batch_size is not None and batch_size > 0 and n_samples > batch_size:
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_t = torch.from_numpy(x_norm[start_idx:end_idx]).to(torch_device)
                out_t = model(batch_t)
                preds_list.append(out_t.detach().cpu().numpy())
            pred_all = np.concatenate(preds_list, axis=0).astype(np.float64)
        else:
            in_t = torch.from_numpy(x_norm).to(torch_device)
            out_t = model(in_t)
            pred_all = out_t.detach().cpu().numpy().astype(np.float64)

    # Output dimensions: [0]=re_norm, [1]=sin(phi), [2]=cos(phi)
    re_pred_norm = pred_all[:, 0]
    sin_pred = pred_all[:, 1]
    cos_pred = pred_all[:, 2]

    # 3. Inverse transform Re to physical scale
    re_pred = re_normalizer.inverse_transform(re_pred_norm).astype(np.float64)

    # 4. Circular phase recovery in [0, 2*pi)
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
    package: LegacyMLPPackage,
    case: ReCaseData,
    device: Union[str, torch.device] = "cpu",
    batch_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience helper: extract sensors from ReCaseData and predict (Re, phase).

    Parameters:
        package: Loaded LegacyMLPPackage containing model and normalizers.
        case: ReCaseData instance.
        device: PyTorch device ('cpu', 'cuda').
        batch_size: Optional chunked batch size.

    Returns:
        Tuple of (re_pred, phase_pred) each of shape [Nt].
    """
    sensor_inputs = case.get_sensors(package.sensor_indices, flatten=True)
    return predict_re_phase(
        model=package.model,
        sensor_inputs=sensor_inputs,
        x_normalizer=package.x_normalizer,
        re_normalizer=package.re_normalizer,
        device=device,
        batch_size=batch_size,
    )  # type: ignore[return-value]
