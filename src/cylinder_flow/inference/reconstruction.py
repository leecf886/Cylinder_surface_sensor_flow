"""Flow field reconstruction inference pipelines (oracle and closed-loop)."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union
import numpy as np
import torch

from cylinder_flow.artifacts.checkpoints import (
    DeepSetsPackage,
    LegacyMLPPackage,
    LegacyPPINNPackage,
)
from cylinder_flow.data.normalization import Normalizer
from cylinder_flow.data.types import MeshData, ReCaseData
from cylinder_flow.evaluation.cycles import wrap_phase
from cylinder_flow.inference.encoder import EncoderPrediction, predict_re_phase_full
from cylinder_flow.models.deepsets import (
    DynamicSensorBatch,
    DynamicTokenRegressor,
    FixedFlattenedAdapter,
    SetSensorEncoder,
    validate_sensor_indices,
)
from cylinder_flow.models.mlp import SensorMLP
from cylinder_flow.models.ppinn import PPINNModel
from cylinder_flow.physics.scaling import scale_field_to_physical


def reconstruct_field_chunked(
    model: PPINNModel,
    roi_xy: np.ndarray,
    re_series: Union[np.ndarray, Sequence[float], float],
    phi_series: Union[np.ndarray, Sequence[float], float],
    device: Union[str, torch.device] = "cpu",
    chunk_size: int = 16384,
) -> np.ndarray:
    """Reconstruct non-dimensional flow field (u*, v*, p*) across space and time.

    Performs spatial chunking over mesh coordinates to avoid memory and CUDA OOM
    when reconstructing flow fields across many snapshots and large ROI meshes.

    Parameters:
        model: Evaluated PPINNModel instance.
        roi_xy: ROI mesh physical coordinates of shape [Nroi, 2].
        re_series: Physical Reynolds number, either a scalar float or 1D array of shape [Nt].
        phi_series: Shedding phase in radians, either a scalar float or 1D array of shape [Nt].
        device: PyTorch device to execute inference on.
        chunk_size: Number of spatial points to process in each forward chunk.

    Returns:
        Predicted non-dimensional flow field [Nt, Nroi, 3] or [Nroi, 3] if scalar inputs.
    """
    xy_arr = np.asarray(roi_xy, dtype=np.float32)
    if xy_arr.ndim != 2 or xy_arr.shape[1] != 2:
        raise ValueError(f"roi_xy must have shape [Nroi, 2], got {xy_arr.shape}")

    nroi = xy_arr.shape[0]
    x_full = xy_arr[:, 0]
    y_full = xy_arr[:, 1]

    # Normalize re_series and phi_series to 1D arrays
    is_scalar_input = np.ndim(re_series) == 0 and np.ndim(phi_series) == 0
    re_arr = np.atleast_1d(np.asarray(re_series, dtype=np.float32))
    phi_arr = np.atleast_1d(np.asarray(phi_series, dtype=np.float32))

    if re_arr.size == 1 and phi_arr.size > 1:
        re_arr = np.full_like(phi_arr, re_arr.item())
    elif phi_arr.size == 1 and re_arr.size > 1:
        phi_arr = np.full_like(re_arr, phi_arr.item())

    if re_arr.shape[0] != phi_arr.shape[0]:
        raise ValueError(
            f"Mismatched time lengths: re_series has {re_arr.shape[0]} while phi_series has {phi_arr.shape[0]}"
        )

    nt = re_arr.shape[0]
    out = np.empty((nt, nroi, 3), dtype=np.float32)

    dev = torch.device(device)
    model.eval()
    model.to(dev)

    effective_chunk = max(int(chunk_size), 1)

    with torch.no_grad():
        for t in range(nt):
            re_t = float(re_arr[t])
            phi_t = float(wrap_phase(phi_arr[t]))

            for st in range(0, nroi, effective_chunk):
                ed = min(st + effective_chunk, nroi)
                n_chunk = ed - st

                x_t = torch.from_numpy(x_full[st:ed]).to(dev)
                y_t = torch.from_numpy(y_full[st:ed]).to(dev)
                re_t_tensor = torch.full((n_chunk,), re_t, dtype=torch.float32, device=dev)
                phi_t_tensor = torch.full((n_chunk,), phi_t, dtype=torch.float32, device=dev)

                pred = model(x_t, y_t, re_t_tensor, phi_t_tensor)
                out[t, st:ed, :] = pred.detach().cpu().numpy()

    if is_scalar_input:
        return out[0]
    return out


def reconstruct_field_oracle(
    ppinn: Union[PPINNModel, LegacyPPINNPackage],
    mesh_xy: np.ndarray,
    true_re: Union[np.ndarray, Sequence[float], float],
    true_phase: Union[np.ndarray, Sequence[float], float],
    output_scale: str = "nondimensional",
    u_inf: Optional[float] = None,
    rho: Optional[float] = None,
    pressure_scale_mode: Optional[str] = None,
    chunk_size: int = 16384,
    device: Union[str, torch.device] = "cpu",
) -> np.ndarray:
    """Reconstruct flow field in Oracle mode given ground-truth Re and shedding phase.

    Parameters:
        ppinn: Either a PPINNModel instance or loaded LegacyPPINNPackage.
        mesh_xy: ROI mesh physical coordinates [Nroi, 2].
        true_re: Ground truth Reynolds number (scalar or 1D array).
        true_phase: Ground truth shedding phase in radians (scalar or 1D array).
        output_scale: Output unit scale: 'nondimensional' (u*, v*, p*) or 'physical' (m/s, Pa).
        u_inf: Optional free-stream velocity (uses package value or default 1.0 if None).
        rho: Optional fluid density (uses package value or default 1000.0 if None).
        pressure_scale_mode: Pressure scaling mode (uses package value or 'rho_uinf2' if None).
        chunk_size: Spatial chunk size for batched evaluation.
        device: PyTorch device ('cpu', 'cuda').

    Returns:
        Predicted flow field array [Nt, Nroi, 3] or [Nroi, 3] in requested scale.
    """
    if isinstance(ppinn, LegacyPPINNPackage):
        model = ppinn.model
        u_inf = ppinn.u_inf if u_inf is None else u_inf
        rho = ppinn.rho if rho is None else rho
        pressure_scale_mode = ppinn.pressure_scale_mode if pressure_scale_mode is None else pressure_scale_mode
    elif isinstance(ppinn, PPINNModel):
        model = ppinn
        u_inf = 1.0 if u_inf is None else u_inf
        rho = 1000.0 if rho is None else rho
        pressure_scale_mode = "rho_uinf2" if pressure_scale_mode is None else pressure_scale_mode
    else:
        raise TypeError(f"ppinn must be PPINNModel or LegacyPPINNPackage, got {type(ppinn)}")

    field_star = reconstruct_field_chunked(
        model=model,
        roi_xy=mesh_xy,
        re_series=true_re,
        phi_series=true_phase,
        device=device,
        chunk_size=chunk_size,
    )

    scale_mode = str(output_scale).strip().lower()
    if scale_mode == "nondimensional":
        return field_star
    if scale_mode == "physical":
        return scale_field_to_physical(
            uvp_star=field_star,
            u_inf=float(u_inf),
            pressure_scale_mode=str(pressure_scale_mode),
            rho=float(rho),
        ).astype(np.float32)
    raise ValueError(f"Unsupported output_scale: {output_scale!r}. Must be 'nondimensional' or 'physical'.")


def reconstruct_field_closed_loop(
    mlp: Union[
        SensorMLP,
        LegacyMLPPackage,
        DeepSetsPackage,
        DynamicTokenRegressor,
        SetSensorEncoder,
        FixedFlattenedAdapter,
        torch.nn.Module,
    ],
    ppinn: Union[PPINNModel, LegacyPPINNPackage],
    mesh_xy: np.ndarray,
    sensor_inputs: Union[np.ndarray, torch.Tensor, DynamicSensorBatch],
    x_normalizer: Optional[Normalizer] = None,
    re_normalizer: Optional[Normalizer] = None,
    output_scale: str = "nondimensional",
    u_inf: Optional[float] = None,
    rho: Optional[float] = None,
    pressure_scale_mode: Optional[str] = None,
    chunk_size: int = 16384,
    device: Union[str, torch.device] = "cpu",
    positions: Optional[Union[Sequence[Sequence[float]], np.ndarray, torch.Tensor]] = None,
) -> Tuple[np.ndarray, EncoderPrediction]:
    """Reconstruct flow field in closed-loop mode (sensors -> encoder -> pPINN).

    Supports both fixed-layout encoders (SensorMLP, SetSensorEncoder) and dynamic
    encoders (DeepSetsPackage, DynamicSetSensorEncoder) with flat or token inputs.

    Parameters:
        mlp: Trained encoder instance or loaded package (MLP or DeepSets).
        ppinn: Either a PPINNModel instance or loaded LegacyPPINNPackage.
        mesh_xy: ROI mesh physical coordinates [Nroi, 2].
        sensor_inputs: Raw sensor observations or DynamicSensorBatch.
        x_normalizer: Fitted sensor feature normalizer (if mlp is bare SensorMLP).
        re_normalizer: Fitted Reynolds number normalizer (if mlp is bare model).
        output_scale: Output scale: 'nondimensional' or 'physical'.
        u_inf: Optional free-stream velocity.
        rho: Optional fluid density.
        pressure_scale_mode: Pressure scaling mode identifier.
        chunk_size: Spatial chunk size for pPINN forward evaluation.
        device: PyTorch device ('cpu', 'cuda').
        positions: Optional sensor coordinates if passing flat inputs to dynamic model.

    Returns:
        Tuple of:
            - field_pred: Reconstructed flow field [Nt, Nroi, 3] or [Nroi, 3].
            - encoder_pred: EncoderPrediction containing (re, phase, phase_sin, phase_cos).
    """
    # 1. Predict Re and Phase from sensor signals
    encoder_pred = predict_re_phase_full(
        model=mlp,
        sensor_inputs=sensor_inputs,
        x_normalizer=x_normalizer,
        re_normalizer=re_normalizer,
        device=device,
        positions=positions,
    )

    # 2. Reconstruct flow field with pPINN using predicted Re and Phase
    field_pred = reconstruct_field_oracle(
        ppinn=ppinn,
        mesh_xy=mesh_xy,
        true_re=encoder_pred.re,
        true_phase=encoder_pred.phase,
        output_scale=output_scale,
        u_inf=u_inf,
        rho=rho,
        pressure_scale_mode=pressure_scale_mode,
        chunk_size=chunk_size,
        device=device,
    )

    return field_pred, encoder_pred


def reconstruct_case_oracle(
    ppinn_package: LegacyPPINNPackage,
    case: ReCaseData,
    mesh: MeshData,
    time_indices: Optional[Sequence[int]] = None,
    output_scale: str = "nondimensional",
    chunk_size: int = 16384,
    device: Union[str, torch.device] = "cpu",
) -> np.ndarray:
    """Convenience oracle reconstruction wrapper for ReCaseData and MeshData structures.

    Parameters:
        ppinn_package: Loaded LegacyPPINNPackage.
        case: ReCaseData instance.
        mesh: MeshData containing ROI mesh coordinates.
        time_indices: Optional subset of snapshot indices to reconstruct.
        output_scale: 'nondimensional' or 'physical'.
        chunk_size: Spatial chunk size.
        device: PyTorch device.

    Returns:
        Predicted flow field array [N_snapshots, Nroi, 3].
    """
    if time_indices is not None:
        phase_subset = case.phase[time_indices]
        re_subset = np.full_like(phase_subset, case.re)
    else:
        phase_subset = case.phase
        re_subset = case.re

    return reconstruct_field_oracle(
        ppinn=ppinn_package,
        mesh_xy=mesh.roi_xy,
        true_re=re_subset,
        true_phase=phase_subset,
        output_scale=output_scale,
        u_inf=case.u_inf,
        rho=ppinn_package.rho,
        pressure_scale_mode=ppinn_package.pressure_scale_mode,
        chunk_size=chunk_size,
        device=device,
    )


def reconstruct_case_closed_loop(
    mlp_package: Union[LegacyMLPPackage, DeepSetsPackage],
    ppinn_package: LegacyPPINNPackage,
    case: ReCaseData,
    mesh: MeshData,
    time_indices: Optional[Sequence[int]] = None,
    output_scale: str = "nondimensional",
    chunk_size: int = 16384,
    device: Union[str, torch.device] = "cpu",
    *,
    sensor_indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, EncoderPrediction]:
    """Convenience closed-loop reconstruction wrapper for ReCaseData and MeshData structures.

    Supports both LegacyMLPPackage (fixed layout) and DeepSetsPackage (dynamic set).

    Parameters:
        mlp_package: Loaded LegacyMLPPackage or DeepSetsPackage.
        ppinn_package: Loaded LegacyPPINNPackage.
        case: ReCaseData instance.
        mesh: MeshData containing ROI mesh coordinates.
        time_indices: Optional subset of snapshot indices to reconstruct.
        output_scale: 'nondimensional' or 'physical'.
        chunk_size: Spatial chunk size.
        device: PyTorch device.
        sensor_indices: Optional subset of sensor indices. Keyword-only.

    Returns:
        Tuple of (field_pred, encoder_pred).
    """
    if isinstance(mlp_package, LegacyMLPPackage):
        if sensor_indices is not None:
            val_idx = validate_sensor_indices(sensor_indices, candidate_count=case.ncand)
            if list(val_idx) != list(mlp_package.sensor_indices):
                raise ValueError(
                    f"Legacy MLP fixed layout does not support changing sensor layout or order. "
                    f"Expected {mlp_package.sensor_indices}, got {list(val_idx)}"
                )
            idx = val_idx
        else:
            idx = mlp_package.sensor_indices

        sensor_inputs = case.get_sensors(idx, flatten=True)
        if time_indices is not None:
            sensor_inputs = sensor_inputs[time_indices]

    elif isinstance(mlp_package, DeepSetsPackage):
        if case.ncand != mlp_package.candidate_count:
            raise ValueError(
                f"Case candidate count ({case.ncand}) does not match package pool count "
                f"({mlp_package.candidate_count})"
            )
        raw_idx = mlp_package.default_sensor_indices if sensor_indices is None else sensor_indices
        idx = validate_sensor_indices(raw_idx, candidate_count=mlp_package.candidate_count)

        values = case.get_sensors(idx, flatten=False)  # [Nt, K, 3]
        if time_indices is not None:
            values = values[time_indices]
        nt = values.shape[0]
        k = len(idx)
        pos = mlp_package.candidate_positions[idx]
        pos_batch = np.broadcast_to(pos[None, :, :], (nt, k, 2)).astype(np.float32).copy()
        mask = np.ones((nt, k), dtype=bool)
        count = np.full((nt, 1), float(k), dtype=np.float32)
        sensor_inputs = DynamicSensorBatch(values=values.astype(np.float32), pos=pos_batch, mask=mask, count=count)

    else:
        raise TypeError(
            f"mlp_package must be LegacyMLPPackage or DeepSetsPackage, got {type(mlp_package)}"
        )

    return reconstruct_field_closed_loop(
        mlp=mlp_package,
        ppinn=ppinn_package,
        mesh_xy=mesh.roi_xy,
        sensor_inputs=sensor_inputs,
        output_scale=output_scale,
        u_inf=case.u_inf,
        rho=ppinn_package.rho,
        pressure_scale_mode=ppinn_package.pressure_scale_mode,
        chunk_size=chunk_size,
        device=device,
    )
