"""Contract tests for DeepSets permutation invariance, dynamic tokens, and legacy parity.

Verifies:
- Synchronized permutation invariance across all pooling modes (mean, sum, max, mean+max).
- Mixed counts within batches.
- Padded token invariance, strictly ignoring non-finite (NaN, +/-Inf) padding.
- Input validation: shapes, dtypes, all-mask rejection, count mismatch, unit-circle checks.
- Position encoding ablation (cos_sin_theta vs none).
- Fixed-layout SetSensorEncoder and FixedFlattenedAdapter.
- Exact state_dict key compatibility and numerical output parity against legacy model.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Sequence
import unittest

import numpy as np
import pytest
import torch

from cylinder_flow.models.deepsets import (
    DEEPSETS_ACTIVATION,
    DYNAMIC_CHECKPOINT_SCHEMA_NAME,
    DYNAMIC_CHECKPOINT_SCHEMA_VERSION,
    DYNAMIC_COUNT_FEATURES,
    DYNAMIC_ENCODER_NAME,
    DYNAMIC_POSITION_ENCODING,
    DYNAMIC_POSITION_ENCODINGS,
    DYNAMIC_POSITION_FINGERPRINT_METHOD,
    SET_POOL_MODES,
    TWO_PI,
    DeepSetsSensorEncoder,
    DynamicSetSensorEncoder,
    DynamicTokenRegressor,
    FixedFlattenedAdapter,
    SetSensorEncoder,
    candidate_position_fingerprint,
    channel_stats_from_flat,
    dynamic_contract_fingerprint,
    is_dynamic_token_regressor,
    positions_from_theta,
    validate_dynamic_checkpoint_metadata,
    validate_dynamic_token_batch,
)

LEGACY_ENCODER_PATH = Path("E:/code/2d_surface_cylinder_flow/encoder_deepset.py")


def _load_legacy_module():
    """Dynamically import legacy encoder_deepset module read-only if available."""
    if not LEGACY_ENCODER_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location("legacy_encoder_deepset", LEGACY_ENCODER_PATH)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestDeepSetsContracts(unittest.TestCase):
    """Test suite for DeepSets models and input contracts."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        np.random.seed(42)

    def test_permutation_invariance_all_pooling_modes(self) -> None:
        """Active tokens permuted with synchronized values, pos, mask must give identical output."""
        max_n = 10
        actual_n = 6
        theta = np.linspace(0.0, TWO_PI, actual_n, endpoint=False)
        pos = torch.from_numpy(positions_from_theta(theta).astype(np.float32)).unsqueeze(0).expand(2, -1, -1)
        values = torch.randn(2, actual_n, 3, dtype=torch.float32)
        mask = torch.ones(2, actual_n, dtype=torch.bool)
        count = torch.full((2, 1), float(actual_n), dtype=torch.float32)

        for pool in SET_POOL_MODES:
            for count_feature in DYNAMIC_COUNT_FEATURES:
                with self.subTest(pool=pool, count_feature=count_feature):
                    model = DynamicSetSensorEncoder(
                        candidate_count=max_n,
                        embed_dim=16,
                        phi_hidden=[16, 16],
                        rho_hidden=[16, 16],
                        out_dim=3,
                        pool=pool,
                        count_feature=count_feature,
                        chan_mean=[0.1, -0.2, 1.0],
                        chan_std=[0.5, 0.4, 2.0],
                    )
                    model.eval()

                    with torch.no_grad():
                        out_orig = model.forward_tokens(values, pos, mask, count)

                        # Apply multiple random synchronized permutations
                        for _ in range(3):
                            perm = torch.randperm(actual_n)
                            out_perm = model.forward_tokens(
                                values[:, perm, :],
                                pos[:, perm, :],
                                mask[:, perm],
                                count,
                            )
                            self.assertTrue(
                                torch.allclose(out_orig, out_perm, atol=1e-6),
                                f"Permutation failed for pool={pool}, count_feature={count_feature}",
                            )

    def test_padding_invariance_and_nonfinite_padding(self) -> None:
        """Masked padding slots must be completely ignored even if they contain NaN, Inf, or extremes."""
        candidate_count = 8
        n_active = 4
        theta_active = np.linspace(0.0, TWO_PI, n_active, endpoint=False)
        pos_active = torch.from_numpy(positions_from_theta(theta_active).astype(np.float32)).unsqueeze(0).expand(2, -1, -1)
        values_active = torch.randn(2, n_active, 3, dtype=torch.float32)
        mask_active = torch.ones(2, n_active, dtype=torch.bool)
        count_active = torch.full((2, 1), float(n_active), dtype=torch.float32)

        for pool in SET_POOL_MODES:
            model = DynamicSetSensorEncoder(
                candidate_count=candidate_count,
                embed_dim=16,
                phi_hidden=[16],
                rho_hidden=[16],
                out_dim=3,
                pool=pool,
                chan_mean=[0.0, 0.0, 0.0],
                chan_std=[1.0, 1.0, 1.0],
            )
            model.eval()

            with torch.no_grad():
                out_clean = model.forward_tokens(values_active, pos_active, mask_active, count_active)

                # 1. Zero padding
                pad_v_zero = torch.zeros(2, candidate_count - n_active, 3, dtype=torch.float32)
                pad_pos_valid = torch.zeros(2, candidate_count - n_active, 2, dtype=torch.float32)
                pad_pos_valid[..., 0] = 1.0  # valid points on unit circle
                pad_mask = torch.zeros(2, candidate_count - n_active, dtype=torch.bool)

                v_padded = torch.cat([values_active, pad_v_zero], dim=1)
                p_padded = torch.cat([pos_active, pad_pos_valid], dim=1)
                m_padded = torch.cat([mask_active, pad_mask], dim=1)

                out_zero_pad = model.forward_tokens(v_padded, p_padded, m_padded, count_active)
                self.assertTrue(
                    torch.allclose(out_clean, out_zero_pad, atol=1e-6),
                    f"Zero padding failed for pool={pool}",
                )

                # 2. Non-finite padding (NaN, +Inf, -Inf, 1e25) in masked slots
                non_finite_values = [
                    float("nan"),
                    float("inf"),
                    float("-inf"),
                    1e25,
                ]
                for bad_val in non_finite_values:
                    pad_v_bad = torch.full_like(pad_v_zero, bad_val)
                    pad_p_bad = torch.full_like(pad_pos_valid, bad_val)

                    v_bad = torch.cat([values_active, pad_v_bad], dim=1)
                    p_bad = torch.cat([pos_active, pad_p_bad], dim=1)

                    out_bad_pad = model.forward_tokens(v_bad, p_bad, m_padded, count_active)
                    self.assertTrue(
                        torch.allclose(out_clean, out_bad_pad, atol=1e-6),
                        f"Nonfinite padding ({bad_val}) affected output for pool={pool}",
                    )

    def test_mixed_counts_in_batch(self) -> None:
        """Batch rows with different sensor counts must match individual single-row evaluations."""
        candidate_count = 6
        theta = np.linspace(0.0, TWO_PI, candidate_count, endpoint=False)
        pos = torch.from_numpy(positions_from_theta(theta).astype(np.float32)).unsqueeze(0).expand(3, -1, -1)
        values = torch.randn(3, candidate_count, 3, dtype=torch.float32)

        # Row 0: 2 sensors (idx 0, 1)
        # Row 1: 5 sensors (idx 0, 1, 2, 3, 4)
        # Row 2: 1 sensor  (idx 0) singleton
        mask = torch.zeros(3, candidate_count, dtype=torch.bool)
        mask[0, :2] = True
        mask[1, :5] = True
        mask[2, :1] = True

        model = DynamicSetSensorEncoder(
            candidate_count=candidate_count,
            embed_dim=16,
            phi_hidden=[16],
            rho_hidden=[16],
            out_dim=3,
            pool="mean+max",
            count_feature="n_over_candidate_count",
        )
        model.eval()

        with torch.no_grad():
            out_batch = model.forward_tokens(values, pos, mask, None)
            self.assertEqual(out_batch.shape, (3, 3))

            for i in range(3):
                active_len = int(mask[i].sum().item())
                row_val = values[i : i + 1, :active_len, :]
                row_pos = pos[i : i + 1, :active_len, :]
                row_mask = torch.ones(1, active_len, dtype=torch.bool)
                out_single = model.forward_tokens(row_val, row_pos, row_mask, None)
                self.assertTrue(
                    torch.allclose(out_batch[i : i + 1], out_single, atol=1e-6),
                    f"Batch row {i} (N={active_len}) mismatched single-case output",
                )

    def test_count_derivation_and_mismatches(self) -> None:
        """Test count auto-derivation, explicit count match, and mismatch rejections."""
        candidate_count = 4
        theta = np.linspace(0.0, TWO_PI, candidate_count, endpoint=False)
        pos = torch.from_numpy(positions_from_theta(theta).astype(np.float32)).unsqueeze(0).expand(2, -1, -1)
        values = torch.randn(2, candidate_count, 3, dtype=torch.float32)
        mask = torch.zeros(2, candidate_count, dtype=torch.bool)
        mask[0, 0:2] = True
        mask[1, 0:3] = True

        model = DynamicSetSensorEncoder(
            candidate_count=candidate_count,
            embed_dim=8,
            phi_hidden=[8],
            rho_hidden=[8],
            out_dim=3,
        )
        model.eval()

        with torch.no_grad():
            out_auto = model.forward_tokens(values, pos, mask, None)
            explicit_count = mask.sum(dim=1, keepdim=True).to(dtype=torch.float32)
            out_explicit = model.forward_tokens(values, pos, mask, explicit_count)
            self.assertTrue(torch.allclose(out_auto, out_explicit, atol=1e-6))

        # Count mismatch must raise ValueError
        wrong_count = explicit_count.clone()
        wrong_count[0, 0] += 1.0
        with self.assertRaises(ValueError) as ctx:
            model.forward_tokens(values, pos, mask, wrong_count)
        self.assertIn("count must equal the number of True mask entries", str(ctx.exception))

        # Invalid count shapes
        with self.assertRaises(ValueError):
            model.forward_tokens(values, pos, mask, explicit_count.squeeze(1))  # 1D

        # Invalid count dtype
        with self.assertRaises(TypeError):
            model.forward_tokens(values, pos, mask, explicit_count.long())

        # NaN count
        with self.assertRaises(ValueError):
            nan_count = explicit_count.clone()
            nan_count[0, 0] = float("nan")
            model.forward_tokens(values, pos, mask, nan_count)

    def test_all_mask_rejected(self) -> None:
        """Batch containing an all-mask (empty active sensors) row must raise ValueError."""
        candidate_count = 4
        theta = np.linspace(0.0, TWO_PI, candidate_count, endpoint=False)
        pos = torch.from_numpy(positions_from_theta(theta).astype(np.float32)).unsqueeze(0).expand(2, -1, -1)
        values = torch.randn(2, candidate_count, 3, dtype=torch.float32)
        all_mask = torch.zeros(2, candidate_count, dtype=torch.bool)

        model = DynamicSetSensorEncoder(
            candidate_count=candidate_count,
            embed_dim=8,
            phi_hidden=[8],
            rho_hidden=[8],
        )
        with self.assertRaises(ValueError) as ctx:
            model.forward_tokens(values, pos, all_mask, None)
        self.assertIn("all-mask sensor sets are invalid", str(ctx.exception))

    def test_validation_errors(self) -> None:
        """Ensure input validation checks enforce shapes, dtypes, and unit-circle bounds."""
        candidate_count = 4
        theta = np.linspace(0.0, TWO_PI, candidate_count, endpoint=False)
        pos = torch.from_numpy(positions_from_theta(theta).astype(np.float32)).unsqueeze(0).expand(2, -1, -1)
        values = torch.randn(2, candidate_count, 3, dtype=torch.float32)
        mask = torch.ones(2, candidate_count, dtype=torch.bool)

        ref = torch.zeros(1, 1, 3, dtype=torch.float32)

        # 1. Invalid values shape
        with self.assertRaises(ValueError):
            validate_dynamic_token_batch(values[:, :, :2], pos, mask, None, candidate_count=4, reference=ref)
        with self.assertRaises(ValueError):
            validate_dynamic_token_batch(values[0], pos, mask, None, candidate_count=4, reference=ref)

        # 2. Invalid pos shape
        with self.assertRaises(ValueError):
            bad_pos = torch.zeros(2, candidate_count, 3, dtype=torch.float32)
            validate_dynamic_token_batch(values, bad_pos, mask, None, candidate_count=4, reference=ref)

        # 3. Invalid mask shape
        with self.assertRaises(ValueError):
            bad_mask = torch.ones(2, candidate_count + 1, dtype=torch.bool)
            validate_dynamic_token_batch(values, pos, bad_mask, None, candidate_count=4, reference=ref)

        # 4. Exceeds candidate_count
        with self.assertRaises(ValueError):
            validate_dynamic_token_batch(values, pos, mask, None, candidate_count=3, reference=ref)

        # 5. Invalid dtypes
        with self.assertRaises(TypeError):
            validate_dynamic_token_batch(values.long(), pos, mask, None, candidate_count=4, reference=ref)
        with self.assertRaises(TypeError):
            validate_dynamic_token_batch(values, pos, mask.float(), None, candidate_count=4, reference=ref)

        # 6. NaN in valid active values
        bad_val = values.clone()
        bad_val[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError) as ctx:
            validate_dynamic_token_batch(bad_val, pos, mask, None, candidate_count=4, reference=ref)
        self.assertIn("valid values/pos entries contain NaN/Inf", str(ctx.exception))

        # 7. Valid pos not on unit circle
        bad_pos_circle = pos.clone()
        bad_pos_circle[0, 0, :] = 2.0  # norm = 2*sqrt(2) != 1.0
        with self.assertRaises(ValueError) as ctx:
            validate_dynamic_token_batch(values, bad_pos_circle, mask, None, candidate_count=4, reference=ref)
        self.assertIn("unit circle", str(ctx.exception))

    def test_position_encoding_ablation(self) -> None:
        """Verify cos_sin_theta uses positions, whereas none is invariant to position changes."""
        candidate_count = 6
        theta = np.linspace(0.0, TWO_PI, candidate_count, endpoint=False)
        pos = torch.from_numpy(positions_from_theta(theta).astype(np.float32)).unsqueeze(0).expand(2, -1, -1)
        values = torch.randn(2, candidate_count, 3, dtype=torch.float32)
        mask = torch.ones(2, candidate_count, dtype=torch.bool)

        model_cos = DynamicSetSensorEncoder(
            candidate_count=candidate_count,
            embed_dim=8,
            phi_hidden=[8],
            rho_hidden=[8],
            position_encoding="cos_sin_theta",
        )
        model_none = DynamicSetSensorEncoder(
            candidate_count=candidate_count,
            embed_dim=8,
            phi_hidden=[8],
            rho_hidden=[8],
            position_encoding="none",
        )
        model_cos.eval()
        model_none.eval()

        # Token input dim check: 5 for cos_sin_theta, 3 for none
        self.assertEqual(model_cos.phi[0].in_features, 5)
        self.assertEqual(model_none.phi[0].in_features, 3)

        with torch.no_grad():
            out_cos = model_cos(values, pos, mask)
            out_none = model_none(values, pos, mask)

            # Rotate positions by delta angle
            rotated_theta = theta + 0.5
            rotated_pos = torch.from_numpy(positions_from_theta(rotated_theta).astype(np.float32)).unsqueeze(0).expand(2, -1, -1)

            out_cos_rot = model_cos(values, rotated_pos, mask)
            out_none_rot = model_none(values, rotated_pos, mask)

            # cos_sin_theta must change when position rotates
            self.assertFalse(torch.allclose(out_cos, out_cos_rot, atol=1e-5))
            # none must be strictly invariant to position rotates
            self.assertTrue(torch.allclose(out_none, out_none_rot, atol=1e-6))

    def test_fixed_set_sensor_encoder_and_flattened_adapter(self) -> None:
        """Test SetSensorEncoder and FixedFlattenedAdapter with flattened [B, 3*K] inputs."""
        n_sensors = 5
        theta = np.linspace(0.0, TWO_PI, n_sensors, endpoint=False)
        positions = positions_from_theta(theta)

        fixed_model = SetSensorEncoder(
            theta=theta,
            embed_dim=16,
            phi_hidden=[16],
            rho_hidden=[16],
            out_dim=3,
            pool="mean",
        )
        fixed_model.eval()

        # Check ReLU activations only
        relu_modules = [m for net in (fixed_model.phi, fixed_model.rho) for m in net if not isinstance(m, torch.nn.Linear)]
        self.assertTrue(all(isinstance(m, torch.nn.ReLU) for m in relu_modules))

        x_flat = torch.randn(3, n_sensors * 3, dtype=torch.float32)
        with torch.no_grad():
            out_fixed = fixed_model(x_flat)
            self.assertEqual(out_fixed.shape, (3, 3))

            # Permute sensors simultaneously in flat array and in theta
            perm = torch.randperm(n_sensors)
            x_reshaped = x_flat.reshape(3, n_sensors, 3)[:, perm, :].reshape(3, n_sensors * 3)
            perm_model = SetSensorEncoder(
                theta=theta[perm.numpy()],
                embed_dim=16,
                phi_hidden=[16],
                rho_hidden=[16],
                out_dim=3,
                pool="mean",
            )
            # Transfer weights to ensure identical MLPs
            perm_model.phi.load_state_dict(fixed_model.phi.state_dict())
            perm_model.rho.load_state_dict(fixed_model.rho.state_dict())
            perm_model.eval()

            out_perm = perm_model(x_reshaped)
            self.assertTrue(torch.allclose(out_fixed, out_perm, atol=1e-6))

        # Test FixedFlattenedAdapter wrapping DynamicSetSensorEncoder
        dynamic_model = DynamicSetSensorEncoder(
            candidate_count=10,
            embed_dim=16,
            phi_hidden=[16],
            rho_hidden=[16],
            out_dim=3,
            count_feature="none",
        )
        dynamic_model.eval()

        adapter = FixedFlattenedAdapter(dynamic_model, positions)
        with torch.no_grad():
            out_adapter = adapter(x_flat)
            out_method = dynamic_model.forward_flat(x_flat, positions)
            self.assertTrue(torch.allclose(out_adapter, out_method, atol=1e-6))

            # Compare against explicit forward_tokens
            val_tokens = x_flat.reshape(3, n_sensors, 3)
            pos_tokens = torch.from_numpy(positions.astype(np.float32)).unsqueeze(0).expand(3, -1, -1)
            mask_tokens = torch.ones(3, n_sensors, dtype=torch.bool)
            count_tokens = torch.full((3, 1), float(n_sensors), dtype=torch.float32)

            out_tokens = dynamic_model.forward_tokens(val_tokens, pos_tokens, mask_tokens, count_tokens)
            self.assertTrue(torch.allclose(out_adapter, out_tokens, atol=1e-6))

    def test_fixed_flattened_adapter_double_precision_parity(self) -> None:
        """Test double precision parity of FixedFlattenedAdapter and forward_flat vs explicit forward_tokens."""
        # 1. Parity test with multiple sensors and batch size
        n_sensors = 4
        candidate_count = 8
        theta = np.linspace(0.0, TWO_PI, n_sensors, endpoint=False)
        positions = positions_from_theta(theta)

        dynamic_model = DynamicSetSensorEncoder(
            candidate_count=candidate_count,
            embed_dim=16,
            phi_hidden=[16],
            rho_hidden=[16],
            out_dim=3,
            count_feature="n_over_candidate_count",
        ).double()
        dynamic_model.eval()

        bsz = 3
        x_flat = torch.randn(bsz, n_sensors * 3, dtype=torch.float64)

        adapter = FixedFlattenedAdapter(dynamic_model, positions)
        with torch.no_grad():
            out_adapter = adapter(x_flat)
            out_method = dynamic_model.forward_flat(x_flat, positions)

            self.assertEqual(out_adapter.dtype, torch.float64)
            self.assertEqual(out_method.dtype, torch.float64)
            self.assertEqual(out_adapter.shape, (bsz, 3))
            self.assertTrue(torch.allclose(out_adapter, out_method, atol=1e-12))

            # Compare against explicit forward_tokens
            val_tokens = x_flat.reshape(bsz, n_sensors, 3)
            # Match adapter's position tensor: float32 canonical positions aligned to float64
            pos_tokens = (
                torch.as_tensor(np.asarray(positions, dtype=np.float32), dtype=torch.float64)
                .unsqueeze(0)
                .expand(bsz, -1, -1)
            )
            mask_tokens = torch.ones(bsz, n_sensors, dtype=torch.bool)
            count_tokens = torch.full((bsz, 1), float(n_sensors), dtype=torch.float64)

            out_tokens = dynamic_model.forward_tokens(val_tokens, pos_tokens, mask_tokens, count_tokens)
            self.assertEqual(out_tokens.dtype, torch.float64)
            self.assertTrue(torch.allclose(out_adapter, out_tokens, atol=1e-12))

        # 2. Focused test for reproduced failure case: [candidate_count=2, embed_dim=4, [[1.,0.],[0.,1.]]]
        m_repro = DynamicSetSensorEncoder(
            candidate_count=2,
            embed_dim=4,
            phi_hidden=(4,),
            rho_hidden=(4,),
        ).double()
        m_repro.eval()
        x_repro = torch.randn(1, 6, dtype=torch.float64)
        pos_repro = [[1.0, 0.0], [0.0, 1.0]]

        with torch.no_grad():
            out_repro_flat = m_repro.forward_flat(x_repro, pos_repro)
            self.assertEqual(out_repro_flat.dtype, torch.float64)
            self.assertEqual(out_repro_flat.shape, (1, 3))

            pos_repro_tokens = torch.tensor([pos_repro], dtype=torch.float64)
            out_repro_tokens = m_repro.forward_tokens(
                x_repro.reshape(1, 2, 3),
                pos_repro_tokens,
                torch.ones(1, 2, dtype=torch.bool),
                torch.tensor([[2.0]], dtype=torch.float64),
            )
            self.assertTrue(torch.allclose(out_repro_flat, out_repro_tokens, atol=1e-12))

        # 3. Strict rejection of incompatible value dtype / model
        # Passing float32 input to double model must raise TypeError
        x_f32 = torch.randn(1, 6, dtype=torch.float32)
        with self.assertRaises(TypeError) as ctx:
            m_repro.forward_flat(x_f32, pos_repro)
        self.assertIn("values/pos dtype must match model dtype", str(ctx.exception))
        # Verify model was not altered
        self.assertEqual(m_repro.chan_mean.dtype, torch.float64)

        # Passing float64 input to float32 model must raise TypeError
        m_f32 = DynamicSetSensorEncoder(
            candidate_count=2,
            embed_dim=4,
            phi_hidden=(4,),
            rho_hidden=(4,),
        )
        with self.assertRaises(TypeError) as ctx2:
            m_f32.forward_flat(x_repro, pos_repro)
        self.assertIn("values/pos dtype must match model dtype", str(ctx2.exception))
        self.assertEqual(m_f32.chan_mean.dtype, torch.float32)

    def test_registered_adapter_to_behavior(self) -> None:
        """Verify registered FixedFlattenedAdapter .to(...) behavior for dtype and device."""
        model = DynamicSetSensorEncoder(
            candidate_count=4,
            embed_dim=8,
            phi_hidden=[8],
            rho_hidden=[8],
            out_dim=3,
        )
        positions = [[1.0, 0.0], [0.0, 1.0]]
        adapter = FixedFlattenedAdapter(model, positions)

        # Default initialization: float32 on CPU
        self.assertEqual(adapter.positions.dtype, torch.float32)
        self.assertEqual(adapter.positions.device, torch.device("cpu"))
        self.assertEqual(adapter.encoder.chan_mean.dtype, torch.float32)

        # Convert registered adapter to double precision
        adapter.to(torch.float64)
        self.assertEqual(adapter.positions.dtype, torch.float64)
        self.assertEqual(adapter.encoder.chan_mean.dtype, torch.float64)

        adapter.eval()
        x_f64 = torch.randn(2, 6, dtype=torch.float64)
        with torch.no_grad():
            out = adapter(x_f64)
            self.assertEqual(out.dtype, torch.float64)
            self.assertEqual(out.shape, (2, 3))

            # Parity vs explicit forward_tokens
            val_tokens = x_f64.reshape(2, 2, 3)
            pos_tokens = adapter.positions.unsqueeze(0).expand(2, -1, -1)
            out_tokens = adapter.encoder.forward_tokens(
                val_tokens,
                pos_tokens,
                torch.ones(2, 2, dtype=torch.bool),
                torch.full((2, 1), 2.0, dtype=torch.float64),
            )
            self.assertTrue(torch.allclose(out, out_tokens, atol=1e-12))

        # as_fixed_flattened_adapter helper check
        m2 = DynamicSetSensorEncoder(candidate_count=4, embed_dim=8, phi_hidden=[8], rho_hidden=[8])
        ad2 = m2.as_fixed_flattened_adapter(positions)
        ad2.double()
        self.assertEqual(ad2.positions.dtype, torch.float64)
        self.assertEqual(ad2.encoder.chan_mean.dtype, torch.float64)

        if torch.cuda.is_available():
            adapter.to(device="cuda")
            self.assertEqual(adapter.positions.device.type, "cuda")
            self.assertEqual(adapter.encoder.chan_mean.device.type, "cuda")
            x_cuda = torch.randn(2, 6, dtype=torch.float64, device="cuda")
            with torch.no_grad():
                out_cuda = adapter(x_cuda)
                self.assertEqual(out_cuda.device.type, "cuda")
                self.assertEqual(out_cuda.dtype, torch.float64)

    def test_cuda_convenience_forward_flat(self) -> None:
        """Conditional CUDA convenience forward_flat test if CUDA is available."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        positions = [[1.0, 0.0], [0.0, 1.0]]

        # 1. CUDA float32 convenience forward_flat
        model_cuda = DynamicSetSensorEncoder(
            candidate_count=4,
            embed_dim=8,
            phi_hidden=[8],
            rho_hidden=[8],
            out_dim=3,
        ).cuda()
        model_cuda.eval()

        x_cuda = torch.randn(2, 6, device="cuda", dtype=torch.float32)
        with torch.no_grad():
            out_cuda = model_cuda.forward_flat(x_cuda, positions)
            self.assertEqual(out_cuda.device.type, "cuda")
            self.assertEqual(out_cuda.dtype, torch.float32)
            self.assertEqual(out_cuda.shape, (2, 3))

            # Parity vs explicit forward_tokens on CUDA
            pos_tokens = torch.tensor(positions, device="cuda", dtype=torch.float32).unsqueeze(0).expand(2, -1, -1)
            out_tokens = model_cuda.forward_tokens(
                x_cuda.reshape(2, 2, 3),
                pos_tokens,
                torch.ones(2, 2, device="cuda", dtype=torch.bool),
                torch.full((2, 1), 2.0, device="cuda", dtype=torch.float32),
            )
            self.assertTrue(torch.allclose(out_cuda, out_tokens, atol=1e-6))

        # 2. CUDA float64 convenience forward_flat
        model_cuda_d = DynamicSetSensorEncoder(
            candidate_count=4,
            embed_dim=8,
            phi_hidden=[8],
            rho_hidden=[8],
            out_dim=3,
        ).cuda().double()
        model_cuda_d.eval()

        x_cuda_d = torch.randn(2, 6, device="cuda", dtype=torch.float64)
        with torch.no_grad():
            out_cuda_d = model_cuda_d.forward_flat(x_cuda_d, positions)
            self.assertEqual(out_cuda_d.device.type, "cuda")
            self.assertEqual(out_cuda_d.dtype, torch.float64)

        # 3. Device mismatch strict rejection guard
        x_cpu = torch.randn(2, 6, device="cpu", dtype=torch.float32)
        with self.assertRaises(ValueError) as ctx:
            model_cuda.forward_flat(x_cpu, positions)
        self.assertIn("must share model device", str(ctx.exception))
        # Verify model device was not altered
        self.assertEqual(model_cuda.chan_mean.device.type, "cuda")

    def test_legacy_model_parity_and_state_dict_compatibility(self) -> None:
        """Verify 100% parameter name match and numerical output parity against legacy model."""
        old_mod = _load_legacy_module()
        if old_mod is None:
            self.skipTest("Legacy encoder_deepset module not available at E:/code/2d_surface_cylinder_flow")

        candidate_count = 8
        theta = np.linspace(0.0, TWO_PI, candidate_count, endpoint=False)
        pos = torch.from_numpy(positions_from_theta(theta).astype(np.float32)).unsqueeze(0).expand(2, -1, -1)
        values = torch.randn(2, candidate_count, 3, dtype=torch.float32)
        mask = torch.ones(2, candidate_count, dtype=torch.bool)
        mask[1, 5:] = False  # row 1 has padding
        count = mask.sum(dim=1, keepdim=True).float()

        # 1. DynamicSetSensorEncoder parity across pooling modes and count features
        for pool in SET_POOL_MODES:
            for count_feature in DYNAMIC_COUNT_FEATURES:
                with self.subTest(pool=pool, count_feature=count_feature):
                    torch.manual_seed(123)
                    old_model = old_mod.DynamicSetSensorEncoder(
                        candidate_count=candidate_count,
                        embed_dim=16,
                        phi_hidden=[16, 16],
                        rho_hidden=[16, 16],
                        out_dim=3,
                        pool=pool,
                        count_feature=count_feature,
                        chan_mean=[0.1, -0.1, 0.5],
                        chan_std=[0.8, 0.9, 1.2],
                    )
                    old_model.eval()

                    new_model = DynamicSetSensorEncoder(
                        candidate_count=candidate_count,
                        embed_dim=16,
                        phi_hidden=[16, 16],
                        rho_hidden=[16, 16],
                        out_dim=3,
                        pool=pool,
                        count_feature=count_feature,
                        chan_mean=[0.1, -0.1, 0.5],
                        chan_std=[0.8, 0.9, 1.2],
                    )
                    new_model.eval()

                    # Exact state_dict key check
                    self.assertEqual(
                        set(old_model.state_dict().keys()),
                        set(new_model.state_dict().keys()),
                    )

                    # Load old state_dict into new model
                    new_model.load_state_dict(old_model.state_dict())

                    with torch.no_grad():
                        out_old = old_model.forward_tokens(values, pos, mask, count)
                        out_new = new_model.forward_tokens(values, pos, mask, count)

                        diff = torch.max(torch.abs(out_old - out_new)).item()
                        self.assertLess(
                            diff,
                            1e-7,
                            f"Numerical parity discrepancy={diff} for pool={pool}, count={count_feature}",
                        )

        # 2. SetSensorEncoder parity
        n_sensors = 6
        theta_fixed = np.linspace(0.0, TWO_PI, n_sensors, endpoint=False)
        x_flat = torch.randn(2, n_sensors * 3, dtype=torch.float32)

        torch.manual_seed(456)
        old_fixed = old_mod.SetSensorEncoder(
            theta=theta_fixed,
            embed_dim=16,
            phi_hidden=[16],
            rho_hidden=[16],
            out_dim=3,
            pool="mean",
            chan_mean=[0.2, -0.3, 0.4],
            chan_std=[1.1, 0.9, 1.5],
        )
        old_fixed.eval()

        new_fixed = SetSensorEncoder(
            theta=theta_fixed,
            embed_dim=16,
            phi_hidden=[16],
            rho_hidden=[16],
            out_dim=3,
            pool="mean",
            chan_mean=[0.2, -0.3, 0.4],
            chan_std=[1.1, 0.9, 1.5],
        )
        new_fixed.eval()

        self.assertEqual(
            set(old_fixed.state_dict().keys()),
            set(new_fixed.state_dict().keys()),
        )
        new_fixed.load_state_dict(old_fixed.state_dict())

        with torch.no_grad():
            out_old_fixed = old_fixed(x_flat)
            out_new_fixed = new_fixed(x_flat)
            diff_fixed = torch.max(torch.abs(out_old_fixed - out_new_fixed)).item()
            self.assertLess(diff_fixed, 1e-7, f"Fixed encoder disparity: {diff_fixed}")

        # 3. Canonical positions and fingerprint parity
        theta_dense = np.linspace(0.0, TWO_PI, 32, endpoint=False)
        pos_old = old_mod.positions_from_theta(theta_dense)
        pos_new = positions_from_theta(theta_dense)
        np.testing.assert_allclose(pos_old, pos_new, rtol=1e-12, atol=1e-12)

        fp_old = old_mod.candidate_position_fingerprint(pos_old)
        fp_new = candidate_position_fingerprint(pos_new)
        self.assertEqual(fp_old, fp_new)

    def test_metadata_contract_validation_and_fingerprint(self) -> None:
        """Validate dynamic metadata schema and fingerprint generation."""
        cand_count = 10
        theta = np.linspace(0.0, TWO_PI, cand_count, endpoint=False)
        pos = positions_from_theta(theta)
        fp = candidate_position_fingerprint(pos)

        metadata = {
            "schema_version": DYNAMIC_CHECKPOINT_SCHEMA_VERSION,
            "encoder": DYNAMIC_ENCODER_NAME,
            "set_activation": DEEPSETS_ACTIVATION,
            "pool": "mean+max",
            "position_encoding": "cos_sin_theta",
            "count_feature": "n_over_candidate_count",
            "candidate_count": cand_count,
            "candidate_positions": pos.tolist(),
            "candidate_position_fingerprint": fp,
            "candidate_position_fingerprint_method": DYNAMIC_POSITION_FINGERPRINT_METHOD,
            "channel_order": ["u", "v", "p"],
            "chan_mean": [0.0, 0.0, 0.0],
            "chan_std": [1.0, 1.0, 1.0],
            "training_sensor_counts": [2, 4, 6],
            "set_embed_dim": 16,
            "set_phi_hidden": [16],
            "set_rho_hidden": [16],
            "channel_stats_source": "train_split_full_candidate_pool",
            "channel_stats_re_ids": [200, 300],
            "default_sensor_indices": [0, 1, 2],
        }

        fp_contract = dynamic_contract_fingerprint(metadata)
        self.assertIsInstance(fp_contract, str)
        self.assertEqual(len(fp_contract), 64)

        metadata["dynamic_contract_fingerprint"] = fp_contract
        validated = validate_dynamic_checkpoint_metadata(metadata)
        self.assertEqual(validated["candidate_count"], cand_count)

        # Tampered metadata must raise ValueError
        bad_meta = dict(metadata)
        bad_meta["candidate_count"] = cand_count + 1
        with self.assertRaises(ValueError):
            validate_dynamic_checkpoint_metadata(bad_meta)


if __name__ == "__main__":
    unittest.main()
