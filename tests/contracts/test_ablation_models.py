"""Contract tests for architecture ablation models, features, and accumulators.

Verifies:
- All 8 variant specifications, parameter counts, and output shapes.
- Synchronized permutation invariance across all pooling modes including slot scatter.
- Masked padding unaffectedness (padding tokens ignored).
- Slot duplicate and unmatched positions rejection for slot-scatter variants.
- Fourier-K2 feature extraction analytical correctness and NumPy/PyTorch parity.
- StreamingRidgeAccumulator closed-form Cholesky solve and MSE against independent NumPy.
- FixedFlattenedAdapter wrapping and execution.
- Checkpoint metadata contract generation, fingerprinting, and strict validation.

Self-contained portable contract test suite (no external data files, no sibling repo dependencies).
"""

from __future__ import annotations

import math
from typing import Sequence
import unittest

import numpy as np
import torch
import torch.nn as nn

from cylinder_flow.models.ablations import (
    ARCH_ABLATION_ACTIVATION_POLICY,
    ARCH_ABLATION_CHANNEL_ORDER,
    ARCH_ABLATION_ENCODER_NAME,
    ARCH_ABLATION_OUTPUT_ORDER,
    ARCH_ABLATION_SCHEMA_NAME,
    ARCH_ABLATION_SCHEMA_VERSION,
    ARCH_ABLATION_VARIANT_IDS,
    CANDIDATE_COUNT,
    EMBED_DIM,
    EXPECTED_PARAM_COUNTS,
    FOURIER_FEATURE_DIM,
    FOURIER_FEATURE_NAMES,
    FOURIER_K,
    OUT_DIM,
    RIDGE_ALPHA,
    SLOT_MATCH_ATOL,
    TOKEN_DIM,
    DynamicArchitectureAblationRegressor,
    FourierK2FeatureExtractor,
    FourierRidgeRegressor,
    StreamingRidgeAccumulator,
    VariantSpec,
    arch_ablation_contract_fingerprint,
    arch_ablation_contract_payload,
    build_arch_ablation_model,
    count_trainable_parameters,
    describe_variant,
    get_variant_spec,
    validate_arch_ablation_checkpoint_metadata,
)
from cylinder_flow.models.deepsets import (
    DynamicSensorBatch,
    FixedFlattenedAdapter,
    candidate_position_fingerprint,
)


def _make_candidate_positions(n: int = CANDIDATE_COUNT) -> np.ndarray:
    """Generate deterministic candidate sensor positions on the unit circle."""
    theta = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False, dtype=np.float64)
    return np.stack([np.cos(theta), np.sin(theta)], axis=-1)


class TestAblationModels(unittest.TestCase):
    """Test suite for architecture ablation models, feature extraction, and Ridge fitting."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        np.random.seed(42)
        self.cand_pos = _make_candidate_positions(CANDIDATE_COUNT)
        self.chan_mean = np.array([0.15, -0.05, 1.02], dtype=np.float64)
        self.chan_std = np.array([0.45, 0.35, 1.80], dtype=np.float64)

    def test_all_8_variant_specs_and_registry(self) -> None:
        """Verify that all 8 immutable variant IDs are registered with expected properties."""
        self.assertEqual(len(ARCH_ABLATION_VARIANT_IDS), 8)
        expected_ids = (
            "arch_main",
            "arch_identity_pool_mlp",
            "arch_identity_pool_linear",
            "arch_linear_phi_pool_mlp",
            "arch_main_phi_pool_linear",
            "arch_main_phi_slot_flat_linear",
            "arch_fourier_k2_ridge",
            "arch_main_phi_slot_flat_mlp",
        )
        self.assertEqual(ARCH_ABLATION_VARIANT_IDS, expected_ids)

        for vid in ARCH_ABLATION_VARIANT_IDS:
            spec = get_variant_spec(vid)
            desc = describe_variant(vid)
            self.assertEqual(spec.variant_id, vid)
            self.assertEqual(spec.expected_trainable_params, EXPECTED_PARAM_COUNTS[vid])
            self.assertEqual(desc["variant_id"], vid)
            self.assertEqual(desc["expected_trainable_params"], EXPECTED_PARAM_COUNTS[vid])

        with self.assertRaises(ValueError):
            get_variant_spec("unknown_variant_id")

    def test_all_8_model_shapes_and_parameter_counts(self) -> None:
        """Verify trainable parameter counts and forward shape [B, 3] for all 8 variants."""
        bsz = 3
        n_sensors = 8
        sel_idx = [2, 7, 15, 23, 31, 39, 47, 55]
        active_pos = self.cand_pos[sel_idx]  # [8, 2]

        pos_t = torch.as_tensor(active_pos, dtype=torch.float32).unsqueeze(0).expand(bsz, -1, -1)
        values_t = torch.randn(bsz, n_sensors, 3, dtype=torch.float32)
        mask_t = torch.ones(bsz, n_sensors, dtype=torch.bool)
        count_t = torch.full((bsz, 1), float(n_sensors), dtype=torch.float32)

        for vid in ARCH_ABLATION_VARIANT_IDS:
            with self.subTest(variant_id=vid):
                torch.manual_seed(1234)
                model = build_arch_ablation_model(
                    vid,
                    self.cand_pos,
                    self.chan_mean,
                    self.chan_std,
                )
                spec = get_variant_spec(vid)
                actual_params = count_trainable_parameters(model)
                self.assertEqual(
                    actual_params,
                    spec.expected_trainable_params,
                    f"Parameter count mismatch for {vid}: expected {spec.expected_trainable_params}, got {actual_params}",
                )

                if spec.closed_form_ridge:
                    self.assertIsInstance(model, FourierRidgeRegressor)
                    self.assertFalse(model.is_fitted)
                    with self.assertRaises(RuntimeError):
                        model.forward_tokens(values_t, pos_t, mask_t, count_t)

                    # Provide dummy fitted solution
                    model.set_fitted_solution(
                        feature_mean=np.zeros(FOURIER_FEATURE_DIM),
                        feature_scale=np.ones(FOURIER_FEATURE_DIM),
                        coef=np.ones((FOURIER_FEATURE_DIM, OUT_DIM)),
                        intercept=np.zeros(OUT_DIM),
                        fitted_exposure_count=100,
                    )
                    self.assertTrue(model.is_fitted)

                model.eval()
                with torch.no_grad():
                    out = model.forward_tokens(values_t, pos_t, mask_t, count_t)
                    self.assertEqual(tuple(out.shape), (bsz, OUT_DIM))

                    out_forward = model(values_t, pos_t, mask_t, count_t)
                    torch.testing.assert_close(out, out_forward)

                    # DynamicSensorBatch container execution
                    batch_container = DynamicSensorBatch(values=values_t, pos=pos_t, mask=mask_t, count=count_t)
                    out_batch = model.forward_tokens(batch_container)
                    torch.testing.assert_close(out, out_batch)

                    # FixedFlattenedAdapter execution
                    adapter = model.as_fixed_adapter(active_pos)
                    self.assertIsInstance(adapter, FixedFlattenedAdapter)
                    self.assertEqual(adapter.n_sensors, n_sensors)
                    flat_values = values_t.reshape(bsz, n_sensors * 3)
                    out_adapter = adapter(flat_values)
                    torch.testing.assert_close(out, out_adapter)

    def test_synchronized_permutation_invariance_all_variants(self) -> None:
        """Active tokens permuted with synchronized values and positions must give identical output."""
        bsz = 2
        n_sensors = 10
        sel_idx = np.array([1, 6, 12, 19, 25, 32, 38, 44, 51, 58])
        active_pos = self.cand_pos[sel_idx]  # [10, 2]

        pos_orig = torch.as_tensor(active_pos, dtype=torch.float32).unsqueeze(0).expand(bsz, -1, -1)
        values_orig = torch.randn(bsz, n_sensors, 3, dtype=torch.float32)
        mask_orig = torch.ones(bsz, n_sensors, dtype=torch.bool)
        count_orig = torch.full((bsz, 1), float(n_sensors), dtype=torch.float32)

        # Apply a fixed permutation to tokens
        perm = torch.as_tensor([4, 0, 7, 2, 9, 1, 8, 3, 6, 5], dtype=torch.long)
        pos_perm = pos_orig[:, perm, :]
        values_perm = values_orig[:, perm, :]
        mask_perm = mask_orig[:, perm]

        for vid in ARCH_ABLATION_VARIANT_IDS:
            with self.subTest(variant_id=vid):
                torch.manual_seed(999)
                model = build_arch_ablation_model(
                    vid,
                    self.cand_pos,
                    self.chan_mean,
                    self.chan_std,
                )
                if get_variant_spec(vid).closed_form_ridge:
                    model.set_fitted_solution(
                        feature_mean=np.zeros(FOURIER_FEATURE_DIM),
                        feature_scale=np.ones(FOURIER_FEATURE_DIM),
                        coef=np.random.randn(FOURIER_FEATURE_DIM, OUT_DIM),
                        intercept=np.array([0.5, -0.3, 0.2]),
                        fitted_exposure_count=50,
                    )

                model.eval()
                with torch.no_grad():
                    out_orig = model.forward_tokens(values_orig, pos_orig, mask_orig, count_orig)
                    out_perm = model.forward_tokens(values_perm, pos_perm, mask_perm, count_orig)
                    torch.testing.assert_close(
                        out_orig,
                        out_perm,
                        atol=1e-5,
                        rtol=1e-5,
                        msg=f"Permutation invariance violated for variant {vid}",
                    )

    def test_masked_padding_unaffected_all_variants(self) -> None:
        """Padded tokens (mask=False) with arbitrary values must have zero effect on predictions."""
        bsz = 2
        k_active = 6
        n_total = 12
        sel_idx = np.array([3, 11, 22, 33, 44, 55])
        active_pos = self.cand_pos[sel_idx]  # [6, 2]

        pos_active = torch.as_tensor(active_pos, dtype=torch.float32).unsqueeze(0).expand(bsz, -1, -1)
        values_active = torch.randn(bsz, k_active, 3, dtype=torch.float32)
        mask_active = torch.ones(bsz, k_active, dtype=torch.bool)
        count = torch.full((bsz, 1), float(k_active), dtype=torch.float32)

        # Pad with 6 extra tokens
        dummy_pos = torch.as_tensor(self.cand_pos[:6], dtype=torch.float32).unsqueeze(0).expand(bsz, -1, -1)
        pos_padded = torch.cat([pos_active, dummy_pos], dim=1)
        # Put arbitrary / garbage values in padded slots
        garbage_values = torch.full((bsz, 6, 3), 999.0, dtype=torch.float32)
        values_padded = torch.cat([values_active, garbage_values], dim=1)
        mask_padded = torch.cat([mask_active, torch.zeros(bsz, 6, dtype=torch.bool)], dim=1)

        for vid in ARCH_ABLATION_VARIANT_IDS:
            with self.subTest(variant_id=vid):
                torch.manual_seed(777)
                model = build_arch_ablation_model(
                    vid,
                    self.cand_pos,
                    self.chan_mean,
                    self.chan_std,
                )
                if get_variant_spec(vid).closed_form_ridge:
                    model.set_fitted_solution(
                        feature_mean=np.zeros(FOURIER_FEATURE_DIM),
                        feature_scale=np.ones(FOURIER_FEATURE_DIM),
                        coef=np.random.randn(FOURIER_FEATURE_DIM, OUT_DIM),
                        intercept=np.array([1.0, 2.0, 3.0]),
                        fitted_exposure_count=100,
                    )

                model.eval()
                with torch.no_grad():
                    out_unpadded = model.forward_tokens(values_active, pos_active, mask_active, count)
                    out_padded = model.forward_tokens(values_padded, pos_padded, mask_padded, count)
                    torch.testing.assert_close(
                        out_unpadded,
                        out_padded,
                        atol=1e-5,
                        rtol=1e-5,
                        msg=f"Masked padding influenced output for variant {vid}",
                    )

    def test_slot_scatter_unmatched_and_duplicate_rejection(self) -> None:
        """Slot-scatter models (A6, A8) must reject unmatched positions and duplicate active slots."""
        slot_variants = ("arch_main_phi_slot_flat_linear", "arch_main_phi_slot_flat_mlp")
        for vid in slot_variants:
            with self.subTest(variant_id=vid):
                model = build_arch_ablation_model(
                    vid,
                    self.cand_pos,
                    self.chan_mean,
                    self.chan_std,
                )
                model.eval()

                # (1) Unmatched position: unit-circle point not matching any candidate within SLOT_MATCH_ATOL
                bad_pos = np.array([math.cos(0.05), math.sin(0.05)], dtype=np.float32)  # between cand 0 and 1
                pos_t = torch.as_tensor(self.cand_pos[:4], dtype=torch.float32).unsqueeze(0).clone()
                pos_t[0, 2] = torch.as_tensor(bad_pos)
                values_t = torch.randn(1, 4, 3, dtype=torch.float32)
                mask_t = torch.ones(1, 4, dtype=torch.bool)
                count_t = torch.full((1, 1), 4.0, dtype=torch.float32)

                with self.assertRaises(ValueError) as ctx:
                    model.forward_tokens(values_t, pos_t, mask_t, count_t)
                self.assertIn("slot position match exceeded atol", str(ctx.exception))

                # When the unmatched token is masked, it must NOT fail
                mask_t[0, 2] = False
                count_t[0, 0] = 3.0
                with torch.no_grad():
                    out = model.forward_tokens(values_t, pos_t, mask_t, count_t)
                    self.assertEqual(tuple(out.shape), (1, OUT_DIM))

                # (2) Duplicate active slot in the same row
                dup_pos = torch.as_tensor(self.cand_pos[[5, 10, 5, 20]], dtype=torch.float32).unsqueeze(0)
                mask_dup = torch.ones(1, 4, dtype=torch.bool)
                count_dup = torch.full((1, 1), 4.0, dtype=torch.float32)
                with self.assertRaises(ValueError) as ctx:
                    model.forward_tokens(values_t, dup_pos, mask_dup, count_dup)
                self.assertIn("duplicate valid slots", str(ctx.exception))

                # When one duplicate is masked, it must NOT fail
                mask_dup[0, 2] = False
                count_dup[0, 0] = 3.0
                with torch.no_grad():
                    out_dup = model.forward_tokens(values_t, dup_pos, mask_dup, count_dup)
                    self.assertEqual(tuple(out_dup.shape), (1, OUT_DIM))

    def test_fourier_feature_extractor_analytical_and_parity(self) -> None:
        """Verify FourierK2FeatureExtractor analytical moments and NumPy/PyTorch exactness."""
        # 4 orthogonal positions: theta = 0, pi/2, pi, 3pi/2
        theta = np.array([0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0], dtype=np.float64)
        pos = np.stack([np.cos(theta), np.sin(theta)], axis=-1)  # [4, 2]
        # Channels: u, v, p
        values = np.array(
            [
                [1.0, 10.0, -1.0],
                [2.0, 20.0, -2.0],
                [3.0, 30.0, -3.0],
                [4.0, 40.0, -4.0],
            ],
            dtype=np.float64,
        )
        chan_mean = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        chan_std = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        mask = np.ones((1, 4), dtype=bool)
        count = np.array([[4.0]], dtype=np.float64)

        feats_np = FourierK2FeatureExtractor.extract_numpy(
            values[None, ...],
            pos[None, ...],
            mask,
            count,
            chan_mean,
            chan_std,
            candidate_count=60,
        )
        self.assertEqual(feats_np.shape, (1, 16))

        # Check channel 0 (u = [1, 2, 3, 4]):
        # c1 = [1, 0, -1, 0], s1 = [0, 1, 0, -1]
        # c2 = [1, -1, 1, -1], s2 = [0, 0, 0, 0]
        # mean(u) = (1+2+3+4)/4 = 2.5
        # mean(u * c1) = (1*1 + 2*0 + 3*(-1) + 4*0)/4 = -0.5
        # mean(u * s1) = (1*0 + 2*1 + 3*0 + 4*(-1))/4 = -0.5
        # mean(u * c2) = (1*1 + 2*(-1) + 3*1 + 4*(-1))/4 = -0.5
        # mean(u * s2) = 0.0
        expected_u = [2.5, -0.5, -0.5, -0.5, 0.0]
        np.testing.assert_allclose(feats_np[0, :5], expected_u, rtol=1e-12, atol=1e-12)

        # Check count feature: 4 / 60
        self.assertAlmostEqual(feats_np[0, 15], 4.0 / 60.0, places=12)

        # Check PyTorch implementation parity
        v_t = torch.as_tensor(values[None, ...], dtype=torch.float32)
        p_t = torch.as_tensor(pos[None, ...], dtype=torch.float32)
        m_t = torch.as_tensor(mask, dtype=torch.bool)
        c_t = torch.as_tensor(count, dtype=torch.float32)
        cm_t = torch.as_tensor(chan_mean, dtype=torch.float32)
        cs_t = torch.as_tensor(chan_std, dtype=torch.float32)

        feats_torch = FourierK2FeatureExtractor.extract_torch(
            v_t, p_t, m_t, c_t, cm_t, cs_t, candidate_count=60
        )
        np.testing.assert_allclose(feats_np, feats_torch.cpu().numpy(), rtol=1e-6, atol=1e-6)

        # Verify feature names count
        self.assertEqual(len(FOURIER_FEATURE_NAMES), 16)
        self.assertEqual(FourierK2FeatureExtractor.feature_dim, 16)

    def test_streaming_ridge_accumulator_vs_independent_numpy(self) -> None:
        """StreamingRidgeAccumulator must match independent NumPy Ridge regression to float64 machine epsilon."""
        np.random.seed(2026)
        n_samples = 400
        p_features = 16
        q_outputs = 3
        alpha = 1e-3

        X = np.random.randn(n_samples, p_features).astype(np.float64)
        true_w = np.random.randn(p_features, q_outputs).astype(np.float64)
        true_b = np.array([12.5, -4.2, 0.8], dtype=np.float64)
        noise = np.random.randn(n_samples, q_outputs).astype(np.float64) * 0.05
        Y = X @ true_w + true_b + noise

        # Independent reference solve in pure NumPy:
        mu_x = np.mean(X, axis=0)
        std_x = np.std(X, axis=0, ddof=0)  # ddof=0
        Z = (X - mu_x) / std_x

        mu_y = np.mean(Y, axis=0)
        Y_c = Y - mu_y

        lhs = (Z.T @ Z) / float(n_samples) + alpha * np.eye(p_features, dtype=np.float64)
        rhs = (Z.T @ Y_c) / float(n_samples)
        ref_coef = np.linalg.solve(lhs, rhs)
        ref_intercept = mu_y

        ref_pred = Z @ ref_coef + ref_intercept
        ref_mse = np.mean((Y - ref_pred) ** 2, axis=0)

        # Accumulator with chunked streaming updates (chunks of 100, 150, 150)
        acc = StreamingRidgeAccumulator(feature_dim=p_features, out_dim=q_outputs)
        acc.update(X[:100], Y[:100])
        acc.update(X[100:250], Y[100:250])
        acc.update(X[250:], Y[250:])
        self.assertEqual(acc.n, n_samples)

        fm, fs, coef, intercept = acc.fit(alpha=alpha)

        np.testing.assert_allclose(fm, mu_x, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(fs, std_x, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(coef, ref_coef, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(intercept, ref_intercept, rtol=1e-12, atol=1e-12)

        acc_mse = acc.mse_by_output(fm, fs, coef, intercept)
        np.testing.assert_allclose(acc_mse, ref_mse, rtol=1e-11, atol=1e-11)

    def test_contract_payload_fingerprint_and_validator(self) -> None:
        """Verify contract payload generation, SHA-256 fingerprinting, and strict validation."""
        counts = [6, 12, 24, 48]
        re_ids = [300, 400, 500, 600]

        for vid in ARCH_ABLATION_VARIANT_IDS:
            with self.subTest(variant_id=vid):
                payload = arch_ablation_contract_payload(
                    variant_id=vid,
                    candidate_positions=self.cand_pos,
                    chan_mean=self.chan_mean,
                    chan_std=self.chan_std,
                    training_sensor_counts=counts,
                    channel_stats_re_ids=re_ids,
                )
                self.assertEqual(payload["encoder"], ARCH_ABLATION_ENCODER_NAME)
                self.assertEqual(payload["schema_name"], ARCH_ABLATION_SCHEMA_NAME)
                self.assertEqual(payload["schema_version"], ARCH_ABLATION_SCHEMA_VERSION)
                self.assertEqual(payload["arch_variant"], vid)

                fp = arch_ablation_contract_fingerprint(payload)
                self.assertIsInstance(fp, str)
                self.assertEqual(len(fp), 64)

                payload_with_fp = dict(payload)
                payload_with_fp["arch_ablation_contract_fingerprint"] = fp
                validated = validate_arch_ablation_checkpoint_metadata(payload_with_fp)
                self.assertEqual(validated["arch_ablation_contract_fingerprint"], fp)

        # Tampering tests
        base_payload = arch_ablation_contract_payload(
            variant_id="arch_main",
            candidate_positions=self.cand_pos,
            chan_mean=self.chan_mean,
            chan_std=self.chan_std,
            training_sensor_counts=counts,
            channel_stats_re_ids=re_ids,
        )
        base_fp = arch_ablation_contract_fingerprint(base_payload)
        base_payload["arch_ablation_contract_fingerprint"] = base_fp

        # Corrupted schema_version
        bad_version = dict(base_payload)
        bad_version["schema_version"] = 2
        with self.assertRaises(ValueError):
            validate_arch_ablation_checkpoint_metadata(bad_version)

        # Corrupted fingerprint
        bad_fp = dict(base_payload)
        bad_fp["arch_ablation_contract_fingerprint"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_arch_ablation_checkpoint_metadata(bad_fp)

        # Missing fingerprint
        no_fp = dict(base_payload)
        del no_fp["arch_ablation_contract_fingerprint"]
        with self.assertRaises(ValueError):
            validate_arch_ablation_checkpoint_metadata(no_fp)

        # Corrupted candidate position
        bad_pos_payload = dict(base_payload)
        bad_pos = np.copy(self.cand_pos)
        bad_pos[0, 0] += 0.1
        bad_pos_payload["candidate_positions"] = bad_pos.tolist()
        with self.assertRaises(ValueError):
            validate_arch_ablation_checkpoint_metadata(bad_pos_payload)

    def test_slot_forward_chunking_parity(self) -> None:
        """Batch size exceeding SLOT_FORWARD_BATCH_CHUNK must give identical result to unchunked forward."""
        model = build_arch_ablation_model(
            "arch_main_phi_slot_flat_linear",
            self.cand_pos,
            self.chan_mean,
            self.chan_std,
        )
        model.eval()

        # Test chunking parity on batch of size 8 by manually chunking vs unchunked
        bsz = 8
        n_sensors = 4
        sel_idx = [0, 15, 30, 45]
        active_pos = self.cand_pos[sel_idx]
        pos_t = torch.as_tensor(active_pos, dtype=torch.float32).unsqueeze(0).expand(bsz, -1, -1)
        values_t = torch.randn(bsz, n_sensors, 3, dtype=torch.float32)
        mask_t = torch.ones(bsz, n_sensors, dtype=torch.bool)
        count_t = torch.full((bsz, 1), float(n_sensors), dtype=torch.float32)

        with torch.no_grad():
            out_full = model.forward_tokens(values_t, pos_t, mask_t, count_t)
            out_c1 = model.forward_tokens(values_t[:4], pos_t[:4], mask_t[:4], count_t[:4])
            out_c2 = model.forward_tokens(values_t[4:], pos_t[4:], mask_t[4:], count_t[4:])
            out_chunked = torch.cat([out_c1, out_c2], dim=0)
            torch.testing.assert_close(out_full, out_chunked)


if __name__ == "__main__":
    unittest.main()
