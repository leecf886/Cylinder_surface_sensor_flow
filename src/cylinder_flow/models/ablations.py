"""Architecture ablation encoders for shared dynamic Re/phase regression (ReLU).

Defines eight immutable architecture variants under an independent checkpoint
schema (dynamic_arch_ablation_schema_v1). Does not alter WP-B deepsets_dynamic.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn

from .deepsets import (
    DYNAMIC_POSITION_FINGERPRINT_METHOD,
    DynamicSensorBatch,
    DynamicTokenRegressor,
    FixedFlattenedAdapter,
    candidate_position_fingerprint,
    make_mlp,
    validate_dynamic_token_batch,
)

ARCH_ABLATION_ENCODER_NAME = "dynamic_arch_ablation"
ARCH_ABLATION_SCHEMA_NAME = "dynamic_arch_ablation_schema_v1"
ARCH_ABLATION_SCHEMA_VERSION = 1
ARCH_ABLATION_ACTIVATION_POLICY = "relu_for_all_existing_hidden_nonlinearities"
ARCH_ABLATION_OUTPUT_ORDER = ["re_norm", "phase_sin", "phase_cos"]
ARCH_ABLATION_CHANNEL_ORDER = ["u", "v", "p"]

CANDIDATE_COUNT = 60
TOKEN_DIM = 5
EMBED_DIM = 128
PHI_HIDDEN = (128, 128)
RHO_HIDDEN = (128, 128)
OUT_DIM = 3
SLOT_MATCH_ATOL = 1e-5
SLOT_FORWARD_BATCH_CHUNK = 1024
RIDGE_ALPHA = 1e-3
FOURIER_K = 2
FOURIER_FEATURE_DIM = 16  # 3 channels * 5 coeffs + count

EXPECTED_PARAM_COUNTS = {
    "arch_main": 83715,
    "arch_identity_pool_mlp": 18435,
    "arch_identity_pool_linear": 36,
    "arch_linear_phi_pool_mlp": 50691,
    "arch_main_phi_pool_linear": 34566,
    "arch_main_phi_slot_flat_linear": 57018,
    "arch_fourier_k2_ridge": 0,
    "arch_main_phi_slot_flat_mlp": 1033987,
}

FOURIER_FEATURE_NAMES = [
    "u_a0",
    "u_c1",
    "u_s1",
    "u_c2",
    "u_s2",
    "v_a0",
    "v_c1",
    "v_s1",
    "v_c2",
    "v_s2",
    "p_a0",
    "p_c1",
    "p_s1",
    "p_c2",
    "p_s2",
    "count_over_60",
]


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    token_transform: str  # main_mlp | identity | linear_affine
    aggregation: str  # mean_max_count | slot_flat_presence_count | fourier_k2
    head: str  # mlp | linear | ridge_closed_form
    h_dim: int
    z_dim: int
    expected_trainable_params: int
    expected_effective_params: int
    uses_relu_in_phi: bool
    uses_relu_in_head: bool
    optimizer_used: bool
    closed_form_ridge: bool

    def structure_dict(self) -> Dict[str, object]:
        return asdict(self)


VARIANT_REGISTRY: Dict[str, VariantSpec] = {
    "arch_main": VariantSpec(
        variant_id="arch_main",
        token_transform="main_mlp",
        aggregation="mean_max_count",
        head="mlp",
        h_dim=EMBED_DIM,
        z_dim=EMBED_DIM * 2 + 1,
        expected_trainable_params=EXPECTED_PARAM_COUNTS["arch_main"],
        expected_effective_params=EXPECTED_PARAM_COUNTS["arch_main"],
        uses_relu_in_phi=True,
        uses_relu_in_head=True,
        optimizer_used=True,
        closed_form_ridge=False,
    ),
    "arch_identity_pool_mlp": VariantSpec(
        variant_id="arch_identity_pool_mlp",
        token_transform="identity",
        aggregation="mean_max_count",
        head="mlp",
        h_dim=TOKEN_DIM,
        z_dim=TOKEN_DIM * 2 + 1,
        expected_trainable_params=EXPECTED_PARAM_COUNTS["arch_identity_pool_mlp"],
        expected_effective_params=EXPECTED_PARAM_COUNTS["arch_identity_pool_mlp"],
        uses_relu_in_phi=False,
        uses_relu_in_head=True,
        optimizer_used=True,
        closed_form_ridge=False,
    ),
    "arch_identity_pool_linear": VariantSpec(
        variant_id="arch_identity_pool_linear",
        token_transform="identity",
        aggregation="mean_max_count",
        head="linear",
        h_dim=TOKEN_DIM,
        z_dim=TOKEN_DIM * 2 + 1,
        expected_trainable_params=EXPECTED_PARAM_COUNTS["arch_identity_pool_linear"],
        expected_effective_params=EXPECTED_PARAM_COUNTS["arch_identity_pool_linear"],
        uses_relu_in_phi=False,
        uses_relu_in_head=False,
        optimizer_used=True,
        closed_form_ridge=False,
    ),
    "arch_linear_phi_pool_mlp": VariantSpec(
        variant_id="arch_linear_phi_pool_mlp",
        token_transform="linear_affine",
        aggregation="mean_max_count",
        head="mlp",
        h_dim=EMBED_DIM,
        z_dim=EMBED_DIM * 2 + 1,
        expected_trainable_params=EXPECTED_PARAM_COUNTS["arch_linear_phi_pool_mlp"],
        expected_effective_params=EXPECTED_PARAM_COUNTS["arch_linear_phi_pool_mlp"],
        uses_relu_in_phi=False,
        uses_relu_in_head=True,
        optimizer_used=True,
        closed_form_ridge=False,
    ),
    "arch_main_phi_pool_linear": VariantSpec(
        variant_id="arch_main_phi_pool_linear",
        token_transform="main_mlp",
        aggregation="mean_max_count",
        head="linear",
        h_dim=EMBED_DIM,
        z_dim=EMBED_DIM * 2 + 1,
        expected_trainable_params=EXPECTED_PARAM_COUNTS["arch_main_phi_pool_linear"],
        expected_effective_params=EXPECTED_PARAM_COUNTS["arch_main_phi_pool_linear"],
        uses_relu_in_phi=True,
        uses_relu_in_head=False,
        optimizer_used=True,
        closed_form_ridge=False,
    ),
    "arch_main_phi_slot_flat_linear": VariantSpec(
        variant_id="arch_main_phi_slot_flat_linear",
        token_transform="main_mlp",
        aggregation="slot_flat_presence_count",
        head="linear",
        h_dim=EMBED_DIM,
        z_dim=CANDIDATE_COUNT * EMBED_DIM + CANDIDATE_COUNT + 1,
        expected_trainable_params=EXPECTED_PARAM_COUNTS["arch_main_phi_slot_flat_linear"],
        expected_effective_params=EXPECTED_PARAM_COUNTS["arch_main_phi_slot_flat_linear"],
        uses_relu_in_phi=True,
        uses_relu_in_head=False,
        optimizer_used=True,
        closed_form_ridge=False,
    ),
    "arch_fourier_k2_ridge": VariantSpec(
        variant_id="arch_fourier_k2_ridge",
        token_transform="identity",
        aggregation="fourier_k2",
        head="ridge_closed_form",
        h_dim=TOKEN_DIM,
        z_dim=FOURIER_FEATURE_DIM,
        expected_trainable_params=0,
        expected_effective_params=FOURIER_FEATURE_DIM * OUT_DIM + OUT_DIM,
        uses_relu_in_phi=False,
        uses_relu_in_head=False,
        optimizer_used=False,
        closed_form_ridge=True,
    ),
    "arch_main_phi_slot_flat_mlp": VariantSpec(
        variant_id="arch_main_phi_slot_flat_mlp",
        token_transform="main_mlp",
        aggregation="slot_flat_count",
        head="mlp",
        h_dim=EMBED_DIM,
        z_dim=CANDIDATE_COUNT * EMBED_DIM + 1,
        expected_trainable_params=EXPECTED_PARAM_COUNTS["arch_main_phi_slot_flat_mlp"],
        expected_effective_params=EXPECTED_PARAM_COUNTS["arch_main_phi_slot_flat_mlp"],
        uses_relu_in_phi=True,
        uses_relu_in_head=True,
        optimizer_used=True,
        closed_form_ridge=False,
    ),
}

ARCH_ABLATION_VARIANT_IDS = tuple(VARIANT_REGISTRY.keys())


def get_variant_spec(variant_id: str) -> VariantSpec:
    key = str(variant_id).strip()
    if key not in VARIANT_REGISTRY:
        raise ValueError(
            f"Unknown arch ablation variant {variant_id!r}; "
            f"expected one of {list(ARCH_ABLATION_VARIANT_IDS)}"
        )
    return VARIANT_REGISTRY[key]


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def build_token_transform(spec: VariantSpec) -> nn.Module:
    if spec.token_transform == "main_mlp":
        return make_mlp(TOKEN_DIM, PHI_HIDDEN, EMBED_DIM, act=nn.ReLU)
    if spec.token_transform == "identity":
        return nn.Identity()
    if spec.token_transform == "linear_affine":
        return nn.Linear(TOKEN_DIM, EMBED_DIM, bias=True)
    raise ValueError(f"Unsupported token_transform: {spec.token_transform}")


def build_head(spec: VariantSpec) -> nn.Module:
    if spec.head == "mlp":
        return make_mlp(spec.z_dim, RHO_HIDDEN, OUT_DIM, act=nn.ReLU)
    if spec.head == "linear":
        return nn.Linear(spec.z_dim, OUT_DIM, bias=True)
    if spec.head == "ridge_closed_form":
        return nn.Identity()
    raise ValueError(f"Unsupported head: {spec.head}")


def _masked_mean_max(
    embedding: torch.Tensor,
    mask: torch.Tensor,
    valid_count: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    valid = mask.unsqueeze(-1)
    mean_pool = (embedding * valid.to(dtype=embedding.dtype)).sum(dim=1) / valid_count.to(
        dtype=embedding.dtype
    )
    max_pool = embedding.masked_fill(~valid, -torch.inf).max(dim=1).values
    return mean_pool, max_pool


def normalize_tokens(
    values: torch.Tensor,
    pos: torch.Tensor,
    mask: torch.Tensor,
    chan_mean: torch.Tensor,
    chan_std: torch.Tensor,
) -> torch.Tensor:
    valid = mask.unsqueeze(-1)
    safe_values = torch.where(valid, values, torch.zeros_like(values))
    normalized = (safe_values - chan_mean) / chan_std
    safe_pos = torch.where(valid, pos, torch.zeros_like(pos))
    return torch.cat([normalized, safe_pos], dim=-1)


class DynamicArchitectureAblationRegressor(DynamicTokenRegressor):
    """Neural architecture ablation variants A1-A6 and A8."""

    def __init__(
        self,
        variant_id: str,
        candidate_positions: Sequence[Sequence[float]] | np.ndarray,
        chan_mean: Sequence[float] | np.ndarray,
        chan_std: Sequence[float] | np.ndarray,
    ) -> None:
        super().__init__()
        self.spec = get_variant_spec(variant_id)
        if self.spec.closed_form_ridge:
            raise ValueError(
                f"{variant_id} is closed-form Ridge; use FourierRidgeRegressor instead"
            )

        positions = np.asarray(candidate_positions, dtype=np.float64)
        if positions.shape != (CANDIDATE_COUNT, 2):
            raise ValueError(
                f"candidate_positions must have shape [{CANDIDATE_COUNT},2], got {positions.shape}"
            )
        self.candidate_count = CANDIDATE_COUNT
        self.variant_id = self.spec.variant_id
        self.register_buffer(
            "candidate_positions",
            torch.as_tensor(positions, dtype=torch.float32),
        )
        self.candidate_position_fingerprint = candidate_position_fingerprint(positions)

        cm = torch.as_tensor(chan_mean, dtype=torch.float32).reshape(3)
        cs = torch.as_tensor(chan_std, dtype=torch.float32).reshape(3)
        if cm.numel() != 3 or cs.numel() != 3:
            raise ValueError("chan_mean/chan_std must each contain 3 values")
        if not torch.isfinite(cm).all() or not torch.isfinite(cs).all() or torch.any(cs <= 0):
            raise ValueError("chan_mean/chan_std must be finite and chan_std positive")
        self.register_buffer("chan_mean", cm.reshape(1, 1, 3))
        self.register_buffer("chan_std", cs.reshape(1, 1, 3))

        self.phi = build_token_transform(self.spec)
        self.head = build_head(self.spec)
        actual = count_trainable_parameters(self)
        if actual != self.spec.expected_trainable_params:
            raise ValueError(
                f"{self.variant_id} parameter count mismatch: expected "
                f"{self.spec.expected_trainable_params}, got {actual}"
            )

    def reset_parameters(self, seed: Optional[int] = None) -> None:
        """Reset all trainable parameters with optional deterministic seed."""
        if seed is not None:
            torch.manual_seed(int(seed))
        for module in (self.phi, self.head):
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
            elif isinstance(module, nn.Sequential):
                for layer in module:
                    if hasattr(layer, "reset_parameters"):
                        layer.reset_parameters()

    def as_fixed_adapter(
        self,
        positions: Sequence[Sequence[float]] | np.ndarray | torch.Tensor,
    ) -> FixedFlattenedAdapter:
        """Return an nn.Module adapter taking [B, 3*K] flattened sensor observations."""
        return FixedFlattenedAdapter(self, positions)

    def as_fixed_flattened_adapter(
        self,
        positions: Sequence[Sequence[float]] | np.ndarray | torch.Tensor,
    ) -> FixedFlattenedAdapter:
        """Alias for as_fixed_adapter for dynamic token regressor compatibility."""
        return self.as_fixed_adapter(positions)

    def _scatter_to_slots(
        self,
        embedding: torch.Tensor,
        pos: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, n_tok, h_dim = embedding.shape
        if h_dim != EMBED_DIM:
            raise ValueError(f"slot scatter expects embed_dim={EMBED_DIM}, got {h_dim}")
        cand = self.candidate_positions  # [60,2]
        # Vectorized across the whole batch. Host synchronization occurs only
        # on a failing batch, not once per sample row during normal GPU training.
        scores = pos @ cand.T  # [B,N,60]
        slot_ids = torch.argmax(scores, dim=-1)  # [B,N]
        matched = cand[slot_ids]
        abs_err = (matched - pos).abs().amax(dim=-1)
        bad_match = mask & (abs_err > SLOT_MATCH_ATOL)
        if torch.any(bad_match):
            bad = torch.nonzero(bad_match, as_tuple=False)[0]
            b, token = int(bad[0].item()), int(bad[1].item())
            raise ValueError(
                f"slot position match exceeded atol={SLOT_MATCH_ATOL} "
                f"on batch row={b}, tokens={[token]}"
            )

        presence_count = embedding.new_zeros((bsz, self.candidate_count))
        presence_count.scatter_add_(1, slot_ids, mask.to(dtype=embedding.dtype))
        duplicate = presence_count > 1.0
        if torch.any(duplicate):
            bad = torch.nonzero(duplicate, as_tuple=False)[0]
            b = int(bad[0].item())
            valid_slots = slot_ids[b, mask[b]].detach().cpu().tolist()
            raise ValueError(f"duplicate valid slots in batch row={b}: {valid_slots}")

        h_slot = embedding.new_zeros((bsz, self.candidate_count, h_dim))
        scatter_index = slot_ids.unsqueeze(-1).expand(-1, -1, h_dim)
        masked_embedding = embedding * mask.unsqueeze(-1).to(dtype=embedding.dtype)
        h_slot.scatter_add_(1, scatter_index, masked_embedding)
        presence = (presence_count > 0).to(dtype=embedding.dtype)
        return h_slot, presence

    def encode_pooled(
        self,
        values: Union[torch.Tensor, DynamicSensorBatch],
        pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(values, DynamicSensorBatch):
            values, pos, mask, count = values.to_tensors(
                device=self.chan_mean.device, dtype=self.chan_mean.dtype
            )
        if pos is None or mask is None:
            raise ValueError("pos and mask are required when values is a torch.Tensor")
        valid_count, effective_count = validate_dynamic_token_batch(
            values,
            pos,
            mask,
            count,
            candidate_count=self.candidate_count,
            reference=self.chan_mean,
        )
        tokens = normalize_tokens(values, pos, mask, self.chan_mean, self.chan_std)
        embedding = self.phi(tokens)
        count_feat = effective_count / float(self.candidate_count)

        if self.spec.aggregation == "mean_max_count":
            mean_pool, max_pool = _masked_mean_max(embedding, mask, valid_count)
            return torch.cat([mean_pool, max_pool, count_feat.to(dtype=mean_pool.dtype)], dim=-1)

        if self.spec.aggregation == "slot_flat_presence_count":
            h_slot, presence = self._scatter_to_slots(embedding, pos, mask)
            flat = h_slot.reshape(int(h_slot.shape[0]), -1)
            return torch.cat([flat, presence, count_feat.to(dtype=flat.dtype)], dim=-1)

        if self.spec.aggregation == "slot_flat_count":
            h_slot, _presence = self._scatter_to_slots(embedding, pos, mask)
            flat = h_slot.reshape(int(h_slot.shape[0]), -1)
            return torch.cat([flat, count_feat.to(dtype=flat.dtype)], dim=-1)

        raise ValueError(f"Unsupported aggregation for neural model: {self.spec.aggregation}")

    def forward_tokens(
        self,
        values: Union[torch.Tensor, DynamicSensorBatch],
        pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(values, DynamicSensorBatch):
            v, p, m, c = values.to_tensors(
                device=self.chan_mean.device, dtype=self.chan_mean.dtype
            )
            return self.forward_tokens(v, p, m, c)
        if pos is None or mask is None:
            raise ValueError("pos and mask are required when values is a torch.Tensor")
        # A6/A8 use high-dimensional fixed-slot representations. Evaluation
        # passes an entire Re case (roughly 1k-2k snapshots, and the combined
        # train panel can be much larger) in one call. Building the full
        # [B,60,128] scatter tensor at once causes multi-GB temporary tensors
        # and severe allocator/BLAS slowdown. Chunk only the batch dimension;
        # this is mathematically identical and preserves gradients.
        if (
            self.spec.aggregation in {"slot_flat_presence_count", "slot_flat_count"}
            and int(values.shape[0]) > SLOT_FORWARD_BATCH_CHUNK
        ):
            outputs: List[torch.Tensor] = []
            for start in range(0, int(values.shape[0]), SLOT_FORWARD_BATCH_CHUNK):
                end = min(start + SLOT_FORWARD_BATCH_CHUNK, int(values.shape[0]))
                outputs.append(
                    self.forward_tokens(
                        values[start:end],
                        pos[start:end],
                        mask[start:end],
                        None if count is None else count[start:end],
                    )
                )
            return torch.cat(outputs, dim=0)
        z = self.encode_pooled(values, pos, mask, count)
        if int(z.shape[-1]) != self.spec.z_dim:
            raise RuntimeError(
                f"{self.variant_id} z_dim mismatch: expected {self.spec.z_dim}, got {int(z.shape[-1])}"
            )
        return self.head(z)

    def forward(
        self,
        values: Union[torch.Tensor, DynamicSensorBatch],
        pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward_tokens(values, pos, mask, count)


class FourierK2FeatureExtractor:
    """Stateless Fourier-K2 feature builder shared by A7 train/infer."""

    feature_dim = FOURIER_FEATURE_DIM
    feature_names = list(FOURIER_FEATURE_NAMES)

    @staticmethod
    def extract_numpy(
        values: np.ndarray,
        pos: np.ndarray,
        mask: np.ndarray,
        count: np.ndarray,
        chan_mean: np.ndarray,
        chan_std: np.ndarray,
        candidate_count: int = CANDIDATE_COUNT,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        pos = np.asarray(pos, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool)
        count = np.asarray(count, dtype=np.float64).reshape(-1, 1)
        chan_mean = np.asarray(chan_mean, dtype=np.float64).reshape(3)
        chan_std = np.asarray(chan_std, dtype=np.float64).reshape(3)
        if values.ndim != 3 or values.shape[-1] != 3:
            raise ValueError(f"values must be [B,N,3], got {values.shape}")
        bsz, n_tok, _ = values.shape
        if pos.shape != (bsz, n_tok, 2) or mask.shape != (bsz, n_tok):
            raise ValueError("pos/mask must align with values")
        if count.shape != (bsz, 1):
            raise ValueError(f"count must be [B,1], got {count.shape}")

        feats = np.zeros((bsz, FOURIER_FEATURE_DIM), dtype=np.float64)
        for b in range(bsz):
            valid = mask[b]
            n_valid = int(valid.sum())
            if n_valid <= 0:
                raise ValueError(f"all-mask row {b} is invalid for Fourier features")
            if abs(float(count[b, 0]) - float(n_valid)) > 1e-5:
                raise ValueError(f"count mismatch on row {b}")
            q = (values[b, valid] - chan_mean) / chan_std
            c1 = pos[b, valid, 0]
            s1 = pos[b, valid, 1]
            c2 = c1 * c1 - s1 * s1
            s2 = 2.0 * c1 * s1
            cols: List[float] = []
            for ch in range(3):
                qc = q[:, ch]
                cols.extend(
                    [
                        float(qc.mean()),
                        float((qc * c1).mean()),
                        float((qc * s1).mean()),
                        float((qc * c2).mean()),
                        float((qc * s2).mean()),
                    ]
                )
            cols.append(float(n_valid) / float(candidate_count))
            feats[b] = np.asarray(cols, dtype=np.float64)
        return feats

    @staticmethod
    def extract_torch(
        values: torch.Tensor,
        pos: torch.Tensor,
        mask: torch.Tensor,
        count: torch.Tensor,
        chan_mean: torch.Tensor,
        chan_std: torch.Tensor,
        candidate_count: int = CANDIDATE_COUNT,
    ) -> torch.Tensor:
        valid_count, effective_count = validate_dynamic_token_batch(
            values,
            pos,
            mask,
            count,
            candidate_count=candidate_count,
            reference=chan_mean,
        )
        del valid_count  # validated above
        bsz = int(values.shape[0])
        valid = mask.unsqueeze(-1)
        q = (values - chan_mean.reshape(1, 1, 3)) / chan_std.reshape(1, 1, 3)
        q = torch.where(valid, q, torch.zeros_like(q))
        c1 = pos[..., 0]
        s1 = pos[..., 1]
        basis = torch.stack(
            [
                torch.ones_like(c1),
                c1,
                s1,
                c1 * c1 - s1 * s1,
                2.0 * c1 * s1,
            ],
            dim=-1,
        )
        basis = torch.where(mask.unsqueeze(-1), basis, torch.zeros_like(basis))
        # [B,N,3,1] * [B,N,1,5] -> channel-major [B,3,5].
        moments = torch.sum(q.unsqueeze(-1) * basis.unsqueeze(-2), dim=1)
        moments = moments / effective_count.reshape(bsz, 1, 1)
        count_feature = effective_count / float(candidate_count)
        return torch.cat([moments.reshape(bsz, 15), count_feature], dim=-1)


class StreamingRidgeAccumulator:
    """Float64 streaming sufficient statistics for multi-output Ridge."""

    def __init__(self, feature_dim: int = FOURIER_FEATURE_DIM, out_dim: int = OUT_DIM) -> None:
        self.feature_dim = int(feature_dim)
        self.out_dim = int(out_dim)
        self.n = 0
        # Keep all BLAS-backed sufficient-statistic operations in PyTorch. On
        # the supported Windows environment, entering NumPy's BLAS after
        # PyTorch has loaded its OpenMP runtime can abort with OMP Error #15.
        self.sum_x = torch.zeros(self.feature_dim, dtype=torch.float64)
        self.sum_y = torch.zeros(self.out_dim, dtype=torch.float64)
        self.sum_y2 = torch.zeros(self.out_dim, dtype=torch.float64)
        self.xtx = torch.zeros((self.feature_dim, self.feature_dim), dtype=torch.float64)
        self.xty = torch.zeros((self.feature_dim, self.out_dim), dtype=torch.float64)

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        x_t = torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=torch.float64)
        y_t = torch.as_tensor(np.asarray(y, dtype=np.float64), dtype=torch.float64)
        if x_t.ndim != 2 or x_t.shape[1] != self.feature_dim:
            raise ValueError(f"x must be [n,{self.feature_dim}], got {tuple(x_t.shape)}")
        if y_t.ndim != 2 or y_t.shape != (x_t.shape[0], self.out_dim):
            raise ValueError(
                f"y must be [n,{self.out_dim}] matching x rows, got {tuple(y_t.shape)}"
            )
        if x_t.shape[0] == 0:
            return
        self.n += int(x_t.shape[0])
        self.sum_x += x_t.sum(dim=0)
        self.sum_y += y_t.sum(dim=0)
        self.sum_y2 += torch.square(y_t).sum(dim=0)
        # Feature dimension is only 16. Explicit outer products avoid
        # loading a second vendor BLAS/OpenMP runtime in the Windows process.
        self.xtx += torch.sum(x_t[:, :, None] * x_t[:, None, :], dim=0)
        self.xty += torch.sum(x_t[:, :, None] * y_t[:, None, :], dim=0)

    def fit(self, alpha: float = RIDGE_ALPHA) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.n <= 0:
            raise ValueError("Cannot fit Ridge with zero exposure count")
        alpha = float(alpha)
        if alpha <= 0:
            raise ValueError("ridge alpha must be positive")
        n = float(self.n)
        mean_x = self.sum_x / n
        # Population variance (ddof=0) from raw second moments:
        # E[x^2] = diag(X^T X)/n ; var = E[x^2] - mean^2
        ex2 = torch.diagonal(self.xtx) / n
        var = torch.clamp(ex2 - mean_x * mean_x, min=0.0)
        scale = torch.clamp(torch.sqrt(var), min=1e-8)
        mean_y = self.sum_y / n

        # Centered Gram: (X-mu)^T (X-mu) = XTX - n * mu mu^T
        xtx_c = self.xtx - n * torch.outer(mean_x, mean_x)
        # (X-mu)^T (Y - my) = XTY - n * mu my^T
        xty_c = self.xty - n * torch.outer(mean_x, mean_y)

        # Z = (X - mu) / scale => Z^T Z = S^{-1} xtx_c S^{-1}
        inv_scale = 1.0 / scale
        ztz = (xtx_c * inv_scale.reshape(-1, 1)) * inv_scale.reshape(1, -1)
        zty = xty_c * inv_scale.reshape(-1, 1)

        a = ztz / n + alpha * torch.eye(self.feature_dim, dtype=torch.float64)
        b = zty / n
        # ``a`` is symmetric positive definite because alpha > 0. A small
        # explicit Cholesky solve is both deterministic and avoids importing
        # NumPy/PyTorch LAPACK backends whose duplicate Intel OpenMP runtimes
        # abort the supported Windows environment. d=16, so the loop cost is
        # negligible relative to feature extraction.
        chol = torch.zeros_like(a)
        try:
            for i in range(self.feature_dim):
                for j in range(i + 1):
                    residual = a[i, j] - torch.sum(chol[i, :j] * chol[j, :j])
                    if i == j:
                        if not torch.isfinite(residual) or float(residual) <= 0.0:
                            raise ArithmeticError(
                                f"non-positive Cholesky pivot at index {i}: {float(residual)}"
                            )
                        chol[i, j] = torch.sqrt(residual)
                    else:
                        chol[i, j] = residual / chol[j, j]

            forward = torch.zeros_like(b)
            for i in range(self.feature_dim):
                correction = torch.sum(chol[i, :i, None] * forward[:i], dim=0)
                forward[i] = (b[i] - correction) / chol[i, i]

            coef_z = torch.zeros_like(b)
            for i in range(self.feature_dim - 1, -1, -1):
                correction = torch.sum(chol[i + 1 :, i, None] * coef_z[i + 1 :], dim=0)
                coef_z[i] = (forward[i] - correction) / chol[i, i]
        except (ArithmeticError, RuntimeError) as exc:
            raise RuntimeError(f"Ridge closed-form Cholesky solve failed with alpha={alpha}") from exc
        # Keep the fitted coefficients in standardized-feature coordinates.
        # FourierRidgeRegressor.forward_tokens applies the same Z transform,
        # so the unpenalized intercept is simply the centered target mean.
        intercept = mean_y
        return (
            mean_x.numpy().copy(),
            scale.numpy().copy(),
            coef_z.numpy().copy(),
            intercept.numpy().copy(),
        )

    def mse_by_output(
        self,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        coef_z: np.ndarray,
        intercept_z: np.ndarray,
    ) -> np.ndarray:
        """Return cumulative per-output MSE without replaying the random loader."""
        if self.n <= 0:
            raise ValueError("Cannot evaluate Ridge loss with zero exposure count")
        mean_x = torch.as_tensor(
            np.asarray(feature_mean, dtype=np.float64), dtype=torch.float64
        ).reshape(self.feature_dim)
        scale = torch.as_tensor(
            np.asarray(feature_scale, dtype=np.float64), dtype=torch.float64
        ).reshape(self.feature_dim)
        coef_z_t = torch.as_tensor(
            np.asarray(coef_z, dtype=np.float64), dtype=torch.float64
        ).reshape(self.feature_dim, self.out_dim)
        intercept_z_t = torch.as_tensor(
            np.asarray(intercept_z, dtype=np.float64), dtype=torch.float64
        ).reshape(self.out_dim)
        if torch.any(scale <= 0) or not torch.all(torch.isfinite(scale)):
            raise ValueError("feature_scale must be positive and finite")

        # Convert only for the closed-form SSE calculation. Runtime inference
        # continues to use standardized features and coef_z.
        coef_x = coef_z_t / scale.reshape(-1, 1)
        intercept_x = intercept_z_t - torch.sum(mean_x[:, None] * coef_x, dim=0)
        linear_cross = torch.sum(coef_x * self.xty, dim=0)
        quadratic = torch.sum(
            self.xtx[:, :, None] * coef_x[:, None, :] * coef_x[None, :, :],
            dim=(0, 1),
        )
        mean_cross = torch.sum(coef_x * self.sum_x[:, None], dim=0)
        sse = (
            self.sum_y2
            - 2.0 * linear_cross
            - 2.0 * intercept_x * self.sum_y
            + quadratic
            + 2.0 * intercept_x * mean_cross
            + float(self.n) * torch.square(intercept_x)
        )
        return (torch.clamp(sse, min=0.0) / float(self.n)).numpy().copy()


class FourierRidgeRegressor(DynamicTokenRegressor):
    """Closed-form Fourier-K2 + Ridge baseline (A7)."""

    def __init__(
        self,
        candidate_positions: Sequence[Sequence[float]] | np.ndarray,
        chan_mean: Sequence[float] | np.ndarray,
        chan_std: Sequence[float] | np.ndarray,
        ridge_alpha: float = RIDGE_ALPHA,
    ) -> None:
        super().__init__()
        self.spec = get_variant_spec("arch_fourier_k2_ridge")
        positions = np.asarray(candidate_positions, dtype=np.float64)
        if positions.shape != (CANDIDATE_COUNT, 2):
            raise ValueError(
                f"candidate_positions must have shape [{CANDIDATE_COUNT},2], got {positions.shape}"
            )
        self.candidate_count = CANDIDATE_COUNT
        self.variant_id = self.spec.variant_id
        self.ridge_alpha = float(ridge_alpha)
        if self.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be positive")
        self.register_buffer(
            "candidate_positions",
            torch.as_tensor(positions, dtype=torch.float32),
        )
        self.candidate_position_fingerprint = candidate_position_fingerprint(positions)

        cm = torch.as_tensor(chan_mean, dtype=torch.float32).reshape(3)
        cs = torch.as_tensor(chan_std, dtype=torch.float32).reshape(3)
        if not torch.isfinite(cm).all() or not torch.isfinite(cs).all() or torch.any(cs <= 0):
            raise ValueError("chan_mean/chan_std must be finite and chan_std positive")
        self.register_buffer("chan_mean", cm.reshape(1, 1, 3))
        self.register_buffer("chan_std", cs.reshape(1, 1, 3))

        self.register_buffer("feature_mean", torch.zeros(FOURIER_FEATURE_DIM, dtype=torch.float64))
        self.register_buffer("feature_scale", torch.ones(FOURIER_FEATURE_DIM, dtype=torch.float64))
        self.register_buffer("coef", torch.zeros(FOURIER_FEATURE_DIM, OUT_DIM, dtype=torch.float64))
        self.register_buffer("intercept", torch.zeros(OUT_DIM, dtype=torch.float64))
        self.register_buffer("_fitted_flag", torch.zeros((), dtype=torch.bool))
        self.fitted_exposure_count = 0
        self.best_epoch: Optional[int] = None

    @property
    def is_fitted(self) -> bool:
        return bool(self._fitted_flag.item())

    def reset_parameters(self, seed: Optional[int] = None) -> None:
        """Reset fitted flag and statistics for closed-form baseline."""
        self._fitted_flag.fill_(False)
        self.fitted_exposure_count = 0
        self.best_epoch = None
        self.feature_mean.zero_()
        self.feature_scale.fill_(1.0)
        self.coef.zero_()
        self.intercept.zero_()

    def as_fixed_adapter(
        self,
        positions: Sequence[Sequence[float]] | np.ndarray | torch.Tensor,
    ) -> FixedFlattenedAdapter:
        """Return an nn.Module adapter taking [B, 3*K] flattened sensor observations."""
        return FixedFlattenedAdapter(self, positions)

    def as_fixed_flattened_adapter(
        self,
        positions: Sequence[Sequence[float]] | np.ndarray | torch.Tensor,
    ) -> FixedFlattenedAdapter:
        """Alias for as_fixed_adapter for dynamic token regressor compatibility."""
        return self.as_fixed_adapter(positions)

    def set_fitted_solution(
        self,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        coef: np.ndarray,
        intercept: np.ndarray,
        *,
        fitted_exposure_count: int,
        best_epoch: Optional[int] = None,
    ) -> None:
        fm = np.asarray(feature_mean, dtype=np.float64).reshape(FOURIER_FEATURE_DIM)
        fs = np.asarray(feature_scale, dtype=np.float64).reshape(FOURIER_FEATURE_DIM)
        cf = np.asarray(coef, dtype=np.float64).reshape(FOURIER_FEATURE_DIM, OUT_DIM)
        ic = np.asarray(intercept, dtype=np.float64).reshape(OUT_DIM)
        if np.any(fs <= 0) or not np.all(np.isfinite(fs)):
            raise ValueError("feature_scale must be positive and finite")
        self.feature_mean.copy_(torch.as_tensor(fm, dtype=torch.float64))
        self.feature_scale.copy_(torch.as_tensor(fs, dtype=torch.float64))
        self.coef.copy_(torch.as_tensor(cf, dtype=torch.float64))
        self.intercept.copy_(torch.as_tensor(ic, dtype=torch.float64))
        self._fitted_flag.fill_(True)
        self.fitted_exposure_count = int(fitted_exposure_count)
        self.best_epoch = None if best_epoch is None else int(best_epoch)

    def extract_features(
        self,
        values: Union[torch.Tensor, DynamicSensorBatch],
        pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(values, DynamicSensorBatch):
            values, pos, mask, count = values.to_tensors(
                device=self.chan_mean.device, dtype=self.chan_mean.dtype
            )
        if pos is None or mask is None:
            raise ValueError("pos and mask are required when values is a torch.Tensor")
        if count is None:
            count = mask.sum(dim=1, keepdim=True).to(dtype=values.dtype)
        return FourierK2FeatureExtractor.extract_torch(
            values,
            pos,
            mask,
            count,
            self.chan_mean,
            self.chan_std,
            candidate_count=self.candidate_count,
        )

    def forward_tokens(
        self,
        values: Union[torch.Tensor, DynamicSensorBatch],
        pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(values, DynamicSensorBatch):
            v, p, m, c = values.to_tensors(
                device=self.chan_mean.device, dtype=self.chan_mean.dtype
            )
            return self.forward_tokens(v, p, m, c)
        if not self.is_fitted:
            raise RuntimeError("FourierRidgeRegressor called before closed-form fit")
        feats = self.extract_features(values, pos, mask, count).to(dtype=torch.float64)
        z = (feats - self.feature_mean) / self.feature_scale
        pred = torch.sum(z[:, :, None] * self.coef[None, :, :], dim=1) + self.intercept
        return pred.to(dtype=torch.float32)

    def forward(
        self,
        values: Union[torch.Tensor, DynamicSensorBatch],
        pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward_tokens(values, pos, mask, count)


def build_arch_ablation_model(
    variant_id: str,
    candidate_positions: Sequence[Sequence[float]] | np.ndarray,
    chan_mean: Sequence[float] | np.ndarray,
    chan_std: Sequence[float] | np.ndarray,
) -> DynamicTokenRegressor:
    spec = get_variant_spec(variant_id)
    if spec.closed_form_ridge:
        return FourierRidgeRegressor(
            candidate_positions=candidate_positions,
            chan_mean=chan_mean,
            chan_std=chan_std,
            ridge_alpha=RIDGE_ALPHA,
        )
    return DynamicArchitectureAblationRegressor(
        variant_id=variant_id,
        candidate_positions=candidate_positions,
        chan_mean=chan_mean,
        chan_std=chan_std,
    )


def arch_ablation_contract_payload(
    *,
    variant_id: str,
    candidate_positions: Sequence[Sequence[float]] | np.ndarray,
    chan_mean: Sequence[float] | np.ndarray,
    chan_std: Sequence[float] | np.ndarray,
    training_sensor_counts: Sequence[int],
    channel_stats_re_ids: Sequence[int],
    default_sensor_indices: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    spec = get_variant_spec(variant_id)
    positions = np.asarray(candidate_positions, dtype=np.float64)
    if positions.shape != (CANDIDATE_COUNT, 2):
        raise ValueError(
            f"candidate_positions must have shape [{CANDIDATE_COUNT},2], got {positions.shape}"
        )
    counts = sorted({int(x) for x in training_sensor_counts})
    if not counts or counts != [int(x) for x in training_sensor_counts]:
        raise ValueError("training_sensor_counts must be a non-empty sorted unique list")
    re_ids = sorted({int(x) for x in channel_stats_re_ids})
    if not re_ids:
        raise ValueError("channel_stats_re_ids must be non-empty")
    cm = np.asarray(chan_mean, dtype=np.float64).reshape(3)
    cs = np.asarray(chan_std, dtype=np.float64).reshape(3)
    if np.any(cs <= 0) or not np.all(np.isfinite(cm)) or not np.all(np.isfinite(cs)):
        raise ValueError("chan_mean/chan_std must be finite with positive std")
    defaults = (
        [int(x) for x in default_sensor_indices]
        if default_sensor_indices is not None
        else list(range(CANDIDATE_COUNT))
    )
    payload: Dict[str, object] = {
        "encoder": ARCH_ABLATION_ENCODER_NAME,
        "schema_name": ARCH_ABLATION_SCHEMA_NAME,
        "schema_version": ARCH_ABLATION_SCHEMA_VERSION,
        "arch_variant": spec.variant_id,
        "activation_policy": ARCH_ABLATION_ACTIVATION_POLICY,
        "variant_spec": spec.structure_dict(),
        "token_dim": TOKEN_DIM,
        "embed_dim": EMBED_DIM,
        "phi_hidden": list(PHI_HIDDEN),
        "rho_hidden": list(RHO_HIDDEN),
        "out_dim": OUT_DIM,
        "output_order": list(ARCH_ABLATION_OUTPUT_ORDER),
        "candidate_count": CANDIDATE_COUNT,
        "candidate_positions": positions.astype(np.float64).tolist(),
        "candidate_position_fingerprint": candidate_position_fingerprint(positions),
        "candidate_position_fingerprint_method": DYNAMIC_POSITION_FINGERPRINT_METHOD,
        "channel_order": list(ARCH_ABLATION_CHANNEL_ORDER),
        "sensor_channels": list(ARCH_ABLATION_CHANNEL_ORDER),
        "chan_mean": cm.tolist(),
        "chan_std": cs.tolist(),
        "channel_mean": cm.tolist(),
        "channel_std": cs.tolist(),
        "channel_stats_source": "train_split_full_candidate_pool",
        "channel_stats_re_ids": re_ids,
        "training_sensor_counts": counts,
        "default_sensor_indices": defaults,
        "position_encoding": "cos_sin_theta",
        "count_feature": "n_over_candidate_count",
        "pool": "mean+max" if spec.aggregation == "mean_max_count" else "none",
        "slot_match_atol": SLOT_MATCH_ATOL,
        "slot_presence_appended": spec.aggregation == "slot_flat_presence_count",
        "fourier_order": FOURIER_K if spec.closed_form_ridge else None,
        "fourier_k": FOURIER_K if spec.closed_form_ridge else None,
        "fourier_feature_names": list(FOURIER_FEATURE_NAMES) if spec.closed_form_ridge else None,
        "fourier_feature_dim": FOURIER_FEATURE_DIM if spec.closed_form_ridge else None,
        "ridge_alpha": RIDGE_ALPHA if spec.closed_form_ridge else None,
        "ridge_standardize": "train_exposure_mean_std_ddof0" if spec.closed_form_ridge else None,
        "ridge_intercept_unpenalized": True if spec.closed_form_ridge else None,
        "closed_form_ridge": bool(spec.closed_form_ridge),
        "optimizer_used": bool(spec.optimizer_used),
        "supports_dynamic_token_input": True,
    }
    return payload


def _normalise_arch_ablation_contract(metadata: Mapping[str, object]) -> Dict[str, object]:
    required = {
        "encoder",
        "schema_name",
        "schema_version",
        "arch_variant",
        "activation_policy",
        "variant_spec",
        "token_dim",
        "embed_dim",
        "phi_hidden",
        "rho_hidden",
        "out_dim",
        "candidate_count",
        "candidate_positions",
        "candidate_position_fingerprint",
        "candidate_position_fingerprint_method",
        "channel_order",
        "sensor_channels",
        "chan_mean",
        "chan_std",
        "channel_mean",
        "channel_std",
        "training_sensor_counts",
        "channel_stats_source",
        "channel_stats_re_ids",
        "output_order",
        "position_encoding",
        "count_feature",
        "pool",
        "slot_match_atol",
        "slot_presence_appended",
        "fourier_order",
        "fourier_k",
        "fourier_feature_names",
        "fourier_feature_dim",
        "ridge_alpha",
        "ridge_standardize",
        "ridge_intercept_unpenalized",
        "closed_form_ridge",
        "optimizer_used",
        "supports_dynamic_token_input",
    }
    missing = sorted(required - set(metadata.keys()))
    if missing:
        raise ValueError(f"Architecture ablation checkpoint missing contract fields: {missing}")

    encoder = str(metadata["encoder"]).lower()
    if encoder != ARCH_ABLATION_ENCODER_NAME:
        raise ValueError(f"encoder must be {ARCH_ABLATION_ENCODER_NAME!r}, got {encoder!r}")
    schema_name = str(metadata["schema_name"])
    if schema_name != ARCH_ABLATION_SCHEMA_NAME:
        raise ValueError(f"schema_name must be {ARCH_ABLATION_SCHEMA_NAME!r}, got {schema_name!r}")
    schema_version = int(metadata["schema_version"])
    if schema_version != ARCH_ABLATION_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {ARCH_ABLATION_SCHEMA_VERSION}, got {schema_version}"
        )
    variant_id = str(metadata["arch_variant"])
    spec = get_variant_spec(variant_id)
    activation_policy = str(metadata["activation_policy"])
    if activation_policy != ARCH_ABLATION_ACTIVATION_POLICY:
        raise ValueError(
            f"activation_policy must be {ARCH_ABLATION_ACTIVATION_POLICY!r}, got {activation_policy!r}"
        )

    expected_structure = spec.structure_dict()
    recorded_structure = metadata["variant_spec"]
    if not isinstance(recorded_structure, Mapping) or dict(recorded_structure) != expected_structure:
        raise ValueError(
            f"variant_spec mismatch for {variant_id}: expected {expected_structure}, "
            f"got {recorded_structure}"
        )
    scalar_contract = {
        "token_dim": TOKEN_DIM,
        "embed_dim": EMBED_DIM,
        "out_dim": OUT_DIM,
        "position_encoding": "cos_sin_theta",
        "count_feature": "n_over_candidate_count",
        "pool": "mean+max" if spec.aggregation == "mean_max_count" else "none",
        "closed_form_ridge": bool(spec.closed_form_ridge),
        "optimizer_used": bool(spec.optimizer_used),
        "supports_dynamic_token_input": True,
    }
    for key, expected in scalar_contract.items():
        actual = metadata[key]
        if actual != expected:
            raise ValueError(f"{key} must be {expected!r}, got {actual!r}")
    if [int(x) for x in metadata["phi_hidden"]] != list(PHI_HIDDEN):  # type: ignore[union-attr]
        raise ValueError(f"phi_hidden must be {list(PHI_HIDDEN)}")
    if [int(x) for x in metadata["rho_hidden"]] != list(RHO_HIDDEN):  # type: ignore[union-attr]
        raise ValueError(f"rho_hidden must be {list(RHO_HIDDEN)}")

    candidate_count = int(metadata["candidate_count"])
    if candidate_count != CANDIDATE_COUNT:
        raise ValueError(f"candidate_count must be {CANDIDATE_COUNT}, got {candidate_count}")
    positions = np.asarray(metadata["candidate_positions"], dtype=np.float64)
    if positions.shape != (CANDIDATE_COUNT, 2):
        raise ValueError(f"candidate_positions shape mismatch: {positions.shape}")
    computed_fp = candidate_position_fingerprint(positions)
    recorded_fp = str(metadata["candidate_position_fingerprint"])
    method = str(metadata["candidate_position_fingerprint_method"])
    if method != DYNAMIC_POSITION_FINGERPRINT_METHOD:
        raise ValueError(
            f"fingerprint method mismatch: expected {DYNAMIC_POSITION_FINGERPRINT_METHOD!r}, got {method!r}"
        )
    if recorded_fp != computed_fp:
        raise ValueError(
            f"candidate_position_fingerprint mismatch: recorded={recorded_fp}, computed={computed_fp}"
        )

    channel_order = [str(x) for x in metadata["channel_order"]]  # type: ignore[union-attr]
    if channel_order != ARCH_ABLATION_CHANNEL_ORDER:
        raise ValueError(f"channel_order must be {ARCH_ABLATION_CHANNEL_ORDER}, got {channel_order}")
    sensor_channels = [str(x) for x in metadata["sensor_channels"]]  # type: ignore[union-attr]
    if sensor_channels != ARCH_ABLATION_CHANNEL_ORDER:
        raise ValueError(
            f"sensor_channels must be {ARCH_ABLATION_CHANNEL_ORDER}, got {sensor_channels}"
        )
    output_order = [str(x) for x in metadata["output_order"]]  # type: ignore[union-attr]
    if output_order != ARCH_ABLATION_OUTPUT_ORDER:
        raise ValueError(f"output_order must be {ARCH_ABLATION_OUTPUT_ORDER}, got {output_order}")

    cm = np.asarray(metadata["chan_mean"], dtype=np.float64).reshape(3)
    cs = np.asarray(metadata["chan_std"], dtype=np.float64).reshape(3)
    if np.any(cs <= 0) or not np.all(np.isfinite(cm)) or not np.all(np.isfinite(cs)):
        raise ValueError("chan_mean/chan_std invalid")
    if str(metadata["channel_stats_source"]) != "train_split_full_candidate_pool":
        raise ValueError("channel_stats_source must be train_split_full_candidate_pool")
    channel_mean = np.asarray(metadata["channel_mean"], dtype=np.float64).reshape(3)
    channel_std = np.asarray(metadata["channel_std"], dtype=np.float64).reshape(3)
    if not np.array_equal(channel_mean, cm) or not np.array_equal(channel_std, cs):
        raise ValueError("channel_mean/channel_std must exactly match chan_mean/chan_std")
    training_sensor_counts = sorted({int(x) for x in metadata["training_sensor_counts"]})  # type: ignore[union-attr]
    if training_sensor_counts != [int(x) for x in metadata["training_sensor_counts"]]:  # type: ignore[union-attr]
        raise ValueError("training_sensor_counts must be sorted unique")
    channel_stats_re_ids = sorted({int(x) for x in metadata["channel_stats_re_ids"]})  # type: ignore[union-attr]
    if not channel_stats_re_ids:
        raise ValueError("channel_stats_re_ids must be non-empty")

    defaults_obj = metadata.get("default_sensor_indices", [])
    defaults = [int(x) for x in defaults_obj]  # type: ignore[union-attr]

    atol = float(metadata["slot_match_atol"])
    if atol != SLOT_MATCH_ATOL:
        raise ValueError(f"slot_match_atol must be {SLOT_MATCH_ATOL}, got {atol}")
    expected_presence = spec.aggregation == "slot_flat_presence_count"
    if bool(metadata["slot_presence_appended"]) != expected_presence:
        raise ValueError(f"slot_presence_appended must be {expected_presence}")

    expected_fourier = {
        "fourier_order": FOURIER_K if spec.closed_form_ridge else None,
        "fourier_k": FOURIER_K if spec.closed_form_ridge else None,
        "fourier_feature_names": list(FOURIER_FEATURE_NAMES) if spec.closed_form_ridge else None,
        "fourier_feature_dim": FOURIER_FEATURE_DIM if spec.closed_form_ridge else None,
        "ridge_alpha": RIDGE_ALPHA if spec.closed_form_ridge else None,
        "ridge_standardize": "train_exposure_mean_std_ddof0" if spec.closed_form_ridge else None,
        "ridge_intercept_unpenalized": True if spec.closed_form_ridge else None,
    }
    for key, expected in expected_fourier.items():
        actual = metadata[key]
        if actual != expected:
            raise ValueError(f"{key} must be {expected!r}, got {actual!r}")

    return {
        "encoder": encoder,
        "schema_name": schema_name,
        "schema_version": schema_version,
        "arch_variant": spec.variant_id,
        "activation_policy": activation_policy,
        "variant_spec": dict(recorded_structure),
        "token_dim": TOKEN_DIM,
        "embed_dim": EMBED_DIM,
        "phi_hidden": list(PHI_HIDDEN),
        "rho_hidden": list(RHO_HIDDEN),
        "out_dim": OUT_DIM,
        "output_order": list(ARCH_ABLATION_OUTPUT_ORDER),
        "candidate_count": CANDIDATE_COUNT,
        "candidate_position_fingerprint": recorded_fp,
        "candidate_position_fingerprint_method": method,
        "channel_order": channel_order,
        "sensor_channels": sensor_channels,
        "chan_mean": cm.tolist(),
        "chan_std": cs.tolist(),
        "channel_mean": channel_mean.tolist(),
        "channel_std": channel_std.tolist(),
        "training_sensor_counts": training_sensor_counts,
        "channel_stats_source": "train_split_full_candidate_pool",
        "channel_stats_re_ids": channel_stats_re_ids,
        "default_sensor_indices": defaults,
        "position_encoding": "cos_sin_theta",
        "count_feature": "n_over_candidate_count",
        "pool": "mean+max" if spec.aggregation == "mean_max_count" else "none",
        "slot_match_atol": SLOT_MATCH_ATOL,
        "slot_presence_appended": bool(spec.aggregation == "slot_flat_presence_count"),
        "fourier_order": FOURIER_K if spec.closed_form_ridge else None,
        "fourier_k": FOURIER_K if spec.closed_form_ridge else None,
        "fourier_feature_names": list(FOURIER_FEATURE_NAMES) if spec.closed_form_ridge else None,
        "fourier_feature_dim": FOURIER_FEATURE_DIM if spec.closed_form_ridge else None,
        "ridge_alpha": RIDGE_ALPHA if spec.closed_form_ridge else None,
        "ridge_standardize": "train_exposure_mean_std_ddof0" if spec.closed_form_ridge else None,
        "ridge_intercept_unpenalized": True if spec.closed_form_ridge else None,
        "closed_form_ridge": bool(spec.closed_form_ridge),
        "optimizer_used": bool(spec.optimizer_used),
        "supports_dynamic_token_input": True,
    }


def arch_ablation_contract_fingerprint(metadata: Mapping[str, object]) -> str:
    payload = _normalise_arch_ablation_contract(metadata)
    payload.pop("default_sensor_indices", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_arch_ablation_checkpoint_metadata(metadata: Mapping[str, object]) -> Dict[str, object]:
    payload = _normalise_arch_ablation_contract(metadata)
    recorded = str(metadata.get("arch_ablation_contract_fingerprint", ""))
    if not recorded:
        raise ValueError("Architecture ablation checkpoint missing arch_ablation_contract_fingerprint")
    computed = arch_ablation_contract_fingerprint(metadata)
    if recorded != computed:
        raise ValueError(
            "Architecture ablation contract fingerprint mismatch "
            f"(recorded={recorded}, computed={computed})"
        )
    payload["arch_ablation_contract_fingerprint"] = recorded
    return payload


def describe_variant(variant_id: str) -> Dict[str, object]:
    spec = get_variant_spec(variant_id)
    return {
        "variant_id": spec.variant_id,
        "structure": spec.structure_dict(),
        "expected_trainable_params": spec.expected_trainable_params,
        "expected_effective_params": spec.expected_effective_params,
        "z_dim": spec.z_dim,
        "h_dim": spec.h_dim,
    }


__all__ = [
    "ARCH_ABLATION_ACTIVATION_POLICY",
    "ARCH_ABLATION_CHANNEL_ORDER",
    "ARCH_ABLATION_ENCODER_NAME",
    "ARCH_ABLATION_OUTPUT_ORDER",
    "ARCH_ABLATION_SCHEMA_NAME",
    "ARCH_ABLATION_SCHEMA_VERSION",
    "ARCH_ABLATION_VARIANT_IDS",
    "CANDIDATE_COUNT",
    "EMBED_DIM",
    "EXPECTED_PARAM_COUNTS",
    "FOURIER_FEATURE_DIM",
    "FOURIER_FEATURE_NAMES",
    "FOURIER_K",
    "OUT_DIM",
    "PHI_HIDDEN",
    "RHO_HIDDEN",
    "RIDGE_ALPHA",
    "SLOT_FORWARD_BATCH_CHUNK",
    "SLOT_MATCH_ATOL",
    "TOKEN_DIM",
    "VARIANT_REGISTRY",
    "DynamicArchitectureAblationRegressor",
    "FourierK2FeatureExtractor",
    "FourierRidgeRegressor",
    "StreamingRidgeAccumulator",
    "VariantSpec",
    "arch_ablation_contract_fingerprint",
    "arch_ablation_contract_payload",
    "build_arch_ablation_model",
    "build_head",
    "build_token_transform",
    "count_trainable_parameters",
    "describe_variant",
    "get_variant_spec",
    "normalize_tokens",
    "validate_arch_ablation_checkpoint_metadata",
]
