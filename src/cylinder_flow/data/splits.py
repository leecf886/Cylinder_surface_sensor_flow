"""Reynolds number split management and data domain protection."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)


@dataclass
class ReSplit:
    """Reynolds number split specification with integrity and anti-leakage guards.

    Attributes:
        train_re: Reynolds numbers designated for model training.
        val_re: Reynolds numbers designated for checkpoint selection and sensor layout evaluation.
        test_re: Strictly held-out Reynolds numbers designated for final evaluation only.
        fold_id: Identifier for the cross-validation or train/val fold (e.g. 'default', 'loo_450').
        purpose: Optional description of the fold purpose.
    """

    train_re: List[int]
    val_re: List[int]
    test_re: List[int]
    fold_id: str = "default"
    purpose: str = ""

    def __post_init__(self) -> None:
        self.train_re = [int(x) for x in self.train_re]
        self.val_re = [int(x) for x in self.val_re]
        self.test_re = [int(x) for x in self.test_re]
        self.validate_disjoint(strict=True)

    def validate_disjoint(self, strict: bool = True) -> bool:
        """Verify that train, validation, and test Re sets are strictly mutually disjoint.

        Args:
            strict: If True, raises ValueError upon any intersection.
                    If False, returns False.

        Returns:
            True if all three sets are disjoint, False otherwise (in non-strict mode).
        """
        train_set = set(self.train_re)
        val_set = set(self.val_re)
        test_set = set(self.test_re)

        tv_overlap = train_set & val_set
        tt_overlap = train_set & test_set
        vt_overlap = val_set & test_set
        total_overlap = tv_overlap | tt_overlap | vt_overlap

        if total_overlap:
            msg = (
                f"Non-disjoint Re split detected in fold '{self.fold_id}':\n"
                f"  Train-Val overlap: {sorted(tv_overlap)}\n"
                f"  Train-Test overlap: {sorted(tt_overlap)}\n"
                f"  Val-Test overlap: {sorted(vt_overlap)}"
            )
            if strict:
                raise ValueError(msg)
            logger.warning(msg)
            return False
        return True

    def assert_no_test_leakage(self, candidate_re_list: Sequence[int]) -> None:
        """Guard against test set leakage during training or sensor selection.

        Args:
            candidate_re_list: Sequence of Re values requested for loading or evaluation.

        Raises:
            PermissionError: If any Re in candidate_re_list belongs to test_re.
        """
        test_set = set(self.test_re)
        leaked = sorted({int(x) for x in candidate_re_list} & test_set)
        if leaked:
            raise PermissionError(
                f"Data domain violation: Attempted to access locked test Re {leaked} "
                f"during model training or sensor layout selection!"
            )

    def is_test_locked(self, allow_test: bool = False) -> None:
        """Enforce strict lock on test partition unless explicitly unlocked.

        Args:
            allow_test: Explicit flag authorizing test set access.

        Raises:
            PermissionError: If allow_test is False.
        """
        if not allow_test:
            raise PermissionError(
                f"Test partition {self.test_re} for fold '{self.fold_id}' is locked. "
                "Explicit user/workflow authorization (allow_test=True) is required."
            )


def check_re_split_disjoint(
    train_re: Sequence[int],
    val_re: Sequence[int],
    test_re: Sequence[int],
    strict: bool = True,
    fold_id: str = "default",
) -> Dict[str, Any]:
    """Check that train/val/test Re sets are mutually disjoint (legacy-compatible helper).

    Args:
        train_re: Re numbers for training.
        val_re: Re numbers for validation.
        test_re: Re numbers for testing.
        strict: If True, raises ValueError on intersection.
        fold_id: Identifier for metadata tracking.

    Returns:
        Dict detailing disjoint status and any pairwise overlaps.
    """
    s_train = {int(x) for x in train_re}
    s_val = {int(x) for x in val_re}
    s_test = {int(x) for x in test_re}

    tv = sorted(s_train & s_val)
    tt = sorted(s_train & s_test)
    vt = sorted(s_val & s_test)
    disjoint = not (tv or tt or vt)

    if not disjoint and strict:
        raise ValueError(
            f"Overlapping Re split in '{fold_id}': "
            f"train_val={tv}, train_test={tt}, val_test={vt}"
        )

    return {
        "disjoint": disjoint,
        "train_val_overlap": tv,
        "train_test_overlap": tt,
        "val_test_overlap": vt,
        "strict": strict,
        "re_fold_id": fold_id,
    }


def _resolve_default_folds_path() -> Path:
    # splits.py -> data -> cylinder_flow -> src -> repo_root
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    return repo_root / "configs" / "folds.json"


def load_re_splits(config_path: Optional[Union[str, Path]] = None) -> Dict[str, ReSplit]:
    """Load and validate all Re splits from JSON configuration.

    Args:
        config_path: Path to folds.json. Defaults to configs/folds.json in repo root.

    Returns:
        Dictionary mapping fold_id to validated ReSplit instances.
    """
    path = Path(config_path) if config_path is not None else _resolve_default_folds_path()
    if not path.is_file():
        raise FileNotFoundError(f"Re splits configuration not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    splits: Dict[str, ReSplit] = {}
    for fold_id, item in data.items():
        if not isinstance(item, dict):
            continue
        splits[fold_id] = ReSplit(
            train_re=item.get("train_re", []),
            val_re=item.get("val_re", []),
            test_re=item.get("test_re", []),
            fold_id=fold_id,
            purpose=item.get("purpose", ""),
        )
    return splits


def select_re_fold(
    fold_id: str = "default",
    config_path: Optional[Union[str, Path]] = None,
    available_re: Optional[Sequence[int]] = None,
    strict: bool = True,
) -> ReSplit:
    """Select a specific Re split fold and optionally verify against available files.

    Args:
        fold_id: Identifier of the fold to select (e.g. 'default', 'loo_450', 'loo_800').
        config_path: Optional path to custom folds.json.
        available_re: Optional sequence of Re values actually present in the data directory.
        strict: If True, raises ValueError if required training/val Re are missing.

    Returns:
        Validated ReSplit instance.
    """
    splits = load_re_splits(config_path)
    if fold_id not in splits:
        raise KeyError(
            f"Fold '{fold_id}' not found in splits configuration. "
            f"Available folds: {list(splits.keys())}"
        )

    split = splits[fold_id]

    if available_re is not None:
        avail_set = {int(x) for x in available_re}
        missing_train = sorted(set(split.train_re) - avail_set)
        missing_val = sorted(set(split.val_re) - avail_set)
        if (missing_train or missing_val) and strict:
            raise ValueError(
                f"Missing required Re cases in data directory for fold '{fold_id}': "
                f"train_missing={missing_train}, val_missing={missing_val}"
            )

    return split
