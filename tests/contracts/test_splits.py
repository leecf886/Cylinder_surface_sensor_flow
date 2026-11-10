"""Contracts and validation tests for Reynolds number splits and data isolation."""
import json
from pathlib import Path
import tempfile
import unittest

from cylinder_flow.data.splits import (
    ReSplit,
    check_re_split_disjoint,
    load_re_splits,
    select_re_fold,
)


class TestReSplitContract(unittest.TestCase):
    """Contract tests for ReSplit dataclass and anti-leakage protections."""

    def setUp(self):
        self.train_re = [200, 300, 400]
        self.val_re = [450]
        self.test_re = [250, 650]

    def test_valid_split_initialization(self):
        split = ReSplit(
            train_re=self.train_re,
            val_re=self.val_re,
            test_re=self.test_re,
            fold_id="test_fold",
            purpose="Unit testing fold",
        )
        self.assertEqual(split.train_re, [200, 300, 400])
        self.assertEqual(split.val_re, [450])
        self.assertEqual(split.test_re, [250, 650])
        self.assertEqual(split.fold_id, "test_fold")

    def test_disjoint_validation_raises_on_overlap(self):
        # 1. Train and Val overlap
        with self.assertRaises(ValueError):
            ReSplit(train_re=[200, 300, 450], val_re=[450], test_re=[250, 650])

        # 2. Train and Test overlap
        with self.assertRaises(ValueError):
            ReSplit(train_re=[200, 250, 300], val_re=[450], test_re=[250, 650])

        # 3. Val and Test overlap
        with self.assertRaises(ValueError):
            ReSplit(train_re=[200, 300], val_re=[450, 650], test_re=[250, 650])

    def test_check_re_split_disjoint_helper(self):
        # Valid disjoint
        res = check_re_split_disjoint(
            train_re=[200, 300], val_re=[400], test_re=[500], strict=True
        )
        self.assertTrue(res["disjoint"])
        self.assertEqual(res["train_val_overlap"], [])

        # Non-strict detection
        res = check_re_split_disjoint(
            train_re=[200, 300], val_re=[300, 400], test_re=[500], strict=False
        )
        self.assertFalse(res["disjoint"])
        self.assertEqual(res["train_val_overlap"], [300])

        # Strict raises
        with self.assertRaises(ValueError):
            check_re_split_disjoint(
                train_re=[200, 300], val_re=[300, 400], test_re=[500], strict=True
            )

    def test_assert_no_test_leakage_guard(self):
        split = ReSplit(train_re=[200, 300], val_re=[400], test_re=[250, 650])

        # Candidate Re completely outside test set
        split.assert_no_test_leakage([200, 300, 400])

        # Candidate Re contains test Re -> must raise PermissionError
        with self.assertRaises(PermissionError) as ctx:
            split.assert_no_test_leakage([200, 250, 300])
        self.assertIn("250", str(ctx.exception))

        with self.assertRaises(PermissionError) as ctx:
            split.assert_no_test_leakage([650])
        self.assertIn("650", str(ctx.exception))

    def test_is_test_locked_guard(self):
        split = ReSplit(train_re=[200, 300], val_re=[400], test_re=[250, 650])

        # Disallowed access
        with self.assertRaises(PermissionError):
            split.is_test_locked(allow_test=False)

        # Allowed access
        split.is_test_locked(allow_test=True)


class TestReSplitConfigLoading(unittest.TestCase):
    """Tests for loading splits from configuration files."""

    def test_load_default_folds_json(self):
        splits = load_re_splits()
        self.assertIn("default", splits)
        self.assertIn("loo_450", splits)
        self.assertIn("loo_800", splits)

        default_split = splits["default"]
        self.assertIn(200, default_split.train_re)
        self.assertEqual(default_split.val_re, [450, 800])
        self.assertEqual(default_split.test_re, [250, 650])

        # Check disjointness for all configured splits
        for split in splits.values():
            self.assertTrue(split.validate_disjoint())

    def test_select_re_fold(self):
        split = select_re_fold("default")
        self.assertEqual(split.fold_id, "default")
        self.assertEqual(split.val_re, [450, 800])

        # Nonexistent fold raises KeyError
        with self.assertRaises(KeyError):
            select_re_fold("non_existent_fold_123")

    def test_select_re_fold_missing_re_validation(self):
        split = select_re_fold(
            "default", available_re=[200, 225, 275, 300, 350, 400, 500, 550, 600, 700, 750, 850, 900, 1000, 450, 800]
        )
        self.assertIsNotNone(split)

        # Strict check raises if training Re is missing
        with self.assertRaises(ValueError):
            select_re_fold("default", available_re=[200, 450, 800], strict=True)


if __name__ == "__main__":
    unittest.main()

