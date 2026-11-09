"""Reynolds number split management and data domain protection."""
from dataclasses import dataclass
from typing import List


@dataclass
class ReSplit:
    train_re: List[int]
    val_re: List[int]
    test_re: List[int]
    fold_id: str = "default"

    def validate_disjoint(self, strict: bool = True) -> bool:
        train_set = set(self.train_re)
        val_set = set(self.val_re)
        test_set = set(self.test_re)
        overlap = (train_set & val_set) | (train_set & test_set) | (val_set & test_set)
        if overlap:
            if strict:
                raise ValueError(f"Overlapping Re in split: {overlap}")
            return False
        return True
