"""Input, coordinate, and target normalization states."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Normalizer:
    mean: float
    scale: float

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.scale

    def inverse_transform(self, x_norm: np.ndarray) -> np.ndarray:
        return x_norm * self.scale + self.mean
