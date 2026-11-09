"""Nondimensionalization, pressure scaling, and Re-Strouhal relations."""
import numpy as np


def compute_st_from_re(re: float, a: float = -0.09758, b: float = 0.00244) -> float:
    """Empirical St(Re) formula guard."""
    return a + b * np.sqrt(re)
