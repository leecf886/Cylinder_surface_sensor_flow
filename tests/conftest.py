"""Shared pytest fixtures for synthetic data and models."""
import pytest
import numpy as np


@pytest.fixture
def synthetic_mesh():
    roi_xy = np.random.uniform(-1, 2, size=(50, 2))
    sensor_theta = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    return {"roi_xy": roi_xy, "sensor_theta": sensor_theta, "cylinder_diameter": 1.0}
