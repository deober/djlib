import pytest
import numpy as np
import djlib.clex.clex as cl


def test_calculate_slopes():
    # Test that the slopes are calculated correctly
    points = np.array([[1, -1], [0, 0], [2, 0], [3, 2], [4, 5]])

    slopes = cl.calculate_slopes(x_coords=points[:, 0], y_coords=points[:, 1])
    assert np.allclose(slopes, [-1.0, 1.0, 2.0, 3.0])

