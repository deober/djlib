import pytest
import numpy as np
import djlib.clex.clex as cl
import thermocore.geometry.hull as thull


@pytest.fixture
def points():
    return np.array([[1, -1], [0, 0], [2, 0], [3, 2], [4, 5]])


def test_calculate_slopes(points):
    # Test that the slopes are calculated correctly

    slopes = cl.calculate_slopes(x_coords=points[:, 0], y_coords=points[:, 1])
    assert np.allclose(slopes, [-1.0, 1.0, 2.0, 3.0])


def test_stable_chemical_potential_windows_binary(points):
    # Test that the stable chemical potential windows are calculated correctly

    hull = thull.full_hull(
        np.reshape(points[:, 0], (len(points), 1)), np.ravel(points[:, 1])
    )
    windows = cl.stable_chemical_potential_windows_binary(hull)
    assert np.allclose(windows, [2.0, 1.0, 1.0])
