from lib2to3.pytree import convert
import numpy as np
from scipy.spatial import ConvexHull
import thermofitting as tfit


def collect_ground_state_indices(
    comp: np.ndarray, predicted_energies: np.ndarray,
) -> list:
    """Stores ground state indices for each set of ECI. 
    Parameters:
    -----------
    comp: np.ndarray
        nxm Composition matrix from casm query: n = number of configurations, m = number of composition axes
    predicted_energies: np.ndarray
        sxn matrix of n predicted energies for each of s sets of ECI. 
    Returns:
    --------
    ground_state_indices: list
        List of lists of ground state indices for each set of ECI
    """
    ground_state_indices = []
    for index, energy_set in enumerate(predicted_energies):
        points = np.hstack(comp, energy_set.reshape(-1, 1))
        hull = ConvexHull(points)
        hull_vertices, hull_simplices = tfit.hull.lower_hull(hull)
        ground_state_indices.append(hull_vertices)
    return ground_state_indices

