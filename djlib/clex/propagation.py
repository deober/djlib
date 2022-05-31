import numpy as np
from scipy.spatial import ConvexHull
import djlib.clex.clex as cl


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
        points = np.hstack((comp, np.reshape(energy_set, (-1, 1))))
        hull = ConvexHull(points)
        hull_simplices, hull_vertices = cl.lower_hull(hull)
        ground_state_indices.append(hull_vertices)
    return ground_state_indices

