import numpy as np
from scipy.spatial import ConvexHull
import djlib.clex.clex as cl


def collect_ground_state_indices(
    compositions: np.ndarray, predicted_energies: np.ndarray,
) -> list:
    """Stores ground state indices for each set of ECI.
    Parameters:
    -----------
    compositions: np.ndarray
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
        points = np.hstack((compositions, np.reshape(energy_set, (-1, 1))))
        hull = ConvexHull(points)
        hull_simplices, hull_vertices = cl.lower_hull(hull)
        ground_state_indices.append(hull_vertices)
    return ground_state_indices


def binning_posterior_ground_state_domains(
    compositions: np.ndarray, predicted_energies: np.ndarray
) -> dict:
    """Tallies all unique ground state sets explored by a stan posterior markov chain in ECI space. 

    Parameters:
    -----------
    compositions: np.ndarray
        nxm Composition matrix from casm query: n = number of configurations, m = number of composition axes
    predicted_energies: np.ndarray
        sxn matrix of n predicted energies for each of s sets of ECI.

    Returns:
    --------
    ground_state_set_tally: dict
        Dictionary of unique ground state sets and their counts. Key is the unique set as a string, value is the count.
    """

    # Find the ground state indices for each set of predicted energies (a set of predicted energies corresponds to a set of ECI)
    ground_states = np.array(
        collect_ground_state_indices(
            compositions, predicted_energies=predicted_energies
        ),
        dtype=object,
    )

    # Find the unique ground state sets
    ground_state_sets = []
    for element in ground_states:
        ground_state_sets.append(set(element))
    unique_ground_states = np.unique(ground_state_sets)

    # Initialize the tally at zero
    ground_state_set_tally = {}
    for element in unique_ground_states:
        ground_state_set_tally[str(element)] = 0

    # Count the number of times each unique ground state set is found
    for element in ground_state_sets:
        ground_state_set_tally[str(element)] += 1

    return ground_state_set_tally
