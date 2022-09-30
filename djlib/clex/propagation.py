import numpy as np
from scipy.spatial import ConvexHull
import djlib.clex.clex as cl
import matplotlib.pyplot as plt
import seaborn as sns
import thermocore.geometry.hull as thull


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
        hull = thull.full_hull(compositions=compositions, energies=energy_set)

        hull_vertices, hull_simplices = thull.lower_hull(hull)
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


def partition_posterior_by_ground_states(
    corr: np.ndarray, compositions: np.ndarray, posterior_distribution: np.ndarray
) -> dict:
    """Takes a posterior markov chain of points in ECI space and returns those points grouped by their ground state sets.

    Parameters
    ----------
    corr: np.ndarray
        nxk matrix of n configurations, k = number of correlations (number of ECI in a single ECI vector)
    compositions : np.ndarray
        nxm Composition matrix from casm query: n = number of configurations, m = number of composition axes
    posterior_distribution : np.ndarray
        sxm matrix of m ECI vectors, s = number of samples in the posterior markov chain


    Returns
    -------
    ground_state_set_partition: dict
        Dictionary of unique ground state sets and their counts. Key is the unique set as a string, value is .
    """

    # Initialize the dictionary
    ground_state_set_partition = {}

    # For each ECI vector in posterior distribution, find the ground state set.
    # If the ground state set is not already a key in the dictionary, add it and initialize the list of ECI vectors to an empty list.
    # Append the ECI vector to the list of ECI vectors for that ground state set.
    # If the ground state set is already a key in the dictionary, append the ECI vector to the list of ECI vectors for that ground state set.
    for eci_vector in posterior_distribution:

        # Find the ground state set for the ECI vector
        predicted_energies = corr @ eci_vector
        predicted_hull = thull.full_hull(
            compositions=compositions, energies=predicted_energies
        )
        hull_vertices, _ = thull.lower_hull(predicted_hull)
        ground_state_set = str(set(hull_vertices))

        # If the ground state set is not already a key in the dictionary, add it and initialize the list of ECI vectors to an empty list.
        if ground_state_set not in ground_state_set_partition.keys():
            ground_state_set_partition[ground_state_set] = []
        ground_state_set_partition[ground_state_set].append(eci_vector)

    return ground_state_set_partition


def plot_eci_covariance_matrix(eci_matrix: np.ndarray) -> np.ndarray:
    """Plots the covariance matrix of the ECI space, given an array of multiple ECI vectors.

    Parameters:
    -----------
    eci_matrix: np.ndarray
        kxm matrix of m ECI vectors, k = number of correlations (number of ECI in a single ECI vector)
    
    Returns:
    --------    
    covariance_matrix_plot: 
    """

    covariance_matrix = np.cov(eci_matrix)

    plt.imshow(covariance_matrix, cmap="bwr")
    plt.title("ECI Covariance Matrix", fontsize=30)
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    fig = plt.gcf()
    fig.set_size_inches(15, 15)
    return fig
