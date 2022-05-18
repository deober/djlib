import djlib.djlib as dj
import numpy as np
import os


def collect_ground_state_indices(corr:np.ndarray, comp:np.ndarray, true_ground_state_indices:np.ndarray, predicted_energies:np.ndarray):
    """Stores ground state indices for each set of ECI. 
    Parameters:
    -----------
    corr: np.ndarray
        nxk Correlation matrix from casm query: n = number of configurations, k = number of ECIs
    comp: np.ndarray
        nxm Composition matrix from casm query: n = number of configurations, m = number of composition axes
    true_ground_state_indices: np.ndarray
        Vector of "true" ground state configurations.
    predicted_energies: np.ndarray
        sxn matrix of n predicted energies for each of s sets of ECI. 
    Returns:
    --------
    ground_state_indices: list
        List of lists of ground state indices for each set of ECI
    """
    ground_state_indices = []
    for index, energy_set in enumerate(predicted_energies):
        