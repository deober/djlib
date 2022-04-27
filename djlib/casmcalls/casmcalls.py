import os

"""Clunky collection of casm cli calls that are frequently used

"""


def genetic_fit_call(fit_directory):
    """genetic_fit_call(fit_directory)
    Run a casm genetic algorithm fit. Assumes that the fit settings file already exists.

    Args:
        fit_directory (str): absolute path to the current genetic fit directory.

    Returns:
        none.
    """
    os.chdir(fit_directory)
    print("Removing old data for individual 0")
    os.system(
        "rm check.0; rm checkhull_genetic_alg_settings_0_*; rm genetic_alg_settings_*"
    )
    print("Running new fit")
    os.system("casm-learn -s genetic_alg_settings.json > fit.out")
    print("Writing data for individual 0")
    os.system("casm-learn -s genetic_alg_settings.json --checkhull --indiv 0 > check.0")


def set_active_eci(fit_directory, hall_of_fame_index):
    """set_active_eci(fit_directory, hall_of_fame_index)

    Sets the casm project active ECI to those that are defined in the fit_directory.

    Args:
        fit_directory (str): absolute path to the current genetic fit directory.
        hall_of_fame_index (str): Integer index as a string

    Returns:
        none.
    """
    hall_of_fame_index = str(hall_of_fame_index)
    os.chdir(fit_directory)
    os.system(
        "casm-learn -s genetic_alg_settings.json --select %s > select_fit_eci.out"
        % hall_of_fame_index
    )


def full_formation_file_call(fit_directory):
    """full_formation_file_call

    Casm query to generate composition of species "A", formation energy, DFT hull distance, cluster expansion energies and cluster expansion hull distance.

    Args:
        fit_directory (str): absolute path to the current genetic fit directory.

    Returns:
        none.
    """
    os.chdir(fit_directory)
    os.system(
        "casm query -k comp formation_energy hull_dist clex clex_hull_dist -o full_formation_energies.txt"
    )
