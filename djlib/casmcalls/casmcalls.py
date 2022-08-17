from __future__ import annotations
import os
import djlib.djlib as dj
import numpy as np

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


def end_state_supercell_calc_setup(
    vasp_calc_dir: str,
    primitive_structure_poscar: str,
    transformation_matrix: np.ndarray,
    incar_template: str,
    kpoints_template: str,
    potcar_path: str,
    job_time_hours: int,
    job_name: str,
    output_dir: str,
    cpus_per_task: int,
    user_command: str,
) -> None:
    """Set up static vasp calculation for composition end state structure in a larger supercell in order to reduce kpoint noise in formation energy values.

    Parameters:
    -----------
    vasp_calc_dir: str
        Path to the vasp calculation directory
    primitive_structure_poscar: str
        Path to the primitive structure file- this file will be scaled by transformation_matrix to create the end state supercell
    scel_name: str
        Name of the supercell.
    transformation_matrix_dict: dict
        transformation_matrix for the given supercell.
    incar_template_path: str
        Path to the incar template file. 
    kpoints_template_path: str
        Path to the kpoints template file.
    potcar_path: str
        Path to the POTCAR file.
    
    Returns:
    --------
    None
    """

    # create the vasp calculation directory if it does not exist
    if not os.path.exists(vasp_calc_dir):
        os.makedirs(vasp_calc_dir, exist_ok=True)

        # copy potcar, incar and kpoint file to vasp calculation directory
        os.system("cp %s %s" % (potcar_path, vasp_calc_dir))
        os.system("cp %s %s" % (incar_template, vasp_calc_dir))
        os.system("cp %s %s" % (kpoints_template, vasp_calc_dir))

    # write a status.json file if it does not exist. If the file does not exist, initialize the status as "not_submitted"
    if not os.path.exists(os.path.join(vasp_calc_dir, "status.json")):
        with open(os.path.join(vasp_calc_dir, "status.json"), "w") as f:
            f.write('{"status": "not_submitted"}')

    # write the transformation matrix to generate this SCEL
    transformation_matrix = transformation_matrix.astype(int)
    transformation_matrix = np.reshape(transformation_matrix, (3, 3))
    transformation_matrix_file = os.path.join(
        vasp_calc_dir, "transformation_matrix.txt"
    )
    np.savetxt(transformation_matrix_file, transformation_matrix, fmt="%d")

    # Generate the poscar for this transformation matrix  using casm super
    poscar_output_file = os.path.join(vasp_calc_dir, "POSCAR")
    os.system(
        "casm super --structure %s --transf-mat %s --vasp5 > %s"
        % (primitive_structure_poscar, transformation_matrix_file, poscar_output_file)
    )

    # Write a slurm submission script for the vasp calculation
    dj.format_slurm_job(
        jobname=job_name,
        hours=job_time_hours,
        user_command=user_command,
        output_dir=output_dir,
        cpus_per_task=cpus_per_task,
    )
