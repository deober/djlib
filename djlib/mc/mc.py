from __future__ import annotations
import json
from re import T
import matplotlib.pyplot as plt
import os
from scipy import integrate
import scipy.interpolate
import scipy.optimize
import numpy as np
from glob import glob
import pathlib
import math as m
import djlib.djlib as dj
from typing import List, Tuple


mc_lib_dir = pathlib.Path(__file__).parent.resolve()


def find(lst: List, a: float):
    """Finds the index of an element that matches a specified value.
    Args:
        a(float): The value to search for
    Returns:
        match_list[0](int): The index that a match occurrs, assuming there is only one match.
    """
    tolerance = 1e-14
    match_list = [
        i for i, x in enumerate(lst) if np.isclose([x], [a], rtol=tolerance)[0]
    ]
    if len(match_list) == 1:
        return match_list[0]
    elif len(match_list) > 1:
        print("Found more than one match. This is not expected.")
    elif len(match_list) == 0:
        print(
            "\nWARNING: Search value does not match any value in the provided list.\n"
        )


def read_mc_results_file(
    results_file_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Function to parse mc results.json files.
    Args;
        results_file_path(str): Path to the results.json file for the given monte carlo simulation.
    Returns:
        x: np.ndarray
            Vector of compostitions
        b: np.ndarray 
            Vector of beta values
        temperature: np.ndarray 
            Vecor of temperature values (K)
        potential_energy: np.ndarray: 
            Vector of potential energy values (E-mu*x)
    """
    with open(results_file_path) as f:
        results = json.load(f)

    mu = np.array(results["param_chem_pot(a)"])
    x = np.array(results["<comp(a)>"])
    b = np.array(results["Beta"])
    temperature = np.array(results["T"])
    potential_energy = np.array(results["<potential_energy>"])
    formation_energy = np.array(results["<formation_energy>"])
    return (mu, x, b, temperature, potential_energy, formation_energy)

def mc_run_namer(run_params:dict):
    """
    Function to generate a name for a MC simulation based on the run parameters.

    Parameters:
    -----------
    run_params: dict
        Dictionary of run parameters.
        Should include: mu start, stop, & increment; T start, stop, & increment, and supercell transformation matrix.

    Returns:
    --------
    name: str
        Name of the non-lte simulation.
    """
    name = f"mu_{run_params['mu_start']}_{run_params['mu_stop']}T_{run_params['T_start']}+{run_params['T_stop']}"
    return name

def mc_status_updater(run_dir):
    """
        Update status.json in a run directory. Updates to "complete" if results.json exists and is of the right size. Updates to "incomplete" if results.json exists and is of the wrong size.
    """
    status_file = os.path.join(run_dir, "status.json")
    settings_file = os.path.join(run_dir, "mc_settings.json")
    length = read_mc_settings(settings_file)[0].shape[0]
    results_file = os.path.join(run_dir, "results.json")
    if os.path.exists(results_file):
        results_length = read_mc_results_file(results_file)[0].shape[0]
        print(f"length: {length}, results_length: {results_length}")
        if results_length == length:
            status = "complete"
        else:
            status = "incomplete"
    else:
        status = "not_submitted"
    with open(status_file, "w") as f:
        f.write({'status':status})

def read_lte_results(
    results_file_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Takes a lte results.json file and returns outputs from the simulation.

    Parameters:
    -----------
    results_file_path: str
        Path to lte results.json file.

    Returns:
    --------
    mu: np.ndarray
        Vector of chemical potentials (species "a")
    b: np.ndarray
        Vector of Beta values (1/Temperature).
    t: np.ndarray 
        Vector of temperatures.
    x: np.ndarray 
        Vector of compositions.
    pot_eng: np.ndarray 
        Vector of phi values (grand canonical potential energy)
    """
    with open(results_file_path) as f:
        results = json.load(f)

    mu = np.array(results["param_chem_pot(a)"])
    b = np.array(results["Beta"])
    t = np.array(results["T"])
    x = np.array(results["gs_comp(a)"])
    pot_eng = np.array(results["phi_LTE"])

    return (mu, b, t, x, pot_eng)


def read_mc_settings(settings_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function to read chemical potential and temperature values from a mc_settings.json file.

    Parameters:
    -----------
    settings_file: str
        Path to mc_settings.json file.
    
    Returns:
    --------
    mu: np.ndarray
        Vector of chemical potentials (species "a")
    t: np.ndarray
        Vector of temperatures.
    """
    with open(settings_file) as f:
        settings = json.load(f)

    mu_start = settings["driver"]["initial_conditions"]["param_chem_pot"]["a"]
    mu_stop = settings["driver"]["final_conditions"]["param_chem_pot"]["a"]
    mu_increment = settings["driver"]["incremental_conditions"]["param_chem_pot"]["a"]
    t_start = settings["driver"]["initial_conditions"]["temperature"]
    t_stop = settings["driver"]["final_conditions"]["temperature"]
    t_increment = settings["driver"]["incremental_conditions"]["temperature"]

    if mu_increment != 0:
        mu_length = int(np.abs((mu_start - mu_stop) / mu_increment))
    else:
        mu_length = 1
    if t_increment != 0:
        t_length = int(np.abs((t_start - t_stop) / t_increment))
    else:
        t_length = 1

    if mu_length == 1:
        mu_values = np.ones(t_length) * mu_start
    elif mu_length > 1:
        mu_values = np.linspace(mu_start, mu_stop, mu_length)
    if t_length == 1:
        t_values = np.ones(mu_length) * t_start
    elif t_length > 1:
        t_values = np.linspace(t_start, t_stop, t_length)

    return (mu_values, t_values)


def read_superdupercell(mc_settings_file: str) -> List:
    """Function to read mu / temperature values as well as superdupercell from a monte carlo settings.json file.

    Parameters:
    -----------
    mc_settings_file: str
        Path to mc_settings.json file.
    Returns:
    --------
    superdupercell: List
        Matrix (list of 3 lists) that describes a supercell for the monte carlo simulation. 
    """
    with open(mc_settings_file) as f:
        settings = json.load(f)
    superdupercell = settings["supercell"]
    return superdupercell


class lte_run:
    """Class to parse CASM results from Grand Canonical monte carlo low temperature expansion (lte) calculation results.
    
    Attributes:
    -----------
    path: str   
        Path to the directory containing the lte results files.

    Methods:
    --------
    read_lte_results:
        Function to parse lte results.json files.
    
    
    
    """

    def __init__(self, lte_dir):
        self.path = lte_dir
        self.read_lte_results()

    def read_lte_results(self):
        results_file = os.path.join(self.path, "results.json")
        with open(results_file) as f:
            results = json.load(f)
        self.mu = results["param_chem_pot(a)"]
        self.b = results["Beta"]
        self.t = results["T"]
        self.x = results["gs_comp(a)"]
        self.pot_eng = results["phi_LTE"]
        self.superdupercell = read_superdupercell(
            os.path.join(self.path, "mc_settings.json")
        )


class constant_t_run:
    """Class to parse CASM results from constant temperature Grand Canonical monte carlo calculations.
    
    Attributes:
    -----------
    path: str
        Path to the directory containing the constant temperature results files.
    
    Methods:
    --------
    integrate_constant_temp_grand_canonical:
        Function to integrate the Grand Canonical Free Energy over varying chemical potential at constant temperature.
    
    """

    def __init__(self, const_t_dir):
        self.path = const_t_dir
        results_file_path = os.path.join(self.path, "results.json")
        (
            self.mu,
            self.x,
            self.b,
            self.t,
            self.pot_eng,
            self.formation_energy,
        ) = read_mc_results_file(results_file_path)
        self.integrate_constant_temp_grand_canonical()
        self.superdupercell = read_superdupercell(
            os.path.join(self.path, "mc_settings.json")
        )

    def integrate_constant_temp_grand_canonical(self):
        """Function to integrate across mu values at a constant temperature
        Args:
            x(list): Vector of compostitions
            b(list): Vector of beta values
            potential_energy(list): Vector of potential energy values (E-mu*x)
            mu(list): Vector of mu values

        Returns:
            integrated_potential(list): List of grand canonical free energy values corresponding to a fixed temperature / beta value.
        """
        free_energy_reference = self.pot_eng[0]
        integrated_potential = []
        for index, value in enumerate(self.mu):
            index = index + 1
            if index > 0:
                current_mu = self.mu[0:index]
                current_b = self.b[0:index]
                current_x = self.x[0:index]
                integrated_potential.append(
                    (1 / current_b[-1])
                    * (
                        self.b[0] * free_energy_reference
                        + integrate.simpson((-1 * current_b * current_x), current_mu)
                    )
                )
        self.integ_grand_canonical = np.asarray(integrated_potential)


class heating_run:
    """Class to parse CASM results from heating Grand Canonical monte carlo calculations at constant chemical potential.

    Attributes:
    -----------
    path: str
        Path to the directory containing the heating results files.
    
    Methods:
    --------
    get_lte_reference_energy:
        Function to look up the lte reference energy at a given chemical potential, from a lte run object.
    integrate_heating_grand_canonical_from_lte:
        Function to integrate the Grand Canonical Free Energy over varying temperature from the end state of a lte run; all at constant chemical potential.

    """

    def __init__(self, heating_dir, lte_run):
        self.path = heating_dir
        results_file_path = os.path.join(self.path, "results.json")
        (
            self.mu,
            self.x,
            self.b,
            self.t,
            self.pot_eng,
            self.formation_energy,
        ) = read_mc_results_file(results_file_path)
        self.get_lte_reference_energy(lte_run)
        self.integrate_heating_grand_canonical_from_lte()
        self.superdupercell = read_superdupercell(
            os.path.join(self.path, "mc_settings.json")
        )

    def get_lte_reference_energy(self, lte_run):
        mu_index = find(lte_run.mu, self.mu[0])
        self.energy_reference = lte_run.pot_eng[mu_index]

    def integrate_heating_grand_canonical_from_lte(self):
        """Function to integrate the grand canonical free energy from monte carlo heating run results.
        Args:
            x(list): Vector of compostitions
            b(list): Vector of beta values
            potential_energy(list): Vector of potential energy values (E-mu*x)
            mu(list): Vector of mu values

        Returns:
            integrated_potential(list): List of grand canonical free energy values corresponding to a fixed mu value.
        """

        self.pot_eng[0] = self.energy_reference

        integrated_potential = []
        for index in range(len(self.b)):
            index = index + 1
            if index > 0:
                current_b = self.b[0:index]
                current_potential_energy = self.pot_eng[0:index]
                integrated_potential.append(
                    (1 / current_b[-1])
                    * (
                        self.b[0] * self.energy_reference
                        + integrate.simpson(current_potential_energy, current_b)
                    )
                )
        self.integ_grand_canonical = np.asarray(integrated_potential)


class cooling_run:
    """Class to parse CASM results from cooling Grand Canonical monte carlo calculations at constant chemical potential.

    Attributes:
    -----------
    path: str
        Path to the directory containing the cooling results files.

    Methods:    
    --------
    get_constant_t_reference_energy:
        Function to look up the lte reference energy at a given chemical potential, from a constant temperature run object.
    """

    def __init__(self, cooling_dir, constant_t_run):
        self.path = cooling_dir
        results_file_path = os.path.join(self.path, "results.json")
        self.mu, self.x, self.b, self.t, self.pot_eng = read_mc_results_file(
            results_file_path
        )
        self.get_constant_t_reference_energy(constant_t_run)
        self.integrate_cooling_from_const_t_run()
        self.superdupercell = read_superdupercell(
            os.path.join(self.path, "mc_settings.json")
        )

    def get_constant_t_reference_energy(self, constant_t_run):
        mu_index = find(constant_t_run.mu, self.mu[0])
        self.energy_reference = constant_t_run.integ_grand_canonical[mu_index]

    def integrate_cooling_from_const_t_run(self):
        free_energy_reference = self.energy_reference
        integrated_potential = []
        for index, value in enumerate(self.b):
            index = index + 1
            if index > 0:
                current_b = self.b[0:index]
                current_potential_energy = self.pot_eng[0:index]
                integrated_potential.append(
                    (1 / current_b[-1])
                    * (
                        self.b[0] * free_energy_reference
                        + integrate.simpson(current_potential_energy, current_b)
                    )
                )
        self.integ_grand_canonical = np.asarray(integrated_potential)


def format_mc_settings(
    superdupercell: list,
    mu_init: float,
    mu_final: float,
    mu_increment: float,
    temp_init: float,
    temp_final: float,
    temp_increment: float,
    output_file: str,
    start_config_path: bool = False,
) -> None:
    """Function to format the CASM monte carlo settings json file file for a monte carlo run.
    
    Parameters:
    -----------
    superdupercell: list
        Tranformation matrix to apply to the CASM project primitive cell. Represented as a list of lists.
    mu_init: float
        Initial chemical potential value.
    mu_final: float
        Final chemical potential value.
    mu_increment: float
        Chemical potential increment value. Sign matters: if positive, it must follow that mu_final > mu_init. If negative, it must follow that mu_final < mu_init.
    temp_init: float
        Initial temperature value.
    temp_final: float
        Final temperature value.
    temp_increment: float
        Temperature increment value. Sign matters: if positive, it must follow that temp_final > temp_init. If negative, it must follow that temp_final < temp_init.
    output_file: str
        Path to the output file.
    start_config_path: str, optional
        Path to the starting configuration file, contained within one of the conditions.* files.

    Returns:
    --------
    None.
    """

    templates_path = os.path.join(mc_lib_dir, "../templates")
    # Read template
    with open(os.path.join(templates_path, "mc_grand_canonical_template.json")) as f:
        mc_settings = json.load(f)

    # Write settings
    mc_settings["supercell"] = superdupercell
    mc_settings["driver"]["initial_conditions"]["param_chem_pot"]["a"] = mu_init
    mc_settings["driver"]["initial_conditions"]["temperature"] = temp_init
    mc_settings["driver"]["final_conditions"]["param_chem_pot"]["a"] = mu_final
    mc_settings["driver"]["final_conditions"]["temperature"] = temp_final
    mc_settings["driver"]["incremental_conditions"]["param_chem_pot"][
        "a"
    ] = mu_increment
    mc_settings["driver"]["incremental_conditions"]["temperature"] = temp_increment

    if (start_config_path != False) and (start_config_path != None):
        mc_settings["driver"]["motif"]["configdof"] = start_config_path

    # write settings file
    with open(output_file, "w") as f:
        json.dump(mc_settings, f, indent="")


def run_cooling_from_const_temperature(
    mu_values: np.ndarray,
    mc_cooling_dir: str,
    const_temp_run_dir: str,
    temp_final: float,
    temperature_increment: float,
    job_scheduler: str = "slurm",
    submit_job: bool = False,
) -> None:
    """Runs many cooling Grand Canonical monte carlo calculations; each run is at different chemical potential. The calculations begin at the end state (atomic configuration) of a constant temperature calculation.
    
    Parameters:
    -----------
    mu_values: np.ndarray
        Array of chemical potential values to run the cooling calculation at.
    mc_cooling_dir: str
        Path to write all cooling calculations to.
    const_temp_run_dir: str
        Path to the directory containing the constant temperature run.
    temp_final: float
        Final temperature value.
    temperature_increment: float
        Temperature increment value. Sign matters: if positive, it must follow that temp_final > temp_init. If negative, it must follow that temp_final < temp_init.
    job_scheduler: str, optional
        Scheduler to use for the job. if not specified, will use slurm.
    submit_job: bool, optional
        Whether to submit the job. If False, will just write the job file. Default is False.
    """

    # read mu values, temperature information from the existing settings file
    (const_t_mu, x, b, temperature_values, potential_energy) = read_mc_results_file(
        os.path.join(const_temp_run_dir, "results.json")
    )
    superdupercell = read_superdupercell(
        os.path.join(const_temp_run_dir, "mc_settings.json")
    )
    # for each mu value, start a cooling run with the condition.# final state as the initial state (condition indexing starts at 0)
    for mu in mu_values:

        # Set up run directory
        run_name = "mu_%.4f_%.4f_T_%d_%d" % (mu, mu, temperature_values[0], temp_final)
        current_dir = os.path.join(mc_cooling_dir, run_name)
        if os.path.isfile(os.path.join(current_dir, "results.json")) == False:

            os.makedirs(current_dir, exist_ok=True)
            os.chdir(current_dir)

            # get const_t_mu index that matches mu
            mu_index = find(const_t_mu, mu)
            # Write settings file
            settings_file = os.path.join(current_dir, "mc_settings.json")
            start_config_path = os.path.join(
                const_temp_run_dir, "conditions.%d" % mu_index, "final_state.json"
            )

            format_mc_settings(
                superdupercell,
                mu,
                mu,
                0,
                temperature_values[0],
                temp_final,
                temperature_increment,
                settings_file,
                start_config_path,
            )

            # Run MC cooling
            if job_scheduler == "slurm":
                user_command = "casm monte -s mc_settings.json > mc_results.out"
                dj.format_slurm_job(
                    jobname="cool_" + run_name,
                    hours=20,
                    user_command=user_command,
                    output_dir=current_dir,
                    delete_submit_script=False,
                )
                if submit_job:
                    dj.submit_slurm_job(current_dir)
            elif job_scheduler == "pbs":
                user_command = "casm monte -s mc_settings.json > mc_results.out"
                format_pbs_job(
                    jobname=run_name,
                    hours=20,
                    user_command=user_command,
                    output_dir=current_dir,
                    delete_submit_script=False,
                )
                if submit_job:
                    submit_pbs_job(current_dir)
            """
            print("Submitting: ", end="")
            print(current_dir)
            os.system("casm monte -s mc_settings.json > mc_results.out &")
            """


def run_heating(
    mc_heating_dir: str,
    mu_values: np.ndarray,
    superdupercell: list,
    temp_init: float,
    temp_final: float,
    temp_increment: float,
    scheduler: str = "slurm",
    submit_job: bool = False,
) -> None:
    """Runs many heating Grand Canonical monte carlo calculations; each run is at different chemical potential.
    
    Parameters:
    -----------
    mc_heating_dir: str
        Path to write all heating calculations to.
    mu_values: np.ndarray
        Array of chemical potential values to run the heating calculation at.
    superdupercell: list
        Transformation matrix to apply on the CASM project primitive cell. Represented as a list of lists.
    temp_init: float
        Initial temperature value.
    temp_final: float
        Final temperature value.
    temp_increment: float
        Temperature increment value. Sign matters: if positive, it must follow that temp_final > temp_init. If negative, it must follow that temp_final < temp_init.
    scheduler: str, optional
        Scheduler to use for the job. if not specified, will use slurm.
    submit_job: bool, optional
        Whether to submit the job. If False, will just write the job file. Default is False.

    Returns:
    --------    
    None.
    """

    for mu_value in mu_values:

        run_name = "mu_%.4f_%.4f_T_%d_%d" % (mu_value, mu_value, temp_init, temp_final)
        current_dir = os.path.join(mc_heating_dir, run_name)

        if os.path.isfile(os.path.join(current_dir, "results.json")) == False:
            os.makedirs(current_dir, exist_ok=True)
            os.chdir(current_dir)

            # Format settings file for this heating run
            settings_file = os.path.join(current_dir, "mc_settings.json")
            format_mc_settings(
                superdupercell,
                mu_value,
                mu_value,
                0,
                temp_init,
                temp_final,
                temp_increment,
                settings_file,
                start_config_path=False,
            )

            # Run MC heating
            user_command = "casm monte -s mc_settings.json > mc_results.out"
            if scheduler == "slurm":
                dj.format_slurm_job(
                    jobname="heat_" + run_name,
                    hours=20,
                    user_command=user_command,
                    output_dir=current_dir,
                    delete_submit_script=False,
                )
                if submit_job:
                    dj.submit_slurm_job(current_dir)
            elif scheduler == "pbs":
                format_pbs_job(
                    jobname=run_name,
                    hours=20,
                    user_command=user_command,
                    output_dir=current_dir,
                    delete_submit_script=False,
                )
                if submit_job:
                    submit_pbs_job(current_dir)
            """
            print("Submitting: ", end="")
            print(current_dir)
            os.system("casm monte -s mc_settings.json > mc_results.out &")
            """


def plot_const_t_x_vs_mu(const_t_left, const_t_right):

    full_mu = np.concatenate((const_t_left.mu, const_t_right.mu))
    full_x = np.concatenate((const_t_left.x, const_t_right.x))

    plt.scatter(full_x, full_mu, color="xkcd:crimson")
    plt.xlabel("Composition (a)", fontsize=18)
    plt.ylabel("Chemical Potential (a)", fontsize=18)
    fig = plt.gcf()
    fig.set_size_inches(15, 19)

    return fig


def plot_heating_and_cooling(heating_run, cooling_run):
    bullet_size = 3
    if heating_run.mu[0] != cooling_run.mu[0]:
        print(
            "WARNING: Chemical potentials do not match between the heating and cooling runs"
        )
    plt.title("Constant Mu: %.4f" % heating_run.mu[0])
    plt.xlabel("Temperature (K)", fontsize=18)
    plt.ylabel("Grand Canonical Free Energy", fontsize=18)
    plt.scatter(cooling_run.t, cooling_run.integ_grand_canonical, s=bullet_size)
    plt.scatter(heating_run.t, heating_run.integ_grand_canonical, s=bullet_size)
    plt.legend(["Cooling", "Heating"])
    fig = plt.gcf()
    fig.set_size_inches(15, 19)
    return fig


def predict_mu_vs_free_energy_crossing(
    const_t_run_1: constant_t_run, const_t_run_2: constant_t_run
) -> Tuple[float, float, float, float]:
    """Function to predict the free energy crossing chemical potential and compositions between two mu vs grand canonical plots

    Parameters:
    -----------
    const_t_run_1: constant_t_run
        A djlib constant_t_run object, containing the data for a constant temperature run
    const_t_run_2: constant_t_run
        A djlib constant_t_run object, containing the data for a constant temperature run

    Returns:
    --------
    mu_intersect_predict: float
        Predicted chemical potential at the crossing point in (chemical_potential vs grand canonical free energy) space
    energy_intersect_predict: float
        predicted grand canonical free energy at the crossing point in (chemical_potential vs grand canonical free energy) space
    run_1_comp_intersect: float
        composition for the first constant temperature run that is closest to the corresponding chemical potential at the crossing point
    run_2_comp_intersect: float
        composition for the second constant temperature run that is closest to the corresponding chemical potential at the crossing point
    

    """
    # Assert that the chemical potentials are the same
    assert np.allclose(
        np.sort(const_t_run_1.mu), np.sort(const_t_run_2.mu)
    ), "Chemical potentials do not match"

    # Ensure that all data is sorted by chemical potential
    mu_1 = const_t_run_1.mu[np.argsort(const_t_run_1.mu)]
    gc_free_energy_1 = const_t_run_1.integ_grand_canonical[np.argsort(const_t_run_1.mu)]
    x_1 = const_t_run_1.x[np.argsort(const_t_run_1.mu)]

    mu_2 = const_t_run_2.mu[np.argsort(const_t_run_2.mu)]
    gc_free_energy_2 = const_t_run_2.integ_grand_canonical[np.argsort(const_t_run_2.mu)]
    x_2 = const_t_run_2.x[np.argsort(const_t_run_2.mu)]

    # Fit a spline to the chemical potential vs grand canonical free energy data
    # Spline requires that domain (chemical potential) is strictly increasing
    interp_run_1 = scipy.interpolate.InterpolatedUnivariateSpline(
        mu_1, gc_free_energy_1
    )
    interp_run_2 = scipy.interpolate.InterpolatedUnivariateSpline(
        mu_2, gc_free_energy_2
    )

    # Define a difference function to be used in the root finding algorithm
    def difference(m):
        return np.abs(interp_run_1(m) - interp_run_2(m))

    # Find an initial guess for the root finding algorithm
    m0_index = np.argmin(abs(gc_free_energy_1 - gc_free_energy_2))
    m0_guess = mu_1[m0_index]

    # find the root of the difference function (chemical potential at crossing)
    mu_intersect_predict = scipy.optimize.fsolve(difference, x0=m0_guess)
    energy_intersect_predict = interp_run_1(mu_intersect_predict)

    # Find the calculated chemical potential that is closest to the predicted chemical potential
    mu_intersect_index_1 = np.argmin(abs(mu_intersect_predict - const_t_run_1.mu))
    mu_intersect_index_2 = np.argmin(abs(mu_intersect_predict - const_t_run_2.mu))

    # Find the crossing composition at a point mu, x that is actually calculated
    difference = np.abs(-mu_intersect_predict)

    run_1_comp_intersect = const_t_run_1.x[mu_intersect_index_1]
    run_2_comp_intersect = const_t_run_2.x[mu_intersect_index_2]
    return (
        mu_intersect_predict,
        energy_intersect_predict,
        run_1_comp_intersect,
        run_2_comp_intersect,
    )


def predict_free_energy_crossing(heating_run, cooling_run):
    """Function to find crossing point between two (energy vs T) curves.
    Args:
        heating_run(djlib.mc.heating_run):  Heating run object defined in djlib.mc
        cooling_run(djlib.mc.cooling_run): Cooling run object defined in djlib.mc
    Returns:
        tuple(
            t_intersect_predict,
            energy_intersect_predict, composition_intersect_predict
        )

    """
    # Check that lengths of all vectors match and that temp_heating == temp_cooling (i.e., they're not the reverse of each other)
    if (
        heating_run.integ_grand_canonical.shape[0]
        == heating_run.t.shape[0]
        == cooling_run.integ_grand_canonical.shape[0]
        == cooling_run.t.shape[0]
        == heating_run.x.shape[0]
        == cooling_run.x.shape[0]
    ):

        find_intersection = False
        if np.allclose(heating_run.t, cooling_run.t):
            find_intersection = True
        else:
            # If the temperature axes arent the same, try swapping the order of temp_cooling and cooling_integrated_free_energy.
            cooling_run.t = np.flip(cooling_run.t)
            cooling_run.integ_grand_canonical = np.flip(
                cooling_run.integ_grand_canonical
            )
            cooling_run.x = np.flip(cooling_run.x)

            # If the temperature axes still aren't the same, cancel the function.
            if np.allclose(heating_run.t, cooling_run.t):
                find_intersection = True
            else:
                print(
                    "Heating and cooling run temperature vectors are the same length but do not match. See printout below:\ntemp_heating  temp_cooling"
                )
                for idx, value in enumerate(heating_run.t):
                    print("%.3f  %.3f" % heating_run.t[idx], cooling_run.t[idx])

        if find_intersection:
            # TODO: Check that there isn't more than one intersection (complete overlap) or no intersection.

            # fit spline to each dataset, calculate intersection
            interp_heating = scipy.interpolate.InterpolatedUnivariateSpline(
                heating_run.t, heating_run.integ_grand_canonical
            )
            interp_heating_comp = scipy.interpolate.InterpolatedUnivariateSpline(
                heating_run.t, heating_run.x
            )
            interp_cooling = scipy.interpolate.InterpolatedUnivariateSpline(
                cooling_run.t, cooling_run.integ_grand_canonical
            )
            interp_cooling_comp = scipy.interpolate.InterpolatedUnivariateSpline(
                cooling_run.t, cooling_run.x
            )

            # define a difference function to calculate the root
            def difference(t):
                return np.abs(interp_heating(t) - interp_cooling(t))

            # Provide a composition x0 as a guess for the root finder
            # This will break if there are multiple identical minimum values
            t0_index = np.argmin(
                abs(
                    heating_run.integ_grand_canonical
                    - cooling_run.integ_grand_canonical
                )
            )
            t0_guess = heating_run.t[t0_index]

            # Calculate the intersection point
            t_intersect_predict = scipy.optimize.fsolve(difference, x0=t0_guess)
            energy_intersect_predict = interp_heating(t_intersect_predict)
            composition_intersect_predict = interp_heating_comp(t_intersect_predict)

            return (
                t_intersect_predict,
                energy_intersect_predict,
                composition_intersect_predict,
            )

    else:
        print(
            "The free energies and composition vectors do not have the same lengths.\nCurrent lengths are:"
        )
        print("length of temp_heating: %d" % len(temp_heating))
        print(
            "length of heating_integrated_free_energy: %d"
            % len(heating_integrated_free_energy)
        )
        print("length of temp_cooling: %d" % len(temp_cooling))
        print(
            "length of cooling_integrated_free_energy: %d"
            % len(cooling_integrated_free_energy)
        )


def find_crossing_composition(
    temperature, integrated_energies, t_intersect_predict, energy_intersect_predict
):
    """Given an interpolated point in (energy vs temperature) space, find the closest existing (energy, temperature) and return the corresponding composition x and corresponding temperature.
    Args:
        integrated_energies(ndarray): Vector of integrated energy values.
        temperature(ndarray): Vector of temperature values (K).
        x(ndarray): Vector of composition values.
        t_intersect_predict(float): Interpolated prediction of the free energy crossing temperature between a heating and cooling grand canonical monte carlo simulation.
        energy_intersect_predict(float): Interpolated prediction of the free energy at the crossing temperature between a heating and cooling grand canonical monte carlo simulation.

    Returns:
        tuple(
            x_at_crossing(float): Composition at the actual coordinates closest to the predicted
            t_at_crossing(float): Temperature (K) at the actual coordinates closest to the predicted
        )
    """

    temperature_and_energy = np.zeros((len(temperature), 2))
    temperature_and_energy[:, 0] = temperature
    temperature_and_energy[:, 1] = integrated_energies

    prediction_point = np.array([t_intersect_predict, energy_intersect_predict])

    difference = temperature_and_energy - prediction_point

    distance = np.sum(np.abs(difference) ** 2, axis=-1) ** (1 / 2)
    closest_point_index = np.argmin(distance)

    x_at_crossing = x[closest_point_index]
    t_at_crossing = temperature[closest_point_index]
    return (x_at_crossing, t_at_crossing)


def simulation_is_complete(mc_run_dir):
    """Check that a grand canonical monte carlo simulation has finished
    Args:
        mc_run_dir(str): Path to a monte carlo simulation directory.

    Returns:
        simulation_status(bool): simulation is complete (True) or simulation is not complete (False)
    """
    # Check the number of conditions (mu, temp) that should be completed
    # TODO: read the results.json file instead of the number of conditions directories to see if a simulaiton is complete.
    mc_settings_file = os.path.join(mc_run_dir, "mc_settings.json")
    mu_values, temperature_values = read_mc_settings(mc_settings_file)

    target_t_length = temperature_values.shape[0]

    if os.path.isfile(os.path.join(mc_run_dir, "results.json")):
        mu, x, b, temperature, potential_energy = read_mc_results_file(
            os.path.join(mc_run_dir, "results.json")
        )
        if temperature.shape[0] == target_t_length.shape:
            simulation_status = True
        else:
            simulation_status = False
    else:
        print("Cannot find %s" % os.path.join(mc_run_dir, "results.json"))
        simulation_status = False
    return simulation_status


def plot_t_vs_x_rainplot(mc_runs_directory: str, show_labels: bool = False):
    """plot_rain_plots(mc_runs_directory, save_image_path=False, show_labels=False)

    Generate a single (T vs composition) plot using all monte carlo runs in mc_runs_directory.
    Args:
        mc_runs_directory(str): Path to the directory containing all grand canonical monte carlo runs.
        same_image(bool): Whether the image will be saved to the run directory or not.
        show_labels(bool): Whether or not the plot legend displays.

    Returns:
        fig(matplotlib.pyplot figure object): 2D plot object. Can do fig.show() to display the plot.
    """
    labels = []
    run_list = glob(os.path.join(mc_runs_directory, "mu*"))
    for run in run_list:
        results_file = os.path.join(run, "results.json")
        if os.path.isfile(results_file):
            with open(results_file) as f:
                data = json.load(f)
                f.close()
                current_mc = run.split("/")[-1]
                labels.append(current_mc)
                composition = data["<comp(a)>"]
                temperature = data["T"]
                plt.scatter(composition, temperature)

    if show_labels:
        plt.legend(labels)
    plt.xlabel("Composition", fontsize=18)
    plt.ylabel("Temperature (K)", fontsize=18)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10)
    return fig


def submit_pbs_job(run_dir):
    submit_file = os.path.join(run_dir, "submit_pbs.sh")
    os.system("qsub %s" % submit_file)


def format_pbs_job(
    jobname, hours, user_command, output_dir, delete_submit_script=False
):
    """
    Formats a pbs (TORQUE) submit file.
    Args:
        jobname(str): Name of the slurm job.
        hours(int): number of hours to run the job. Only accepts integer values.
        user_command(str): command line command submitted by the user as a string.
        output_dir(str): Path to the directory that will contain the submit file. Assumes that submit file will be named "submit.sh"
        delete_submit_script(bool): Whether the submission script should delete itself upon completion.
    Returns:
        None.
    """
    submit_file_path = os.path.join(output_dir, "submit_pbs.sh")
    templates_path = os.path.join(mc_lib_dir, "../templates")
    with open(os.path.join(templates_path, "single_task_pbs_template.sh")) as f:
        template = f.read()

        if delete_submit_script:
            delete_submit_script = "rm %s" % submit_file_path
        else:
            delete_submit_script = ""

        hours = int(m.ceil(hours))
        s = template.format(
            jobname=jobname,
            rundir=output_dir,
            hours=hours,
            user_command=user_command,
            delete_submit_script=delete_submit_script,
        )
    with open(submit_file_path, "w") as f:
        f.write(s)
    os.system("chmod +x %s " % submit_file_path)
