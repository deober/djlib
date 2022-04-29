import json
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

mc_lib_dir = pathlib.Path(__file__).parent.resolve()


def find(lst, a):
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


def read_mc_results_file(results_file_path):
    """Function to parse mc results.json files.
    Args;
        results_file_path(str): Path to the results.json file for the given monte carlo simulation.
    Returns:
        x(ndarray): Vector of compostitions
        b(ndarray): Vector of beta values
        temperature(ndarray): Vecor of temperature values (K)
        potential_energy(ndarray): Vector of potential energy values (E-mu*x)
    """
    with open(results_file_path) as f:
        results = json.load(f)

    mu = np.array(results["param_chem_pot(a)"])
    x = np.array(results["<comp(a)>"])
    b = np.array(results["Beta"])
    temperature = np.array(results["T"])
    potential_energy = np.array(results["<potential_energy>"])
    return (mu, x, b, temperature, potential_energy)


def read_lte_results(results_file_path):
    """
    Takes a lte results.json file and returns outputs from the simulation.

    Args:
        results_file_path(str): Path to the results.json file.
    Returns:
        tuple(
            mu(ndarray): Vector of chemical potentials (species "a")
            b(ndarray): Vector of Beta values.
            t(ndarray): Vector of temperatures.
            x(ndarray): Vector of compositions.
            pot_eng(ndarray): Vector of phi values (grand canonical potential energy)
        )
    """
    with open(results_file_path) as f:
        results = json.load(f)

    mu = np.array(results["param_chem_pot(a)"])
    b = np.array(results["Beta"])
    t = np.array(results["T"])
    x = np.array(results["gs_comp(a)"])
    pot_eng = np.array(results["phi_LTE"])

    return (mu, b, t, x, pot_eng)


def read_mc_settings(settings_file):
    """
    Function to read chemical potential and temperature values from a mc_settings.json file.

    Args:
        settings_file(str): Path to a mc_settings.json file.
    Returns:
        tuple(
            mu_values,      (ndarray): Vector of chemical potential values.
            t_values        (ndarray): Vector of temperature values.
        )
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


def read_superdupercell(mc_settings_file):
    """Function to read mu / temperature values as well as superdupercell from a monte carlo settings.json file.
    Args:
        mc_settings_file(str): Path to mc_settings.json file.
    Returns:
        superdupercell(list): Matrix (list of 3 lists) that describes a supercell for the monte carlo simulation
    """
    with open(mc_settings_file) as f:
        settings = json.load(f)
    superdupercell = settings["supercell"]
    return superdupercell


class lte_run:
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
    def __init__(self, const_t_dir):
        self.path = const_t_dir
        results_file_path = os.path.join(self.path, "results.json")
        self.mu, self.x, self.b, self.t, self.pot_eng = read_mc_results_file(
            results_file_path
        )
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
    def __init__(self, heating_dir, lte_run):
        self.path = heating_dir
        results_file_path = os.path.join(self.path, "results.json")
        self.mu, self.x, self.b, self.t, self.pot_eng = read_mc_results_file(
            results_file_path
        )
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
    superdupercell,
    mu_init,
    mu_final,
    mu_increment,
    temp_init,
    temp_final,
    temp_increment,
    output_file,
    start_config_path,
):

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
    mu_values,
    mc_cooling_dir,
    const_temp_run_dir,
    temp_final=20,
    temperature_increment=-5,
    job_scheduler="slurm",
):

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

            os.makedirs(current_dir)
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
                dj.submit_slurm_job(current_dir)
            elif job_scheduler == "braid":
                user_command = "casm monte -s mc_settings.json > mc_results.out"
                format_pbs_job(
                    jobname=run_name,
                    hours=20,
                    user_command=user_command,
                    output_dir=current_dir,
                    delete_submit_script=False,
                )
                submit_pbs_job(current_dir)
            elif job_scheduler == "pod":
                print("In the works")
            """
            print("Submitting: ", end="")
            print(current_dir)
            os.system("casm monte -s mc_settings.json > mc_results.out &")
            """


def run_heating(
    mc_heating_dir,
    mu_values,
    superdupercell,
    temp_init=20,
    temp_final=2000,
    temp_increment=5,
    scheduler="slurm",
):

    for mu_value in mu_values:

        run_name = "mu_%.4f_%.4f_T_%d_%d" % (mu_value, mu_value, temp_init, temp_final)
        current_dir = os.path.join(mc_heating_dir, run_name)

        if os.path.isfile(os.path.join(current_dir, "results.json")) == False:
            os.makedirs(current_dir)
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
                dj.submit_slurm_job(current_dir)
            elif scheduler == "pbs":
                format_pbs_job(
                    jobname=run_name,
                    hours=20,
                    user_command=user_command,
                    output_dir=current_dir,
                    delete_submit_script=False,
                )
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


# TODO: Function to check that a free energy crossing


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


def plot_t_vs_x_rainplot(mc_runs_directory, save_image=False, show_labels=False):
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
    title = (
        "Chemical Potential and Temperature Sweep Rain Plot: %s"
        % mc_runs_directory.split("/")[-1]
    )
    plt.title("Chemical Potential and Temperature Sweep Rain Plot", fontsize=30)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10)
    if save_image:
        fig.savefig(os.path.join(mc_runs_directory, title + ".png"), dpi=100)
        print("Saving image to file: ", end="")
        print(os.path.join(mc_runs_directory, title + ".png"))
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


def setup_dos_calculation(
    config_name,
    training_dir,
    hours,
    spin=1,
    slurm=True,
    run_jobs=False,
    delete_submit_script=False,
):
    """
    Runs a static DOS calculation for a relaxed configuration. (Should have a run.final relaxation already in the folder)

    Parameters
    ----------
    config_name: str
        Name of the configuration to run the DOS calculation on.
    training_dir: str
        Path to the training directory (contains the configuration[s])
    slurm: bool
        Whether to submit the job with slurm. 
    run_jobs: bool
        Whether to run the job. If False, will only setup the folders and submit scripts.

    Returns
    -------
    None.
    """
    # TODO: check for bugs

    print("Setting up DOS calculation for %s" % config_name)

    calc_dir = os.path.join(training_dir, config_name, "calctype.default")
    print("Making static_charge_calc directory in %s" % calc_dir)
    os.makedirs(os.path.join(calc_dir, "static_charge_calc"), exist_ok=True)
    templates_path = os.path.join(mc_lib_dir, "../templates")
    # format INCAR
    with open(os.path.join(templates_path, "INCAR_static_charge.template")) as f:
        template = f.read()

        with open(os.path.join(calc_dir, "run.final", "INCAR")) as g:
            incar = g.readlines()

        for line in incar:
            if "ENCUT" in line:
                encut = line.split("=")[1].strip()
            if "ISMEAR" in line:
                ismear = line.split("=")[1].strip()

        s = template.format(encut=encut, spin=spin, ismear=ismear)

    with open(os.path.join(calc_dir, "static_charge_calc", "INCAR"), "w") as f:
        f.write(s)

    # format KPOINTS
    os.system(
        "cp %s %s"
        % (
            os.path.join(calc_dir, "run.final/KPOINTS"),
            os.path.join(calc_dir, "static_charge_calc", "KPOINTS"),
        )
    )

    # format POTCAR
    os.system(
        "cp %s %s"
        % (
            os.path.join(calc_dir, "run.final/POTCAR"),
            os.path.join(calc_dir, "static_charge_calc", "POTCAR"),
        )
    )

    # format POSCAR
    os.system(
        "cp %s %s"
        % (
            os.path.join(calc_dir, "run.final/CONTCAR"),
            os.path.join(calc_dir, "static_charge_calc", "POSCAR"),
        )
    )

    # define user command

    # script will submit static charge calc, then copy chgcar and results to DOS calc [changing INCAR as needed]

    user_command = """cd %s
    info_file=test.info
    echo \"HOSTNAME=$(hostname)\" >> $info_file
    echo \"STARTTIME=$(date --iso-8601=ns)\" >> $info_file

    mpirun vasp >& vasp.out

    cd ..
    mkdir dos_calc

    cp static_charge_calc/CONTCAR dos_calc/POSCAR
    cp static_charge_calc/INCAR dos_calc/INCAR
    cp static_charge_calc/POTCAR dos_calc/POTCAR
    cp static_charge_calc/KPOINTS dos_calc/KPOINTS
    cp static_charge_calc/CHGCAR dos_calc/CHGCAR

    cd dos_calc

    sed -i \"s/LORBIT.*/LORBIT = 11/g\" INCAR
    sed -i \"s/ICHARG.*/ICHARG = 11/g\" INCAR

    mpirun vasp >& vasp.out

    """ % (
        os.path.join(calc_dir, "static_charge_calc")
    )

    # format submit script
    if slurm:
        dj.format_slurm_job(
            jobname=config_name,
            hours=hours,
            user_command=user_command,
            output_dir=calc_dir,
            delete_submit_script=delete_submit_script,
        )

        if run_jobs:
            dj.submit_slurm_job(calc_dir)
