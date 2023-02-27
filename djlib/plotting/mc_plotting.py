import matplotlib.pyplot as plt
import thermocore.geometry.hull as thull
from glob import glob
import os
import numpy as np
import json
import djlib.mc.mc as mc


def plot_const_t_x_vs_mu(
    const_t_left: mc.constant_t_run, const_t_right: mc.constant_t_run
) -> plt.figure:
    full_mu = np.concatenate((const_t_left.mu, const_t_right.mu))
    full_x = np.concatenate((const_t_left.x, const_t_right.x))

    plt.scatter(full_x, full_mu, color="xkcd:crimson")
    plt.xlabel("Composition (a)", fontsize=18)
    plt.ylabel("Chemical Potential (a)", fontsize=18)
    fig = plt.gcf()
    fig.set_size_inches(15, 19)
    return fig


def plot_heating_and_cooling(
    heating_run: mc.heating_run, cooling_run: mc.cooling_run
) -> plt.figure:
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


def plot_t_vs_x_rainplot(
    mc_runs_directory: str,
    show_labels: bool = False,
    show_chemical_potential_labels: bool = False,
) -> plt.figure:
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
                chemical_potential = data["param_chem_pot(a)"][0]
                plt.scatter(composition, temperature)
                # Also, plot chemical potential as a number in a text box next to the highest temperature point.
                if show_chemical_potential_labels:
                    max_temp_index = np.argmax(temperature)
                    plt.text(
                        composition[max_temp_index],
                        temperature[max_temp_index],
                        chemical_potential,
                        fontsize=7,
                        rotation=90,
                    )

    if show_labels:
        plt.legend(labels)
    plt.xlabel("Composition", fontsize=18)
    plt.ylabel("Temperature (K)", fontsize=18)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10)
    return fig


def sgcmc_full_project_diagnostic_plots(sgcmc_project_data_dictionary: dict) -> None:
    """Takes a dictionary of run data from a full MC project (generated from propagate_gcmc.py propagation_project_parser) and generates diagnostic plots.
    These plots include: chemical potential vs composition, Integrated sgc free energy vs temperature, heat capacity plots, and rainplot. 

    Parameters
    ----------
    sgcmc_project_data_dictionary : dict
        Dictionary of run data from a full MC project (generated from propagate_gcmc.py propagation_project_parser)
    
    Returns
    -------
    Matplotlib.pyplot figure object
        Diagnostic plots.
    """
    # First, integrate the semi grand canonical free energy for all runs:
    integrated_data = mc.full_project_integration(sgcmc_project_data_dictionary)

    # Make a 4x3 grid of subplots
    fig, axs = plt.subplots(4, 3)
    fig.set_size_inches(22, 30)

    # Plot chemical potential vs composition for the constant temperature runs
    for run in integrated_data["T_const"]:
        axs[0, 0].scatter(
            run["<comp(a)>"],
            run["param_chem_pot(a)"],
            s=3,
            color="k",
            label="Constant T",
        )

    # Plot integrated sgc free energy vs temperature for the heating and cooling runs
    for run in integrated_data["heating"]:
        axs[0, 1].scatter(
            run["T"],
            run["integrated_potential_energy"],
            s=3,
            color="r",
            label="Heating",
        )
    for run in integrated_data["cooling"]:
        axs[0, 1].scatter(
            run["T"],
            run["integrated_potential_energy"],
            s=3,
            color="b",
            label="Cooling",
        )

    # Plot heat capacity vs temperature for the heating and cooling runs. Also plot a vertical line at the temperature with the maximum heat capacity.
    for run in integrated_data["heating"]:
        axs[1, 0].scatter(
            run["T"], run["heat_capacity"], s=3, color="r", label="Heating"
        )
        axs[1, 0].axvline(
            x=run["T"][np.argmax(run["heat_capacity"])],
            color="r",
            linestyle=":",
            label="Max Heat Capacity",
        )
    for run in integrated_data["cooling"]:
        axs[1, 0].scatter(
            run["T"], run["heat_capacity"], s=3, color="b", label="Cooling"
        )
        axs[1, 0].axvline(
            x=run["T"][np.argmax(run["heat_capacity"])],
            color="b",
            linestyle=":",
            label="Max Heat Capacity",
        )

    # Plot rainplot
    for constant_temperature in integrated_data["T_const"]:
        axs[1, 1].scatter(
            constant_temperature["<comp(a)>"],
            constant_temperature["T"],
            s=3,
            color="k",
            label="Constant T",
        )
    for run in integrated_data["heating"]:
        axs[1, 1].scatter(run["<comp(a)>"], run["T"], s=3, color="r", label="Heating")
    for run in integrated_data["cooling"]:
        axs[1, 1].scatter(
            run["<comp(a)>"], run["T"], s=3, color="b", alpha=0.5, label="Cooling"
        )
    # Also, plot the chemical potential as a number in a text box next to the highest temperature point.
    for run in integrated_data["heating"]:
        max_temp_index = np.argmax(run["T"])
        axs[1, 1].text(
            run["<comp(a)>"][max_temp_index],
            run["T"][max_temp_index],
            run["param_chem_pot(a)"][0],
            fontsize=7,
            rotation=90,
        )
    for run in integrated_data["cooling"]:
        max_temp_index = np.argmax(run["T"])
        axs[1, 1].text(
            run["<comp(a)>"][max_temp_index],
            run["T"][max_temp_index],
            run["param_chem_pot(a)"][0],
            fontsize=7,
            rotation=90,
        )

    # Plot gibbs vs composition for the constant temperature runs
    for run in integrated_data["T_const"]:
        axs[0, 2].scatter(
            run["<comp(a)>"], run["gibbs"], s=3, color="k", label="Constant T"
        )

    # Plot semi grand canonical free energy vs chemical potential for the constant temperature runs
    for run in integrated_data["T_const"]:
        axs[1, 2].scatter(
            run["param_chem_pot(a)"],
            run["integrated_potential_energy"],
            s=3,
            color="k",
            label="Constant T",
        )

    # Plot the gibbs - formation energy divided by kT vs composition for the constant temperature runs
    k = 8.617333262e-5  # eV/K
    for run in integrated_data["T_const"]:
        mc_entropy = (
            -1
            * np.array(np.array(run["gibbs"]) - np.array(run["<formation_energy>"]))
            / (k * np.array(run["T"]))
        )
        ideal_mixing_entropy = -1 * (
            np.array(run["<comp(a)>"]) * np.log(np.array(run["<comp(a)>"]))
            + (1 - np.array(run["<comp(a)>"])) * np.log(1 - np.array(run["<comp(a)>"]))
        )
        axs[2, 0].scatter(
            run["<comp(a)>"], mc_entropy, s=3, color="k", label="MC Entropy"
        )
        axs[2, 0].scatter(
            run["<comp(a)>"],
            ideal_mixing_entropy,
            s=3,
            color="orange",
            label="Ideal Solution Entropy",
        )

    # Plot the formation energy vs chemical potential for the constant temperature runs
    for run in integrated_data["T_const"]:
        axs[2, 1].scatter(
            run["param_chem_pot(a)"],
            run["<formation_energy>"],
            s=3,
            color="k",
            label="Constant T",
        )

    # Plot the potential energy vs chemical potential for the constant temperature runs
    for run in integrated_data["T_const"]:
        axs[2, 2].scatter(
            run["param_chem_pot(a)"],
            run["<potential_energy>"],
            s=3,
            color="k",
            label="Constant T",
        )

    # For all cooling runs, find the composition and formation energy at the lowest temperature. Store the composition and formation energy in a list.
    # Do the same for all heating runs.
    # If there is more than one cooling run, find the lower convex hull of the cooling compositions and formation energies.
    # If there is more than one heating run, find the lower convex hull of the heating compositions and formation energies.
    # Plot all this data on the same plot.
    cooling_compositions = []
    cooling_formation_energies = []
    for cooling_run in integrated_data["cooling"]:
        cooling_compositions.append(
            cooling_run["<comp(a)>"][np.argmin(cooling_run["T"])]
        )
        cooling_formation_energies.append(
            cooling_run["<formation_energy>"][np.argmin(cooling_run["T"])]
        )
    cooling_compositions = np.array(cooling_compositions)
    cooling_formation_energies = np.array(cooling_formation_energies)
    axs[3, 0].scatter(
        cooling_compositions,
        cooling_formation_energies,
        marker="1",
        color="b",
        label="Cooling",
    )
    if len(integrated_data["cooling"]) > 1:
        cooling_hull = thull.full_hull(
            compositions=cooling_compositions.reshape(-1, 1),
            energies=cooling_formation_energies,
        )
        cooling_hull_vertices, _ = thull.lower_hull(cooling_hull)

        hull_vertices_argsort = np.argsort(cooling_compositions[cooling_hull_vertices])

        axs[3, 0].plot(
            cooling_compositions[cooling_hull_vertices][hull_vertices_argsort],
            cooling_formation_energies[cooling_hull_vertices][hull_vertices_argsort],
            color="b",
            label="Cooling Hull",
        )

    heating_compositions = []
    heating_formation_energies = []
    for heating_run in integrated_data["heating"]:
        heating_compositions.append(
            heating_run["<comp(a)>"][np.argmin(heating_run["T"])]
        )
        heating_formation_energies.append(
            heating_run["<formation_energy>"][np.argmin(heating_run["T"])]
        )
    heating_compositions = np.array(heating_compositions)
    heating_formation_energies = np.array(heating_formation_energies)
    axs[3, 0].scatter(
        heating_compositions,
        heating_formation_energies,
        marker="2",
        color="r",
        label="Heating",
    )
    if len(integrated_data["heating"]) > 1:
        heating_hull = thull.full_hull(
            compositions=heating_compositions.reshape(-1, 1),
            energies=heating_formation_energies,
        )
        heating_hull_vertices, _ = thull.lower_hull(heating_hull)
        heating_vertices_argsort = np.argsort(
            heating_compositions[heating_hull_vertices]
        )
        axs[3, 0].plot(
            heating_compositions[heating_hull_vertices][heating_vertices_argsort],
            heating_formation_energies[heating_hull_vertices][heating_vertices_argsort],
            color="r",
            label="Heating Hull",
        )

    # Set labels
    axs[0, 0].set_xlabel("Composition", fontsize=18)
    axs[0, 0].set_ylabel("Chemical Potential", fontsize=18)
    axs[0, 0].legend(fontsize=18)
    axs[0, 1].set_xlabel("Temperature (K)", fontsize=18)
    axs[0, 1].set_ylabel("Semi Grand Canonical Free Energy", fontsize=18)
    axs[0, 1].legend(fontsize=18)
    axs[1, 0].set_xlabel("Temperature (K)", fontsize=18)
    axs[1, 0].set_ylabel("Heat Capacity", fontsize=18)
    axs[1, 0].legend(fontsize=18)
    axs[1, 1].set_xlabel("Composition", fontsize=18)
    axs[1, 1].set_ylabel("Temperature (K)", fontsize=18)
    axs[1, 1].legend(fontsize=18)
    axs[0, 2].set_xlabel("Composition", fontsize=18)
    axs[0, 2].set_ylabel("Gibbs Free Energy", fontsize=18)
    axs[0, 2].legend(fontsize=18)
    axs[1, 2].set_xlabel("Chemical Potential", fontsize=18)
    axs[1, 2].set_ylabel("Semi Grand Canonical Free Energy", fontsize=18)
    axs[1, 2].legend(fontsize=18)
    axs[2, 0].set_xlabel("Composition", fontsize=18)
    axs[2, 0].set_ylabel("MC Entropy and Ideal Entropy", fontsize=18)
    axs[2, 0].legend(fontsize=18)
    axs[2, 1].set_xlabel("Chemical Potential", fontsize=18)
    axs[2, 1].set_ylabel("Formation Energy", fontsize=18)
    axs[2, 1].legend(fontsize=18)
    axs[2, 2].set_xlabel("Chemical Potential", fontsize=18)
    axs[2, 2].set_ylabel("Potential Energy", fontsize=18)
    axs[2, 2].legend(fontsize=18)
    axs[3, 0].set_xlabel("Composition", fontsize=18)
    axs[3, 0].set_ylabel("Formation Energy", fontsize=18)
    axs[3, 0].legend(fontsize=18)
    return fig

