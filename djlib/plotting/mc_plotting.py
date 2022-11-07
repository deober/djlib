import matplotlib.pyplot as plt
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
    mc_runs_directory: str, show_labels: bool = False
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
                plt.scatter(composition, temperature)

    if show_labels:
        plt.legend(labels)
    plt.xlabel("Composition", fontsize=18)
    plt.ylabel("Temperature (K)", fontsize=18)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10)
    return fig
