from __future__ import annotations

import matplotlib
import djlib.djlib as dj
import json
import os
import numpy as np
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, LassoLarsCV
from sklearn.metrics import mean_squared_error
import csv
from glob import glob
import pickle
from string import Template
import arviz as ar
import thermocore.geometry.hull as thull
import pathlib
from warnings import warn
from typing import Callable, List, Tuple, Sequence
import stan
from sklearn.decomposition import PCA
from sklearn.linear_model import BayesianRidge


def boltzmann(hulldist, coef, beta, temperature):
    return coef * np.exp(-(beta * hulldist) / temperature)


def weighted_feature_and_target_arrays(
    corr: np.ndarray,
    formation_energy: np.ndarray,
    hulldists: np.ndarray,
    A: float = 1,
    B: float = 1,
    kT: float = 1,
):
    """Given boltzmann weighting parameters, returns a weighted feature matrix and corresponding target vector.

    Parameters
    ----------
    A : float
        Boltzmann weighting coefficient.
    B : float
        Boltzmann weighting coefficient.
    kT : float
        Boltzmann weighting coefficient.
    corr : numpy.ndarray
        CASM nxk correlation matrix, n = number of configurations, k = number of ECI.
    formation_energy : numpy.ndarray
        n-dimensional vector of formation energies.
    hulldists : numpy.ndarray
        n-dimensional vector of hull distances.

    Returns
    -------
    x_prime : numpy.ndarray
        Weighted CASM nxk correlation matrix, n = number of configurations, k = number of ECI.
    y_prime : numpy.ndarray
        Weighted n-dimensional vector of formation energies.
    """

    weight = np.identity(formation_energy.shape[0])
    for config_index in range(formation_energy.shape[0]):
        weight[config_index, config_index] = boltzmann(
            hulldist=hulldists[config_index], coef=A, beta=B, temperature=kT
        )
    l = np.linalg.cholesky(weight)
    y_prime = np.ravel(l @ formation_energy.reshape(-1, 1))
    x_prime = l @ corr
    return (x_prime, y_prime)


def general_weighted_feature_and_target_arrays(
    feature_matrix: np.ndarray, target_array: np.ndarray, weight_matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """A more abstracted version of weighted_feature_and_target_arrays, where the weight matrix is passed in directly.

    Parameters
    ----------
    feature_matrix : np.ndarray
        nxk feature matrix, n = number of configurations, k = number of features.
    target_array : np.ndarray
        n-dimensional target vector.
    weight_matrix : np.ndarray
        n x n weight matrix.

    Returns
    -------
    Tuple[np.ndarray,np.ndarray]
        Weighted feature matrix and target vector.
    """
    l = np.linalg.cholesky(weight_matrix)
    y_prime = np.ravel(l @ target_array.reshape(-1, 1))
    x_prime = l @ feature_matrix
    return (x_prime, y_prime)


def lower_hull(hull: ConvexHull, energy_index=-2) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the lower convex hull (with respect to energy direction) given  complete convex hull.
    Parameters
    ----------
        hull : scipy.spatial.ConvexHull
            Complete convex hull object.
        energy_index : int
            index of energy dimension of points within 'hull'.
    Returns
    -------
        lower_hull_simplices : numpy.ndarray of ints, shape (nfacets, ndim)
            indices of the facets forming the lower convex hull.

        lower_hull_vertices : numpy.ndarray of ints, shape (nvertices,)
            Indices of the vertices forming the vertices of the lower convex hull.
    """
    warn(
        "This function is deprecated. Use thermocore.geometry.hull.lower_hull instead."
    )
    # Note: energy_index is used on the "hull.equations" output, which has a scalar offset.
    # (If your energy is the last column of "points", either use the direct index, or use "-2". Using "-1" as energy_index will not give the proper lower hull.)
    lower_hull_simplices = hull.simplices[hull.equations[:, energy_index] < 0]
    lower_hull_vertices = np.unique(np.ravel(lower_hull_simplices))
    return (lower_hull_vertices, lower_hull_simplices)


def checkhull(
    hull_comps: np.ndarray,
    hull_energies: np.ndarray,
    test_comp: np.ndarray,
    test_energy: np.ndarray,
) -> np.ndarray:
    """Calculates hull distance for each configuration
    Parameters
    ----------
        hull_comps : numpy.ndarray shape(number_configurations, number_composition_axes)
            Coordinate in composition space for all configurations.
        hull_energies : numpy.ndarray shape(number_configurations,)
            Formation energy for each configuration.
    Returns
    -------
        hull_dist : numpy.ndarray shape(number_of_configurations,)
            The distance of each configuration from the convex hull described by the current cluster expansion of DFT-determined convex hull configurations.

    """
    # Need to reshape to column vector to work properly.
    # Test comp should also be a column vector.
    test_energy = np.reshape(test_energy, (-1, 1))
    # Fit linear grid
    interp_hull = griddata(hull_comps, hull_energies, test_comp, method="linear")

    # Check if the test_energy points are above or below the hull
    hull_dist = test_energy - interp_hull
    return np.ravel(np.array(hull_dist))


def run_lassocv(corr: np.ndarray, formation_energy: np.ndarray) -> np.ndarray:
    reg = LassoCV(fit_intercept=False, n_jobs=4, max_iter=50000).fit(
        corr, formation_energy
    )
    eci = reg.coef_
    return eci


def find_proposed_ground_states(
    corr: np.ndarray,
    comp: np.ndarray,
    formation_energy: np.ndarray,
    eci_set: np.ndarray,
) -> np.ndarray:
    """Collects indices of configurations that fall 'below the cluster expansion prediction of DFT-determined hull configurations'.

    Parameters
    ----------
    corr: numpy.ndarray
        CASM nxk correlation matrix, n = number of configurations, k = number of ECI. Order matters.

    comp: numpy.ndarray
        nxl matrix of compositions, n = number of configurations, l = number of varying species

    formation_energy: numpy.ndarray
        Vector of n DFT-calculated formation energies. Order matters.

    eci_set: numpy.ndarray
        mxk matrix, m = number of monte carlo sampled ECI sets, k = number of ECI.


    Returns
    -------
    proposed_ground_state_indices: numpy.ndarray
        Vector of indices denoting configurations which appeared below the DFT hull across all of the Monte Carlo steps.
    """

    # Read data from casm query json output
    # data = read_corr_comp_formation(corr_comp_energy_file)
    # corr = data["corr"]
    # formation_energy = data["formation_energy"]
    # comp = data["comp"]

    proposed_ground_states_indices = np.array([])

    # Dealing with compatibility: Different descriptors for un-calculated formation energy (1.1.2->{}, 1.2-> null (i.e. None))
    uncalculated_energy_descriptor = None
    if {} in formation_energy:
        uncalculated_energy_descriptor = {}

    # Downsampling only the calculated configs:
    downsample_selection = formation_energy != uncalculated_energy_descriptor
    corr_calculated = corr[downsample_selection]
    formation_energy_calculated = formation_energy[downsample_selection]
    comp_calculated = comp[downsample_selection]

    # Find and store correlations for DFT-predicted hull states:
    points = np.zeros(
        (formation_energy_calculated.shape[0], comp_calculated.shape[1] + 1)
    )
    points[:, 0:-1] = comp_calculated
    points[:, -1] = formation_energy_calculated
    hull = ConvexHull(points)
    dft_hull_config_indices, dft_hull_simplices = thull.lower_hull(hull)
    dft_hull_corr = corr_calculated[dft_hull_config_indices]
    dft_hull_vertices = hull.points[dft_hull_config_indices]

    # Temporarily removed- requires too much memory
    # sampled_hulldist = []

    # Collect proposed ground state indices
    for current_eci in eci_set:
        full_predicted_energy = np.matmul(corr, current_eci)

        # Predict energies of DFT-determined hull configs using current ECI selection
        dft_hull_clex_predict_energies = np.matmul(dft_hull_corr, current_eci)

        hulldist = checkhull(
            dft_hull_vertices[:, 0:-1],
            dft_hull_clex_predict_energies,
            comp,
            full_predicted_energy,
        )

        # Temporarily removed. Requires too much memory
        # sampled_hulldist.append(hulldist)

        # Find configurations that break the convex hull
        below_hull_selection = hulldist < 0
        below_hull_indices = np.ravel(np.array(below_hull_selection.nonzero()))
        proposed_ground_states_indices = np.concatenate(
            (proposed_ground_states_indices, below_hull_indices)
        )
    return proposed_ground_states_indices


def stan_model_formatter(
    eci_variance_is_fixed: bool,
    model_variance_is_fixed: bool,
    eci_parameters: list,
    model_parameters: list,
    start_stop_indices: list,
) -> str:
    """Formats the Stan model for use in the Stan Fit class.
    Parameters:
    -----------
    eci_variance_is_fixed: bool
        If True, the function will not create any eci variance variables.
        If False, the function will create eci variance variables.
    model_variance_is_fixed: bool
        If True, the function will not create any model variance variables.
        If False, the function will create model variance variables.
    eci_parameters: list
        A list of ECI parameters; each element is a string.
    model_parameters: list
        A list of model parameters; each element is a string.
    start_stop_indices: list
        list start and stop indices to divide configurations into groups with different model variances.
        Each element is a list of two integers.

    Returns:
    --------
    stan_model_template: str
        Formatted Stan model template, ready to be passed to the Stan Fit class.


    Notes:
    ------
    eci_parameters:
        If eci_variance_is_fixed == True, each element should describe the prior distribution for ECI. If only one element is provided, it will be used as the prior for all ECI.
        If eci_variance_is_fixed == False, each element should describe the hyperdistribution for the ECI variance.
        If only one element is provided, it will be used as the prior for all ECI.
        Otherwise, there should be one element for each ECI.
        example: eci_variance_is_fixed==True: ['~ normal(0, #)']
        example: eci_variance_is_fixed==False: ['~ gamma(1, #)']
    model_parameters:
        If model_variance_is_fixed == True, each element should be a string of a number quantifying the model variance.
        If model_variance_is_fixed == False, each element should be a string of a hyperparameter for the model variance.
        If only one element is provided, it will be used as the prior for all model variances.
        If more than one element is provided, the user must specify start-stop indices, denoting the range of energies to use for each model variance parameter.


    """

    assert all(type(x) == str for x in eci_parameters)
    assert all(type(x) == str for x in model_parameters)

    # Template parameters section
    parameters_string = "\t" + "vector[K] eci;\n"
    if eci_variance_is_fixed == False:
        parameters_string += "\t" + "vector<lower=0>[K] eci_variance;\n"
    if model_variance_is_fixed == False:
        parameters_string += "\t" + "vector<lower=0>[n_configs] sigma;"

    # Template model section
    model_string = ""
    optimize_eci_miultiply = False
    optimize_model_multiply = False
    if len(eci_parameters) == 1:
        # Assign ECI in a for loop with the same prior.
        optimize_eci_miultiply = True
    if len(model_parameters) == 1:
        # Use a single model variance for all configurations, allowing for a matrix multiply
        optimize_model_multiply = True

    if optimize_eci_miultiply:
        # If all ECI priors are the same
        model_string += """for (k in 1:K){\n"""
        if eci_variance_is_fixed:
            model_string += "\t\t" + "eci[k] " + eci_parameters[0] + ";\n\t}\n"
        else:
            model_string += "\t\t" + "eci_variance[k] " + eci_parameters[0] + ";\n"
            model_string += "\t\t" + "eci[k] ~ normal(0,eci_variance[k]);\n\t}\n"
    else:
        # If ECI priors are different
        if eci_variance_is_fixed:
            for i, parameter in enumerate(eci_parameters):
                model_string += "\t" + "eci[{i}] ".format(i=i + 1) + parameter + ";\n"
        else:
            for i, parameter in enumerate(eci_parameters):
                model_string += (
                    "\t\t" + "eci_variance[{i}] ".format(i=i + 1) + parameter + ";\n"
                )
                model_string += (
                    "\t\t"
                    + "eci[{i}] ~ normal(0,eci_variance[{i}]);\n\t}\n".format(i=i + 1)
                )

    if optimize_model_multiply:
        # If there is only one model variance sigma^2
        if model_variance_is_fixed:
            model_string += "\t" + "real sigma = " + str(model_parameters[0]) + ";\n"
        else:
            model_string += "\t" + "sigma " + model_parameters[0] + ";\n"
        model_string += "\t" + "energies ~ normal(corr * eci, sigma);\n"
    else:
        # If there are multiple model variances
        if model_variance_is_fixed:
            for sigma_index, start_stop in enumerate(start_stop_indices):
                model_string += (
                    "energies[{start}:{stop}] ~ normal(corr[{start}:{stop}]*eci, {sigma}) ".format(
                        start=start_stop[0],
                        stop=start_stop[1],
                        sigma=model_parameters[sigma_index],
                    )
                    + ";\n"
                )
        else:
            for sigma_index, model_param in enumerate(model_parameters):
                model_string += (
                    "sigma[{sigma_index}] ".format(sigma_index=sigma_index + 1)
                    + model_param
                    + ";\n"
                )
            for sigma_index, start_stop in enumerate(start_stop_indices):
                model_string += (
                    "energies[{start}:{stop}] ~ normal(corr[{start}:{stop}}]*eci, sigma[{sigma_index}]) ".format(
                        start=start_stop[0],
                        stop=start_stop[1],
                        sigma_index=sigma_index + 1,
                    )
                    + ";\n"
                )
    # Load template from templates directory
    clex_lib_dir = pathlib.Path(__file__).parent.resolve()
    templates = os.path.join(clex_lib_dir, "../templates")

    with open(os.path.join(templates, "stan_model_template.txt"), "r") as f:
        template = Template(f.read())
    return template.substitute(
        formatted_parameters=parameters_string, formatted_model=model_string
    )


def format_stan_executable_script(
    data_file: str,
    stan_model_file: str,
    eci_output_file: str,
    num_samples: int,
    energy_tag="formation_energy",
    num_chains=4,
) -> str:
    """
    Parameters
    ----------
    data_file: string
        Path to casm query output containing correlations, compositions and formation energies
    stan_model_file: string
        Path to text file containing stan model specifics
    eci_output_file: string
        Path to file where Stan will write the sampled ECI
    num_samples: int
        Number of samples in the stan monte carlo process
    num_chains: int
        Number of simultaneous markov chains

    Returns
    -------
    executable_file: str
        Executable python stan command, formatted as a string
    """
    template = Template(
        """
import pickle 
import stan
import djlib.djlib as dj
import djlib.clex.clex as cl
import numpy as np
import time

# Time the process
start_time = time.time()

# Load Casm Data
data_file = '$data_file'
data = dj.casm_query_reader(data_file)
corr = np.squeeze(np.array(data["corr"]))
corr = tuple(map(tuple, corr))
energies = tuple(data['$energy_tag'])

#Format Stan Model
n_configs = len(energies)
k = len(corr[0])
ce_data = {"K": k, "n_configs": n_configs, "corr": corr, "energies": energies}
with open('$stan_model_file', 'r') as f:
    ce_model = f.read()
posterior = stan.build(ce_model, data=ce_data)

# Run MCMC
fit = posterior.sample(num_chains=$num_chains, num_samples=$num_samples)

# Write results
with open('$eci_output_file', "wb") as f:
    pickle.dump(fit, f, protocol=pickle.HIGHEST_PROTOCOL)

# Print execution time
end_time = time.time()
print("Run time is: ", end_time - start_time, " seconds")
"""
    )
    executable_file = template.substitute(
        data_file=data_file,
        stan_model_file=stan_model_file,
        num_chains=int(num_chains),
        num_samples=int(num_samples),
        eci_output_file=eci_output_file,
        energy_tag=energy_tag,
    )
    return executable_file


def bayes_train_test_analysis(run_dir: str) -> dict:
    """Calculates training and testing rms for cross validated fitting.

    Parameters:
    -----------
    run_dir: str
        Path to single set of partitioned data and the corresponding results.

    Returns:
    --------
    dict{
        training_rms: numpy.ndarray
            RMS values for each ECI vector compared against training data.
        testing_rms: numpy.ndarray
            RMS values for each ECI vector compared against testing data.
        training_set: numpy.ndarray
            Indices of configurations partitioned as training data from the original dataset.
        test_set:
            Indices of configurations partitioned as testing data from the original dataset.
        eci_variance_args: list
            Arguments for ECI prior distribution. Currently assumes Gamma distribution with two arguments: first argument is the gamma shape parameter, second argument is the gamma shrinkage parameter.
        data_source: str
            Path to the original data file used to generate the training and testing datasets.
        random_seed: int
            Value used for the random seed generator.
    }
    """

    # Load run information to access query data file and train / test indices
    with open(os.path.join(run_dir, "run_info.json"), "r") as f:
        run_info = json.load(f)
    with open(run_info["data_source"], "r") as f:
        query_data = np.array(json.load(f))

    # Load training and testing data
    testing_data = dj.casm_query_reader(
        casm_query_json_data=query_data[run_info["test_set"]]
    )
    training_data = dj.casm_query_reader(
        casm_query_json_data=query_data[run_info["training_set"]]
    )

    training_corr = np.squeeze(np.array(training_data["corr"]))
    training_energies = np.array(training_data["formation_energy"])

    testing_corr = np.squeeze(np.array(testing_data["corr"]))
    testing_energies = np.array(testing_data["formation_energy"])

    with open(os.path.join(run_dir, "results.pkl"), "rb") as f:
        eci = pickle.load(f)["eci"]

    train_data_predict = np.transpose(training_corr @ eci)
    test_data_predict = np.transpose(testing_corr @ eci)

    # Calculate RMS on testing data using only the mean ECI values
    eci_mean = np.mean(eci, axis=1)
    eci_mean_prediction = testing_corr @ eci_mean
    eci_mean_rms = np.sqrt(mean_squared_error(testing_energies, eci_mean_prediction))

    # Calculate rms for training and testing datasets
    training_rms = np.array(
        [
            np.sqrt(mean_squared_error(training_energies, train_data_predict[i]))
            for i in range(train_data_predict.shape[0])
        ]
    )
    testing_rms = np.array(
        [
            np.sqrt(mean_squared_error(testing_energies, test_data_predict[i]))
            for i in range(test_data_predict.shape[0])
        ]
    )

    # Collect parameters unique to this kfold run
    with open(os.path.join(run_dir, "run_info.json"), "r") as f:
        run_info = json.load(f)

    # Collect all run information in a single dictionary and return.
    kfold_data = {}
    kfold_data.update(
        {
            "training_rms": training_rms,
            "testing_rms": testing_rms,
            "eci_mean_testing_rms": eci_mean_rms,
            "eci_means": eci_mean,
        }
    )
    kfold_data.update(run_info)
    return kfold_data


def kfold_analysis(kfold_dir: str) -> dict:
    """Collects statistics across k fits.

    Parameters:
    -----------
    kfold_dir: str
        Path to directory containing the k bayesian calibration runs as subdirectories.

    Returns:
    --------
    train_rms: np.array
        Average training rms value for each of the k fitting runs.
    test_rms: np.array
        Average testing rms value for each of the k fitting runs.
    """
    train_rms_values = []
    test_rms_values = []
    eci_mean_testing_rms = []
    eci_mean = None
    invalid_rhat_tally = []

    kfold_subdirs = glob(os.path.join(kfold_dir, "*"))
    for run_dir in kfold_subdirs:
        if os.path.isdir(run_dir):
            if os.path.isfile(os.path.join(run_dir, "results.pkl")):

                run_data = bayes_train_test_analysis(run_dir)
                train_rms_values.append(np.mean(run_data["training_rms"]))
                test_rms_values.append(np.mean(run_data["testing_rms"]))
                eci_mean_testing_rms.append(run_data["eci_mean_testing_rms"])
                if type(eci_mean) == type(None):
                    eci_mean = run_data["eci_means"]
                else:
                    eci_mean = (eci_mean + run_data["eci_means"]) / 2
                with open(os.path.join(run_dir, "results.pkl"), "rb") as f:
                    results = pickle.load(f)
                    rhat_check_results = rhat_check(results)
                    invalid_rhat_tally.append(rhat_check_results["total_count"])
                with open(os.path.join(run_dir, "run_info.json"), "r") as f:
                    run_info = json.load(f)
                    run_info["rhat_summary"] = rhat_check_results
                with open(os.path.join(run_dir, "run_info.json"), "w") as f:
                    json.dump(run_info, f)
    eci_mean_testing_rms = np.mean(np.array(eci_mean_testing_rms), axis=0)
    return {
        "train_rms": train_rms_values,
        "test_rms": test_rms_values,
        "eci_mean_testing_rms": eci_mean_testing_rms,
        "kfold_avg_eci_mean": eci_mean,
        "invalid_rhat_tally": invalid_rhat_tally,
    }


def plot_eci_uncertainty(eci: np.ndarray, title=False) -> plt.figure:
    """
    Parameters
    ----------
    eci: numpy.array
        nxm matrix of ECI values: m sets of n ECI

    Returns
    -------
    fig: matplotlib.pyplot figure
    """
    print(
        "clex.clex.plot_eci_uncertainty() is deprecated. Use plotting.clex_plotting.plot_eci_uncertainty() instead."
    )
    stats = np.array([[np.mean(row), np.std(row)] for row in eci])
    print(stats.shape)
    means = stats[:, 0]
    std = stats[:, 1]
    index = list(range(eci.shape[0]))

    plt.scatter(index, means, color="xkcd:crimson", label="Means")
    plt.errorbar(index, means, std, ls="none", color="k", label="Stddev")

    plt.xlabel("ECI index (arbitrary)", fontsize=22)
    plt.ylabel("ECI magnitude (eV)", fontsize=22)
    if title:
        plt.title(title, fontsize=30)
    else:
        plt.title("STAN ECI", fontsize=30)
        plt.legend(fontsize=21)
    fig = plt.gcf()
    fig.set_size_inches(15, 10)

    return fig


def general_binary_convex_hull_plotter(
    composition: np.ndarray, true_energies: np.ndarray, predicted_energies=[None]
) -> matplotlib.figure.Figure:
    """Plots a 2D convex hull for any 2D dataset. Can optionally include predicted energies to compare true and predicted formation energies and conved hulls.

    Parameters:
    -----------
    composition: numpy.ndarray
        Vector of composition values, same length as true_energies.
    true_energies: numpy.ndarray
        Vector of formation energies. Required.
    predicted_energies: numpy.ndarray
        None by default. If a vector of energies are provided, it must be the same length as composition. RMSE score will be reported.
    """
    print(
        "clex.clex.general_binary_convex_hull_plotter() is deprecated. Use plotting.hull_plotting.general_binary_convex_hull_plotter() instead."
    )
    predicted_color = "red"
    predicted_label = "Predicted Energies"

    plt.scatter(
        composition, true_energies, color="k", marker="x", label='"True" Energies'
    )

    dft_hull = ConvexHull(
        np.hstack((composition.reshape(-1, 1), np.reshape(true_energies, (-1, 1))))
    )

    if any(predicted_energies):
        predicted_hull = ConvexHull(
            np.hstack(
                (composition.reshape(-1, 1), np.reshape(predicted_energies, (-1, 1)))
            )
        )

    dft_lower_hull_vertices = thull.lower_hull(dft_hull)[0]
    if any(predicted_energies):
        predicted_lower_hull_vertices = thull.lower_hull(predicted_hull)[0]

    dft_lower_hull = dj.column_sort(dft_hull.points[dft_lower_hull_vertices], 0)

    if any(predicted_energies):
        predicted_lower_hull = dj.column_sort(
            predicted_hull.points[predicted_lower_hull_vertices], 0
        )

    plt.plot(
        dft_lower_hull[:, 0], dft_lower_hull[:, 1], marker="D", markersize=15, color="k"
    )
    if any(predicted_energies):
        plt.plot(
            predicted_lower_hull[:, 0],
            predicted_lower_hull[:, 1],
            marker="D",
            markersize=10,
            color=predicted_color,
        )
    if any(predicted_energies):
        plt.scatter(
            composition,
            predicted_energies,
            color="red",
            marker="x",
            label=predicted_label,
        )

        rmse = np.sqrt(mean_squared_error(true_energies, predicted_energies))
        plt.text(
            min(composition),
            0.9 * min(np.concatenate((true_energies, predicted_energies))),
            "RMSE: " + str(rmse) + " eV",
            fontsize=21,
        )

    plt.xlabel("Composition X", fontsize=21)
    plt.ylabel("Formation Energy (eV)", fontsize=21)
    plt.legend(fontsize=21)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    fig = plt.gcf()
    fig.set_size_inches(19, 14)
    return fig


def rhat_check(posterior_fit_object: stan.fit.Fit, rhat_tolerance=1.05) -> dict:
    """Counts the number of stan parameters with rha values greater than the provided rhat_tolerance.

    Parameters:
    -----------
    posterior_fit_object: stan.fit.Fit
        Posterior fit object from stan.
    rhat_tolerance: float
        Value to compare rhat metrics against. Default is 1.05




    Returns:
    --------
    rhat_summary: dict{
        total_count: int
            total number of parameters above rhat_tolerance.
        *_count: int
            Number of parameters in parameter vector * which are above the rhat tolerance. * is representative of a posterior parameter vector.
    }

    Notes:
    ------
    The rhat metric describes sampling convergence across the posterior distribution. Rhat values greater than 1.05 indicate that those parameters are not reasonably converged. Rhat below 1.05 is okay, but closer to 1.0 indicates better convergence.
    """
    rhat = ar.rhat(posterior_fit_object)
    rhat_keys = list(rhat.keys())
    total_count = 0
    rhat_summary = {}
    for key in rhat_keys:
        tally = np.count_nonzero(rhat[key] > rhat_tolerance)
        rhat_summary[key + "_count"] = tally
        total_count += tally
    rhat_summary["total_count"] = total_count
    return rhat_summary


def calculate_hulldist_corr(
    corr: np.ndarray, comp: np.ndarray, formation_energy: np.ndarray
) -> np.ndarray:
    """Calculated the effective correlations to predict hull distance instead of absolute formation energy.
    Parameters:
    -----------
    corr: np.array
        nxk correlation matrix, where n is the number of configurations and k is the number of ECI.
    comp: np.array
        nxc matrix of compositions, where n is the number of configurations and c is the number of composition axes.
    formation_energy: np.array
        nx1 matrix of formation energies.

    Returns:
    --------
    hulldist_corr: np.array
        nxk matrix of effective correlations describing hull distance instead of absolute formation energy. n is the number of configurations and k is the number of ECI.
    """

    # Build convex hull
    points = np.hstack((comp, formation_energy.reshape(-1, 1)))
    hull = ConvexHull(points)

    # Get convex hull simplices
    lower_vertices, lower_simplices = thull.lower_hull(hull)

    hulldist_corr = np.zeros(corr.shape)

    for config_index in list(range(corr.shape[0])):

        # Find the simplex that contains the current configuration's composition, and find the hull energy for that composition
        relevant_simplex_index, hull_energy = thull.lower_hull_simplex_containing(
            compositions=comp[config_index].reshape(1, -1),
            convex_hull=hull,
            lower_hull_simplex_indices=lower_simplices,
        )

        relevant_simplex_index = relevant_simplex_index[0]

        # Find vectors defining the corners of the simplex which contains the curent configuration's composition.
        simplex_corners = comp[hull.simplices[relevant_simplex_index]]
        interior_point = np.array(comp[config_index]).reshape(1, -1)

        # Enforce that the sum of weights is equal to 1.
        simplex_corners = np.hstack(
            (simplex_corners, np.ones((simplex_corners.shape[0], 1)))
        )
        interior_point = np.hstack((interior_point, np.ones((1, 1))))

        # Project the interior point onto the vectors that define the simplex corners.
        weights = interior_point @ np.linalg.pinv(simplex_corners)

        # Form the hull distance correlations by taking a linear combination of simplex corners.

        hulldist_corr[config_index] = (
            corr[config_index] - weights @ corr[hull.simplices[relevant_simplex_index]]
        )

    return hulldist_corr


def variance_mean_ratio_eci_ranking(posterior_eci: np.ndarray) -> np.ndarray:
    """Calculates the variance mean ratio for each ECI and ranks them from highest to lowest variance mean ratio.

    Parameters:
    -----------
    posterior_eci: np.ndarray
        nxk matrix of ECI, where n is the number of posterior samples and k is the number of ECI.

    Returns:
    --------
    eci_ranking: np.ndarray
        Vector of ECI indices ranked from highest to lowest variance mean ratio.
    """
    eci_variance = np.var(posterior_eci, axis=1)
    eci_mean = np.mean(posterior_eci, axis=1)
    eci_vmr = eci_variance / np.abs(eci_mean)
    eci_ranking = np.argsort(eci_vmr)
    return eci_ranking


def principal_component_analysis_eci_ranking(posterior_eci: np.ndarray) -> np.ndarray:
    """Runs Principal Component Analysis on the ECI Posterior distribution, then computes the normalized inverse of the pca explained variance. 
       The PCA contributing the top (explained_variance_tolerance) of the normalized inverse explained variance are summed. This produces a vector of length k (Number of ECI). 
       This vector is then ranked from lowest to highest and returned. 

    Parameters
    ----------
    posterior_eci : np.ndarray
        nxk matrix of ECI, where n is the number of posterior samples and k is the number of ECI.
    explained_variance_tolerance : float
        The fraction of the normalized inverse of the explained variance, which will decide how many PCA components to sum.

    Returns:
    --------
    pca_explained_variance_ranking: np.ndarray
        Vector of ECI indices ranked from lowest to highest magnitude in the PCA component with the lowest explained variance
        from the posterior distribution. The PCA with the lowest explained variance shows the ECI direction which varies the least,
        meaning that the ECI which contribute to this direction are "well pinned" by the data.
    """
    pca = PCA().fit(posterior_eci.T)
    pca_explained_variance_ranking = np.argsort(np.abs(pca.components_[-1]))

    return pca_explained_variance_ranking


def vmr_bayesian_ridge(bayesian_ridge_fit: BayesianRidge.fit()) -> np.ndarray:
    """Sorts the ECI by variance mean ratio, and returns the array of sorted indices. 

    Parameters
    ----------
    bayesian_ridge_fit : sklearn.linear_model.BayesianRidge.fit()
        Bayesian Ridge fit object.
    
    Returns
    -------
    eci_ranking : np.ndarray
        Vector of ECI indices ranked from highest to lowest variance mean ratio.
    """
    br_stddev = np.sqrt(np.diagonal(bayesian_ridge_fit.sigma_))
    br_prune_order_indices = np.argsort(br_stddev / np.abs(bayesian_ridge_fit.coef_))
    return br_prune_order_indices


def iteratively_prune_eci_by_importance_array(
    mean_eci: np.ndarray,
    prune_order_indices: np.ndarray,
    comp,
    corr,
    true_energies,
    fit_each_iteration: bool = False,
    sorter_function: Callable = None,
) -> np.ndarray:
    """Iteratively prunes ECI by importance array.

    Parameters
    ----------
    mean_eci : np.ndarray
        Vector of mean ECI.
    prune_order_indices : np.ndarray
        Vector of ECI indices, in the order that they should be pruned.
    comp : np.ndarray
        nxm matrix of compositions, where n is the number of configurations and m is the number of composition axes.
    corr : np.ndarray
        nxk correlation matrix, where n is the number of configurations and k is the number of ECI.
    true_energies : np.ndarray
        nx1 matrix of true formation energies.
    fit_each_iteration : bool, optional
        Whether to fit the model after each iteration, by default False
    sorter_function : Callable, optional
        Function to sort the ECI by; must accept a BayesianRidge.fit() object, and return an array of indices. None by default. 
    

    Returns
    -------
    pruning_record: dict
        Dictionary containing rmse, ground states, and eci_record for each pruning iteration.
    """

    mask = np.ones(mean_eci.shape[0])
    rmse = []
    ground_state_indices = []
    pruned_eci_record = []
    predicted_energy_record = []
    br_prune_order_indices = np.ones(mean_eci.shape[0])
    bayesian_ridge_corr = corr.copy()
    for index, entry in enumerate(prune_order_indices[0:-3]):
        mask[entry] = 0
        pruned_eci = mean_eci * mask
        pruned_eci_record.append(pruned_eci)
        if fit_each_iteration:
            bayesian_ridge_corr = bayesian_ridge_corr[
                :, br_prune_order_indices.astype(bool)
            ]
            bayesian_ridge_fit = BayesianRidge(fit_intercept=False,).fit(
                bayesian_ridge_corr, true_energies
            )

            predicted_energy = bayesian_ridge_corr @ bayesian_ridge_fit.coef_
            predicted_energy_record.append(predicted_energy)
            br_prune_order_indices = sorter_function(bayesian_ridge_fit)
        else:
            predicted_energy = corr @ pruned_eci
            predicted_energy_record.append(predicted_energy)
        rmse.append(np.sqrt(mean_squared_error(true_energies, predicted_energy)))
        hull = thull.full_hull(compositions=comp, energies=predicted_energy)
        vertices, _ = thull.lower_hull(hull)
        ground_state_indices.append(vertices)
    pruning_record = {
        "rmse": rmse,
        "ground_states": ground_state_indices,
        "eci_record": pruned_eci_record,
        "predicted_energy_record": predicted_energy_record,
    }
    return pruning_record


def calculate_slopes(x_coords: np.ndarray, y_coords: np.ndarray):
    """Calculates the slope for each line segment in a series of connected points.
    
    Parameters:
    -----------
    x_coords: np.ndarray
        Array of x coordinates.
    y_coords: np.ndarray
        Array of y coordinates.
    
    Returns:
    --------
    slopes: np.ndarray
        Array of slopes.
    """

    # sort x_coords and y_coords by x_coords
    x_coords, y_coords = zip(*sorted(zip(x_coords, y_coords)))

    slopes = np.zeros(len(x_coords) - 1)
    for i in range(len(x_coords) - 1):
        slopes[i] = (y_coords[i + 1] - y_coords[i]) / (x_coords[i + 1] - x_coords[i])

    return slopes


def stable_chemical_potential_windows_binary(hull: ConvexHull) -> np.ndarray:
    """Takes a convex hull and returns the stable chemical potential windows of the lower convex hull, excluding the end states.
    
    Parameters
    ----------
    hull: ConvexHull
        A convex hull object.
    
    Returns
    -------
    windows: np.ndarray
        An array of scalars representing the magnitude of the stable chemical potential windows. 
        Returned in order of increasing composition.
        End state chemical potential windows are not included. 
    
    """
    lower_hull_vertices, _ = thull.lower_hull(hull)
    p = hull.points[lower_hull_vertices]
    slopes = calculate_slopes(np.ravel(p[:, 0]), np.ravel(p[:, 1]))
    return np.array([slopes[i + 1] - slopes[i] for i in range(len(slopes) - 1)])


def ranking_by_stable_chemical_potential_window_binary(
    compositions: np.ndarray, energies: np.ndarray
) -> np.ndarray:
    """Calculates the stable chemical potential window for each ECI, and returns the array of sorted indices.

    Parameters
    ----------
    compositions : np.ndarray
        nx1 vector of compositions, where n is the number of configurations and m is the number of composition axes.

    Returns
    -------
    eci_ranking : np.ndarray
        Vector of ECI indices ranked from highest to lowest stable chemical potential window.
    """
    # Calculate the convex hull
    hull = thull.full_hull(compositions, energies)
    lower_hull_vertices, _ = thull.lower_hull(hull)

    # Sort the stable chemical potential windows by composition, and drop the first and last elements
    # (The first and last elements are the end states and have unbounded chemical potential windows)
    sorting_indices = np.argsort(np.ravel(compositions[lower_hull_vertices]))[1:-1]
    lower_hull_vertices = lower_hull_vertices[sorting_indices]

    # Calculate the stable chemical potential windows
    stable_chemical_potential_windows = stable_chemical_potential_windows_binary(hull)

    # Now, sort the lower_hull_vertices by high to low stable chemical potential window
    return lower_hull_vertices[np.argsort(stable_chemical_potential_windows)[::-1]]


def upscale_eci_vector(ecis: np.ndarray, mask: np.ndarray):
    """Intended for cases when the ECI vector is pruned, and a user would like to then upscale the ECI vector back to its original size.

    Parameters
    ----------
    ecis : np.ndarray
        Vector of ECI values.
    mask : np.ndarray
        Vector of Booleans, where True indicates the ECI is included in the vector. Number of True should equal the length of the ECI vector. Otherwise, the function will return an error.
    
    Returns 
    -------
    ecis_upscaled : np.ndarray
        Vector of ECI values, upscaled to the original size.
    """
    if np.sum(mask) != len(ecis):
        raise ValueError(
            "The number of True elements in the mask must equal the length of the ECI vector."
        )
    else:
        # Create a new vector of zeros. The length of the new vector will be the same as the mask.
        # Find the indices of the mask where the value is 1
        # Replace the zeros in the new vector with the ECI values, in the order of the indices of the mask where the value is 1
        indices = np.nonzero(mask)[0]
        ecis_upscaled = np.zeros(len(mask))
        ecis_upscaled[indices] = ecis
        return ecis_upscaled


def ground_state_accuracy_metric(
    composition_predicted: np.ndarray,
    energy_predicted: np.ndarray,
    true_ground_state_indices: np.ndarray,
) -> float:
    """Computes a scalar ground state accuracy metric. The metric varies between [0,1], where 1 is perfect accuracy. The metric is a fraction. 
        The denominator is the sum across the stable chemical potential windows (slopes) for each configuration predicted on the convex hull.
        The numerator is the sum across the stable chemical potential windows (slopes) for each configuration predicted on the convex hull, which are ALSO ground states in DFT data.

    Parameters
    ----------
    composition_predicted : np.ndarray
        nxm matrix of compositions, where n is the number of configurations and m is the number of composition axes.
    energy_predicted : np.ndarray
        nx1 matrix of predicted formation energies.
    true_ground_state_indices : np.ndarray
        nx1 matrix of true ground state indices.

    Returns
    -------
    float
        Ground state accuracy metric.
    """
    hull = thull.full_hull(
        compositions=composition_predicted, energies=energy_predicted
    )
    vertices, _ = thull.lower_hull(hull)

    slopes = calculate_slopes(
        composition_predicted[vertices], energy_predicted[vertices]
    )
    stable_chem_pot_windows = [
        slopes[i + 1] - slopes[i] for i in range(len(slopes) - 1)
    ]

    # End states will always be on the convex hull and have an infinite stable chemical potential window. Exclude these from the
    vertices = np.sort(vertices)[2:]

    vertex_indices_ordered_by_comp = np.argsort(
        np.ravel(composition_predicted[vertices])
    )

    numerator = 0
    for vertex_index in vertex_indices_ordered_by_comp:
        if vertices[vertex_index] in true_ground_state_indices:
            numerator += stable_chem_pot_windows[vertex_index]

    return numerator / np.sum(stable_chem_pot_windows)


def ground_state_accuracy_fraction_correct(
    predicted_ground_state_indices, true_ground_state_indices
) -> float:
    """Computes a scalar ground state accuracy metric. The metric varies between [0,1], where 1 is perfect accuracy. 
        The denominator is the number of ground state configurations predicted by DFT.
        The numerator is the number of predicted ground state configurations which are ALSO ground states predicted by DFT.

    Parameters
    ----------
    predicted_ground_state_indices : np.ndarray
        1D array of predicted ground state indices.
    true_ground_state_indices : np.ndarray
        1D array of true ground state indices.

    Returns
    -------
    float
        Ground state accuracy metric, between 0 and 1. 1 is perfect accuracy.
    """
    numerator = 0
    for vertex_index in predicted_ground_state_indices:
        if vertex_index in true_ground_state_indices:
            numerator += 1

    return numerator / len(true_ground_state_indices)


def gsa_fraction_correct_DFT_mu_window_binary(
    predicted_comp: np.ndarray,
    predicted_energies: np.ndarray,
    true_comp: np.ndarray,
    true_energies: np.ndarray,
) -> float:
    """Normalized sum over DFT-predicted stable chemical potential windows for all configurations on the convex hull, excluding ground states. 
        The denominator is the sum across the DFT-determined stable chemical potential windows (slopes) for each configuration on the convex hull, excluding ground states.
        The numerator is the sum across the stable chemical potential windows (slopes) for each configuration that is predicted (by both cluster expansion AND DFT) to be on the convex hull, excluding ground states.
        The metric varies between [0,1], where 1 is perfect accuracy. The metric is a fraction.
        (1c ground state accuracy metric)
    Parameters
    ----------
    predicted_comp : np.ndarray
        nx1 matrix of compositions, where n is the number of configurations.
    predicted_energies : np.ndarray
        Vector of n predicted formation energies.
    true_comp : np.ndarray
        nx1 matrix of compositions, where n is the number of configurations.
    true_energies : np.ndarray
        Vector of n "true" formation energies.

    Returns
    -------
    float
        Ground state accuracy metric, between [0,1]. 1 is perfect accuracy.
    """
    # Calculate the lower convex hull vertices for the predicted and true convex hulls
    predicted_hull = thull.full_hull(
        compositions=predicted_comp, energies=predicted_energies
    )
    predicted_vertices, _ = thull.lower_hull(predicted_hull)
    true_hull = thull.full_hull(compositions=true_comp, energies=true_energies)
    true_vertices, _ = thull.lower_hull(true_hull)

    # Calculate the slope windows for the true convex hull
    true_chemical_potential_windows = stable_chemical_potential_windows_binary(
        true_hull
    )

    # The end state structures with composition 0 and 1 will always be on the convex hull. However, they are not useful for this metric.
    # Check the compositions of the true and predicted convex hull vertices.
    # If the compositions of any of the vertices are 0 or 1, delete them from the array.

    true_indices_to_remove = np.union1d(
        np.where(true_comp[true_vertices] == 0),
        np.where(true_comp[true_vertices] == 1),
    )
    true_vertices = np.delete(true_vertices, true_indices_to_remove)
    del true_indices_to_remove

    predicted_energies_to_remove = np.union1d(
        np.where(predicted_comp[predicted_vertices] == 0),
        np.where(predicted_comp[predicted_vertices] == 1),
    )
    predicted_vertices = np.delete(predicted_vertices, predicted_energies_to_remove)

    # Sort the true convex hull vertices by their compositions
    true_vertices_ordered_by_comp = np.argsort(np.ravel(true_comp[true_vertices]))

    # Calculate the ground state accuracy metric
    numerator = 0
    for vertex_index in true_vertices_ordered_by_comp:
        if true_vertices[vertex_index] in predicted_vertices:
            numerator += true_chemical_potential_windows[vertex_index]
    return numerator / np.sum(true_chemical_potential_windows)


def gsa_fraction_correct_predicted_mu_window_binary(
    predicted_comp: np.ndarray,
    predicted_energies: np.ndarray,
    true_comp: np.ndarray,
    true_energies: np.ndarray,
) -> float:
    """
    Normalized sum over predicted chemical potential windows. 
    The numerator sums across chemical potential windows (slope change across a convex hull vertex) of the predicted convex hull. A chemical potential window is only included if its corresponding configuration is also on the true convex hull.
    The denominator sums across the PREDICTED slope windows of data points that are considered the true ground states of the system. 
    Unlike the numerator, the denominator does not require that the configurations lie on the predicted convex hull. It is simply a collection of slope windows across the 'true' ground states of the system. 
    Because the end states have unbounded chemical potential windows, they are excluded from this metric entirely. 
    """
    # Calculate the lower convex hull vertices for the predicted and true convex hulls
    predicted_hull = thull.full_hull(
        compositions=predicted_comp, energies=predicted_energies
    )
    predicted_vertices, _ = thull.lower_hull(predicted_hull)
    true_hull = thull.full_hull(compositions=true_comp, energies=true_energies)
    true_vertices, _ = thull.lower_hull(true_hull)

    # Form a convex hull object of the true ground state indices within the predicted data
    true_ground_staes_within_predicted = thull.full_hull(
        compositions=predicted_comp[true_vertices],
        energies=predicted_energies[true_vertices],
    )

    # Calculate the slope windows for the predicted convex hull
    predicted_chemical_potential_windows = stable_chemical_potential_windows_binary(
        predicted_hull
    )
    chem_pot_windows_of_true_slice_of_predictions = stable_chemical_potential_windows_binary(
        true_ground_staes_within_predicted
    )

    # Sort the true convex hull vertices by their compositions
    true_vertices_ordered_by_comp = np.argsort(np.ravel(true_comp[true_vertices]))

    # If the elements 1 or 0 are in the true convex hull vertices, delete them from the array. Do the same for predicted vertices.
    if 0 in true_vertices:
        true_vertices = np.delete(true_vertices, np.where(true_vertices == 0))
    if 1 in true_vertices:
        true_vertices = np.delete(true_vertices, np.where(true_vertices == 1))
    if 0 in predicted_vertices:
        predicted_vertices = np.delete(
            predicted_vertices, np.where(predicted_vertices == 0)
        )
    if 1 in predicted_vertices:
        predicted_vertices = np.delete(
            predicted_vertices, np.where(predicted_vertices == 1)
        )

    # Calculate the ground state accuracy metric
    numerator = 0
    for vertex_index in true_vertices_ordered_by_comp:
        if true_vertices[vertex_index] in predicted_vertices:
            numerator += predicted_chemical_potential_windows[vertex_index]
    return numerator / np.sum(chem_pot_windows_of_true_slice_of_predictions)


def gsa_number_incorrect(predicted_ground_state_indices, true_ground_state_indices):
    """Returns the number of incorrectly predicted ground state indices

    Parameters
    ----------
    predicted_ground_state_indices : np.ndarray
        1D array of predicted ground state indices.
    true_ground_state_indices : np.ndarray
        1D array of true ground state indices.

    Returns
    -------
    int
        Number of incorrectly predicted ground state indices
    """
    return np.setdiff1d(predicted_ground_state_indices, true_ground_state_indices).shape


def gsa_fraction_intersection_over_union(
    predicted_ground_state_indices, true_ground_state_indices
):
    """Normalized fraction of the set intersection over the set union of the predicted and true ground state indices.

    Parameters
    ----------
    predicted_ground_state_indices : np.ndarray
        1D array of predicted ground state indices.
    true_ground_state_indices : np.ndarray
        1D array of true ground state indices.

    Returns
    -------
    float
        Normalized fraction of the set intersection over the set union of the predicted and true ground state indices.
    """
    return (
        np.intersect1d(predicted_ground_state_indices, true_ground_state_indices).shape[
            0
        ]
        / np.union1d(predicted_ground_state_indices, true_ground_state_indices).shape[0]
    )


def ground_state_accuracy_fraction_of_top_n_stable_configurations(
    predicted_ground_state_indices: np.ndarray,
    composition_true: np.ndarray,
    energy_true: np.ndarray,
    n: int,
) -> float:
    """Computes a scalar metric between [0,1] which measures the fraction of the top n stable configurations which are also ground states in DFT data. 1 is perfect accuracy.
        First, DFT-predicted ground state configurations are ranked by their stable chemical potential window. Only the top n of these configurations are considered in the accuracy metric. 
        The numerator is the number of elements in the set intersection between the predicted ground states and the top n DFT predicted ground states.
        The denominator is n (the number of configurations considered in the accuracy metric).
        

    Parameters
    ----------
    predicted_ground_state_indices : np.ndarray
        1D array of predicted ground state indices.
    composition_true : np.ndarray
        nxm matrix of compositions, where n is the number of configurations and m is the number of composition axes.
    energy_true : np.ndarray
        nx1 matrix of true formation energies.  
    n:int
        Number of DFT-predicted ground state configurations to compare agains in the ground state accuracy metric. 
        Configurations with the largest stable chemical potential window are chose first. 
        By the construction of the convex hull, end states will always be predicted. Therefore, end states are never included in the metric. 
    

    Returns
    -------
    float
        Ground state accuracy metric, between 0 and 1. 1 is perfect accuracy.
    """

    # Calculate the true convex hull, find the vertices, and calculate the stable chemical potential windows for each vertex.
    true_hull = thull.full_hull(compositions=composition_true, energies=energy_true)
    true_vertices, _ = thull.lower_hull(true_hull)
    true_slopes = calculate_slopes(
        composition_true[true_vertices], energy_true[true_vertices]
    )
    true_stable_chem_pot_windows = [
        true_slopes[i + 1] - true_slopes[i] for i in range(len(true_slopes) - 1)
    ]

    # Sort true_vertices, drop the end states.
    # Chemical potential windows are sorted by composition, true vertices are not. Sort by composition (ascending) so that they match order.
    # Then sort by stable chemical potential window (descending)
    true_vertices = np.sort(true_vertices)[2:]
    true_vertices = true_vertices[np.argsort(np.ravel(composition_true[true_vertices]))]
    true_vertices = true_vertices[np.argsort(true_stable_chem_pot_windows)[::-1]]

    # Check that n is not larger than the number of true_vertices.
    # If it is, set n to the number of true_vertices.
    # Then set the true_vertices to the first n elements in true_vertices.
    if n > len(true_vertices):
        n = len(true_vertices)
    true_vertices = true_vertices[:n]

    # Calculated the predicted lower hull vertices, and count the number of vertices which are also in true_vertices. This is the numerator.
    numerator = 0
    for vertex_index in predicted_ground_state_indices:
        if vertex_index in true_vertices:
            numerator += 1

    return numerator / n
