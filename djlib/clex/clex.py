from __future__ import annotations
import djlib.djlib as dj
import djlib.clex.clex as cl
import json
import os
import numpy as np
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
import csv
from glob import glob
import pickle
from string import Template
from sklearn.model_selection import ShuffleSplit
import arviz as ar
import thermofitting.hull.lower_hull as thull
import pathlib
from warnings import warn


def lower_hull(hull: ConvexHull, energy_index=-2):
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
    # Note: energy_index is used on the "hull.equations" output, which has a scalar offset.
    # (If your energy is the last column of "points", either use the direct index, or use "-2". Using "-1" as energy_index will not give the proper lower hull.)
    lower_hull_simplices = hull.simplices[hull.equations[:, energy_index] < 0]
    lower_hull_vertices = np.unique(np.ravel(lower_hull_simplices))
    return (lower_hull_simplices, lower_hull_vertices)


def checkhull(
    hull_comps: np.ndarray,
    hull_energies: np.ndarray,
    test_comp: np.ndarray,
    test_energy: np.ndarray,
):
    """Find if specified coordinates are above, on or below the specified lower convex hull.
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
    corr: numpy.ndarray,
    comp: numpy.ndarray,
    formation_energy: numpy.ndarray,
    eci_set: numpy.ndarray,
) -> numpy.ndarray:
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
    dft_hull_simplices, dft_hull_config_indices = lower_hull(hull, energy_index=-2)
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


def plot_eci_hist(eci_data, xmin=None, xmax=None):
    plt.hist(x=eci_data, bins="auto", color="xkcd:crimson", alpha=0.7, rwidth=0.85)
    if xmin and xmax:
        plt.xlim(xmin, xmax)
    plt.xlabel("ECI value (eV)", fontsize=18)
    plt.ylabel("Count", fontsize=18)
    # plt.show()
    fig = plt.gcf()
    return fig


def plot_eci_covariance(eci_data_1, eci_data_2):
    plt.scatter(eci_data_1, eci_data_2, color="xkcd:crimson")
    plt.xlabel("ECI 1 (eV)", fontsize=18)
    plt.ylabel("ECI 2 (eV)", fontsize=18)
    fig = plt.gcf()
    return fig


def plot_clex_hull_data_1_x(
    fit_dir,
    hall_of_fame_index,
    full_formation_energy_file="full_formation_energies.txt",
    custom_title="False",
):
    """plot_clex_hull_data_1_x(fit_dir, hall_of_fame_index, full_formation_energy_file='full_formation_energies.txt')

    Function to plot DFT energies, cluster expansion energies, DFT convex hull and cluster expansion convex hull.

    Args:
        fit_dir (str): absolute path to a casm cluster expansion fit.
        hall_of_fame_index (int or str): Integer index. "hall of fame" index for a specific fit (corresponding to a set of "Effective Cluster Interactions" or ECI).
        full_formation_energy_file (str): filename that contains the formation energy of all configurations of interest. Generated using a casm command

    Returns:
        fig: a python figure object.
    """
    # TODO: Definitely want to re-implement this with json input
    # Pre-define values to pull from data files
    # title is intended to be in the form of "casm_root_name_name_of_specific_fit_directory".
    if custom_title:
        title = custom_title
    else:
        title = fit_dir.split("/")[-3] + "_" + fit_dir.split("/")[-1]
    dft_scel_names = []
    clex_scel_names = []
    dft_hull_data = []
    clex_hull_data = []
    cv = None
    rms = None
    wrms = None
    below_hull_exists = False
    hall_of_fame_index = str(hall_of_fame_index)

    # Read necessary files
    os.chdir(fit_dir)
    files = glob("*")
    for f in files:
        if "_%s_dft_gs" % hall_of_fame_index in f:
            dft_hull_path = os.path.join(fit_dir, f)
            dft_hull_data = np.genfromtxt(
                dft_hull_path, skip_header=1, usecols=list(range(1, 10))
            ).astype(float)
            with open(dft_hull_path, "r") as dft_dat_file:
                dft_scel_names = [
                    row[0] for row in csv.reader(dft_dat_file, delimiter=" ")
                ]
                dft_scel_names = dft_scel_names[1:]

        if "_%s_clex_gs" % hall_of_fame_index in f:
            clex_hull_path = os.path.join(fit_dir, f)
            clex_hull_data = np.genfromtxt(
                clex_hull_path, skip_header=1, usecols=list(range(1, 10))
            ).astype(float)
            with open(clex_hull_path, "r") as clex_dat_file:
                clex_scel_names = [
                    row[0] for row in csv.reader(clex_dat_file, delimiter=" ")
                ]
                clex_scel_names = clex_scel_names[1:]

        if "_%s_below_hull" % hall_of_fame_index in f:
            below_hull_exists = True
            below_hull_path = os.path.join(fit_dir, f)
            below_hull_data = np.reshape(
                np.genfromtxt(
                    below_hull_path, skip_header=1, usecols=list(range(1, 10))
                ).astype(float),
                ((-1, 9)),
            )
            with open(below_hull_path, "r") as below_hull_file:
                below_hull_scel_names = [
                    row[0] for row in csv.reader(below_hull_file, delimiter=" ")
                ]
                below_hull_scel_names = below_hull_scel_names[1:]

        if "check.%s" % hall_of_fame_index in f:
            checkfile_path = os.path.join(fit_dir, f)
            with open(checkfile_path, "r") as checkfile:
                linecount = 0
                cv_rms_wrms_info_line = int
                for line in checkfile.readlines():
                    if (
                        line.strip() == "-- Check: individual 0  --"
                    ):  # % hall_of_fame_index:
                        cv_rms_wrms_info_line = linecount + 3

                    if linecount == cv_rms_wrms_info_line:
                        cv = float(line.split()[3])
                        rms = float(line.split()[4])
                        wrms = float(line.split()[5])
                    linecount += 1

    # Generate the plot
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.text(
        0.80,
        0.90 * min(dft_hull_data[:, 5]),
        "CV:      %.10f\nRMS:    %.10f\nWRMS: %.10f" % (cv, rms, wrms),
        fontsize=20,
    )
    labels = []
    if custom_title:
        plt.title(custom_title, fontsize=30)
    else:
        plt.title(title, fontsize=30)
    plt.xlabel(r"Composition", fontsize=22)
    plt.ylabel(r"Energy $\frac{eV}{prim}$", fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.plot(dft_hull_data[:, 1], dft_hull_data[:, 5], marker="o", color="xkcd:crimson")
    labels.append("DFT Hull")
    plt.plot(
        clex_hull_data[:, 1],
        clex_hull_data[:, 8],
        marker="o",
        linestyle="dashed",
        color="b",
    )
    labels.append("ClEx Hull")
    plt.scatter(dft_hull_data[:, 1], dft_hull_data[:, 8], color="k")
    labels.append("Clex Prediction of DFT Hull")

    if full_formation_energy_file:
        # format:
        # run casm query -k comp formation_energy hull_dist clex clex_hull_dist -o full_formation_energies.txt
        #            configname    selected           comp(a)    formation_energy    hull_dist(MASTER,atom_frac)        clex()    clex_hull_dist(MASTER,atom_frac)
        datafile = full_formation_energy_file
        data = np.genfromtxt(datafile, skip_header=1, usecols=list(range(2, 7))).astype(
            float
        )
        composition = data[:, 0]
        dft_formation_energy = data[:, 1]
        clex_formation_energy = data[:, 3]
        plt.scatter(composition, dft_formation_energy, color="salmon")
        labels.append("DFT energies")
        plt.scatter(composition, clex_formation_energy, marker="x", color="skyblue")
        labels.append("ClEx energies")

    # TODO: This implementation is wrong. This is the distance below the hull (energy difference) not the actual energy.
    if below_hull_exists:
        plt.scatter(below_hull_data[:, 1], below_hull_data[:, 7], marker="+", color="k")
        labels.append("Clex Below Clex Prediction of DFT Hull Configs")
    else:
        print("'_%s_below_hull' file doesn't exist" % hall_of_fame_index)

    plt.legend(labels, loc="lower left", fontsize=20)

    fig = plt.gcf()
    return fig


def stan_model_formatter(
    eci_variance_is_fixed: bool,
    model_variance_is_fixed: bool,
    eci_parameters: list,
    model_parameters: list,
    sigma_indices: list,
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
    sigma_indices: list
        A 1-D list of indices, length equal to the number of configurations. Each element is an index of the sigma parameter to use for the corresponding configuration.
        Only used if there is more than one element in model_parameters.
    
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
        If more than one element is provided, the user must specify a sigma_indices list to match each configuration to a model variance.
    sigma_indices:
        Only used if more than one sigma is provided in the model_parameters list. Identifies which sigma to use for each configuration.

    """

    templates = os.path.join(clex_lib_dir, "../templates")
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
            model_string += "\t\t" + "eci_variance[K] " + eci_parameters[0] + ";\n"
            model_string += "\t\t" + "eci[K] ~ normal(0,eci_variance[k]);\n\t}\n"
    else:
        # If ECI priors are different
        if eci_variance_is_fixed:
            for i, parameter in enumerate(eci_parameters):
                model_string += "\t" + "eci[{i}] ".format(i=i) + parameter + ";\n"
        else:
            for i, parameter in enumerate(eci_parameters):
                model_string += (
                    "\t\t" + "eci_variance[{i}] ".format(i) + parameter + ";\n"
                )
                model_string += (
                    "\t\t"
                    + "eci[{i}] ~ normal(0,eci_variance[{i}]);\n\t}\n".format(i=i)
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
            for config_index, sigma_index in enumerate(sigma_indices):
                model_string += (
                    "energies[{i}] ~ normal(corr[{i}]*eci, {sigma}) ".format(
                        i=config_index, sigma=model_parameters[sigma_index]
                    )
                    + ";\n"
                )
        else:
            for sigma_index, model_param in enumerate(model_parameters):
                model_string += (
                    "sigma[{sigma_index}] ".format(sigma_index=sigma_index)
                    + model_param
                    + ";\n"
                )
            for config_index, sigma_index in enumerate(sigma_indices):
                model_string += (
                    "energies[{i}] ~ normal(corr[{i}]*eci, sigma[{sigma_index}]) ".format(
                        i=config_index, sigma_index=sigma_index
                    )
                    + ";\n"
                )
    # Load template from templates directory
    clex_lib_dir = pathlib.Path(__file__).parent.resolve()
    with open(os.path.join(templates, "stan_model_template.txt"), "r") as f:
        template = Template(f.read())
    return template.substitute(
        formatted_parameters=parameters_string, formatted_model=model_string
    )


# format_stan_model is deprecated. Use stan_model_formatter instead.
def format_stan_model(
    eci_variance_args,
    likelihood_variance_args,
    eci_prior="normal",
    eci_variance_prior="gamma",
    likelihood_variance_prior="gamma",
    fixed_variance=False,
) -> str:
    """
    Parameters
    ----------
    eci_variance_args: tuple
        if fixed_variance is True, this should be a single float for eci variance. eg. eci_variance_args = 0.0016
        if fixed_variance is False, give arguments for the distribution as a tuple. eg. eci_variance_args = (1,1)
    likelihood_variance_args: tuple
        if fixed_variance is True, this should be a single float for the model variance. eg. likelihood_variance_args = 0.005
        if fixed_variance is False, give arguments for the distribution as a tuple. eg. eci_variance_args = (1,1)
    eci_prior: string
        Distribution type for ECI priors
    eci_variance_prior: string
        Distribution type for ECI variance prior
    likelihood_variance_prior: string
        Distribution type for model variance prior
    fixed_variance: Bool
        If True, model and ECI variance are fixed values. If false, they follow a distribution governed by hyperparameters.
        This choice will affect the eci and likelihood variance arg inputs; please read documentation for both.

    Returns
    -------
    model_template : str
        Formatted stan model template
    """
    warn(
        'This functinon "format_stan_model() is deprecated. Use "stan_model_formatter()" instead.',
        DeprecationWarning,
    )

    # Old args:
    # TODO: Add filter on string arguments

    supported_eci_priors = ["normal"]
    supported_eci_variance_priors = ["gamma"]
    supported_model_variance_priors = ["gamma"]

    assert eci_prior in supported_eci_priors, "Specified ECI prior is not suported."
    assert (
        eci_variance_prior in supported_eci_variance_priors
    ), "Specified ECI variance prior is not supported."

    assert (
        likelihood_variance_prior in supported_model_variance_priors
    ), "Specified model variance prior is not supported."
    # TODO: make this a single string that can be modified to allow any combination of fixed / non-fixed ECI and model variance priors.

    if fixed_variance:
        # If model and ECI variance are fixed to scalar values
        formatted_sigma = str(likelihood_variance_args)
        formatted_eci_variance = str(eci_variance_args)

        ce_model = Template(
            """data {
        int K; 
        int n_configs;
        matrix[n_configs, K] corr;
        vector[n_configs] energies;
    }
parameters {
        vector[K] eci;
    }
model 
    {
        real sigma = $formatted_sigma;
        for (k in 1:K){
            eci[k] ~ normal(0,$formatted_eci_variance);
        }
        energies ~ normal(corr * eci, sigma);
    }"""
        )
    else:
        # If model and ECI variance are not fixed (follows a distribution)
        formatted_sigma = str(likelihood_variance_args)
        formatted_eci_variance = eci_variance_prior + str(eci_variance_args)
        ce_model = Template(
            """data {
        int K; 
        int n_configs;
        matrix[n_configs, K] corr;
        vector[n_configs] energies;
    }
parameters {
        vector[K] eci;
        vector<lower=0>[K] eci_variance;
    }
model 
    {
        real sigma = $formatted_sigma;
        for (k in 1:K){
            eci_variance[k] ~ $formatted_eci_variance ;
            eci[k] ~ normal(0,eci_variance[k]);
        }
        energies ~ normal(corr * eci, sigma);
    }"""
        )
    model_template = ce_model.substitute(
        formatted_sigma=formatted_sigma, formatted_eci_variance=formatted_eci_variance,
    )
    # model_template = ce_model.substitute(formatted_eci_variance=formatted_eci_variance)
    return model_template


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


def cross_validate_stan_model(
    data_file: str,
    num_samples: int,
    eci_variance_args,
    likelihood_variance_args,
    cross_val_directory: str,
    random_seed=5,
    eci_prior="normal",
    eci_variance_prior="gamma",
    likelihood_variance_prior="gamma",
    stan_model_file="stan_model.txt",
    eci_output_file="results.pkl",
    energy_tag="formation_energy",
    num_chains=4,
    kfold=5,
    submit_with_slurm=True,
    fixed_variance=False,
):
    """Perform kfold cross validation on a specific stan model. Wraps around format_stan_model() and format_stan_executable_script().

    Parameters:
    -----------
    data_file: string
        Path to casm query output containing correlations, compositions and formation energies
    num_samples: int
        Number of samples in the stan monte carlo process
    eci_variance_args: tuple
        arguments for gamma distribution as a tuple. eg. eci_variance_args = (1,1)
    cross_val_directory: str
        Path to directory where the kfold cross validation runs will write data.
    random_seed: int
        Random number seed for randomized kfold data splitting. Providing the same seed will result in identical training / testing data partitions.
    eci_prior: string
        Distribution type for ECI priors
    eci_variance_prior: string
        Distribution type for ECI variance prior
    stan_model_file: string
        Path to text file containing stan model specifics
    eci_output_file: string
        Path to file where Stan will write the sampled ECI
    energy_tag: string
        Tag for the energy column in the casm query output (Can be formation_energy, energy, energy_per_atom, formation_energy_per_atom, etc.)
    num_chains: int
        Number of simultaneous markov chains
    kfold: int
        Number of "bins" to split training data into.
    submit_with_slurm: bool
        Decides if the function will submit with slurm. Defaults to true.

    Returns:
    --------
    None
    """
    # create directory for kfold cross validation
    os.makedirs(cross_val_directory, exist_ok=True)

    # load data
    with open(data_file) as f:
        data = np.array(json.load(f))

    # setup kfold batches, format for stan input
    data_length = data.shape[0]
    ss = ShuffleSplit(n_splits=kfold, random_state=random_seed)
    indices = range(data_length)

    count = 0
    for train_index, test_index in ss.split(indices):

        # make run directory for this iteration of the kfold cross validation
        this_run_path = os.path.join(cross_val_directory, "crossval_" + str(count))
        os.makedirs(this_run_path, exist_ok=True)

        # slice data; write training and testing data in separate files.
        training_data = data[train_index].tolist()
        testing_data = data[test_index].tolist()
        training_data_path = os.path.join(this_run_path, "training_data.json")
        with open(training_data_path, "w") as f:
            json.dump(training_data, f)
        with open(os.path.join(this_run_path, "testing_data.json"), "w") as f:
            json.dump(testing_data, f)

        # Also write training/ testing indices for easier post processing.
        run_info = {
            "training_set": train_index.tolist(),
            "test_set": test_index.tolist(),
            "eci_variance_args": eci_variance_args,
            "likelihood_variance_args": likelihood_variance_args,
            "data_source": data_file,
            "random_seed": random_seed,
        }
        with open(os.path.join(this_run_path, "run_info.json"), "w") as f:
            json.dump(run_info, f)

        # Write model info

        # format and write stan model
        formatted_stan_model = format_stan_model(
            eci_variance_args=eci_variance_args,
            eci_prior=eci_prior,
            eci_variance_prior=eci_variance_prior,
            likelihood_variance_args=likelihood_variance_args,
            fixed_variance=fixed_variance,
        )
        with open(os.path.join(this_run_path, stan_model_file), "w") as f:
            f.write(formatted_stan_model)

        # format and write stan executable python script
        formatted_stan_script = format_stan_executable_script(
            data_file=training_data_path,
            stan_model_file=stan_model_file,
            eci_output_file=eci_output_file,
            num_samples=num_samples,
            energy_tag=energy_tag,
            num_chains=4,
        )

        with open(os.path.join(this_run_path, "run_stan.py"), "w") as f:
            f.write(formatted_stan_script)

        # format and write slurm submission file
        user_command = "python run_stan.py"

        likelihood_variance_name = str(likelihood_variance_args)
        if type(eci_variance_args) == type(tuple([1])):
            eci_name = eci_variance_args[1]
        else:
            eci_name = str(eci_variance_args)
        dj.format_slurm_job(
            jobname="eci_var_"
            + str(eci_name)
            + "likelihood_"
            + likelihood_variance_name
            + "_crossval_"
            + str(count),
            hours=20,
            user_command=user_command,
            output_dir=this_run_path,
        )
        if submit_with_slurm:
            dj.submit_slurm_job(this_run_path)
        count += 1


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
    eci_mean_testing_rms = np.mean(np.array(eci_mean_testing_rms), axis=0)
    return {
        "train_rms": train_rms_values,
        "test_rms": test_rms_values,
        "eci_mean_testing_rms": eci_mean_testing_rms,
        "kfold_avg_eci_mean": eci_mean,
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


def write_eci_json(eci: np.ndarray, basis_json: dict):
    """Writes supplied ECI to the eci.json file for use in grand canonical monte carlo. Written for CASM 1.2.0

    Parameters:
    -----------
    eci: numpy.ndarray
        Vector of ECI values.

    basis_json_path: str
        Path to the casm-generated basis.json file.

    Returns:
    --------
    data: dict
        basis.json dictionary formatted with provided eci's
    """

    for index, orbit in enumerate(basis_json["orbits"]):
        basis_json["orbits"][index]["cluster_functions"][0]["eci"] = eci[index]

    return basis_json


def general_binary_convex_hull_plotter(
    composition: np.ndarray, true_energies: np.ndarray, predicted_energies=[None]
):
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

    predicted_color = "red"
    predicted_label = "Predicted Energies"

    plt.scatter(
        composition, true_energies, color="k", marker="x", label='"True" Energies'
    )
    if any(predicted_energies):
        plt.scatter(
            composition,
            predicted_energies,
            color="red",
            marker="x",
            label=predicted_label,
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

    dft_lower_hull_vertices = cl.lower_hull(dft_hull)[1]
    if any(predicted_energies):
        predicted_lower_hull_vertices = cl.lower_hull(predicted_hull)[1]

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

    fig = plt.gcf()
    fig.set_size_inches(19, 14)


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

