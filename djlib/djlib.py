from __future__ import annotations
import numpy as np
import os
import pathlib
import math as m
from glob import glob
import json
from typing import List, Tuple
import shutil
import warnings

libpath = pathlib.Path(__file__).parent.resolve()


def regroup_dicts_by_keys(list_of_dictionaries: list) -> dict:
    """Groups CASM query data by property instead of by configuration.

    Parameters
    ----------
    list_of_dictionaries: list
        List of dictionaries.

    Returns
    -------
    results: dict
        Dictionary of all data grouped by keys (not grouped by configuraton)

    Notes
    ------
    This function assumes that all dictionaries have the same keys.
    It sorts all properties by those keys instead of by list index.
    Properties that are a single value or string are passed as a list of those properties.
    Properties that are arrays are passed as a list of lists (2D matrices) even if the
    property only has one value (a matrix of one column).
    """
    data = list_of_dictionaries
    keys = data[0].keys()
    data_collect = []
    for i in range(len(keys)):
        data_collect.append([])

    for element_dict in data:
        for index, key in enumerate(keys):
            data_collect[index].append(element_dict[key])
    return dict(zip(keys, data_collect))


def ungroup_dicts_by_keys(dict_with_keys: dict) -> list:
    """Ungroups property sorted data (as dict) to a list of dictionaries (by configuration). Essentially does the reverse of regroup_dicts_by_keys.

    Parameters
    ----------
    dict_with_keys: dict
        Dictionary of all data grouped by keys (not grouped by configuraton)

    Returns
    -------
    list_of_dictionaries: list
        List of dictionaries. Each dictionary has the same keys as the input dictionary, but only a single entry for each key (aka each dictionary represents a single configuration).

    Notes
    ------
    This function assumes that each key in the dictionary is of the same length and will abort if this is not the case.
    """
    keys = dict_with_keys.keys()
    # verify that the keys are all the same length
    key_lengths = [len(dict_with_keys[key]) for key in keys]
    if len(set(key_lengths)) != 1:
        raise ValueError("All keys must be the same length")
    list_of_dictionaries = []
    for i in range(key_lengths[0]):
        list_of_dictionaries.append({})
    for key in keys:
        for index, value in enumerate(dict_with_keys[key]):
            list_of_dictionaries[index][key] = value
    return list_of_dictionaries


def casm_query_reader(casm_query_json_path="pass", casm_query_json_data=None):
    """Reads keys and values from casm query json dictionary.

    Parameters
    ----------
    casm_query_json_path: str
        Absolute path to casm query json file.
        Defaults to 'pass' which means that the function will look to take a dictionary directly.
    casm_query_json_data: dict
        Can also directly take the casm query json dictionary.
        Default is None.

    Returns
    -------
    results: dict
        Dictionary of all data grouped by keys (not grouped by configuraton)
    """
    print(
        "This function is deprecated. As an alternative, please load a casm query json file, then pass the dictionary to regroup_dicts_by_keys."
    )
    if casm_query_json_data is None:
        with open(casm_query_json_path) as f:
            data = json.load(f)
    else:
        data = casm_query_json_data
    keys = data[0].keys()
    data_collect = []
    for i in range(len(keys)):
        data_collect.append([])

    for element_dict in data:
        for index, key in enumerate(keys):
            data_collect[index].append(element_dict[key])

    results = dict(zip(keys, data_collect))
    if "comp" in results.keys():
        results["comp"] = np.array(results["comp"])
        if len(results["comp"].shape) > 2:
            results["comp"] = np.squeeze(results["comp"])
        if len(results["comp"].shape) == 1:
            results["comp"] = np.reshape(results["comp"], (-1, 1))
        results["comp"] = results["comp"].tolist()
    if "corr" in results.keys():
        results["corr"] = np.squeeze(results["corr"]).tolist()
    return results


def trim_unknown_energies(casm_query_json: list, keyword="energy"):
    """
    Given a data dictionary from a casm query sorted by property, removes data with null/None values in the designated key
    Parameters
    ----------
    casm_query_json : list of dicts
        A dictionary from a casm query sorted by property. Loaded directly from query json.
    key : str
        The key in the data dictionary that corresponds to the value you want to base entry removal. Defaults to 'energy_per_atom'.

    Returns
    -------
    denulled_data: dict
        A dictionary with the same keys as the input data, but with entries removed for which the key value is null.
    """
    initial_length = len(casm_query_json)
    denulled_data = [
        entry for entry in casm_query_json if entry[keyword] is not None
    ]  #
    final_length = len(denulled_data)
    print(
        "Removed %d entries with null values with key: %s"
        % (initial_length - final_length, keyword)
    )
    return denulled_data


def get_dj_dir():
    libpath = pathlib.Path(__file__).parent.resolve()
    return libpath


def column_sort(matrix: np.ndarray, column_index: int) -> np.ndarray:
    """Sorts a matrix by the values of a specific column. Far left column is column 0.
    Args:
        matrix(numpy_array): mxn numpy array.
        column_index(int): Index of the column to sort by.
    """
    column_index = round(column_index)
    sorted_matrix = matrix[np.argsort(matrix[:, column_index])]
    return sorted_matrix


def find(lst: list, a: float):
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
        print("Search value does not match any value in the provided list.")


def update_properties_files(casm_root_dir: str):
    """Updates json key of property.calc.json files to allow imports.

    Parameters
    ----------
    training_data_dir : str
        Path to a training_data directory in a casm project.

    Returns
    -------
    None.

    Notes
    -----
    Currently, only modifies "coord_mode" -> "coordinate_mode"
    """
    training_data_dir = os.path.join(casm_root_dir, "training_data")
    scels = glob(os.path.join(training_data_dir, "SCEL*"))
    for scel in scels:
        configs = glob(os.path.join(scel, "*"))

        for config in configs:
            properties_path = os.path.join(
                config, "calctype.default/properties.calc.json"
            )

            if os.path.isfile(properties_path):
                with open(properties_path) as f:
                    properties = json.load(f)
                if "coord_mode" in properties:
                    properties["coordinate_mode"] = properties["coord_mode"]
                if properties["atom_properties"]["force"]["value"] == []:
                    atoms = len(properties["atom_type"])
                    fixer = []
                    for i in range(atoms):
                        fixer.append([0.0, 0.0, 0.0])
                    print(len(fixer))
                    properties["atom_properties"]["force"]["value"] = fixer
                    print("Fixed empty forces in %s" % config)
                with open(
                    os.path.join(config, "calctype.default/properties.calc.json"), "w"
                ) as f:
                    json.dump(properties, f, indent="")
            else:
                print("Could not find %s" % properties_path)


def move_calctype_dirs(casm_root_dir: str, calctype="default"):
    """Meant to fix casm import issue where calctype_default is copied within new calctype_default directory. Shifts all the data up one directory.

    Parameters
    ----------
    casm_root_dir : str
        Path to casm project root.
    calctype : str
        Calctype to check for. Defaults to "default"

    Returns
    -------
    None.
    """
    calctype_string = "calctype." + calctype
    scels = glob(os.path.join(casm_root_dir, "training_data/SCEL*"))
    for scel in scels:
        configs = glob(os.path.join(scel, "*"))

        for config in configs:
            if os.path.isdir(
                os.path.join(config, "%s/%s" % (calctype_string, calctype_string))
            ):
                nested_calctype_data = os.path.join(
                    config, "%s/%s/*" % (calctype_string, calctype_string)
                )
                calctype_path = os.path.join(config, calctype_string)
                os.system("mv %s %s" % (nested_calctype_data, calctype_path))
                os.system(
                    "rm -r %s"
                    % os.path.join(config, "%s/%s" % (calctype_string, calctype_string))
                )
            if os.path.isdir(
                os.path.join(config, calctype_string, "relax_loop_to_static")
            ):
                calctype_path = os.path.join(config, calctype_string)
                nested_relax_loop = os.path.join(
                    calctype_path, "relax_loop_to_static/*"
                )
                os.system("mv %s %s" % (nested_relax_loop, calctype_path))
                os.system(
                    "rm -r %s" % os.path.join(calctype_path, "relax_loop_to_static")
                )


def submit_slurm_job(run_dir: str, submit_script_name: str = "submit_slurm.sh"):
    """Submits a job using the slurm job scheduler.
    Parameters
    ----------
    run_dir : str
        Path to the directory that contains the submit file.
    submit_script_name : str, optional
        Name of the submit file. The default is "submit_slurm.sh".

    Returns
    -------
    None.
    """
    submit_file = os.path.join(run_dir, submit_script_name)
    os.system("cd %s" % run_dir)
    os.system("sbatch %s" % submit_file)


def format_slurm_job(
    jobname: str,
    hours: int,
    user_command: str,
    output_dir: str,
    delete_submit_script=False,
    queue="batch",
    nodes=1,
    ntasks_per_node=1,
):
    """
    Formats a slurm job submission script. Assumes that the task only needs one thread.

    Parameters
    ----------
    jobname: str
        Name of the slurm job.
    hours: int
        number of hours to run the job. Only accepts integer values.
    user_command: str
        command line command submitted by the user as a string.
    output_dir: str
        Path to the directory that will contain the submit file. Assumes that submit file will be named "submit.sh"
    delete_submit_script: bool, optional
        Whether the submission script should delete itself upon completion. The default is False.
    queue: str, optional
        Queue to submit the job to. "batch" or "short". The default is "batch".
    nodes: int, optional
        Number of nodes to use. The default is 1.
    ntasks_per_node: int, optional
        Number of tasks per node. The default is 1.

    Returns
    -------
        None.
    """
    submit_file_path = os.path.join(output_dir, "submit_slurm.sh")
    templates_path = os.path.join(libpath, "templates")
    if queue == "batch" or queue == "debug":
        slurm_template_file = "single_task_slurm_template.sh"
    elif queue == "short":
        slurm_template_file = "short_queue_single_task_slurm_template.sh"
    with open(os.path.join(templates_path, slurm_template_file)) as f:
        template = f.read()

        if delete_submit_script:
            delete_submit_script = "rm %s" % submit_file_path
        else:
            delete_submit_script = ""

        hours = round(m.ceil(hours))
        s = template.format(
            jobname=jobname,
            rundir=output_dir,
            hours=hours,
            user_command=user_command,
            delete_submit_script=delete_submit_script,
            nodes=nodes,
            ntasks_per_node=ntasks_per_node,
        )
    with open(submit_file_path, "w") as f:
        f.write(s)
    os.system("chmod 755 %s " % submit_file_path)


def mode(vec: np.ndarray) -> float:
    """Calculates and returns the mode of a vector of continuous data.

    Parameters
    ----------
    vec: numpy.ndarray
        Vector of floats

    Returns
    -------
    hist_mode: float
        Value corresponding to the peak of the histogram.
    """

    hist = np.histogram(vec, bins="auto")
    max_index = np.where(hist[0] == max(hist[0]))[0]
    hist_mode = np.mean((hist[1][max_index], hist[1][max_index + 1]))
    return hist_mode


def analytic_posterior(
    feature_matrix: np.ndarray,
    weight_covariance_matrix: np.ndarray,
    weight_mean_vec: np.ndarray,
    label_covariance_matrix: np.ndarray,
    label_vec: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the posterior distribution (mean and covariance matrix) given the weight mean vector,
    weight covariance matrix, target values vector, and target values covariance matrix.

    Taken from Bishop Pattern Recognition and Machine Learning, 2006, p. 93

    Parameters
    ----------
    weight_covariance_matrix: np.ndarray
        Weight covariance matrix.
    weight_mean_vec: np.ndarray
        Weight mean vector.
    label_covariance_matrix: np.ndarray
        Target values covariance matrix.
    label_vec: np.ndarray
        Target values vector.

    Returns
    -------
    posterior_mean_vec: np.ndarray
        Posterior mean vector.
    posterior_covariance_matrix: np.ndarray
        Posterior covariance matrix.
    """
    # Calculate precision matrices (inverse of covariance matrices)
    weight_precision_matrix = np.linalg.pinv(weight_covariance_matrix)
    label_precision_matrix = np.linalg.pinv(label_covariance_matrix)

    # Calculate the posterior distribution covariance matrix
    posterior_covariance_matrix = np.linalg.pinv(
        weight_precision_matrix
        + feature_matrix.T @ label_precision_matrix @ feature_matrix
    )

    # Calculate the posterior distribution mean vector
    posterior_mean_vec = posterior_covariance_matrix @ (
        feature_matrix.T @ label_precision_matrix @ label_vec
        + weight_precision_matrix @ weight_mean_vec
    )

    return (posterior_mean_vec, posterior_covariance_matrix)


def collect_config_structure_files(
    casm_root: str, config_list: List[str], output_directory
) -> None:
    """Collects the config structure.json files from a list of config names.

    Parameters:
    -----------
    casm_root: str
        Path to the CASM root directory.
    config_list: List[str]
        List of config names.

    Returns:
    --------
    None.
    """

    # make output directory if it doesn't exist
    print("creating output directory: ", output_directory)
    os.makedirs(output_directory, exist_ok=True)

    # For each config, copy the config structure.json file to the output directory
    for config in config_list:
        print("copying config structure.json file for config: ", config)
        os.makedirs(os.path.join(output_directory, config), exist_ok=True)
        shutil.copy(
            os.path.join(casm_root, "training_data", config, "structure.json"),
            os.path.join(output_directory, config),
        )
        # Check if there is a file called "calctype.default/run.final/CONTCAR" within the config directory.
        # If so, copy it to the same location as the structure.json file in the output directory.
        if os.path.isfile(
            os.path.join(
                casm_root,
                "training_data",
                config,
                "calctype.default",
                "run.final",
                "CONTCAR",
            )
        ):
            shutil.copy(
                os.path.join(
                    casm_root,
                    "training_data",
                    config,
                    "calctype.default",
                    "run.final",
                    "CONTCAR",
                ),
                os.path.join(output_directory, config),
            )
    print("done")


class gridspace_manager:
    """
    A class for managing repetitive experiments in a grid of parameter space.

    Parameters
    ----------
    origin_dir : str, optional
        The path to the directory that will contain the grid of experiments. Default is the current directory.
    namer : callable, optional
        A function that maps a dictionary of parameters to a unique string that will be used as a directory name.
    run_parser : callable, optional
        A function that extracts data from the output of each experiment.
    run_creator : callable, optional
        A function that creates input files for each experiment.
    status_updater : callable, optional
        A function that updates the status of each experiment based on its output.
    run_submitter : callable, optional
        A function that runs each experiment.

    Attributes
    ----------
    data : list
        A list containing the output data of each experiment.
    origin_dir : str
        The path to the directory that will contain the grid of experiments.
    namer : callable
        A function that maps a dictionary of parameters to a unique string that will be used as a directory name.
    run_parser : callable
        A function that extracts data from the output of each experiment.
    run_creator : callable
        A function that creates input files for each experiment.
    grid_params : dict
        A dictionary containing the parameters of each experiment.
    status_updater : callable
        A function that updates the status of each experiment based on its output.
    run_submitter : callable
        A function that runs each experiment.

    Methods
    -------
    collect_data()
        Collects data from the output of each experiment.
    format_run_dirs()
        Creates directories for each experiment according to the namer function and the parameters specified in grid_params.
    update_status()
        Updates the status of each experiment based on its output.
    run_valid_calculations()
        Runs each experiment.
    """

    def __init__(
        self,
        origin_dir: str = "./",
        namer: callable = None,
        run_parser: callable = None,
        run_creator: callable = None,
        status_updater: callable = None,
        run_submitter: callable = None,
        grid_params: dict = None,  # list of dictionary, each dictionary corresponds to a run
    ) -> None:
        self.data = None
        self.origin_dir = origin_dir
        self.namer = namer
        self.run_parser = run_parser
        self.run_creator = run_creator
        self.grid_params = grid_params
        self.status_updater = status_updater
        self.run_submitter = run_submitter

    def collect_data(self):
        # Iterate through directories, collecting data from each run.
        self.data = []
        self.dirs = glob(os.path.join(self.origin_dir, "*"))

        for dir in self.dirs:
            try:
                self.data.append(self.run_parser(dir))
            except:
                print("failed to parse: ", dir)
        self.data = [entry for entry in self.data if entry is not None]

    def format_run_dirs(self) -> None:
        for entry in self.grid_params:
            # Make a directory for each entry grid_params, according to the namer function. Overwrite existing directories if they exist.
            run_dir = os.path.join(self.origin_dir, self.namer(entry))
            os.makedirs(run_dir, exist_ok=True)
            self.run_creator(entry, run_dir)

    def update_status(self) -> None:
        # Iterate through directories, updating status of each run according to the status_updater function.
        self.dirs = glob(os.path.join(self.origin_dir, "*"))
        for dir in self.dirs:
            try:
                self.status_updater(dir)
            except:
                print("Failed to update: ", dir)

    def run_valid_calculations(self) -> None:
        # Iterate through directories, submitting each according to the run_submitter function.
        self.dirs = glob(os.path.join(self.origin_dir, "*"))
        for dir in self.dirs:
            try:
                self.run_submitter(dir)
            except:
                print("Failed to run: ", dir)


def cartesian_to_n_sphere(x):
    """
    Converts n-dimensional Cartesian coordinates to n-sphere coordinates.
    Parameters
    ----------
    x : array_like
        Matrix of shape (m,n) where m is the number of points and n is the number of dimensions.
    Returns
    -------
    n_sphere_coords : array_like
        Matrix of shape (m,n) where the first element (index 0) is the radius, elements of indices 1 to n-2 are the polar angles [0,\pi], and the last element (index n-1)
        is the azimuthal angle [0,2\pi).
    """
    r = np.linalg.norm(x, axis=1, keepdims=True)  # Radius is the Euclidean norm
    thetas = np.zeros((x.shape[0], x.shape[1] - 1))  # preallocate theta values

    for i in range(thetas.shape[1]):
        if i < thetas.shape[1] - 1:
            arg1 = np.linalg.norm(x[:, i + 1 :], axis=1)
            thetas[:, i] = np.arctan2(arg1, x[:, i])
        else:
            thetas[:, i] = np.arctan2(x[:, i + 1], x[:, i])
    return np.hstack((r, thetas))


def n_sphere_to_cartesian(n_sphere_coords):
    """
    Converts n-sphere coordinates to n-dimensional Cartesian coordinates.
    Parameters
    ----------
    n_sphere_coords : array_like
        Matrix of shape (m,n), with m points and n dimensions. The first element (index 0) is the radius,
        elements of indices 1 to n-2 are the polar angles [0,\pi], and the last element (index n-1)
        is the azimuthal angle [0,2\pi).
    Returns
    -------
    x : array_like
        Matrix of shape (m,n) where m is the number of points and n is the number of dimensions.
    """
    r = n_sphere_coords[:, 0]
    thetas = n_sphere_coords[:, 1:]

    x = np.zeros(
        (n_sphere_coords.shape[0], n_sphere_coords.shape[1])
    )  # preallocate x values
    for i in range(
        n_sphere_coords.shape[1]
    ):  # iterate over each dimension, populating the x values

        if i < thetas.shape[1]:
            if i == 0:
                x[:, i] = r * np.cos(thetas[:, i])
            else:
                x[:, i] = (
                    r * np.prod(np.sin(thetas[:, :i]), axis=1) * np.cos(thetas[:, i])
                )
        else:
            x[:, i] = r * np.prod(np.sin(thetas[:, :i]), axis=1)

    return x


def edge_finder_binary_search(
    direction_vector, oracle_function, tolerance, max_iterations, relevant_length_scale
) -> np.ndarray:
    """Searches for the edge of a polytope boundary along a given direction vector using binary search.
    The boundary is defined as where the oracle function changes from True to False.
    Parameters
    ----------
    direction_vector : array_like
        The direction vector to search along by scaling.
    oracle_function : function
        A function that takes a single argument, a 1D array_like object (the scaled direction_vector), and returns a boolean value.
    tolerance : float
        The desired tolerance for the edge location. The boundary will be at direction_vector*optimal_scaling +[0, tolerance].
    max_iterations : int
        The maximum number of iterations to perform, in case the binary search does not converge.
    relevant_length_scale : float
        The length scale of the problem, used to set the initial search bounds.
    Returns
    -------
    """
    inside = []
    outside = []
    # initialize the search bounds
    lower_bound = 0
    upper_bound = relevant_length_scale

    # Scale the direction vector by the upper bound until the oracle function returns False
    while oracle_function(direction_vector * upper_bound):
        upper_bound *= 2
        if upper_bound > 1e5:
            raise ValueError(
                "The upper bound is larger than 1e5. The oracle function may not be working properly."
            )

    # Perform the binary search
    for i in range(max_iterations):
        # Compute the midpoint
        midpoint = (lower_bound + upper_bound) / 2
        # If the midpoint is within the tolerance, return it
        if upper_bound - lower_bound < tolerance:
            return {
                "boundary_vector": np.squeeze(midpoint * direction_vector),
                "inside_vectors": inside,
                "outside_vectors": outside,
            }

        # If the midpoint is within the tolerance, return it
        if oracle_function(direction_vector * midpoint):
            lower_bound = midpoint
            inside.append(midpoint * direction_vector)
        else:
            upper_bound = midpoint
            outside.append(midpoint * direction_vector)
    # If the binary search does not converge, return the midpoint as the boundary scale , and send a message that the search did not converge
    print(
        "The binary search did not converge after",
        max_iterations,
        "iterations. Reported boundary is at the final midpoint.",
    )
    return {
        "boundary_vector": np.squeeze(midpoint * direction_vector),
        "inside_vectors": inside,
        "outside_vectors": outside,
    }
