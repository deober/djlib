from __future__ import annotations
import numpy as np
import os
import pathlib
import math as m
from glob import glob
import json

libpath = pathlib.Path(__file__).parent.resolve()


def casm_query_reader(casm_query_json_path="pass", casm_query_json_data=None):
    """Reads keys and values from casm query json dictionary.
    Parameters:
    -----------
    casm_query_json_path: str
        Absolute path to casm query json file.
        Defaults to 'pass' which means that the function will look to take a dictionary directly.
    casm_query_json_data: dict
        Can also directly take the casm query json dictionary.
        Default is None.

    Returns:
    results: dict
        Dictionary of all data grouped by keys (not grouped by configuraton)
    """
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
    column_index = int(column_index)
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
                    properties["atom_properties"]["force"]["value"] = [[0.0, 0.0, 0.0]]
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


def submit_slurm_job(run_dir: str):
    submit_file = os.path.join(run_dir, "submit_slurm.sh")
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
    ntasks=1,
    tasks_per_core=1,
):
    """
    Formats a slurm job submission script. Assumes that the task only needs one thread.
    Args:
        jobname(str): Name of the slurm job.
        hours(int): number of hours to run the job. Only accepts integer values.
        user_command(str): command line command submitted by the user as a string.
        output_dir(str): Path to the directory that will contain the submit file. Assumes that submit file will be named "submit.sh"
        delete_submit_script(bool): Whether the submission script should delete itself upon completion.
        queue(str): Queue to submit the job to. "batch" or "short"
    Returns:
        None.
    """
    submit_file_path = os.path.join(output_dir, "submit_slurm.sh")
    templates_path = os.path.join(libpath, "templates")
    if queue == "batch":
        slurm_template_file = "single_task_slurm_template.sh"
    elif queue == "short":
        slurm_template_file = "short_queue_single_task_slurm_template.sh"
    with open(os.path.join(templates_path, slurm_template_file)) as f:
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
            nodes=nodes,
            ntasks=ntasks,
            tasks_per_core=tasks_per_core,
        )
    with open(submit_file_path, "w") as f:
        f.write(s)
    os.system("chmod 755 %s " % submit_file_path)


def run_submitter(
    run_directory: str,
    output_file_name="results.pkl",
    slurm_submit_script_name="submit_slurm.sh",
):
    """Checks if calculation output file exists. If slurm submission script exists, but calculation output does not, run the submission script.

    Parameters:
    -----------
    run_directory: str
        Path to the run directory.
    output_file_name: str
        Name of the output file (e.g. results.pkl). This is not a path, only the filename.
    slurm_subnmit_script_name: str
        Name of the slurm submit scriptfile (e.g. submit_slurm.sh)
    Returns:
    --------
    None.
    """
    output_exists = os.path.isfile(os.path.join(run_directory, output_file_name))
    slurm_script_exists = os.path.isfile(
        os.path.join(run_directory, slurm_submit_script_name)
    )

    if output_exists == False and slurm_script_exists == True:
        submit_slurm_job(run_dir=run_directory)


def mode(vec: np.ndarray) -> float:
    """Calculates and returns the mode of a vector of continuous data.

    Parameters:
    -----------
    vec: numpy.ndarray
        Vector of floats

    Returns:
    --------
    hist_mode: float
        Value corresponding to the peak of the histogram.
    """

    hist = np.histogram(vec, bins="auto")
    max_index = np.where(hist[0] == max(hist[0]))[0]
    hist_mode = np.mean((hist[1][max_index], hist[1][max_index + 1]))
    return hist_mode
