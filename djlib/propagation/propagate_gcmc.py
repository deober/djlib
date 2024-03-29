import os
import json
import thermocore.io.casm as cio
import djlib.djlib as dj
import djlib.mc.mc as mc
from glob import glob
import warnings
import numpy as np
from typing import List, Dict, Tuple


def propagation_project_namer(propagation_info_dict: dict):
    """Takes a dictionary containing the necessary information to set up a new casm project, and returns a name for that project as a string.
    Intended to be used as the namer argument in a gridspace_manager object.

    Parameters
    ----------
    propagation_info_dict : dict
        Dictionary containing the following keys:
            'template_project_root_path' : str
                Path to the casm project root
            'sample_index' : int
                Index of the eci selection. Also decides the name of the propagaiton directory.
            'eci' : np.ndarray
                ECI vector to write to the casm project
            'propagation_directory' : str
                Path to the directory which will contain all the propagation directories.

    Returns
    -------
    str
        Name of the propagation project
    """
    sample_index = propagation_info_dict["sample_index"]
    return "sample_index_" + str(sample_index)


def propagation_project_parser(
    propagated_casm_project_root_path: str, incomplete_override: bool = False
) -> dict:
    """Pulls all information from grand canonical monte carlo runs, 
    integrates the free energy and returns a dictionary containing all the information.
    Will not parse until all run statuses are complete.
    Intended to be used as the run_parser argument in a gridspace_manager object.

    Parameters
    ----------
    propagated_casm_project_root_path: str
        Path to the casm project root of the propagated project.

    Return
    -------
    all_data : dict
        Dictionary containing all the data from the grand canonical monte carlo project, 
        including all LTE, constant_temperature, cooling, and heating runs.
    """

    all_statuses = []
    # ***Load the status file for the grand canonical monte carlo runs, and check that all runs are complete***
    with open(
        os.path.join(
            propagated_casm_project_root_path, "grand_canonical_monte_carlo/status.json"
        ),
        "r",
    ) as f:
        status = json.load(f)
    # Collect all keys in a list. Each key corresponds to a list of dictionaries. Collect all values from each dictionary in the list, and append to all_statuses
    for key in status.keys():
        for dictionary in status[key]:
            all_statuses.append(list(dictionary.values())[0])
    # Check that all runs are "complete"
    if (
        not all(status == "complete" for status in all_statuses)
        and incomplete_override == False
    ):
        warnings.warn(
            "Not all runs are complete. Cannot parse until all runs are complete."
        )
        return None

    # If all runs are complete, create a gridspace manager object for each type of run, and collect all data in a dictionary.
    lte_gridspace = dj.gridspace_manager(
        origin_dir=os.path.join(
            propagated_casm_project_root_path, "grand_canonical_monte_carlo/MC_LTE"
        ),
        namer=mc.mc_run_namer,
        run_parser=mc.mc_run_parser,
    )
    t_const_gridspace = dj.gridspace_manager(
        origin_dir=os.path.join(
            propagated_casm_project_root_path, "grand_canonical_monte_carlo/MC_t_const"
        ),
        namer=mc.mc_run_namer,
        run_parser=mc.mc_run_parser,
    )
    cooling_gridspace = dj.gridspace_manager(
        origin_dir=os.path.join(
            propagated_casm_project_root_path, "grand_canonical_monte_carlo/MC_cooling"
        ),
        namer=mc.mc_run_namer,
        run_parser=mc.mc_run_parser,
    )
    heating_gridspace = dj.gridspace_manager(
        origin_dir=os.path.join(
            propagated_casm_project_root_path, "grand_canonical_monte_carlo/MC_heating"
        ),
        namer=mc.mc_run_namer,
        run_parser=mc.mc_run_parser,
    )

    # Pull data for all runs
    lte_gridspace.collect_data()
    t_const_gridspace.collect_data()
    cooling_gridspace.collect_data()
    heating_gridspace.collect_data()

    # Combine all data into a single dictionary
    all_data = {}
    all_data["LTE"] = lte_gridspace.data
    all_data["T_const"] = t_const_gridspace.data
    all_data["cooling"] = cooling_gridspace.data
    all_data["heating"] = heating_gridspace.data
    return all_data


def propagation_casm_project_creator(
    propagation_info_dict: dict, propagation_project_root_path: str
) -> None:
    """Copies a pre-templated casm project, writes a specific eci vector to
    project_root/cluster_expansions/clex.formation_energy/calctype.default/ref.default/bset.default/eci.default/eci.json
    and creates all standard directories for typical grand canonical monte carlo simulations.
    Intended to be used as the project_creator argument in a gridspace_manager object.

    NOTE: details in this function, and its helper function are very specific to the material system being studied. 
    Please copy and modify as needed.

    Parameters
    ----------
    propagation_info_dict : dict
        Dictionary containing the following keys:
            'template_project_root_path' : str
                Path to the casm project root
            'markov_chain_index' : int
                Index of the eci selection in the posterior markov chain to be used. Also decides the name of the propagaiton directory.
            'eci' : np.ndarray
                ECI vector to write to the casm project
            'propagation_directory' : str
                Path to the directory which will contain all the propagation directories.
    propagation_project_root_path : str
        Path to the casm project root of the propagation project.

    Returns
    -------
    None
    """

    # Copy the template project to the propagation directory, and name it according to the markov chain index.
    template_project_root_path = propagation_info_dict["template_project_root_path"]
    os.system(
        "cp -r "
        + template_project_root_path
        + "/."
        + " "
        + propagation_project_root_path
    )

    # Load basis.json
    basis_json_path = os.path.join(
        template_project_root_path, "basis_sets/bset.default/basis.json"
    )
    with open(basis_json_path, "r") as f:
        basis_dict = json.load(f)

    # Append eci to the basis dictionary
    eci = propagation_info_dict["eci"]
    eci_dict = cio.append_ECIs_to_basis_data(ecis=eci, basis_data=basis_dict)

    # Write the dictionary to eci.json within the new project.
    with open(
        os.path.join(
            propagation_project_root_path,
            "cluster_expansions/clex.formation_energy/calctype.default/ref.default/bset.default/eci.default/eci.json",
        ),
        "w",
    ) as f:
        json.dump(eci_dict, f)

    # Create a grand canonical monte carlo directory within the new project.
    os.system(
        "mkdir "
        + os.path.join(propagation_project_root_path, "grand_canonical_monte_carlo")
    )

    # Write the propagation_info_dict to a json file to run_info.json within the grand_canonical_monte_carlo directory.
    tmp_propagation_info_dict = propagation_info_dict.copy()
    tmp_propagation_info_dict["eci"] = tmp_propagation_info_dict["eci"].tolist()
    with open(
        os.path.join(
            propagation_project_root_path, "grand_canonical_monte_carlo/run_info.json",
        ),
        "w",
    ) as f:
        json.dump(tmp_propagation_info_dict, f)
    del tmp_propagation_info_dict

    # If it doesn't exist, create a status.json file in the grand canonical monte carlo directory to keep track of all monte carlo run statuses.
    if not os.path.exists(
        os.path.join(
            propagation_project_root_path, "grand_canonical_monte_carlo/status.json"
        )
    ):
        with open(
            os.path.join(
                propagation_project_root_path, "grand_canonical_monte_carlo/status.json"
            ),
            "w",
        ) as f:
            json.dump({}, f)

    # Create an MC_cooling, MC_heating, MC_LTE, and MC_t_const directories within the new grand canonical monte carlo directory.
    os.system(
        "mkdir "
        + os.path.join(
            propagation_project_root_path, "grand_canonical_monte_carlo/MC_cooling"
        )
    )
    os.system(
        "mkdir "
        + os.path.join(
            propagation_project_root_path, "grand_canonical_monte_carlo/MC_heating"
        )
    )
    os.system(
        "mkdir "
        + os.path.join(
            propagation_project_root_path, "grand_canonical_monte_carlo/MC_LTE"
        )
    )
    os.system(
        "mkdir "
        + os.path.join(
            propagation_project_root_path, "grand_canonical_monte_carlo/MC_t_const"
        )
    )
    # Set up MC runs for my specific project
    heating_and_cooling_at_50_percent_ground_state(propagation_project_root_path)


def collect_all_statuses_gcmc(MC_gridspace: str) -> List[Dict]:
    """Collects statuses from all status.json files within the given gridspace. Returns a list of dictionaries,
    where each dictionary key is the run directory name, and value is the run status as a string.

    Parameters
    ----------
    MC_gridspace : str
        Path to the gridspace directory

    Returns
    -------
    status_list : list
        List of dictionaries, where each dictionary key is the run directory name, and value is the run status as a string.
    """
    run_directories = glob(os.path.join(MC_gridspace, "mu_*"))
    status_list = []
    for run_directory in run_directories:
        with open(os.path.join(run_directory, "status.json"), "r") as f:
            status = json.load(f)
        status_list.append({run_directory.split("/")[-1]: status["status"]})
    return status_list


def propagation_casm_project_status_updater(
    propagated_casm_project_root_path: str,
) -> None:
    """Checks and updates status for the grand canonical monte carlo runs of an entire casm project.
    Operates by creating gridspace managers for each run type (heating, cooling, constant temperature, LTE) in a given casm project, 
    then running the status update method for each. 
    Intended to be used as a status updater for a casm project.

    Parameters
    ----------
    propagated_casm_project_root_path : str
        Path to the casm project root

    Returns
    -------
    None
    """
    # Create a gridspace manager object for MC_cooling, MC_heating, MC_LTE, and MC_t_const directories
    # Run the status update method for each gridspace manager object.
    # For each type of monte carlo run, collect the statuses of all runs, and append to grand_canonical_monte_carlo/status.json

    # LTE run update and collect status
    print("Updating LTE runs")
    lte_gridspace_manager = dj.gridspace_manager(
        origin_dir=os.path.join(
            propagated_casm_project_root_path, "grand_canonical_monte_carlo/MC_LTE"
        ),
        namer=mc.mc_run_namer,
        status_updater=mc.mc_status_updater,
    )
    lte_gridspace_manager.update_status()
    with open(
        os.path.join(
            propagated_casm_project_root_path, "grand_canonical_monte_carlo/status.json"
        ),
        "r",
    ) as f:
        status_dict = json.load(f)
    status_dict["MC_LTE"] = collect_all_statuses_gcmc(lte_gridspace_manager.origin_dir)
    with open(
        os.path.join(
            propagated_casm_project_root_path, "grand_canonical_monte_carlo/status.json"
        ),
        "w",
    ) as f:
        json.dump(status_dict, f)

    # T_const update and collect status
    print("Updating constant temperature runs. ")
    t_const_gridspace_manager = dj.gridspace_manager(
        origin_dir=os.path.join(
            propagated_casm_project_root_path, "grand_canonical_monte_carlo/MC_t_const"
        ),
        namer=mc.mc_run_namer,
        status_updater=mc.mc_status_updater,
    )
    t_const_gridspace_manager.update_status()
    with open(
        os.path.join(
            propagated_casm_project_root_path, "grand_canonical_monte_carlo/status.json"
        ),
        "r",
    ) as f:
        status_dict = json.load(f)
    status_dict["MC_t_const"] = collect_all_statuses_gcmc(
        t_const_gridspace_manager.origin_dir
    )
    with open(
        os.path.join(
            propagated_casm_project_root_path, "grand_canonical_monte_carlo/status.json"
        ),
        "w",
    ) as f:
        json.dump(status_dict, f)

    # Heating run update and collect status
    print("Updating heating runs")
    heating_gridspace_manager = dj.gridspace_manager(
        origin_dir=os.path.join(
            propagated_casm_project_root_path, "grand_canonical_monte_carlo/MC_heating"
        ),
        namer=mc.mc_run_namer,
        status_updater=mc.mc_status_updater,
    )
    heating_gridspace_manager.update_status()
    with open(
        os.path.join(
            propagated_casm_project_root_path, "grand_canonical_monte_carlo/status.json"
        ),
        "r",
    ) as f:
        status_dict = json.load(f)
    status_dict["MC_heating"] = collect_all_statuses_gcmc(
        heating_gridspace_manager.origin_dir
    )
    with open(
        os.path.join(
            propagated_casm_project_root_path, "grand_canonical_monte_carlo/status.json"
        ),
        "w",
    ) as f:
        json.dump(status_dict, f)

    # Cooling run update and collect status
    print("Updating cooling runs")
    cooling_gridspace_manager = dj.gridspace_manager(
        origin_dir=os.path.join(
            propagated_casm_project_root_path, "grand_canonical_monte_carlo/MC_cooling"
        ),
        namer=mc.mc_run_namer,
        status_updater=mc.mc_status_updater,
    )
    cooling_gridspace_manager.update_status()
    with open(
        os.path.join(
            propagated_casm_project_root_path, "grand_canonical_monte_carlo/status.json"
        ),
        "r",
    ) as f:
        status_dict = json.load(f)
    status_dict["MC_cooling"] = collect_all_statuses_gcmc(
        cooling_gridspace_manager.origin_dir
    )
    with open(
        os.path.join(
            propagated_casm_project_root_path, "grand_canonical_monte_carlo/status.json"
        ),
        "w",
    ) as f:
        json.dump(status_dict, f, indent=4)


def propagation_casm_project_submitter(propagated_casm_project_root_path: str) -> None:
    """Runs all grand canonical monte carlo simulations for an entire casm project.
    Cooling runs must initialize from completed constant Temperature runs, and heating runs must initialize from completed LTE runs.
    Intended to be used as a submitter for a gridspace manager object.
    
    NOTE:Dependent runs will not submit if necessary dependencies are not complete. 
    Because of this, the gridspace manager "format_run_dirs()" method 
    should be used AGAIN after constant temperature runs are complete, and before cooling runs are submitted. 
    This allows cooling runs to initialize from completed constant temperature runs. 

    Parameters
    ----------
    propagated_casm_project_root_path: str
        Path to the casm project, where grand canonical monte carlo simulations will be run.

    Returns:
        None.
    """

    # Create a gridspace manager object for each type of monte carlo run.
    # Run the submit method for each gridspace manager object.

    # LTE run submit
    lte_gridspace_manager = dj.gridspace_manager(
        origin_dir=os.path.join(
            propagated_casm_project_root_path, "grand_canonical_monte_carlo/MC_LTE"
        ),
        namer=mc.mc_run_namer,
        status_updater=mc.mc_status_updater,
        run_submitter=mc.mc_run_submitter,
    )
    lte_gridspace_manager.run_valid_calculations()

    # T_const run submit
    t_const_gridspace_manager = dj.gridspace_manager(
        origin_dir=os.path.join(
            propagated_casm_project_root_path, "grand_canonical_monte_carlo/MC_t_const"
        ),
        namer=mc.mc_run_namer,
        status_updater=mc.mc_status_updater,
        run_submitter=mc.mc_run_submitter,
    )
    t_const_gridspace_manager.run_valid_calculations()

    # Heating run submit
    heating_gridspace_manager = dj.gridspace_manager(
        origin_dir=os.path.join(
            propagated_casm_project_root_path, "grand_canonical_monte_carlo/MC_heating"
        ),
        namer=mc.mc_run_namer,
        status_updater=mc.mc_status_updater,
        run_submitter=mc.mc_run_submitter,
    )
    heating_gridspace_manager.run_valid_calculations()

    # Cooling run submit
    cooling_gridspace_manager = dj.gridspace_manager(
        origin_dir=os.path.join(
            propagated_casm_project_root_path, "grand_canonical_monte_carlo/MC_cooling"
        ),
        namer=mc.mc_run_namer,
        status_updater=mc.mc_status_updater,
        run_submitter=mc.mc_run_submitter,
    )
    # Collect all constant temperature run statuses in a list. If all are "complete", then submit cooling runs.
    # Otherwise, warn the user that cooling runs cannot be submitted until constant temperature runs are complete.
    # TODO: Change this so that cooling runs only require the constant T run that matches their highest temperature.
    with open(
        os.path.join(
            propagated_casm_project_root_path, "grand_canonical_monte_carlo/status.json"
        ),
        "r",
    ) as f:
        status_dict = json.load(f)
    t_const_statuses = [
        list(status_dict["MC_t_const"][i].values())[0]
        for i in range(len(status_dict["MC_t_const"]))
    ]
    if all(status == "complete" for status in t_const_statuses):
        cooling_gridspace_manager.run_valid_calculations()
    else:
        print(
            "Not all constant temperature runs are complete. Cooling runs cannot be submitted until constant temperature runs are complete. Please check constant temperature runs in %s "
            % os.path.join(
                propagated_casm_project_root_path,
                "grand_canonical_monte_carlo/MC_t_const",
            )
        )


def heating_and_cooling_at_50_percent_ground_state(casm_root_path: str):
    """A very specific helper function to be used with "propagation_casm_project_creator()". Details of this function depend heavily on 
    the specific material system being studied. Users should copy and modify this function to suit their needs.
    
    Writes all necessary files for heating and cooling runs for the ground state at 50% composition. This includes:
        -High temperature constant t runs from very low to very high chemical potential and very high to very low chemical potential
        -Cooling runs from the high temperature constant t runs
        -Low Temperature Expansion (LTE) runs
        -Heating runs that initialize from LTE runs


    Parameters
    ----------
    casm_root_path: str
        Path to the casm project root

    Returns
    -------
    None
    """

    # Create a dj.gridspace_manager object to control lte runs. Then format the run directories.
    lte_dir = os.path.join(casm_root_path, "grand_canonical_monte_carlo/MC_LTE")
    lte_param_list_of_dicts = [
        {
            "mu_start": -0.221,
            "mu_stop": -0.221,
            "mu_increment": 0.0,
            "T_start": 40.0,
            "T_stop": 40.0,
            "T_increment": 0.0,
            "supercell": [[24, 0, 0], [0, 24, 0], [0, 0, 24]],
            "hours": 24,
        }
    ]
    lte_gs = dj.gridspace_manager(
        origin_dir=lte_dir,
        namer=mc.mc_run_namer,
        run_creator=mc.mc_lte_run_creator,
        status_updater=mc.mc_status_updater,
        run_submitter=mc.mc_run_submitter,
        grid_params=lte_param_list_of_dicts,
    )
    lte_gs.format_run_dirs()

    # Create a dj.gridspace_manager object to control high temperature constant t runs. Then format the run directories.
    t_const_dir = os.path.join(casm_root_path, "grand_canonical_monte_carlo/MC_t_const")
    t_const_param_list_of_dicts = [
        {
            "mu_start": -3.0,
            "mu_stop": 3.0,
            "mu_increment": 0.05,
            "T_start": 2000.0,
            "T_stop": 2000.0,
            "T_increment": 0.0,
            "supercell": [[24, 0, 0], [0, 24, 0], [0, 0, 24]],
            "hours": 35,
        }
    ]
    t_const_param_list_of_dicts += [
        {
            "mu_start": 3.0,
            "mu_stop": -3.0,
            "mu_increment": -0.05,
            "T_start": 2000.0,
            "T_stop": 2000.0,
            "T_increment": 0.0,
            "supercell": [[24, 0, 0], [0, 24, 0], [0, 0, 24]],
            "hours": 35,
        }
    ]
    t_const_gs = dj.gridspace_manager(
        origin_dir=t_const_dir,
        namer=mc.mc_run_namer,
        run_creator=mc.mc_run_creator,
        status_updater=mc.mc_status_updater,
        run_submitter=mc.mc_run_submitter,
        grid_params=t_const_param_list_of_dicts,
    )
    t_const_gs.format_run_dirs()

    # Create a dj.gridspace_manager object to control cooling runs
    cooling_dir = os.path.join(casm_root_path, "grand_canonical_monte_carlo/MC_cooling")
    cooling_param_list_of_dicts = [
        {
            "mu_start": -0.221,
            "mu_stop": -0.221,
            "mu_increment": 0.0,
            "T_start": 2000.0,
            "T_stop": 40.0,
            "T_increment": -20.0,
            "supercell": [[24, 0, 0], [0, 24, 0], [0, 0, 24]],
            "hours": 24,
        }
    ]
    cooling_gs = dj.gridspace_manager(
        origin_dir=cooling_dir,
        namer=mc.mc_run_namer,
        run_creator=mc.mc_run_creator,
        status_updater=mc.mc_status_updater,
        run_submitter=mc.mc_run_submitter,
        grid_params=cooling_param_list_of_dicts,
    )
    cooling_gs.format_run_dirs()

    # Make sure the cooling run initializes from the constant temperature run with the closest chemical potential value
    # Assuming there are only two constant temperature runs, that they are at the high temperature, and that they cover the same chemical potential
    # values (in opposite orders)
    # First, look up the constant temperature run with the closest starting chemical potential.
    # Then, find the index of the chemical potential with the closest value in the cooling run.
    # This index marks the conditions directory that the cooling run should initialize from.

    t_const_runs = np.array(glob(os.path.join(t_const_dir, "mu_*")))
    t_const_mu = []
    t_const_temperatures = []
    for t_const_run in t_const_runs:
        mu_temporary = mc.read_mc_settings(
            os.path.join(t_const_run, "mc_settings.json")
        )[0]
        temperature_temporary = mc.read_mc_settings(
            os.path.join(t_const_run, "mc_settings.json")
        )[1][0]
        t_const_mu.append(mu_temporary)
        t_const_temperatures.append(temperature_temporary)
    t_const_mu = np.array(t_const_mu)
    t_const_temperatures = np.array(t_const_temperatures)

    # find the indices of t_const_temperatures that are equal to 2000
    t_const_2000_indices = np.where(t_const_temperatures == 2000)[0]

    # Downsample t_const_mu to only include the mu values at 2000 K
    t_const_mu_2000 = t_const_mu[t_const_2000_indices]

    # Iterate through all cooling runs
    for cooling_run_path in glob(os.path.join(cooling_dir, "mu_*")):
        # Get the chemical potential from the cooling run
        cooling_mu = mc.read_mc_settings(
            os.path.join(cooling_run_path, "mc_settings.json")
        )[0][0]

        # Find the index of the constant temperature run with the closest initial chemical potential
        closest_t_const_index = np.argmin(np.abs(t_const_mu_2000[:, 0] - cooling_mu))

        # Find the closest conditions index in the constant temperature run to initialize the cooling run from.
        closest_conditions_index = np.argmin(
            np.abs(t_const_mu_2000[closest_t_const_index, :] - cooling_mu)
        )

        # First, check that the closest conditions file exists. Raise a warning if it does not.
        if not os.path.exists(
            os.path.join(
                t_const_runs[t_const_2000_indices][closest_t_const_index],
                "conditions.%d/final_state.json" % closest_conditions_index,
            )
        ):
            print(
                "The closest conditions file does not exist. Check that the constant temperature runs have been run."
            )
        else:
            # Write the closest conditions index to the cooling run's mc_settings.json file
            with open(os.path.join(cooling_run_path, "mc_settings.json"), "r") as f:
                cooling_settings = json.load(f)
            cooling_settings["driver"]["motif"]["configdof"] = os.path.join(
                t_const_runs[closest_t_const_index],
                "conditions.%d/final_state.json" % closest_conditions_index,
            )
            cooling_settings["driver"]["motif"].pop("configname", None)
            with open(os.path.join(cooling_run_path, "mc_settings.json"), "w") as f:
                json.dump(cooling_settings, f, indent=4)
    """
    #Old code
    # Iterate through all cooling runs
    for cooling_run_path in glob(os.path.join(cooling_dir, "mu_*")):
        # Get the chemical potential from the cooling run
        cooling_mu = mc.read_mc_settings(
            os.path.join(cooling_run_path, "mc_settings.json")
        )[0][0]

        # Find the index of the constant temperature run with the closest initial chemical potential
        closest_t_const_index = np.argmin(np.abs(t_const_mu[:, 0] - cooling_mu))

        # Find the closest conditions index in the constant temperature run to initialize the cooling run from.
        closest_conditions_index = np.argmin(
            np.abs(t_const_mu[closest_t_const_index, :] - cooling_mu)
        )

        # First, check that the closest conditions file exists. Raise a warning if it does not.
        if not os.path.exists(
            os.path.join(
                t_const_runs[closest_t_const_index],
                "conditions.%d/final_state.json" % closest_conditions_index,
            )
        ):
            print(
                "The closest conditions file does not exist. Check that the constant temperature runs have been run."
            )
        else:
            # Write the closest conditions index to the cooling run's mc_settings.json file
            with open(os.path.join(cooling_run_path, "mc_settings.json"), "r") as f:
                cooling_settings = json.load(f)
            cooling_settings["driver"]["motif"]["configdof"] = os.path.join(
                t_const_runs[closest_t_const_index],
                "conditions.%d/final_state.json" % closest_conditions_index,
            )
            cooling_settings["driver"]["motif"].pop("configname", None)
            with open(os.path.join(cooling_run_path, "mc_settings.json"), "w") as f:
                json.dump(cooling_settings, f, indent=4)
    """

    # Create a dj.gridspace_manager object to control heating runs
    heating_dir = os.path.join(casm_root_path, "grand_canonical_monte_carlo/MC_heating")
    heating_param_list_of_dicts = [
        {
            "mu_start": -0.221,
            "mu_stop": -0.221,
            "mu_increment": 0.0,
            "T_start": 40.0,
            "T_stop": 2000.0,
            "T_increment": 20.0,
            "supercell": [[24, 0, 0], [0, 24, 0], [0, 0, 24]],
            "hours": 24,
        }
    ]
    heating_gs = dj.gridspace_manager(
        origin_dir=heating_dir,
        namer=mc.mc_run_namer,
        run_creator=mc.mc_run_creator,
        status_updater=mc.mc_status_updater,
        run_submitter=mc.mc_run_submitter,
        grid_params=heating_param_list_of_dicts,
    )
    heating_gs.format_run_dirs()


# TODO: write function to find the clex-predicted ground state for a given chemical potential
# TODO: write function to find the transformation matrix that forms the supercell of a given configuration (Will involve a casm call, should probably be placed in casmcalls)


def sgcmc_setup(casm_root_path: str):
    """A very specific function: writes all necessary files to create a phase diagram in the ZrN FCC structure. 
    Modify as necessary for other scenarios. 
    Generally, creates multiple constant temperature runs, LTE runs, cooling runs and heating runs.


    Parameters
    ----------
    casm_root_path: str
        Path to the casm project root

    Returns
    -------
    None
    """

    mu_list = np.linspace(-2, 2, 41)
    print(mu_list)
    # Create a dj.gridspace_manager object to control lte runs. Then format the run directories.
    lte_dir = os.path.join(casm_root_path, "grand_canonical_monte_carlo/MC_LTE")
    lte_param_list_of_dicts = [
        {
            "mu_start": mu,
            "mu_stop": mu,
            "mu_increment": 0.0,
            "T_start": 40.0,
            "T_stop": 40.0,
            "T_increment": 0.0,
            "supercell": [[24, 0, 0], [0, 24, 0], [0, 0, 24]],
            "hours": 24,
        }
        for mu in mu_list
    ]
    lte_gs = dj.gridspace_manager(
        origin_dir=lte_dir,
        namer=mc.mc_run_namer,
        run_creator=mc.mc_lte_run_creator,
        status_updater=mc.mc_status_updater,
        run_submitter=mc.mc_run_submitter,
        grid_params=lte_param_list_of_dicts,
    )
    lte_gs.format_run_dirs()

    # Create a dj.gridspace_manager object to control high temperature constant t runs. Then format the run directories.
    t_const_dir = os.path.join(casm_root_path, "grand_canonical_monte_carlo/MC_t_const")
    t_const_param_list_of_dicts = [
        {
            "mu_start": -3.0,
            "mu_stop": 3.0,
            "mu_increment": 0.05,
            "T_start": 2000.0,
            "T_stop": 2000.0,
            "T_increment": 0.0,
            "supercell": [[24, 0, 0], [0, 24, 0], [0, 0, 24]],
            "hours": 35,
        }
    ]
    t_const_param_list_of_dicts += [
        {
            "mu_start": 3.0,
            "mu_stop": -3.0,
            "mu_increment": -0.05,
            "T_start": 2000.0,
            "T_stop": 2000.0,
            "T_increment": 0.0,
            "supercell": [[24, 0, 0], [0, 24, 0], [0, 0, 24]],
            "hours": 35,
        }
    ]
    t_const_gs = dj.gridspace_manager(
        origin_dir=t_const_dir,
        namer=mc.mc_run_namer,
        run_creator=mc.mc_run_creator,
        status_updater=mc.mc_status_updater,
        run_submitter=mc.mc_run_submitter,
        grid_params=t_const_param_list_of_dicts,
    )
    t_const_gs.format_run_dirs()

    # Create a dj.gridspace_manager object to control cooling runs
    cooling_dir = os.path.join(casm_root_path, "grand_canonical_monte_carlo/MC_cooling")
    cooling_param_list_of_dicts = [
        {
            "mu_start": mu,
            "mu_stop": mu,
            "mu_increment": 0.0,
            "T_start": 2000.0,
            "T_stop": 40.0,
            "T_increment": -20.0,
            "supercell": [[24, 0, 0], [0, 24, 0], [0, 0, 24]],
            "hours": 24,
        }
        for mu in mu_list
    ]
    cooling_gs = dj.gridspace_manager(
        origin_dir=cooling_dir,
        namer=mc.mc_run_namer,
        run_creator=mc.mc_run_creator,
        status_updater=mc.mc_status_updater,
        run_submitter=mc.mc_run_submitter,
        grid_params=cooling_param_list_of_dicts,
    )
    cooling_gs.format_run_dirs()

    # Make sure the cooling run initializes from the constant temperature run with the closest chemical potential value
    # Assuming there are only two constant temperature runs, that they are at the high temperature, and that they cover the same chemical potential
    # values (in opposite orders)
    # First, look up the constant temperature run with the closest starting chemical potential.
    # Then, find the index of the chemical potential with the closest value in the cooling run.
    # This index marks the conditions directory that the cooling run should initialize from.

    t_const_runs = glob(os.path.join(t_const_dir, "mu_*"))
    t_const_mu = []
    for t_const_run in t_const_runs:
        t_const_mu.append(
            mc.read_mc_settings(os.path.join(t_const_run, "mc_settings.json"))[0]
        )
    t_const_mu = np.array(t_const_mu)

    # Iterate through all cooling runs
    for cooling_run_path in glob(os.path.join(cooling_dir, "mu_*")):
        # Get the chemical potential from the cooling run
        cooling_mu = mc.read_mc_settings(
            os.path.join(cooling_run_path, "mc_settings.json")
        )[0][0]

        # Find the index of the constant temperature run with the closest initial chemical potential
        closest_t_const_index = np.argmin(np.abs(t_const_mu[:, 0] - cooling_mu))

        # Find the closest conditions index in the constant temperature run to initialize the cooling run from.
        closest_conditions_index = np.argmin(
            np.abs(t_const_mu[closest_t_const_index, :] - cooling_mu)
        )

        # First, check that the closest conditions file exists. Raise a warning if it does not.
        if not os.path.exists(
            os.path.join(
                t_const_runs[closest_t_const_index],
                "conditions.%d/final_state.json" % closest_conditions_index,
            )
        ):
            print(
                "The closest conditions file does not exist. Check that the constant temperature runs have been run."
            )
        else:
            # Write the closest conditions index to the cooling run's mc_settings.json file
            with open(os.path.join(cooling_run_path, "mc_settings.json"), "r") as f:
                cooling_settings = json.load(f)
            cooling_settings["driver"]["motif"]["configdof"] = os.path.join(
                t_const_runs[closest_t_const_index],
                "conditions.%d/final_state.json" % closest_conditions_index,
            )
            cooling_settings["driver"]["motif"].pop("configname", None)
            with open(os.path.join(cooling_run_path, "mc_settings.json"), "w") as f:
                json.dump(cooling_settings, f, indent=4)

    # Create a dj.gridspace_manager object to control heating runs
    heating_dir = os.path.join(casm_root_path, "grand_canonical_monte_carlo/MC_heating")
    heating_param_list_of_dicts = [
        {
            "mu_start": mu,
            "mu_stop": mu,
            "mu_increment": 0.0,
            "T_start": 40.0,
            "T_stop": 2000.0,
            "T_increment": 20.0,
            "supercell": [[24, 0, 0], [0, 24, 0], [0, 0, 24]],
            "hours": 24,
        }
        for mu in mu_list
    ]
    heating_gs = dj.gridspace_manager(
        origin_dir=heating_dir,
        namer=mc.mc_run_namer,
        run_creator=mc.mc_run_creator,
        status_updater=mc.mc_status_updater,
        run_submitter=mc.mc_run_submitter,
        grid_params=heating_param_list_of_dicts,
    )
    heating_gs.format_run_dirs()


def sgcmc_casm_project_creator(propagation_info_dict, propagation_project_root_path):
    """Copies a pre-templated casm project, writes a specific eci vector to
    project_root/cluster_expansions/clex.formation_energy/calctype.default/ref.default/bset.default/eci.default/eci.json
    and creates all standard directories for typical grand canonical monte carlo simulations.

    Parameters
    ----------
    propagation_info_dict : dict
        Dictionary containing the following keys:
            'template_project_root_path' : str
                Path to the casm project root
            'markov_chain_index' : int
                Index of the eci selection in the posterior markov chain to be used. Also decides the name of the propagaiton directory.
            'eci' : np.ndarray
                ECI vector to write to the casm project
            'propagation_directory' : str
                Path to the directory which will contain all the propagation directories.

    Returns
    -------
    None
    """

    # Copy the template project to the propagation directory, and name it according to the markov chain index.
    template_project_root_path = propagation_info_dict["template_project_root_path"]
    os.system(
        "cp -r "
        + template_project_root_path
        + "/."
        + " "
        + propagation_project_root_path
    )

    # Load basis.json
    basis_json_path = os.path.join(
        template_project_root_path, "basis_sets/bset.default/basis.json"
    )
    with open(basis_json_path, "r") as f:
        basis_dict = json.load(f)

    # Append eci to the basis dictionary
    eci = propagation_info_dict["eci"]
    eci_dict = cio.append_ECIs_to_basis_data(ecis=eci, basis_data=basis_dict)

    # If it doesn't already exist, Write the dictionary to eci.json within the new project.
    if (
        os.path.isfile(
            os.path.join(
                propagation_project_root_path,
                "cluster_expansions/clex.formation_energy/calctype.default/ref.default/bset.default/eci.default/eci.json",
            )
        )
        == False
    ):
        with open(
            os.path.join(
                propagation_project_root_path,
                "cluster_expansions/clex.formation_energy/calctype.default/ref.default/bset.default/eci.default/eci.json",
            ),
            "w",
        ) as f:
            json.dump(eci_dict, f)

    else:
        print("ECI file already exists. Skipping.")

    # Create a grand canonical monte carlo directory within the new project.
    os.system(
        "mkdir "
        + os.path.join(propagation_project_root_path, "grand_canonical_monte_carlo")
    )

    # Write the propagation_info_dict to a json file to run_info.json within the grand_canonical_monte_carlo directory.
    tmp_propagation_info_dict = propagation_info_dict.copy()
    tmp_propagation_info_dict["eci"] = tmp_propagation_info_dict["eci"].tolist()
    with open(
        os.path.join(
            propagation_project_root_path, "grand_canonical_monte_carlo/run_info.json",
        ),
        "w",
    ) as f:
        json.dump(tmp_propagation_info_dict, f)
    del tmp_propagation_info_dict

    # If it doesn't exist, create a status.json file in the grand canonical monte carlo directory to keep track of all monte carlo run statuses.
    if not os.path.exists(
        os.path.join(
            propagation_project_root_path, "grand_canonical_monte_carlo/status.json"
        )
    ):
        with open(
            os.path.join(
                propagation_project_root_path, "grand_canonical_monte_carlo/status.json"
            ),
            "w",
        ) as f:
            json.dump({}, f)

    # Create an MC_cooling, MC_heating, MC_LTE, and MC_t_const directories within the new grand canonical monte carlo directory.
    os.system(
        "mkdir "
        + os.path.join(
            propagation_project_root_path, "grand_canonical_monte_carlo/MC_cooling"
        )
    )
    os.system(
        "mkdir "
        + os.path.join(
            propagation_project_root_path, "grand_canonical_monte_carlo/MC_heating"
        )
    )
    os.system(
        "mkdir "
        + os.path.join(
            propagation_project_root_path, "grand_canonical_monte_carlo/MC_LTE"
        )
    )
    os.system(
        "mkdir "
        + os.path.join(
            propagation_project_root_path, "grand_canonical_monte_carlo/MC_t_const"
        )
    )
    # Set up MC runs for my specific project
    sgcmc_setup(propagation_project_root_path)
