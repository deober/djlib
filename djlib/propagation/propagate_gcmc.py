import os
import json
import thermocore.io.casm as cio
import djlib.djlib as dj
import djlib.mc.mc as mc
from glob import glob
import warnings


def propagation_project_namer(propagation_info_dict):
    """Takes a dictionary containing the necessary information to set up a new casm project, and returns a name for that project as a string. 

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
    str
        Name of the propagation project
    """
    markov_chain_index = propagation_info_dict["markov_chain_index"]
    return "markov_chain_index_" + str(markov_chain_index)


def propagation_project_parser(propagated_casm_project_root_path: str):
    """Pulls all information from grand canonical monte carlo runs, integrates the free energy and returns a dictionary containing all the information. 
    Will not parse until all run statuses are complete. 
    
    Parameters
    ----------
    propagated_casm_project_root_path: str
        Path to the casm project root of the propagated project. 

    Returns
    -------
    
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
    if not all(status == "complete" for status in all_statuses):
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
        run_parser=mc.read_lte_results,
    )
    t_const_gridspace = dj.gridspace_manager(
        origin_dir=os.path.join(
            propagated_casm_project_root_path, "grand_canonical_monte_carlo/MC_T_const"
        ),
        namer=mc.mc_run_namer,
        run_parser=mc.read_mc_results_file,
    )
    cooling_gridspace = dj.gridspace_manager(
        origin_dir=os.path.join(
            propagated_casm_project_root_path, "grand_canonical_monte_carlo/MC_cooling"
        ),
        namer=mc.mc_run_namer,
        run_parser=mc.read_mc_results_file,
    )
    heating_gridspace = dj.gridspace_manager(
        origin_dir=os.path.join(
            propagated_casm_project_root_path, "grand_canonical_monte_carlo/MC_heating"
        ),
        namer=mc.mc_run_namer,
        run_parser=mc.read_mc_results_file,
    )

    # Pull data for all runs
    lte_gridspace.collect_data()
    t_const_gridspace.collect_data()
    cooling_gridspace.collect_data()
    heating_gridspace.collect_data()

    # TODO: integrate the free energy across t_const runs
    # TODO: integrate the free energy across LTE runs
    # TODO: integrate the free energy across cooling runs:
    # Needs to initialize from a t_const run, but there are multiple t_const paths.
    # Choose the path that starts at a chemical potential that is closest to the chemical potential of the cooling run.
    # TODO: integrate the free energy across heating runs:
    # Needs to initialize from a LTE run; make sure that the selected LTE configuration and free energy reference are at the same temperature as the heating run initial temperature.
    # Also, heating runs are sensitive to the initial supercell choice. Heating runs should start from the clex-predicted supercell


def propagation_casm_project_creator(
    propagation_info_dict, propagation_project_root_path
):
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


def collect_all_statuses_gcmc(MC_gridspace):
    """Collects statuses from all status.json files within the given gridspace Returns a list of dictionaries, 
    where each dictionary key is the run directory name, and value is the run status as a string. 

    Parameters
    ----------
    MC_gridspace : str
        Path to the gridspace directory

    Returns
    -------
    list
        List of dictionaries, where each dictionary key is the run directory name, and value is the run status as a string. 
    """
    run_directories = glob(os.path.join(MC_gridspace, "mu_*"))
    status_list = []
    for run_directory in run_directories:
        with open(os.path.join(run_directory, "status.json"), "r") as f:
            status = json.load(f)
        status_list.append({run_directory.split("/")[-1]: status["status"]})
    return status_list


def propagation_casm_project_status_updater(propagated_casm_project_root_path: str):
    """Checks and updates status for the grand canonical monte carlo runs of an entire casm project. 

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


def propagation_casm_project_submitter(propagated_casm_project_root_path: str):
    """Runs all grand canonical monte carlo simulations for an entire casm project.
    Cooling runs must initialize from completed constant Temperature runs, and heating runs must initialize from completed LTE runs. 
    Dependent runs will not submit if necessary dependencies are not complete. 

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
    # TODO: Enforce that heating runs must initialize from completed LTE runs. If LTE is not complete not, do not submit.
    heating_gridspace_manager = dj.gridspace_manager(
        origin_dir=os.path.join(
            propagated_casm_project_root_path, "grand_canonical_monte_carlo/MC_heating"
        ),
        namer=mc.mc_run_namer,
        status_updater=mc.mc_status_updater,
        run_submitter=mc.mc_run_submitter,
    )
    # Collect all LTE run statuses in a list. If all are "complete", then submit heating runs.
    # Otherwise, warn the user that heating runs cannot be submitted until LTE runs are complete.
    with open(
        os.path.join(
            propagated_casm_project_root_path, "grand_canonical_monte_carlo/status.json"
        ),
        "r",
    ) as f:
        status_dict = json.load(f)
    lte_statuses = [
        list(status_dict["MC_LTE"][i].values())[0]
        for i in range(len(status_dict["MC_LTE"]))
    ]
    if all(status == "complete" for status in lte_statuses):
        heating_gridspace_manager.run_valid_calculations()
    else:
        print(
            "Not all LTE runs are complete. Heating runs cannot be submitted until LTE runs are complete."
        )

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
            "Not all constant temperature runs are complete. Cooling runs cannot be submitted until constant temperature runs are complete."
        )


def heating_and_cooling_at_50_percent_ground_state(casm_root_path: str):
    """A very specific function: Writes all necessary files for heating and cooling runs for the ground state at 50% composition. This includes:
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
            "mu_start": 0.0,
            "mu_stop": 0.0,
            "mu_increment": 0.0,
            "T_start": 0.0,
            "T_stop": 100.0,
            "T_increment": 5.0,
            "supercell": [[16, 0, 0], [0, 16, 0], [0, 0, 16]],
            "hours": 24,
        }
    ]
    lte_gs = dj.gridspace_manager(
        origin_dir=lte_dir,
        namer=mc.mc_run_namer,
        run_creator=mc.mc_run_creator,
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
            "supercell": [[16, 0, 0], [0, 16, 0], [0, 0, 16]],
            "hours": 24,
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
            "supercell": [[16, 0, 0], [0, 16, 0], [0, 0, 16]],
            "hours": 24,
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
            "mu_start": 0.0,
            "mu_stop": 0.0,
            "mu_increment": 0.0,
            "T_start": 2000.0,
            "T_stop": 0.0,
            "T_increment": -5.0,
            "supercell": [[16, 0, 0], [0, 16, 0], [0, 0, 16]],
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

    # ***Make sure that the cooling run initializes at the final configuration of the high temperature constant T run***
    # Read the high constant T run settings file to check how many conditions will be generated.
    t_const_runs = glob(t_const_dir + "/*")
    t_const_settings = mc.read_mc_settings(
        os.path.join(t_const_runs[0], "mc_settings.json")
    )
    number_of_conditions = (
        len(t_const_settings[0]) - 1
    )  # -1 is required because conditions indexing starts at 0

    # Check that this conditions file exists
    if not os.path.exists(
        os.path.join(t_const_runs[0], "conditions.%d" % number_of_conditions)
    ):
        warnings.warn(
            "High temperature constant t run final conditions file cannot be found. Please verify that %s exists."
            % os.path.join(t_const_runs[0], "conditions.%d" % number_of_conditions)
        )

    # Reference the final constant T configuration as the starting configuration for the cooling run.
    cooling_runs = glob(cooling_dir + "/*")
    with open(os.path.join(cooling_runs[0], "mc_settings.json"), "r") as f:
        cooling_settings = json.load(f)
    cooling_settings["driver"]["motif"]["configdof"] = os.path.join(
        t_const_runs[0], "conditions.%d" % number_of_conditions
    )

    # Create a dj.gridspace_manager object to control heating runs
    heating_dir = os.path.join(casm_root_path, "grand_canonical_monte_carlo/MC_heating")
    heating_param_list_of_dicts = [
        {
            "mu_start": 0.0,
            "mu_stop": 0.0,
            "mu_increment": 0.0,
            "T_start": 0.0,
            "T_stop": 2000.0,
            "T_increment": 5.0,
            "supercell": [[16, 0, 0], [0, 16, 0], [0, 0, 16]],
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
    # ***Make sure that the heating run initializes at the final configuration of the LTE run***
    # Read the LTE run settings file to check how many conditions will be generated.
    lte_runs = glob(lte_dir + "/*")
    lte_settings = mc.read_mc_settings(os.path.join(lte_runs[0], "mc_settings.json"))
    number_of_conditions = (
        len(lte_settings[0]) - 1
    )  # -1 is required because conditions indexing starts at 0

    # Check that this conditions file exists
    if not os.path.exists(
        os.path.join(lte_runs[0], "conditions.%d" % number_of_conditions)
    ):
        warnings.warn(
            "LTE run final conditions file cannot be found. Please verify that %s exists."
            % os.path.join(lte_runs[0], "conditions.%d" % number_of_conditions)
        )

    # Reference the final LTE configuration as the starting configuration for the heating run.
    heating_runs = glob(heating_dir + "/*")
    with open(os.path.join(heating_runs[0], "mc_settings.json"), "r") as f:
        heating_settings = json.load(f)
    heating_settings["driver"]["motif"]["configdof"] = os.path.join(
        lte_runs[0], "conditions.%d" % number_of_conditions
    )


# TODO: write function to find the clex-predicted ground state for a given chemical potential
# TODO: write function to find the transformation matrix that forms the supercell of a given configuration (Will involve a casm call, should probably be placed in casmcalls)

