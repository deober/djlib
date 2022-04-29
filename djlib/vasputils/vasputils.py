import shutil
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import djlib.djlib as dj


class poscar:
    def __init__(self, poscar_file_path):
        self.poscar_file_path = poscar_file_path
        self.pos_name = ""
        self.species_vec = []
        self.species_count = np.array([])
        self.basis_scaling = 1
        self.basis = np.zeros((3, 3))
        self.coord_style = "Direct"  # in vasp, direct == fractional
        self.coords = []

        lineCount = 0
        readCoords = True
        coord_line = 7
        special_settings = []
        with open(self.poscar_file_path, "r") as pfile:
            for line in pfile:

                if len(line.split()) == 0:
                    # print('ERROR: unexpected empty line at line %s\nScript might not work properly.' % (lineCount +1) )
                    # print("(if this is a CONTCAR and the problem line is after the coordinates, you're fine)\n\n")
                    readCoords = False

                if lineCount == 0:
                    self.pos_name = line
                elif lineCount == 1:
                    self.basis_scaling = float(line)
                elif lineCount > 1 and lineCount < 5:
                    self.basis[lineCount - 2, :] = np.array(line.split()).astype(float)
                elif lineCount == 5:
                    self.species_vec = line.split()
                elif lineCount == 6:
                    self.species_count = np.array(line.split()).astype(int)
                elif lineCount == 7:
                    if line.split()[0][0] == "d" or line.split()[0][0] == "D":
                        self.coord_style = "Direct"
                    elif line.split()[0][0] == "c" or line.split()[0][0] == "C":
                        self.coord_style = "Cartesian"
                    else:
                        special_settings.append(line.strip())
                        coord_line = coord_line + 1

                elif lineCount > coord_line and readCoords:
                    self.coords.append(
                        line.split()[0:3]
                    )  # will chop of any descriptors
                lineCount += 1

        pfile.close()
        self.coords = np.array(self.coords).astype(float)

    def writePoscar(self):
        # writes the poscar to a file
        currentDirectory = ""
        for i in range(len(self.poscar_file_path.split("/")) - 1):
            currentDirectory = (
                currentDirectory + "/" + self.poscar_file_path.split("/")[i]
            )
        currentDirectory = currentDirectory[1:]

        with open(os.path.join(currentDirectory, "newPoscar.vasp"), "w") as newPoscar:
            newPoscar.write("new_poscar_" + self.pos_name)
            newPoscar.write("%f\n" % self.basis_scaling)

            for row in self.basis:
                for element in row:
                    newPoscar.write(str(element) + " ")
                newPoscar.write("\n")

            for species in self.species_vec:
                newPoscar.write(species + " ")
            newPoscar.write("\n")

            for count in self.species_count:
                newPoscar.write(str(count) + " ")
            newPoscar.write("\n")

            newPoscar.write("%s\n" % self.coord_style)

            for row in self.coords:
                if True:  # all(row < 1):
                    for element in row:
                        newPoscar.write(str(element) + " ")
                    newPoscar.write("\n")
        newPoscar.close()


def parse_outcar(outcar):
    """
    Parameters
    ----------
    outcar: str
        Path to a VASP OUTCAR file.

    Returns
    -------
    final_energy: float
        Last energy reported in the OUTCAR file (for sigma->0).
    """
    scf_energies = []
    with open(outcar, "r") as f:
        for line in f:
            if "sigma" in line:
                scf_energies.append(float(line.split()[-1]))

    final_energy = scf_energies[-1]
    return final_energy


def parse_incar(incar):
    """
    Parameters
    ----------
    incar: str
        Path to VASP incar

    Returns
    -------
    encut: int
        Energy Cutoff
    """
    encut = None
    with open(incar, "r") as f:
        for line in f:
            if "ENCUT" in line:
                encut = float(line.split("=")[-1].strip())
    return encut


def parse_kpoints(kpoints):
    """
    Parameters
    ----------
    kpoints: str
        Path to VASP KPOINTS file

    Returns
    -------
    kpoint_Rk: int
        Kpoint density parameter Rk
    """
    kpoint_Rk = None
    read_density = False
    with open(kpoints) as f:
        linecount = 0
        for line in f:
            if linecount == 2 and "A" in line:
                read_density = True
            if linecount == 3 and read_density == True:
                kpoint_Rk = int(float(line.strip()))
            linecount += 1
    assert (
        kpoint_Rk != None
    ), "Could not read kpoint file. Ensure that the file uses an automatic kpoint mesh."
    return kpoint_Rk


def parse_ibzkpts(ibz_file):
    """
    Parameters
    ----------
    ibz_file: str
        Path to VASP IBZKPTS file.

    Returns
    -------
    kpoint_count: int
        Number of kpoints used in the vasp simulation.
    """
    kpoint_count = None
    linecount = 0
    with open(ibz_file) as f:
        for line in f:
            if linecount == 1:
                kpoint_count = int(float(line.strip()))
                break
            linecount += 1
    return kpoint_count


def scrape_vasp_data(run_dir, write_data=True):
    """
    Parameters
    ----------
    run_dir: str
        Path to VASP simulation directory

    Returns
    -------
    scraped_data: dict
    """
    energy = parse_outcar(os.path.join(run_dir, "OUTCAR"))
    encut = parse_incar(os.path.join(run_dir, "INCAR"))
    kdensity = parse_kpoints(os.path.join(run_dir, "KPOINTS"))
    kcount = parse_ibzkpts(os.path.join(run_dir, "IBZKPT"))

    scraped_data = {
        "name": run_dir.split("/")[-1],
        "encut": encut,
        "energy": energy,
        "kdensity": kdensity,
        "kcount": kcount,
    }
    if write_data:
        with open(os.path.join(run_dir, "scraped_data.json"), "w") as f:
            json.dump(scraped_data, f)
    return scraped_data


def collect_convergence_data(convergence_dir, write_data=True):
    """
    Parameters
    ----------
    convergence_dir: str
        Path to a convergence directory containing many VASP simulaitons as subdirectories.

    Returns
    -------
    convergence_data: dict
        Names, energies and kpoint information for a colleciton of convergence simulations.
    """

    subdirs = [x[0] for x in os.walk(convergence_dir)]
    subdirs.remove(convergence_dir)

    names = []
    encuts = []
    energies = []
    kdensity = []
    kcount = []
    for run in subdirs:
        run_data = scrape_vasp_data(run)
        names.append(run_data["name"])
        encuts.append(run_data["encut"])
        energies.append(run_data["energy"])
        kdensity.append(run_data["kdensity"])
        kcount.append(run_data["kcount"])

    convergence_data = {
        "names": names,
        "encuts": encuts,
        "energies": energies,
        "kdensity": kdensity,
    }
    if write_data:
        with open(os.path.join(convergence_dir, "convergence_data.json"), "w") as f:
            json.dump(convergence_data, f)
    return convergence_data


def plot_convergence(x, y, xlabel, ylabel, title, convergence_tolerance=0.0005):

    data = np.zeros((len(x), 2))
    data[:, 0] = x
    data[:, 1] = y

    data = dj.column_sort(data, 0)

    plt.scatter(data[:, 0], data[:, 1], color="xkcd:crimson")
    plt.hlines(
        data[-1, 1] + convergence_tolerance, min(x), max(x), linestyle="--", color="k"
    )
    plt.hlines(
        data[-1, 1] - convergence_tolerance, min(x), max(x), linestyle="--", color="k"
    )
    plt.xlabel(xlabel, fontsize=22)
    plt.ylabel(ylabel, fontsize=22)
    plt.title(title, fontsize=30)
    fig = plt.gcf()
    fig.set_size_inches(13, 10)
    return fig


def collect_final_contcars(config_list_json_path, casm_root_path, deposit_directory):
    """Copies CONTCAR files for the specified configurations to a single directory: (Useful for collecting and examining ground state configuratin CONTCARS)

    Parameters:
    -----------
    config_list_json_path: str
        Path to a casm query output json containing the configurations of interest. 
    
    casm_root_path: str
        Path to the main directory of a CASM project. 
    
    deposit_directory: str
        Path to the directory where the CONTCARs should be copied. 

    Returns:
    --------
    None.
    """

    os.makedirs(deposit_directory, exist_ok=True)
    query_data = casm_query_reader(config_list_json_path)
    config_names = query_data["name"]

    for name in config_names:
        try:
            contcar_path = os.path.join(
                casm_root_path,
                "training_data",
                name,
                "calctype.default/run.final",
                "CONTCAR",
            )
            destination = os.path.join(
                deposit_directory,
                name.split("/")[0] + "_" + name.split("/")[-1] + ".vasp",
            )
            shutil.copy(contcar_path, destination)
        except:
            print("could not find %s " % contcar_path)


def reset_calc_staus(unknowns_file, casm_root):
    """For runs that failed and must be re-submitted; resets status to 'not_submitted'

    Parameters:
    -----------
    unknowns_file: str
        Path to casm query output of configurations to be reset. 
    casm_root: str
        Path to casm project root. 

    Returns:
    --------
    None.
    """
    query_data = casm_query_reader(unknowns_file)
    names = query_data["name"]

    for name in names:
        status_file = os.path.join(
            casm_root, "training_data", name, "calctype.default", "status.json"
        )
        with open(status_file, "r") as f:
            status = json.load(f)
        status["status"] = "not_submitted"
        with open(status_file, "w") as f:
            json.dump(status, f)

