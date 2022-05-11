from __future__ import annotations
import shutil
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import djlib.djlib as dj
import pathlib

vasputils_lib_dir = pathlib.Path(__file__).parent.resolve()

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
    templates_path = os.path.join(vasputils_lib_dir, "../../templates")
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

def setup_scan_calculation_from_existing_run(config_name,
    training_dir,
    hours,
    calctype='SCAN',
    from_settings=False,
    slurm=True,
    run_jobs=False,
    queue="batch",
    delete_submit_script=False,
    encut=500,
    ismear=1,
    spin=1,
    max_relax_steps=20,
    ):
    """
    Sets up a SCAN relax to static calculation in VASP for a specific configuration. (Will search for a preexisting calculation in calctype.default)

    Parameters
    ----------
    config_name: str
        Name of the configuration to run the DOS calculation on.
    training_dir: str
        Path to the training directory (contains the configuration[s])
    hours: int
        Number of hours to run the job for.
    calctype: str
        Defines which calctype folder to run the calculation in. Defaults to "SCAN" for calctype.SCAN.
    slurm: bool
        Whether to submit the job with slurm. 
    run_jobs: bool
        Whether to run the job. If False, will only setup the folders and submit scripts.
    queue: str
        Queue to submit the job to. Defaults to "batch". Options: short, batch
    delete_submit_script: bool
        Whether to delete the submit script after submitting the job. Default: False
    encut: int
        Energy cutoff for the calculation. Default: 500
    ismear: int
        Smearing type for the calculation. Default: 1
    spin: int
        Spin polarization for the calculation. Default: 1
    max_relax_steps: int
        Maximum number of relaxation steps. Default: 20

    Returns
    -------
    None.
    """
    

    print("Setting up SCAN calculation for %s" % config_name)
    default_calc_dir = os.path.join(training_dir, config_name, "calctype.default")
    calc_dir = os.path.join(training_dir, config_name, "calctype.%s" % calctype)
    print("Setting up inputs in %s" % calc_dir)
    os.makedirs(os.path.join(calc_dir, "inputs"), exist_ok=True)
    templates_path = os.path.join(vasputils_lib_dir, "../templates")
    
    # format INCAR
    with open(os.path.join(templates_path, "INCAR_SCAN.template")) as f:
        template = f.read()

    # load information from existing run TODO: assumed to be calctype.default
        with open(os.path.join(default_calc_dir, "run.0", "INCAR")) as g:
            incar = g.readlines()

        for line in incar:
            if "ENCUT" in line:
                encut = line.split("=")[1].strip()
            if "ISMEAR" in line:
                ismear = line.split("=")[1].strip()
            if "ISPIN" in line:
                spin = line.split("=")[1].strip()

        s = template.format(encut=encut, spin=spin, ismear=ismear)

    with open(os.path.join(calc_dir, "INCAR"), "w") as f:
        f.write(s)

    # format KPOINTS
    os.system(
        "cp %s %s"
        % (
            os.path.join(default_calc_dir, "run.final/KPOINTS"),
            os.path.join(calc_dir, "KPOINTS"),
        )
    )

    # format POTCAR
    os.system(
        "cp %s %s"
        % (
            os.path.join(default_calc_dir, "run.final/POTCAR"),
            os.path.join(calc_dir, "POTCAR"),
        )
    )

    # format POSCAR
    os.system(
        "cp %s %s"
        % (
            os.path.join(default_calc_dir, "run.final/CONTCAR"),
            os.path.join(calc_dir, "POSCAR"),
        )
    )

    # define user command

    # script will submit a relax to static calculation

    user_command = """cd %s
info_file=test.info
echo \"HOSTNAME=$(hostname)\" >> $info_file
echo \"STARTTIME=$(date --iso-8601=ns)\" >> $info_file

IMAX=%i #max number of vasp runs after initial relaxation    
printf "STARTED\n" > STATUS

mkdir run.0
cp POSCAR run.0/POSCAR
cp INCAR run.0/INCAR
cp KPOINTS run.0/KPOINTS
cp POTCAR run.0/POTCAR
cd run.0
mpirun vasp >& vasp.out

NSTEPS=$(cat vasp.out | grep E0 | wc -l)
grep "reached required accuracy" OUTCAR
if [ $? -ne 0 ] ; then printf "FAILED TO RELAX\n" >> ../STATUS ; exit ; fi
cd ../

while [ $NSTEPS -ne 1 ] && [ $I -lt $IMAX ]
do
    printf "Run $I had $NSTEPS steps.\n" >> ../../STATUS
    I=$(($I+1))
    cp -r run.$(($I-1)) run.$I
    cd run.$I
    cp CONTCAR POSCAR
    mpirun vasp >& vasp.out
    NSTEPS=$(cat vasp.out | grep E0 | wc -l)
    grep "reached required accuracy" OUTCAR
    if [ $? -ne 0 ] ; then printf "FAILED TO RELAX\n" >> ../STATUS ; exit ; fi
    cd ../
done

I=$(($I+1))
cp -r run.$(($I-1)) run.final
cd run.final
cp CONTCAR POSCAR

sed -i "s/LREAL.*/LREAL = .FALSE./g" INCAR
sed -i "s/IBRION.*/IBRION = -1/g" INCAR
sed -i "s/NSW.*/NSW = 0/g" INCAR
sed -i "s/ISIF.*/ISIF = 0/g" INCAR
sed -i "s/ISMEAR.*/ISMEAR = -5/g" INCAR

mpirun vasp >& vasp.out

cd ../
    
    """ % (
        os.path.join(calc_dir),max_relax_steps
    )

    # format submit script
    if slurm:
        dj.format_slurm_job(
            jobname=config_name,
            hours=hours,
            user_command=user_command,
            output_dir=calc_dir,
            delete_submit_script=delete_submit_script,
            queue=queue,
        )

        if run_jobs:
            dj.submit_slurm_job(calc_dir)
