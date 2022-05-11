from djlib import vasputils
import os

"""
For reference, the structure inside HfN_full/ground_states is 
2 lists of configs FCC_gs.txt and HCP_gs.txt and 2 folders FCC and HCP
each of FCC and HCP are structured as the training_data folder in CASM, with only the ground state configs
ex. HfN_full/ground_states/FCC/SCEL1_1_1_1_0_0_0/1/calctype.default/ is a path
"""

root_dir = "/home/jonnyli/experiments/HfN_full/ground_states"
xtals = ["FCC", "HCP"]

for xtal in xtals:
    training_dir = os.path.join(root_dir, xtal)
    configs_list = os.path.join(root_dir, (xtal + "_gs.txt"))
    for config in open(configs_list):
        print("Config = ", config)
        vasputils.setup_dos_calculation(
            config.strip(),
            training_dir,
            hours=10,
            spin=1,
            slurm=True,
            run_jobs=True,
            delete_submit_script=False,
        )
