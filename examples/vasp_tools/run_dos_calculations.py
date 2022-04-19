from djlib import mc
import os, sys

root_dir = '/home/jonnyli/pod_experiments/HfN_full/ground_states'
xtals = ['FCC','HCP']

for xtal in xtals:
    training_dir = os.path.join(root_dir,xtal)
    configs_list = os.path.join(root_dir,(xtal+"_gs.txt"))
    for config in open(configs_list):
        print("Config = ",config)
        mc.setup_dos_calculation(config.strip(),
        training_dir,
        hours=10,
        spin=1,
        slurm=True,
        run_jobs=False,
        delete_submit_script=False,
    )