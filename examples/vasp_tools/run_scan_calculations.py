from djlib import vasputils
import os

""" Sets up and runs SCAN calculations from a list of training directory paths, each containing a list of configs"""
training_dir_path_list = './TiNO_system_training_dir_paths.txt'

for training_dir in open(training_dir_path_list):
    print("training_dir= ", training_dir)
    for config in open(os.path.join(training_dir.strip(), 'configs_list.txt')):
        print("config= ", config)
        print("Setting up SCAN calculation")
        vasputils.setup_scan_calculation_from_existing_run(config.strip(),training_dir.strip(),hours=10,slurm=True,run_jobs=False,delete_submit_script=False,max_relax_steps=20)
