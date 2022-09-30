#!/bin/bash
#SBATCH --job-name={jobname} # Job name
#SBATCH --nodes={nodes}                    # Run all processes on a single node	
#SBATCH --ntasks-per-node={ntasks_per_node}  # Number of processes per core
#SBATCH --time={hours}:00:00              # Time limit hrs:min:sec
#SBATCH --output={rundir}/slurmjob_%j.log # Standard output and error log

ulimit -s unlimited
cd {rundir}

echo 'submitting from: ' {rundir}
{user_command}

{delete_submit_script}


