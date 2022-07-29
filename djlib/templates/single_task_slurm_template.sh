#!/bin/bash
#SBATCH --job-name={jobname} # Job name
#SBATCH -n {nodes}                    # number of nodes
#SBATCH --time={hours}:00:00              # Time limit hrs:min:sec
#SBATCH --output={rundir}/slurmjob_%j.log # Standard output and error log

ulimit -s unlimited

cd {rundir}

echo 'submitting from: ' {rundir}
{user_command}

{delete_submit_script}


