#!/bin/bash
#SBATCH --job-name={jobname} # Job name
#SBATCH --nodes=1                    # Run all processes on a single node	
#SBATCH --ntasks=1                   # Number of processes
#SBATCH --ntasks-per-core=1
#SBATCH --time={hours}:00:00              # Time limit hrs:min:sec
#SBATCH --output={rundir}/slurmjob_%j.log # Standard output and error log

cd {rundir}

echo 'submitting from: ' {rundir}
{user_command}

{delete_submit_script}


