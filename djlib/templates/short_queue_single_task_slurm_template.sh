#!/bin/bash
#SBATCH --job-name={jobname} # Job name
#SBATCH -n {nodes}                    # number of nodes
#SBATCH -p short
#SBATCH --time=2:00:00              # Time limit hrs:min:sec
#SBATCH --output={rundir}/slurmjob_%j.log # Standard output and error log

ulimit -s unlimited

cd {rundir}

echo 'submitting to short queue from: ' {rundir} 'initially with ' {hours} 'hours'
{user_command}

{delete_submit_script}


