#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime={hours}:00:00
#PBS -j oe
#PBS -N {jobname}
#PBS -V

module load intel/18
ulimit -s unlimited
cat $PBS_NODEFILE > nodes

cd {rundir}
echo 'submitting from: '{rundir}
echo 'Started' > STATUS
{user_command}
echo 'Finished' > STATUS
{delete_submit_script}


