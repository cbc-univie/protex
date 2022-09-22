#!/bin/bash
#SBATCH --gres=gpu
#SBATCH -p lgpu

source /home/florian/anaconda3/bin/activate openmm
#conda activate openmm

inp_file="npt.py"
job_name="npt"
last=12
file_name="npt_test"

python $inp_file -cnt $1 -n ${file_name}

ret_val=$?

if [ $ret_val -eq 0 -a $1 -lt $last ] ; then
    let next=$1+1
    sbatch -J ${job_name}_$next -o out/npt_$next.out run_npt.sh $next
elif [ ! $ret_val -eq 0 ] ; then
    echo "Error in run $1 .." >> out/error.log
fi
