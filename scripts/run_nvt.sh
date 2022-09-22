#!/bin/bash
#SBATCH --gres=gpu
#SBATCH -p lgpu

source /home/florian/anaconda3/bin/activate openmm
#conda activate openmm

inp_file="nvt.py"
#job_name="pt"
#last=3000

python $inp_file

#ret_val=$?
#
#if [ $ret_val -eq 0 -a $1 -lt $last ] ; then
#    let next=$1+1
#    sbatch -J ${job_name}_$next -o out/nvt_$next.out run_nvt.sh $next
#elif [ ! $ret_val -eq 0 ] ; then
#    echo "Error in run $1 .." >> out/error.log
#fi
