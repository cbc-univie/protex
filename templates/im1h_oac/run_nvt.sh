#!/bin/bash
#SBATCH --gres=gpu
#SBATCH -p lgpu

source /home/florian/anaconda3/bin/activate openmm7.6

inp_file="sim_pt.py"
job_name="r1"
last=100 # 500ps*100 = 50ns

python $inp_file $1 $last

ret_val=$?

if [ $ret_val -eq 0 -a $1 -lt $last ] ; then
    let next=$1+1
    sbatch -J ${job_name}v_$next -o out/nvt_$next.out run_nvt.sh $next
elif [ ! $ret_val -eq 0 ] ; then
    echo "Error in run $1 .." >> out/error.log
fi
