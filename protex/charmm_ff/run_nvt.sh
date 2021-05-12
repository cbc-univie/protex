#!/bin/bash
#SBATCH --gres=gpu
#SBATCH -p lgpu

source /home/florian/anaconda3/bin/activate openmm

inp_file="openmm_run.py"
job_name="nvt"
last=101 # 101 wird jeder frame gespeichert fuer ir spektrum
file_name="im1h_oac_150_im1_hoac_350"
params="nvt_specs.inp"
psf=${file_name}".psf"
crd=${file_name}".crd"
toppar="toppar_lj02.str"

python $inp_file -cnt $1 -i $params -p $psf -c $crd -t $toppar -n ${file_name}

ret_val=$?

if [ $ret_val -eq 0 -a $1 -lt $last ] ; then
    let next=$1+1
    sbatch -J ${job_name}_$next -o out/nvt_$next.out run_nvt.sh $next
elif [ ! $ret_val -eq 0 ] ; then
    echo "Error in run $1 .." >> out/error.log
fi
