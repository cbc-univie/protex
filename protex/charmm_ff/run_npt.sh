#!/bin/bash
#SBATCH --gres=gpu
#SBATCH -p lgpu

source /home/florian/anaconda3/bin/activate openmm

inp_file="openmm_run.py"
job_name="npt"
last=1   #first one is only 20ps with dt 0.1fs, last one for mean
file_name="im1h_oac_150_im1_hoac_350"
params="npt_specs.inp"
psf=${file_name}".psf"
crd=${file_name}".crd"
toppar="toppar_lj02.str"

python $inp_file -cnt $1 -i $params -p $psf -c $crd -t $toppar -n ${file_name}

ret_val=$?

if [ $ret_val -eq 0 -a $1 -lt $last ] ; then
    let next=$1+1
    sbatch -J ${job_name}_$next -o out/npt_$next.out run_npt.sh $next
elif [ ! $ret_val -eq 0 ] ; then
    echo "Error in run $1 .." >> out/error.log
fi
