#!/bin/bash
#SBATCH --gres=gpu
#SBATCH -p lgpu

source /home/florian/anaconda3/bin/activate openmm_equil


inp_file="openmm_run.py"
file_name="im1h_oac_150_im1_hoac_350"
psf=${file_name}".psf"
crd=${file_name}".crd"
toppar="../toppar/toppar.str"

job_name="npt"
last_npt=4   #first one is only 20ps with dt 0.1fs
params="npt_specs.inp"

python $inp_file -cnt $1 -i $params -p $psf -c $crd -t $toppar -n ${file_name}

ret_val=$?

if [ $ret_val -eq 0 -a $1 -lt $last_npt ] ; then
    let next=$1+1
    sbatch -J ${job_name}e_$next -o out/npt_$next.out run_npt.sh $next
elif [ ! $ret_val -eq 0 ] ; then
    echo "Error in run $1 .." >> out/error.log
fi
