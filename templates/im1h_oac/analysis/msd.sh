#!/bin/bash

#SBATCH -p gpu
#SBATCH -w n0022

source /home/florian/anaconda3/bin/activate mda2.0

python msd_core.py -i

#run from command line with: sbatch -J jobname -o outfile filename.sh
