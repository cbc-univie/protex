#!/bin/bash

#SBATCH -p lcpu

source /home/florian/anaconda3/bin/activate mda_py3

python msd_charged.py -i

#run from command line with: sbatch -J jobname -o outfile filename.sh
