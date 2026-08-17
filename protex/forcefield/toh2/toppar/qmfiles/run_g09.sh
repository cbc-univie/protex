#!/bin/bash
#SBATCH -p cpuongpu
#SBATCH --mem=0
#SBATCH -n 40
#SBATCH -J g09
#SBATCH -o opt_pol.log

g09 < opt_pol.com

#sbatch -J chelpg -o chelpg_pol.log run_g09.sh
