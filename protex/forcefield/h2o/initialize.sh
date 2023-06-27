#!/bin/bash

###change###
filename="h2o"
#############

echo "Running packmol..."
/home/andras/git/packmol/packmol < packmol.inp > packmol.out
echo "Running pdb2crd..."
/home/marta/bin/pdb2corv2 ${filename}_init.pdb > ${filename}_init.crd
echo "Running CHARMM, create psf, crd..."
charmm -i write_psf_crd.inp -o write_psf_crd.out
