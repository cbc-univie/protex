#!/bin/bash

###change###
filename="hpts_reorient"
#############

echo "Running packmol..."
/home/andras/git_test/packmol/packmol <packmol_reorient.inp >packmol_reorient.out
echo "Running pdb2crd..."
pdb2corv2 ${filename}_init.pdb >${filename}_init.crd
echo "Running CHARMM, create psf, crd..."
charmm -i write_psf_crd_reorient.inp -o write_psf_crd_reorient.out
