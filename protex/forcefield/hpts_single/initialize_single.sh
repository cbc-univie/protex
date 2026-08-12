#!/bin/bash

###change###
filename="hpts_single"
#############

echo "Running packmol..."
/home/andras/git_test/packmol/packmol <packmol_single.inp >packmol_single.out
echo "Running pdb2crd..."
pdb2corv2 ${filename}_init.pdb >${filename}_init.crd
echo "Running CHARMM, create psf, crd..."
charmm -i write_psf_crd_single.inp -o write_psf_crd_single.out
