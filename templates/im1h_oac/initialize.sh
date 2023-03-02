#!/bin/bash

###change###
filename="im1h_oac_150_im1_hoac_350"
#############

echo "Running packmol..."
packmol <packmol.inp >packmol.out
echo "Running pdb2crd..."
./pdb2crd ${filename}_init.pdb >${filename}_init.crd
echo "Running CHARMM, create psf, crd..."
charmm -i write_psf_crd.inp -o write_psf_crd.out
