#!/bin/bash

###change###
filename="im1h_oac_150_im1_hoac_350"
#############

echo "Running packmol..."
packmol <packmol.inp >packmol.out
echo "Running pdb2crd..."
./pdb2crd ${filename}_init.pdb >${filename}.crd
echo "Running Python, create psf"
python generate_psf.py 
