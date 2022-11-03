#!/bin/bash
echo "Running packmol..."
packmol <packmol_im1_hoac_2.inp >packmol_im1_hoac_2.out
echo "Running pdb2crd..."
pdb2crd im1_hoac_2_init.pdb >im1_hoac_2_init.crd
echo "Running CHARMM, create psf, crd..."
charmm -i write_psf_crd_im1_hoac_2.inp -o write_psf_crd_im1_hoac_2.out
