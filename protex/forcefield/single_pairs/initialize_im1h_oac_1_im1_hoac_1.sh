#!/bin/bash
echo "Running packmol..."
packmol <packmol_im1h_oac_1_im1_hoac_1.inp >packmol_im1h_oac_1_im1_hoac_1.out
echo "Running pdb2crd..."
pdb2crd im1h_oac_1_im1_hoac_1_init.pdb >im1h_oac_1_im1_hoac_1_init.crd
echo "Running CHARMM, create psf, crd..."
charmm -i write_psf_crd.inp -o write_psf_crd.out
