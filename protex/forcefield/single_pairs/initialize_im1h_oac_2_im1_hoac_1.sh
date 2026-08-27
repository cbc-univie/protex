#!/bin/bash
echo "Running packmol..."
/home/andras/git_test/packmol/packmol <packmol_im1h_oac_2_im1_hoac_1.inp >packmol_im1h_oac_2_im1_hoac_1.out
echo "Running pdb2crd..."
pdb2crd im1h_oac_2_im1_hoac_1_init.pdb >im1h_oac_2_im1_hoac_1_init.crd
echo "Running CHARMM, create psf, crd..."
charmm -i write_psf_crd_im1h_oac2.inp -o write_psf_crd_im1h_oac_2_im1_hoac_1.out
