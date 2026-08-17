#!/bin/bash
echo "Running packmol..."
/home/andras/git_test/packmol/packmol <packmol_im1h.inp >packmol_im1h.out
echo "Running pdb2crd..."
pdb2crd only_im1h_init.pdb >only_im1h_init.crd
echo "Running CHARMM, create psf, crd..."
charmm -i write_psf_crd_im1h.inp -o write_psf_crd_im1h.out
