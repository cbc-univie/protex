#!/bin/bash
echo "Running packmol..."
packmol <packmol.inp >packmol.out
echo "Running pdb2crd..."
pdb2crd small_box_init.pdb >small_box_init.crd
echo "Running CHARMM, create psf, crd..."
charmm -i write_psf_crd.inp -o write_psf_crd.out
