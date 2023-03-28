#!/bin/bash
echo "Running packmol..."
packmol_mod <packmol.inp >packmol.out #to allow for 5character name
echo "Running pdb2crd..."
pdb2corv2 small_box_init.pdb >small_box_init.crd #to allow for 5character name
echo "Running CHARMM, create psf, crd..."
charmm -i write_psf_crd.inp -o write_psf_crd.out
