import sys

#pip install git+https://github.com/florianjoerg/cbchelpers.git
from cbchelpers.parmed_extension import enable_psf_from_scratch

enable_psf_from_scratch()
from parmed.charmm import CharmmParameterSet, CharmmPsfFile


def main():
    para_files = [
        "../../protex/forcefield/toppar/toppar_drude_master_protein_2013f_lj025.str",
        "../../protex/forcefield/toppar/im1h_d.str",
        "../../protex/forcefield/toppar/oac_d_dummy.str",
        "../../protex/forcefield/toppar/im1_d_dummy.str",
        "../../protex/forcefield/toppar/hoac_d.str",
    ]

    molecules = {
        "IM1H": {"number": 150, "drude_mass": 0.4},
        "OAC": {"number": 150, "drude_mass": 0.4},
        "IM1": {"number": 350, "drude_mass": 0.4},
        "HOAC": {"number": 350, "drude_mass": 0.4},
    }

    params = CharmmParameterSet(*para_files)
    psf = CharmmPsfFile.from_scratch(params, molecules)
    print(len(psf.residues))
    psf.write_psf("im1h_oac_150_im1_hoac_350.psf")

if __name__ == "__main__":
    main()
