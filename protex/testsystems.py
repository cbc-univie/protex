import protex
from protex.constants import bxl


def generate_im1h_oac_system():
    """
    Sets up a solvated and paraterized system for IM1H/OAC
    """

    def load_charmm_files():
        from simtk.openmm.app import CharmmCrdFile, CharmmParameterSet, CharmmPsfFile

        # =======================================================================
        # Force field
        # =======================================================================
        # Loading CHARMM files
        print("Loading CHARMM files...")
        PARA_FILES = [
            "toppar_drude_master_protein_2013f_lj02.str",
            "hoac_d.str",
            "im1h_d_fm_lj_lp.str",
            "im1_d_fm_lj_dummy_lp.str",
            "oac_d_lj.str",
        ]
        base = f"{protex.__path__[0]}/charmm_ff"  # NOTE: this points now to the installed files!
        params = CharmmParameterSet(
            *[f"{base}/toppar/{para_files}" for para_files in PARA_FILES]
        )

        psf = CharmmPsfFile(f"{base}/im1h_oac_150_im1_hoac_350_lp.psf")
        # cooridnates can be provieded by CharmmCrdFile, CharmmRstFile or PDBFile classes
        crd = CharmmCrdFile(f"{base}/im1h_oac_150_im1_hoac_350_lp.crd")
        return psf, crd, params

    def setup_system():
        from simtk.openmm.app import PME, HBonds
        from simtk.unit import angstroms

        psf, crd, params = load_charmm_files()
        xtl = bxl
        psf.setBox(xtl, xtl, xtl)
        system = psf.createSystem(
            params,
            nonbondedMethod=PME,
            nonbondedCutoff=11.0 * angstroms,
            switchDistance=10 * angstroms,
            constraints=HBonds,
        )

        return system

    def setup_simulation():
        from simtk.openmm import DrudeLangevinIntegrator, DrudeNoseHooverIntegrator
        from simtk.openmm.app import Simulation
        from simtk.unit import angstroms, kelvin, picoseconds

        psf, crd, params = load_charmm_files()
        system = setup_system()
        integrator = DrudeNoseHooverIntegrator(
            300 * kelvin,
            5 / picoseconds,
            1 * kelvin,
            100 / picoseconds,
            0.0005 * picoseconds,
        )
        integrator.setMaxDrudeDistance(0.25 * angstroms)
        simulation = Simulation(psf.topology, system, integrator)
        simulation.context.setPositions(crd.positions)
        simulation.context.computeVirtualSites()
        simulation.context.setVelocitiesToTemperature(300 * kelvin)

        return simulation

    return setup_simulation()


IM1H_IM1 = {
    "IM1H": {
        "charge": [
            3.4019,
            -3.1819,
            9.00e-02,
            9.00e-02,
            9.00e-02,
            1.9683,
            -2.5063,
            2.8343,
            -2.6693,
            0.116,
            2.8563,
            -2.6693,
            0.12,
            3.0971,
            -2.7471,
            0.167,
            2.0293,
            -2.5063,
            0.42,
        ],
        "atom_name": "H7",
        "canonical_name": "IM1",
    },
    "IM1": {
        "charge": [
            2.8999,
            -3.1819,
            0.101,
            0.101,
            0.101,
            2.7943,
            -2.5303,
            2.5879,
            -2.8809,
            0.14,
            2.7959,
            -2.8809,
            9.20e-02,
            2.9535,
            -2.7635,
            0.101,
            2.1903,
            -2.6203,
            0,
        ],
        "atom_name": "N2",
        "canonical_name": "IM1",
    },
}

OAC_HOAC = {
    "OAC": {
        "charge": [
            3.1817,
            -2.4737,
            2.9879,
            -3.1819,
            4.00e-03,
            4.00e-03,
            4.00e-03,
            2.0548,
            -2.0518,
            2.0548,
            -2.0518,
            0,
            -0.383,
            -0.383,
            -0.383,
            -0.383,
        ],
        "atom_name": "O1",
        "canonical_name": "OAC",
    },
    "HOAC": {
        "charge": [
            3.5542,
            -2.6962,
            3.2682,
            -3.5682,
            9.20e-02,
            9.20e-02,
            9.20e-02,
            2.3565,
            -2.3565,
            2.7765,
            -2.7765,
            0.374,
            -0.319,
            -0.319,
            -0.285,
            -0.285,
        ],
        "atom_name": "H",
        "canonical_name": "OAC",
    },
}
