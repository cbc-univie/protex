import protex
from simtk.openmm.openmm import System
from simtk.openmm.app.simulation import Simulation


def generate_im1h_oac_system() -> Simulation:
    """
    Sets up a solvated and paraterized system for IM1H/OAC
    """

    def load_charmm_files():
        from simtk.openmm.app import CharmmCrdFile, CharmmParameterSet, CharmmPsfFile
        from simtk.unit import angstroms

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
        xtl = 48.0 * angstroms
        psf.setBox(xtl, xtl, xtl)
        # cooridnates can be provieded by CharmmCrdFile, CharmmRstFile or PDBFile classes
        crd = CharmmCrdFile(f"{base}/im1h_oac_150_im1_hoac_350_lp.crd")
        return psf, crd, params

    def setup_system() -> System:
        from simtk.openmm.app import PME, HBonds
        from simtk.unit import angstroms

        psf, crd, params = load_charmm_files()
        system = psf.createSystem(
            params,
            nonbondedMethod=PME,
            nonbondedCutoff=11.0 * angstroms,
            switchDistance=10 * angstroms,
            constraints=HBonds,
        )

        return system

    def setup_simulation() -> Simulation:
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
            0.0002 * picoseconds,
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
        "atom_name": "H7",
        "canonical_name": "IM1",
    },
    "IM1": {
        "atom_name": "N2",
        "canonical_name": "IM1",
    },
}

OAC_HOAC = {
    "OAC": {
        "atom_name": "O1",
        "canonical_name": "OAC",
    },
    "HOAC": {
        "atom_name": "H",
        "canonical_name": "OAC",
    },
}


def generate_im1h_oac_system_chelpg():
    """
    Sets up a solvated and paraterized system for IM1H/OAC
    """

    def load_charmm_files():
        from simtk.openmm.app import CharmmCrdFile, CharmmParameterSet, CharmmPsfFile
        from simtk.unit import angstroms

        # =======================================================================
        # Force field
        # =======================================================================
        # Loading CHARMM files
        print("Loading CHARMM files...")
        PARA_FILES = [
            "toppar_drude_master_protein_2013f_lj02.str",
            "hoac_d.str",
            "im1h_d_fm_lj_chelpg.str",
            "im1_d_fm_lj_chelpg_withlp.str",
            "oac_d_lj.str",
        ]
        base = f"{protex.__path__[0]}/chelpg_charges"  # NOTE: this points now to the installed files!
        params = CharmmParameterSet(
            *[f"{base}/toppar/{para_files}" for para_files in PARA_FILES]
        )

        psf = CharmmPsfFile(f"{base}/im1h_oac_150_im1_hoac_350.psf")
        xtl = 48.0 * angstroms
        psf.setBox(xtl, xtl, xtl)
        # cooridnates can be provieded by CharmmCrdFile, CharmmRstFile or PDBFile classes
        crd = CharmmCrdFile(f"{base}/im1h_oac_150_im1_hoac_350.crd")
        return psf, crd, params

    def setup_system():
        from simtk.openmm.app import PME, HBonds
        from simtk.unit import angstroms

        psf, crd, params = load_charmm_files()      
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
            0.00025 * picoseconds,
        )
        integrator.setMaxDrudeDistance(0.25 * angstroms)
        simulation = Simulation(psf.topology, system, integrator)
        simulation.context.setPositions(crd.positions)
        simulation.context.computeVirtualSites()
        simulation.context.setVelocitiesToTemperature(300 * kelvin)

        return simulation

    return setup_simulation()


IM1H_IM1_chelpg = {
    "IM1H": {
        "charge": [
            2.48500,
           -2.66700,
           0.135000,
           0.135000,
           0.135000,
            2.35720,
           -2.19920,
            2.44700,
           -2.55400,
           0.195000,
            2.50700,
           -2.55400,
           0.192000,
            2.72410,
           -2.74710,
           0.203000,
            2.04220,
           -2.19920,
           0.363000,
            0.00000,
        ],
        "atom_name": "H7",
        "canonical_name": "IM1",
    },
    "IM1": {
        "charge": [
            2.39060,
           -2.55160,
           0.094000,
           0.094000,
           0.094000,
            2.67030,
           -2.53030,
            2.51190,
           -2.88090,
           0.150000,
            3.06890,
           -2.88090,
           0.053000,
            2.40440,
           -2.28640,
           0.073000,
            2.24930,
           -2.24930,
            0.00000,
           0.474000,
        ],
        "atom_name": "N2",
        "canonical_name": "IM1",
    },
}

OAC_HOAC_chelpg = {
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
