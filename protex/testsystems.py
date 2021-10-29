import protex
from simtk.openmm.openmm import System
from simtk.openmm.app.simulation import Simulation

def generate_im1h_oac_system_chelpg(coll_freq=10, drude_coll_freq=120):
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
        from simtk.openmm import XmlSerializer

        # plugin
        # https://github.com/z-gong/openmm-velocityVerlet
        from velocityverletplugin import VVIntegrator

        psf, crd, params = load_charmm_files()
        system = setup_system()
        # integrator = DrudeNoseHooverIntegrator(
        #    300 * kelvin,
        #    5 / picoseconds,
        #    1 * kelvin,
        #    40 / picoseconds,
        #    0.0005 * picoseconds,
        # )
        # temperature grouped nose hoover thermostat
        # coll_freq = 10
        # drude_coll_freq = 80
        print(
            f"{coll_freq=}, {drude_coll_freq=}"
        )  # tested with 20, 40, 80, 100, 120, 140, 160: 20,40 bad; 80 - 120 good; 140, 160 crashed
        integrator = VVIntegrator(
            300 * kelvin,
            coll_freq / picoseconds,
            1 * kelvin,
            drude_coll_freq / picoseconds,
            0.0005 * picoseconds,
        )
        integrator.setMaxDrudeDistance(0.25 * angstroms)
        simulation = Simulation(psf.topology, system, integrator)
        simulation.context.setPositions(crd.positions)
        # Try with positinos from equilibrated system:
        base = f"{protex.__path__[0]}/chelpg_charges"
        with open(f"{base}/traj/im1h_oac_150_im1_hoac_350_npt_4.rst") as f:
            print(f"Opening restart file {f}")
            simulation.context.setState(XmlSerializer.deserialize(f.read()))
        simulation.context.computeVirtualSites()
        # simulation.context.setVelocitiesToTemperature(300 * kelvin)

        return simulation

    return setup_simulation()


IM1H_IM1_chelpg = {
    "IM1H": {
        "atom_name": "H7",
        "canonical_name": "IM1",
    },
    "IM1": {
        "atom_name": "N2",
        "canonical_name": "IM1",
    },
}

OAC_HOAC_chelpg = {
    "OAC": {
        "atom_name": "O1",
        "canonical_name": "OAC",
    },
    "HOAC": {
        "atom_name": "H",
        "canonical_name": "OAC",
    },
}


# def generate_im1h_oac_system() -> Simulation:
#     """
#     Sets up a solvated and paraterized system for IM1H/OAC
#     """

#     def load_charmm_files():
#         from simtk.openmm.app import CharmmCrdFile, CharmmParameterSet, CharmmPsfFile
#         from simtk.unit import angstroms

#         # =======================================================================
#         # Force field
#         # =======================================================================
#         # Loading CHARMM files
#         print("Loading CHARMM files...")
#         PARA_FILES = [
#             "toppar_drude_master_protein_2013f_lj02.str",
#             "hoac_d.str",
#             "im1h_d_fm_lj_lp.str",
#             "im1_d_fm_lj_dummy_lp.str",
#             "oac_d_lj.str",
#         ]
#         base = f"{protex.__path__[0]}/charmm_ff"  # NOTE: this points now to the installed files!
#         params = CharmmParameterSet(
#             *[f"{base}/toppar/{para_files}" for para_files in PARA_FILES]
#         )

#         psf = CharmmPsfFile(f"{base}/im1h_oac_150_im1_hoac_350_lp.psf")
#         xtl = 48.0 * angstroms
#         psf.setBox(xtl, xtl, xtl)
#         # cooridnates can be provieded by CharmmCrdFile, CharmmRstFile or PDBFile classes
#         crd = CharmmCrdFile(f"{base}/im1h_oac_150_im1_hoac_350_lp.crd")
#         return psf, crd, params

#     def setup_system() -> System:
#         from simtk.openmm.app import PME, HBonds
#         from simtk.unit import angstroms

#         psf, crd, params = load_charmm_files()
#         system = psf.createSystem(
#             params,
#             nonbondedMethod=PME,
#             nonbondedCutoff=11.0 * angstroms,
#             switchDistance=10 * angstroms,
#             constraints=HBonds,
#         )

#         return system

#     def setup_simulation() -> Simulation:
#         from simtk.openmm import DrudeLangevinIntegrator, DrudeNoseHooverIntegrator
#         from simtk.openmm.app import Simulation
#         from simtk.unit import angstroms, kelvin, picoseconds

#         psf, crd, params = load_charmm_files()
#         system = setup_system()
#         integrator = DrudeNoseHooverIntegrator(
#             300 * kelvin,
#             10 / picoseconds,
#             1 * kelvin,
#             40 / picoseconds,
#             0.00025 * picoseconds,
#         )
#         integrator.setMaxDrudeDistance(0.25 * angstroms)
#         simulation = Simulation(psf.topology, system, integrator)
#         simulation.context.setPositions(crd.positions)
#         simulation.context.computeVirtualSites()
#         simulation.context.setVelocitiesToTemperature(300 * kelvin)

#         return simulation

#     return setup_simulation()


# IM1H_IM1 = {
#     "IM1H": {
#         "atom_name": "H7",
#         "canonical_name": "IM1",
#     },
#     "IM1": {
#         "atom_name": "N2",
#         "canonical_name": "IM1",
#     },
# }

# OAC_HOAC = {
#     "OAC": {
#         "atom_name": "O1",
#         "canonical_name": "OAC",
#     },
#     "HOAC": {
#         "atom_name": "H",
#         "canonical_name": "OAC",
#     },
# }