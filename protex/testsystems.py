import protex
import os

try:
    from openmm import OpenMMException
except ImportError:
    from simtk.openmm import OpenMMException


def generate_im1h_oac_system(coll_freq=10, drude_coll_freq=120):
    """
    Sets up a solvated and paraterized system for IM1H/OAC
    """

    def load_charmm_files():
        from openmm.app import CharmmCrdFile, CharmmParameterSet, CharmmPsfFile
        from openmm.unit import angstroms

        # =======================================================================
        # Force field
        # =======================================================================
        # Loading CHARMM files
        print("Loading CHARMM files...")
        PARA_FILES = [
            "toppar_drude_master_protein_2013f_lj025.str",
            "hoac_d.str",
            "im1h_d.str",
            "im1_d.str",
            "oac_d.str",
        ]
        base = f"{protex.__path__[0]}/forcefield"  # NOTE: this points now to the installed files!
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
        from openmm.app import PME, HBonds
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
        from openmm import (
            Platform,
            DrudeLangevinIntegrator,
            DrudeNoseHooverIntegrator,
            XmlSerializer,
        )
        from openmm.app import Simulation
        from simtk.unit import angstroms, kelvin, picoseconds

        # plugin
        # https://github.com/z-gong/openmm-velocityVerlet

        # coll_freq = 10
        # drude_coll_freq = 80

        try:
            from velocityverletplugin import VVIntegrator

            integrator = VVIntegrator(
                300 * kelvin,
                coll_freq / picoseconds,
                1 * kelvin,
                drude_coll_freq / picoseconds,
                0.0005 * picoseconds,
            )

        except ModuleNotFoundError:
            integrator = DrudeNoseHooverIntegrator(
                300 * kelvin,
                coll_freq / picoseconds,
                1 * kelvin,
                drude_coll_freq / picoseconds,
                0.0005 * picoseconds,
            )
            # temperature grouped nose hoover thermostat

        psf, crd, params = load_charmm_files()
        system = setup_system()
        print(
            f"{coll_freq=}, {drude_coll_freq=}"
        )  # tested with 20, 40, 80, 100, 120, 140, 160: 20,40 bad; 80 - 120 good; 140, 160 crashed
        integrator.setMaxDrudeDistance(0.25 * angstroms)
        try:
            platform = Platform.getPlatformByName("CUDA")
            prop = dict(CudaPrecision="single")  # default is single
        except OpenMMException:
            platform = Platform.getPlatformByName("CPU")
            prop = dict()

        simulation = Simulation(
            psf.topology, system, integrator, platform=platform, platformProperties=prop
        )
        simulation.context.setPositions(crd.positions)
        # Try with positions from equilibrated system:
        base = f"{protex.__path__[0]}/forcefield"
        if os.path.exists(f"{base}/traj/im1h_oac_150_im1_hoac_350_npt_7.rst"):
            with open(f"{base}/traj/im1h_oac_150_im1_hoac_350_npt_7.rst") as f:
                print(f"Opening restart file {f}")
                simulation.context.setState(XmlSerializer.deserialize(f.read()))
            simulation.context.computeVirtualSites()
        else:
            print(f"No restart file found. Using initial coordinate file.")
            simulation.context.computeVirtualSites()
            simulation.context.setVelocitiesToTemperature(300 * kelvin)

        return simulation

    return setup_simulation()


def generate_single_im1h_oac_system(coll_freq=10, drude_coll_freq=120):
    """
    Sets up a system with 1 IM1H, 1OAC, 1IM1 and 1 HOAC
    Was for testing the deformation of the imidazole ring -> solved by adding the nonbonded exception to the updates
    """

    def load_charmm_files():
        from openmm.app import CharmmCrdFile, CharmmParameterSet, CharmmPsfFile
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
        base = f"{protex.__path__[0]}/single_pairs"  # NOTE: this points now to the installed files!
        params = CharmmParameterSet(
            *[f"{base}/toppar/{para_files}" for para_files in PARA_FILES]
        )

        psf = CharmmPsfFile(f"{base}/im1h_oac_im1_hoac_1_secondtry.psf")
        xtl = 15.0 * angstroms
        psf.setBox(xtl, xtl, xtl)
        # cooridnates can be provieded by CharmmCrdFile, CharmmRstFile or PDBFile classes
        crd = CharmmCrdFile(f"{base}/im1h_oac_im1_hoac_1_secondtry.crd")
        return psf, crd, params

    def setup_system():
        from openmm.app import PME, HBonds
        from simtk.unit import angstroms

        psf, crd, params = load_charmm_files()
        system = psf.createSystem(
            params,
            nonbondedMethod=PME,
            nonbondedCutoff=3.0 * angstroms,
            switchDistance=2.5 * angstroms,
            constraints=HBonds,
        )

        return system

    def setup_simulation():
        from openmm import (
            Platform,
            DrudeNoseHooverIntegrator,
            XmlSerializer,
        )
        from openmm.app import Simulation
        from simtk.unit import angstroms, kelvin, picoseconds

        # plugin
        # https://github.com/z-gong/openmm-velocityVerlet

        try:
            from velocityverletplugin import VVIntegrator

            integrator = VVIntegrator(
                300 * kelvin,
                coll_freq / picoseconds,
                1 * kelvin,
                drude_coll_freq / picoseconds,
                0.0005 * picoseconds,
            )

        except ModuleNotFoundError:
            integrator = DrudeNoseHooverIntegrator(
                300 * kelvin,
                coll_freq / picoseconds,
                1 * kelvin,
                drude_coll_freq / picoseconds,
                0.0005 * picoseconds,
            )
            # temperature grouped nose hoover thermostat

        psf, crd, params = load_charmm_files()
        system = setup_system()
        print(
            f"{coll_freq=}, {drude_coll_freq=}"
        )  # tested with 20, 40, 80, 100, 120, 140, 160: 20,40 bad; 80 - 120 good; 140, 160 crashed
        integrator.setMaxDrudeDistance(0.25 * angstroms)
        try:
            platform = Platform.getPlatformByName("CUDA")
            prop = dict(CudaPrecision="single")  # default is single
        except OpenMMException:
            platform = Platform.getPlatformByName("CPU")
            prop = dict()

        simulation = Simulation(
            psf.topology,
            system,
            integrator,
            platform=platform,  # platformProperties=prop
        )
        simulation.context.setPositions(crd.positions)
        simulation.context.computeVirtualSites()
        # simulation.context.setVelocitiesToTemperature(300 * kelvin)

        # print(simulation.context.getIntegrator())
        # print(simulation.context.getPlatform().getName())
        # print(
        #    [
        #        f"{i}: {simulation.context.getPlatform().getPropertyValue(simulation.context, i)}"
        #        for i in simulation.context.getPlatform().getPropertyNames()
        #    ]
        # )
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
