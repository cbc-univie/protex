def generate_im1h_oac_system():
    """
    Sets up a solvated and paraterized system for IM1H/OAC
    """

    def load_charmm_files():
        from simtk.openmm.app import CharmmParameterSet, CharmmPsfFile, CharmmCrdFile

        # =======================================================================
        # Force field
        # =======================================================================

        # Loading CHARMM files
        print("Loading CHARMM files...")
        PARA_FILES = [
            "toppar_drude_master_protein_2013f_lj02.str",
            "hoac_d.str",
            "im1h_d_fm_lj.str",
            "im1_d_fm_lj.str",
            "oac_d_lj.str",
        ]
        base = "protex/charmm_ff"
        params = CharmmParameterSet(
            *[f"{base}/toppar/{para_files}" for para_files in PARA_FILES]
        )

        psf = CharmmPsfFile(f"{base}/im1h_oac_150_im1_hoac_350.psf")
        # cooridnates can be provieded by CharmmCrdFile, CharmmRstFile or PDBFile classes
        crd = CharmmCrdFile(f"{base}/im1h_oac_150_im1_hoac_350.crd")
        return psf, crd, params

    def setup_system():
        from simtk.unit import angstroms
        from simtk.openmm.app import PME, HBonds

        psf, crd, params = load_charmm_files()
        xtl = 48.0 * angstroms
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
        from simtk.unit import kelvin, picoseconds, angstroms
        from simtk.openmm.app import Simulation
        from simtk.openmm import DrudeLangevinIntegrator, DrudeNoseHooverIntegrator

        psf, crd, params = load_charmm_files()
        system = setup_system()
        # integrator = DrudeLangevinIntegrator(
        integrator = DrudeNoseHooverIntegrator(
            300 * kelvin,
            10 / picoseconds,
            1 * kelvin,
            200 / picoseconds,
            0.0005 * picoseconds,
        )
        integrator.setMaxDrudeDistance(0.2 * angstroms)
        simulation = Simulation(psf.topology, system, integrator)
        simulation.context.setPositions(crd.positions)
        simulation.context.computeVirtualSites()
        return simulation

    return setup_simulation()