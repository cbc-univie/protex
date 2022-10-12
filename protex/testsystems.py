import os

import protex

try:  # Syntax changed in OpenMM 7.6
    import openmm as mm
    from openmm import (
        Context,
        DrudeLangevinIntegrator,
        DrudeNoseHooverIntegrator,
        OpenMMException,
        Platform,
        XmlSerializer,
    )
    from openmm.app import (
        PME,
        CharmmCrdFile,
        CharmmParameterSet,
        CharmmPsfFile,
        HBonds,
        Simulation,
    )
    from openmm.unit import angstroms, kelvin, picoseconds
except ImportError:
    import simtk.openmm as mm
    from simtk.openmm import (
        Context,
        DrudeLangevinIntegrator,
        DrudeNoseHooverIntegrator,
        OpenMMException,
        Platform,
        XmlSerializer,
    )
    from simtk.openmm.app import (
        PME,
        CharmmCrdFile,
        CharmmParameterSet,
        CharmmPsfFile,
        HBonds,
        Simulation,
    )
    from simtk.unit import angstroms, kelvin, picoseconds


def generate_im1h_oac_system(
    psf_file: str = None,
    crd_file: str = None,
    restart_file: str = None,
    constraints: str = None,
    boxl: float = 48.0,
    para_files: list[str] = None,
    coll_freq=10,
    drude_coll_freq=100,
):
    """
    Sets up a solvated and paraterized system for IM1H/OAC
    """

    def load_charmm_files(
        psf_file=psf_file, crd_file=crd_file, boxl=boxl, para_files=para_files
    ):
        # =======================================================================
        # Force field
        # =======================================================================
        # Loading CHARMM files
        print("Loading CHARMM files...")
        base = f"{protex.__path__[0]}/forcefield"  # NOTE: this points now to the installed files!
        if para_files is not None:
            print("Using user supplied parameter files. Whole path must be given")
            params = CharmmParameterSet(*para_files)
        else:
            PARA_FILES = [
                "toppar_drude_master_protein_2013f_lj025.str",
                "hoac_d.str",
                "im1h_d.str",
                "im1_d.str",
                "oac_d.str",
            ]
            params = CharmmParameterSet(
                *[f"{base}/toppar/{para_files}" for para_files in PARA_FILES]
            )
        if psf_file is None:
            psf_file = f"{base}/im1h_oac_150_im1_hoac_350.psf"
        psf = CharmmPsfFile(psf_file)
        xtl = boxl * angstroms
        psf.setBox(xtl, xtl, xtl)
        # cooridnates can be provieded by CharmmCrdFile, CharmmRstFile or PDBFile classes
        if crd_file is None:
            crd_file = f"{base}/im1h_oac_150_im1_hoac_350.crd"
        crd = CharmmCrdFile(crd_file)
        return psf, crd, params

    def setup_system(constraints=constraints):
        psf, crd, params = load_charmm_files()
        # print(params.atom_types_str["DUM"].epsilon)
        # print(params.atom_types_str["DUM"].rmin)
        # print(params.atom_types_str["DUM"].epsilon_14)
        # print(params.atom_types_str["DUM"].rmin_14)
        # print(params.atom_types_str["DUM"].nbfix)
        # print(params.atom_types_str["DUM"].nbthole)
        # Charmm DUM atomtype has epsilon = 0, but problem with openmm during creatin of system/simulation
        # openmm.OpenMMException: updateParametersInContext: The set of non-excluded exceptions has changed
        # Therefore set it here non zero and zero afterwards
        params.atom_types_str["DUM"].set_lj_params(
            -0.00001,
            params.atom_types_str["DUM"].rmin,
            -0.00001,
            params.atom_types_str["DUM"].rmin_14,
        )
        if constraints is None:
            system = psf.createSystem(
                params,
                nonbondedMethod=PME,
                nonbondedCutoff=11.0 * angstroms,
                switchDistance=10 * angstroms,
                constraints=None,
            )
        elif constraints == "HBonds":
            system = psf.createSystem(
                params,
                nonbondedMethod=PME,
                nonbondedCutoff=11.0 * angstroms,
                switchDistance=10 * angstroms,
                constraints=HBonds,
            )
        else:
            print(
                "Only contraints=None or constraints=HBonds (given as string in function call) implemented"
            )

        for force in system.getForces():
            if type(force).__name__ == "CMMotionRemover":
                # From OpenMM psf file it has automatically ForceGroup 0, which is already used for harmonic bond force
                force.setForceGroup(len(system.getForces()) - 1)
                # print(force.getForceGroup())

        return system

    def setup_simulation(
        restart_file=restart_file, coll_freq=coll_freq, drude_coll_freq=drude_coll_freq
    ):

        psf, crd, params = load_charmm_files()
        system = setup_system()

        # coll_freq = 10
        # drude_coll_freq = 80

        try:
            # plugin
            # https://github.com/z-gong/openmm-velocityVerlet
            from velocityverletplugin import VVIntegrator

            # temperature grouped nose hoover thermostat
            integrator = VVIntegrator(
                300 * kelvin,
                coll_freq / picoseconds,
                1 * kelvin,
                drude_coll_freq / picoseconds,
                0.0005 * picoseconds,
            )
            # test if platform and integrator are compatible -> VVIntegrator only works on cuda
            # If we do not create a context it is not tested if there is cuda availabe for the plugin
            context = Context(system, integrator)
            del context
            # afterwards the integrator is already bound to the context and we need a new one...
            # is there something like integrator.reset()?
            integrator = VVIntegrator(
                300 * kelvin,
                coll_freq / picoseconds,
                1 * kelvin,
                drude_coll_freq / picoseconds,
                0.0005 * picoseconds,
            )
            print("Using VVIntegrator Plugin")

        except (ModuleNotFoundError, OpenMMException) as e:
            integrator = DrudeNoseHooverIntegrator(
                300 * kelvin,
                coll_freq / picoseconds,
                1 * kelvin,
                drude_coll_freq / picoseconds,
                0.0005 * picoseconds,
            )
            print("Using built in DrudeNoseHooverIntegrator")
            print("Some tests might fail")
            print("Plugin not usable, because:")
            print(e)

        # print(
        #    f"{coll_freq=}, {drude_coll_freq=}"
        # )  # tested with 20, 40, 80, 100, 120, 140, 160: 20,40 bad; 80 - 120 good; 140, 160 crashed
        integrator.setMaxDrudeDistance(0.25 * angstroms)
        try:
            platform = Platform.getPlatformByName("CUDA")
            prop = dict(CudaPrecision="single")  # default is single
            # prop = dict(CudaPrecision="double")
            # Moved creating the simulation object inside the try...except block, because i.e.
            # Error loading CUDA module: CUDA_ERROR_INVALID_PTX (218)
            # message was only thrown during simulation creation not by specifying the platform
            simulation = Simulation(
                psf.topology,
                system,
                integrator,
                platform=platform,
                platformProperties=prop,
            )
        except OpenMMException:
            platform = Platform.getPlatformByName("CPU")
            prop = dict()
            simulation = Simulation(
                psf.topology,
                system,
                integrator,
                platform=platform,
                platformProperties=prop,
            )

        simulation.context.setPositions(crd.positions)
        # Try with positions from equilibrated system:
        if restart_file is None:
            base = f"{protex.__path__[0]}/forcefield"
            restart_file = f"{base}/traj/im1h_oac_150_im1_hoac_350_npt_7.rst"
        if os.path.exists(restart_file):
            with open(restart_file) as f:
                print(f"Opening restart file {f}")
                simulation.context.setState(XmlSerializer.deserialize(f.read()))
            simulation.context.computeVirtualSites()
        else:
            print(f"No restart file found. Using initial coordinate file.")
            simulation.context.computeVirtualSites()
            simulation.context.setVelocitiesToTemperature(300 * kelvin)

        # set nonbonded parameters including dummy zero again.
        # had to be  zero durin creating becuase of:
        # openmm.OpenMMException: updateParametersInContext: The set of non-excluded exceptions has changed
        nonbonded_force = [
            f for f in simulation.system.getForces() if isinstance(f, mm.NonbondedForce)
        ][0]
        dummy_atoms = []
        for atom in simulation.topology.atoms():
            if atom.residue.name == "IM1" and atom.name == "H7":
                dummy_atoms.append(atom.index)
                nonbonded_force.setParticleParameters(atom.index, 0.0, 0.0, 0.0)
            if atom.residue.name == "OAC" and atom.name == "H":
                dummy_atoms.append(atom.index)
                nonbonded_force.setParticleParameters(atom.index, 0.0, 0.0, 0.0)
        for exc_id in range(nonbonded_force.getNumExceptions()):
            f = nonbonded_force.getExceptionParameters(exc_id)
            idx1 = f[0]
            idx2 = f[1]
            chargeProd, sigma, epsilon = f[2:]
            if idx1 in dummy_atoms or idx2 in dummy_atoms:
                nonbonded_force.setExceptionParameters(
                    exc_id, idx1, idx2, 0.0, sigma, 0.0
                )

        return simulation

    return setup_simulation()


### new generate
def generate_hpts_system(
    psf_file: str = None,
    crd_file: str = None,
    restart_file: str = None,
    constraints: str = None,
    boxl: float = 48.0,
    para_files: list[str] = None,
    coll_freq=10,
    drude_coll_freq=100,
):
    """
    Sets up a solvated and paraterized system for IM1H/OAC/HPTS
    """

    def load_charmm_files(
        psf_file=psf_file, crd_file=crd_file, boxl=boxl, para_files=para_files
    ):
        # =======================================================================
        # Force field
        # =======================================================================
        # Loading CHARMM files
        print("Loading CHARMM files...")
        if para_files is not None:
            print("Using user supplied parameter files. Whole path must be given")
            params = CharmmParameterSet(*para_files)
        else:
            PARA_FILES = [
                "toppar_drude_master_protein_2013f_lj025_modhpts.str",
                "hoac_d.str",
                "im1h_d.str",
                "im1_d.str",
                "oac_d.str",
                "hpts_d.str",
                "hptsh_d.str",
            ]
            base = f"{protex.__path__[0]}/forcefield"  # NOTE: this points now to the installed files!
            params = CharmmParameterSet(
                *[f"{base}/toppar/{para_files}" for para_files in PARA_FILES]
            )
        if psf_file is None:
            psf_file = f"{base}/hpts.psf"
        psf = CharmmPsfFile(psf_file)
        xtl = boxl * angstroms
        psf.setBox(xtl, xtl, xtl)
        # cooridnates can be provieded by CharmmCrdFile, CharmmRstFile or PDBFile classes
        if crd_file is None:
            crd_file = f"{base}/hpts.crd"
        crd = CharmmCrdFile(crd_file)
        return psf, crd, params

    def setup_system(constraints=constraints):
        psf, crd, params = load_charmm_files()
        if constraints is None:
            system = psf.createSystem(
                params,
                nonbondedMethod=PME,
                nonbondedCutoff=11.0 * angstroms,
                switchDistance=10 * angstroms,
                constraints=None,
            )
        elif constraints == "HBonds":
            system = psf.createSystem(
                params,
                nonbondedMethod=PME,
                nonbondedCutoff=11.0 * angstroms,
                switchDistance=10 * angstroms,
                constraints=HBonds,
            )
        else:
            print(
                "Only contraints=None or constraints=HBonds (given as string in function call) implemented"
            )
        return system

    def setup_simulation(
        restart_file=restart_file, coll_freq=coll_freq, drude_coll_freq=drude_coll_freq
    ):

        psf, crd, params = load_charmm_files()
        system = setup_system()

        # coll_freq = 10
        # drude_coll_freq = 80

        try:
            # plugin
            # https://github.com/z-gong/openmm-velocityVerlet
            from velocityverletplugin import VVIntegrator

            # temperature grouped nose hoover thermostat
            integrator = VVIntegrator(
                300 * kelvin,
                coll_freq / picoseconds,
                1 * kelvin,
                drude_coll_freq / picoseconds,
                0.0005 * picoseconds,
            )
            # test if platform and integrator are compatible -> VVIntegrator only works on cuda
            # If we do not create a context it is not tested if there is cuda availabe for the plugin
            context = Context(system, integrator)
            del context
            # afterwards the integrator is already bound to the context and we need a new one...
            # is there something like integrator.reset()?
            integrator = VVIntegrator(
                300 * kelvin,
                coll_freq / picoseconds,
                1 * kelvin,
                drude_coll_freq / picoseconds,
                0.0005 * picoseconds,
            )

        except (ModuleNotFoundError, OpenMMException):
            integrator = DrudeNoseHooverIntegrator(
                300 * kelvin,
                coll_freq / picoseconds,
                1 * kelvin,
                drude_coll_freq / picoseconds,
                0.0005 * picoseconds,
            )

        print(
            f"{coll_freq=}, {drude_coll_freq=}"
        )  # tested with 20, 40, 80, 100, 120, 140, 160: 20,40 bad; 80 - 120 good; 140, 160 crashed
        integrator.setMaxDrudeDistance(0.25 * angstroms)
        try:
            platform = Platform.getPlatformByName("CUDA")
            prop = dict(CudaPrecision="single")  # default is single
            # Moved creating the simulation object inside the try...except block, because i.e.
            # Error loading CUDA module: CUDA_ERROR_INVALID_PTX (218)
            # message was only thrown during simulation creation not by specifying the platform
            simulation = Simulation(
                psf.topology,
                system,
                integrator,
                platform=platform,
                platformProperties=prop,
            )
        except OpenMMException:
            platform = Platform.getPlatformByName("CPU")
            prop = dict()
            simulation = Simulation(
                psf.topology,
                system,
                integrator,
                platform=platform,
                platformProperties=prop,
            )

        simulation.context.setPositions(crd.positions)
        # Try with positions from equilibrated system:
        if restart_file is None:
            base = f"{protex.__path__[0]}/forcefield"
            restart_file = f"{base}/traj/hpts_npt_7.rst"
        if os.path.exists(restart_file):
            with open(restart_file) as f:
                print(f"Opening restart file {f}")
                simulation.context.setState(XmlSerializer.deserialize(f.read()))
            simulation.context.computeVirtualSites()
        else:
            print(f"No restart file found. Using initial coordinate file.")
            simulation.context.computeVirtualSites()
            simulation.context.setVelocitiesToTemperature(300 * kelvin)

        return simulation

    return setup_simulation()


def generate_single_im1h_oac_system(coll_freq=10, drude_coll_freq=100, psf_file=None):
    """
    Sets up a system with 1 IM1H, 1OAC, 1IM1 and 1 HOAC
    Was for testing the deformation of the imidazole ring -> solved by adding the nonbonded exception to the updates
    """

    def load_charmm_files(psf_file=psf_file):
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
        base = f"{protex.__path__[0]}/forcefield/single_pairs"  # NOTE: this points now to the installed files!
        params = CharmmParameterSet(
            *[f"{base}/toppar/{para_files}" for para_files in PARA_FILES]
        )
        if psf_file is None:
            psf_file = f"{base}/im1h_oac_im1_hoac_1_secondtry.psf"
        psf = CharmmPsfFile(psf_file)
        xtl = 15.0 * angstroms
        psf.setBox(xtl, xtl, xtl)
        # cooridnates can be provieded by CharmmCrdFile, CharmmRstFile or PDBFile classes
        crd = CharmmCrdFile(f"{base}/im1h_oac_im1_hoac_1_secondtry.crd")
        return psf, crd, params

    def setup_system():
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
        psf, crd, params = load_charmm_files()
        system = setup_system()

        try:
            # plugin
            # https://github.com/z-gong/openmm-velocityVerlet
            from velocityverletplugin import VVIntegrator

            integrator = VVIntegrator(
                300 * kelvin,
                coll_freq / picoseconds,
                1 * kelvin,
                drude_coll_freq / picoseconds,
                0.0005 * picoseconds,
            )

            context = Context(system, integrator)
            del context
            integrator = VVIntegrator(
                300 * kelvin,
                coll_freq / picoseconds,
                1 * kelvin,
                drude_coll_freq / picoseconds,
                0.0005 * picoseconds,
            )
            print("Using VVIntegrator Plugin")

        except (ModuleNotFoundError, OpenMMException):
            integrator = DrudeNoseHooverIntegrator(
                300 * kelvin,
                coll_freq / picoseconds,
                1 * kelvin,
                drude_coll_freq / picoseconds,
                0.0005 * picoseconds,
            )
            # temperature grouped nose hoover thermostat
            print("Using built in DrudeNoseHooverIntegrator")

        print(
            f"{coll_freq=}, {drude_coll_freq=}"
        )  # tested with 20, 40, 80, 100, 120, 140, 160: 20,40 bad; 80 - 120 good; 140, 160 crashed
        integrator.setMaxDrudeDistance(0.25 * angstroms)
        try:
            platform = Platform.getPlatformByName("CUDA")
            prop = dict(CudaPrecision="single")  # default is single
            simulation = Simulation(
                psf.topology,
                system,
                integrator,
                platform=platform,  # platformProperties=prop
            )
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


def generate_im1h_oac_dummy_system(
    psf_file: str = None,
    crd_file: str = None,
    restart_file: str = None,
    constraints: str = None,
    boxl: float = 48.0,
    para_files: list[str] = None,
    coll_freq=10,
    drude_coll_freq=100,
):
    """
    Sets up a solvated and paraterized system for IM1H/OAC
    """

    def load_charmm_files(
        psf_file=psf_file, crd_file=crd_file, boxl=boxl, para_files=para_files
    ):
        # =======================================================================
        # Force field
        # =======================================================================
        # Loading CHARMM files
        print("Loading CHARMM files...")
        base = f"{protex.__path__[0]}/forcefield/dummy"  # NOTE: this points now to the installed files!
        if para_files is not None:
            print("Using user supplied parameter files. Whole path must be given")
            params = CharmmParameterSet(*para_files)
        else:
            PARA_FILES = [
                "toppar_drude_master_protein_2013f_lj025.str",
                "hoac_d.str",
                "im1h_d.str",
                "im1_d_dummy.str",
                "oac_d_dummy.str",
            ]
            params = CharmmParameterSet(
                *[f"{base}/toppar/{para_files}" for para_files in PARA_FILES]
            )
        if psf_file is None:
            psf_file = f"{base}/im1h_oac_im1_hoac_1.psf"
        psf = CharmmPsfFile(psf_file)
        xtl = boxl * angstroms
        psf.setBox(xtl, xtl, xtl)
        # cooridnates can be provieded by CharmmCrdFile, CharmmRstFile or PDBFile classes
        if crd_file is None:
            crd_file = f"{base}/im1h_oac_im1_hoac_1.crd"
        crd = CharmmCrdFile(crd_file)
        return psf, crd, params

    def setup_system(constraints=constraints):
        psf, crd, params = load_charmm_files()
        # print(params.atom_types_str["DUM"].epsilon)
        # print(params.atom_types_str["DUM"].rmin)
        # print(params.atom_types_str["DUM"].epsilon_14)
        # print(params.atom_types_str["DUM"].rmin_14)
        # print(params.atom_types_str["DUM"].nbfix)
        # print(params.atom_types_str["DUM"].nbthole)
        params.atom_types_str["DUM"].set_lj_params(
            -0.00001,
            params.atom_types_str["DUM"].rmin,
            -0.00001,
            params.atom_types_str["DUM"].rmin_14,
        )
        # print(params.atom_types_str["DUM"].epsilon)
        # print(params.atom_types_str["DUM"].rmin)
        # print(params.atom_types_str["DUM"].epsilon_14)
        # print(params.atom_types_str["DUM"].rmin_14)
        if constraints is None:
            system = psf.createSystem(
                params,
                nonbondedMethod=PME,
                nonbondedCutoff=11.0 * angstroms,
                switchDistance=10 * angstroms,
                constraints=None,
            )
        elif constraints == "HBonds":
            system = psf.createSystem(
                params,
                nonbondedMethod=PME,
                nonbondedCutoff=11.0 * angstroms,
                switchDistance=10 * angstroms,
                constraints=HBonds,
            )
        else:
            print(
                "Only contraints=None or constraints=HBonds (given as string in function call) implemented"
            )

        for force in system.getForces():
            if type(force).__name__ == "CMMotionRemover":
                # From OpenMM psf file it has automatically ForceGroup 0, which is already used for harmonic bond force
                force.setForceGroup(len(system.getForces()) - 1)
                # print(force.getForceGroup())

        return system

    def setup_simulation(
        restart_file=restart_file, coll_freq=coll_freq, drude_coll_freq=drude_coll_freq
    ):

        psf, crd, params = load_charmm_files()
        system = setup_system()

        # coll_freq = 10
        # drude_coll_freq = 80

        try:
            # plugin
            # https://github.com/z-gong/openmm-velocityVerlet
            from velocityverletplugin import VVIntegrator

            # temperature grouped nose hoover thermostat
            integrator = VVIntegrator(
                300 * kelvin,
                coll_freq / picoseconds,
                1 * kelvin,
                drude_coll_freq / picoseconds,
                0.0005 * picoseconds,
            )
            # test if platform and integrator are compatible -> VVIntegrator only works on cuda
            # If we do not create a context it is not tested if there is cuda availabe for the plugin
            context = Context(system, integrator)
            del context
            # afterwards the integrator is already bound to the context and we need a new one...
            # is there something like integrator.reset()?
            integrator = VVIntegrator(
                300 * kelvin,
                coll_freq / picoseconds,
                1 * kelvin,
                drude_coll_freq / picoseconds,
                0.0005 * picoseconds,
            )
            print("Using VVIntegrator Plugin")

        except (ModuleNotFoundError, OpenMMException) as e:
            integrator = DrudeNoseHooverIntegrator(
                300 * kelvin,
                coll_freq / picoseconds,
                1 * kelvin,
                drude_coll_freq / picoseconds,
                0.0005 * picoseconds,
            )
            print("Using built in DrudeNoseHooverIntegrator")
            print("Some tests might fail")
            print("Plugin not usable, because:")
            print(e)

        # print(
        #    f"{coll_freq=}, {drude_coll_freq=}"
        # )  # tested with 20, 40, 80, 100, 120, 140, 160: 20,40 bad; 80 - 120 good; 140, 160 crashed
        integrator.setMaxDrudeDistance(0.25 * angstroms)
        try:
            platform = Platform.getPlatformByName("CUDA")
            prop = dict(CudaPrecision="single")  # default is single
            # prop = dict(CudaPrecision="double")
            # Moved creating the simulation object inside the try...except block, because i.e.
            # Error loading CUDA module: CUDA_ERROR_INVALID_PTX (218)
            # message was only thrown during simulation creation not by specifying the platform
            simulation = Simulation(
                psf.topology,
                system,
                integrator,
                platform=platform,
                platformProperties=prop,
            )
        except OpenMMException:
            platform = Platform.getPlatformByName("CPU")
            prop = dict()
            simulation = Simulation(
                psf.topology,
                system,
                integrator,
                platform=platform,
                platformProperties=prop,
            )

        simulation.context.setPositions(crd.positions)
        # Try with positions from equilibrated system:
        # if restart_file is None:
        #    base = f"{protex.__path__[0]}/forcefield"
        #    restart_file = f"{base}/traj/im1h_oac_150_im1_hoac_350_npt_7.rst"
        # if os.path.exists(restart_file):
        #    with open(restart_file) as f:
        #        print(f"Opening restart file {f}")
        #        simulation.context.setState(XmlSerializer.deserialize(f.read()))
        #    simulation.context.computeVirtualSites()
        # else:
        #    print(f"No restart file found. Using initial coordinate file.")
        #    simulation.context.computeVirtualSites()
        #    simulation.context.setVelocitiesToTemperature(300 * kelvin)

        nonbonded_force = [
            f for f in simulation.system.getForces() if isinstance(f, mm.NonbondedForce)
        ][0]
        dummy_atoms = []
        for atom in simulation.topology.atoms():
            if atom.residue.name == "IM1" and atom.name == "H7":
                dummy_atoms.append(atom.index)
                nonbonded_force.setParticleParameters(atom.index, 0.0, 0.0, 0.0)
            if atom.residue.name == "OAC" and atom.name == "H":
                dummy_atoms.append(atom.index)
                nonbonded_force.setParticleParameters(atom.index, 0.0, 0.0, 0.0)
        for exc_id in range(nonbonded_force.getNumExceptions()):
            f = nonbonded_force.getExceptionParameters(exc_id)
            idx1 = f[0]
            idx2 = f[1]
            chargeProd, sigma, epsilon = f[2:]
            if idx1 in dummy_atoms or idx2 in dummy_atoms:
                nonbonded_force.setExceptionParameters(
                    exc_id, idx1, idx2, 0.0, sigma, 0.0
                )

        return simulation

    return setup_simulation()


IM1H_IM1 = {
    "IM1H": {
        "atom_name": "H7",
        "canonical_name": "IM1",  # Not needed?
    },
    "IM1": {
        "atom_name": "N2",
        "canonical_name": "IM1",  # Not needed?
    },
}

OAC_HOAC = {
    "OAC": {
        "atom_name": "O2",
        "canonical_name": "OAC",  # Not needed?
    },
    "HOAC": {
        "atom_name": "H",
        "canonical_name": "OAC",  # Not needed?
    },
}

HPTSH_HPTS = {
    "HPTSH": {
        "atom_name": "H7",
        "canonical_name": "HPTS",
    },
    "HPTS": {
        "atom_name": "O7",
        "canonical_name": "HPTS",
    },
}
