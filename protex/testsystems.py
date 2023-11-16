import os
from itertools import combinations

import protex

try:  # Syntax changed in OpenMM 7.6
    import openmm as mm
    from openmm import (
        Context,
        DrudeNoseHooverIntegrator,
        MonteCarloBarostat,
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
    from openmm.unit import angstroms, atmosphere, kelvin, picoseconds
except ImportError:
    import simtk.openmm as mm
    from simtk.openmm import (
        Context,
        DrudeNoseHooverIntegrator,
        MonteCarloBarostat,
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


# general functions
def load_charmm_files(
    psf_file: str,
    crd_file: str,
    para_files: list[str],
    boxl: float = 48.0,
):
    print("Loading CHARMM files...")
    params = CharmmParameterSet(*para_files)
    psf = CharmmPsfFile(psf_file)
    xtl = boxl * angstroms
    psf.setBox(xtl, xtl, xtl)
    crd = CharmmCrdFile(crd_file)
    return psf, crd, params


def setup_system(
    psf: CharmmPsfFile,
    params: CharmmParameterSet,
    constraints=None,
    dummy_atom_type: str = "DUMH",
    cutoff: float = 11.0,
    switch: float = 10.0,
    ensemble = "nVT"
):
    if dummy_atom_type is not None:
        # print(params.atom_types_str["DUM"].epsilon)
        # print(params.atom_types_str["DUM"].rmin)
        # print(params.atom_types_str["DUM"].epsilon_14)
        # print(params.atom_types_str["DUM"].rmin_14)
        # print(params.atom_types_str["DUM"].nbfix)
        # print(params.atom_types_str["DUM"].nbthole)
        # Charmm DUM atomtype has epsilon = 0, but problem with openmm during creatin of system/simulation
        # openmm.OpenMMException: updateParametersInContext: The set of non-excluded exceptions has changed
        # Therefore set it here non zero and zero afterwards
        print(
            f"Changing atom type {dummy_atom_type} temporarily to not zero for dummy things"
        )
        params.atom_types_str[dummy_atom_type].set_lj_params(
            -0.00001,
            params.atom_types_str[dummy_atom_type].rmin,
            -0.00001,
            params.atom_types_str[dummy_atom_type].rmin_14,
        )
    if constraints is None:
        system = psf.createSystem(
            params,
            nonbondedMethod=PME,
            nonbondedCutoff=cutoff * angstroms,
            switchDistance=switch * angstroms,
            constraints=None,
        )
    elif constraints == "HBonds":
        system = psf.createSystem(
            params,
            nonbondedMethod=PME,
            nonbondedCutoff=cutoff * angstroms,
            switchDistance=switch * angstroms,
            constraints=HBonds,
        )
    else:
        print(
            "Only contraints=None or constraints=HBonds (given as string in function call) implemented"
        )

    if ensemble == "npT":
        barostat = MonteCarloBarostat(1*atmosphere, 300*kelvin)
        system.addForce(barostat)

    for force in system.getForces():
        if type(force).__name__ == "CMMotionRemover":
            # From OpenMM psf file it has automatically ForceGroup 0, which is already used for harmonic bond force
            force.setForceGroup(len(system.getForces()) - 1)
            # print(force.getForceGroup())

    return system


def setup_simulation(
    psf: CharmmPsfFile,
    crd: CharmmCrdFile,
    system: mm.System,
    restart_file: str = None,
    coll_freq: int = 10,
    drude_coll_freq: int = 100,
    dummies: list[tuple[str, str]] = [("IM1", "H7"), ("OAC", "H")],
    use_plugin: bool = True,
    platformname: str = "CUDA",
    cuda_precision: str = "single",
):
    if use_plugin and platformname != "CUDA":
        assert "Plugin only available with CUDA"
    if use_plugin:
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
        print("Using VVIntegrator Plugin")
    else:
        integrator = DrudeNoseHooverIntegrator(
            300 * kelvin,
            coll_freq / picoseconds,
            1 * kelvin,
            drude_coll_freq / picoseconds,
            0.0001 * picoseconds,
        )
        print("Using built in DrudeNoseHooverIntegrator")
        print("Some tests might fail")

    # print(
    #    f"{coll_freq=}, {drude_coll_freq=}"
    # )  # tested with 20, 40, 80, 100, 120, 140, 160: 20,40 bad; 80 - 120 good; 140, 160 crashed
    integrator.setMaxDrudeDistance(0.25 * angstroms)
    try:
        platform = Platform.getPlatformByName(platformname)
        if platformname == "CUDA":
            prop = dict(CudaPrecision=cuda_precision)  # default is single
            print(f"Using precision {cuda_precision}")
        else:
            prop = dict()
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
    except OpenMMException as e:
        print(e)
        platform = Platform.getPlatformByName("CPU")
        prop = dict()
        simulation = Simulation(
            psf.topology,
            system,
            integrator,
            platform=platform,
            platformProperties=prop,
        )
    print(f"Using platform: {platform.getName()}")
    simulation.context.setPositions(crd.positions)
    # Try with positions from equilibrated system:
    if restart_file is not None and os.path.exists(restart_file):
        with open(restart_file) as f:
            print(f"Opening restart file {f}")
            simulation.context.setState(XmlSerializer.deserialize(f.read()))
        simulation.context.computeVirtualSites()
    else:
        print("No restart file found. Using initial coordinate file.")
        simulation.context.computeVirtualSites()
        simulation.context.setVelocitiesToTemperature(300 * kelvin)
    if dummies is not None:
        # set nonbonded parameters including dummy zero again.
        # had to be  zero durin creating becuase of:
        # openmm.OpenMMException: updateParametersInContext: The set of non-excluded exceptions has changed
        nonbonded_force = [
            f for f in simulation.system.getForces() if isinstance(f, mm.NonbondedForce)
        ][0]
        dummy_atoms = []
        for atom in simulation.topology.atoms():
            name_tuple = (atom.residue.name, atom.name)
            if name_tuple in dummies:
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


def generate_complete_system(  # currently not in use
    psf_file: str,
    crd_file: str,
    restart_file: str,  # | None,
    constraints: str,  # | None,
    boxl: float,
    para_files: list[str],
    coll_freq: int,
    drude_coll_freq: int,
    dummy_atom_type: str,
    dummies: list[tuple[str, str]],
    use_plugin: bool,
):
    psf, crd, params = load_charmm_files(
        psf_file=psf_file,
        crd_file=crd_file,
        para_files=para_files,
        boxl=boxl,
    )
    system = setup_system(
        psf, params, constraints=constraints, dummy_atom_type=dummy_atom_type
    )

    simulation = setup_simulation(
        psf,
        crd,
        system,
        restart_file=restart_file,
        coll_freq=coll_freq,
        drude_coll_freq=drude_coll_freq,
        dummies=dummies,
        use_plugin=use_plugin,
    )

    return simulation


# def generate_im1h_oac_system( # using generate_complete_system
#     psf_file: str = None,
#     crd_file: str = None,
#     restart_file: str = None,
#     constraints: str = None,
#     boxl: float = 48.0,
#     para_files: list[str] = None,
#     coll_freq: int=10,
#     drude_coll_freq: int=100,
#     dummy_atom_type: str="DUMH",
#     dummies: list[tuple[str,str]]=[("IM1", "H7"), ("OAC", "H")],
#     use_plugin: bool =True
# ):
#     """Set up a solvated and parametrized system for IM1H/OAC."""
#     base = f"{protex.__path__[0]}/forcefield/dummy"
#     if psf_file is None:
#         psf_file = f"{base}/im1h_oac_150_im1_hoac_350.psf"
#     if crd_file is None:
#         crd_file = f"{base}/im1h_oac_150_im1_hoac_350.crd"
#     if para_files is None:
#         PARA_FILES = [
#             "toppar_drude_master_protein_2013f_lj025.str",
#             "hoac_d.str",
#             "im1h_d.str",
#             "im1_d_dummy.str",
#             "oac_d_dummy.str",
#         ]
#     para_files = [f"{base}/toppar/{para_files}" for para_files in PARA_FILES]

#     if restart_file is None:
#         base = f"{protex.__path__[0]}/forcefield"
#         restart_file = f"{base}/traj/im1h_oac_150_im1_hoac_350_npt_7.rst"

#     simulation = generate_complete_system(
#         psf_file = psf_file,
#         crd_file = crd_file,
#         restart_file = restart_file,
#         constraints = constraints,
#         boxl = boxl,
#         para_files = para_files,
#         coll_freq=coll_freq,
#         drude_coll_freq=drude_coll_freq,
#         dummy_atom_type=dummy_atom_type,
#         dummies=dummies,
#         use_plugin =use_plugin
#     )

#     return simulation


def generate_im1h_oac_system(
    psf_file: str = None,
    crd_file: str = None,
    restart_file: str = None,
    constraints: str = None,
    boxl: float = 48.0,
    para_files: list[str] = None,
    coll_freq: int = 10,
    drude_coll_freq: int = 100,
    dummy_atom_type: str = "DUMH",
    dummies: list[tuple[str, str]] = [("IM1", "H7"), ("OAC", "H")],
    use_plugin: bool = True,
    platformname="CUDA",
    cuda_precision="single",
):
    """Set up a solvated and parametrized system for IM1H/OAC."""
    base = f"{protex.__path__[0]}/forcefield/"
    if psf_file is None:
        psf_file = f"{base}/dummy/im1h_oac_150_im1_hoac_350.psf"
    if crd_file is None:
        crd_file = f"{base}/dummy/im1h_oac_150_im1_hoac_350.crd"
    if para_files is None:
        PARA_FILES = [
            "toppar_drude_master_protein_2013f_lj025.str",
            "hoac_d.str",
            "im1h_d.str",
            "im1_d_dummy.str",
            "oac_d_dummy.str",
        ]
        para_files = [f"{base}/toppar/{para_files}" for para_files in PARA_FILES]

    psf, crd, params = load_charmm_files(
        psf_file=psf_file,
        crd_file=crd_file,
        para_files=para_files,
        boxl=boxl,
    )
    system = setup_system(
        psf, params, constraints=constraints, dummy_atom_type=dummy_atom_type
    )

    if restart_file is None:
        restart_file = f"{base}/traj/im1h_oac_150_im1_hoac_350_npt_7.rst"

    simulation = setup_simulation(
        psf,
        crd,
        system,
        restart_file=restart_file,
        coll_freq=coll_freq,
        drude_coll_freq=drude_coll_freq,
        dummies=dummies,
        use_plugin=use_plugin,
        platformname=platformname,
        cuda_precision=cuda_precision,
    )

    return simulation


def generate_h2o_system(
    psf_file: str = None,
    crd_file: str = None,
    restart_file: str = None,
    constraints: str = None,
    boxl: float = 10.0,
    para_files: list[str] = None,
    coll_freq: int = 10,
    drude_coll_freq: int = 100,
    dummy_atom_type: str = "DUMH",
    dummies: list[tuple[str, str]] = [("OH", "H2"), ("OH", "H3"), ("OH", "H4"), ("H2O", "H3"), ("H2O", "H4"), ("H3O", "H4")], # NOTE: simulation created at start of every run, won't work like this
    use_plugin: bool = True,
    platformname="CUDA",
    cuda_precision="single",
):
    """Set up a solvated and parametrized system for OH/H2O/H3O."""
    base = f"{protex.__path__[0]}/forcefield/"
    if psf_file is None:
        psf_file = f"{base}/h2o/h2o.psf"
    if crd_file is None:
        crd_file = f"{base}/h2o/h2o.crd"
    if para_files is None:
        PARA_FILES = [
            "toppar_drude_master_protein_2013f_lj025_modhpts_chelpg.str",
            "h2o_d.str",
            "h3o_d.str",
            "oh_d.str",
            "cl_d.str",
            "na_d.str",
        ]
        para_files = [f"{base}/h2o/{para_files}" for para_files in PARA_FILES]

    psf, crd, params = load_charmm_files(
        psf_file=psf_file,
        crd_file=crd_file,
        para_files=para_files,
        boxl=boxl,
    )
    system = setup_system(
        psf,
        params,
        constraints=constraints,
        dummy_atom_type=dummy_atom_type,
        cutoff=3,
        switch=2,
    )

    # if restart_file is None:
    #    restart_file = f"{base}/traj/im1h_oac_150_im1_hoac_350_npt_7.rst"

    simulation = setup_simulation(
        psf,
        crd,
        system,
        # restart_file=restart_file,
        coll_freq=coll_freq,
        drude_coll_freq=drude_coll_freq,
        dummies=dummies,
        use_plugin=use_plugin,
        platformname=platformname,
        cuda_precision=cuda_precision,
    )

    return simulation

def generate_toh2_system(
    psf_file: str = None,
    crd_file: str = None,
    restart_file: str = None,
    constraints: str = None,
    boxl: float = 32.0,
    para_files: list[str] = None,
    coll_freq: int = 10,
    drude_coll_freq: int = 100,
    dummy_atom_type: str = "DUMH",
    dummies: list[tuple[str, str]] = [("OH", "H2"), ("OH", "H3"), ("OH", "H4"), ("H2O", "H3"), ("H2O", "H4"), ("H3O", "H4")], # NOTE: simulation created at start of every run, won't work like this
    use_plugin: bool = True,
    platformname="CUDA",
    cuda_precision="single",
):
    """Set up a solvated and parametrized system for OH/H2O/H3O."""
    base = f"{protex.__path__[0]}/forcefield"
    if psf_file is None:
        psf_file = f"{base}/toh2/h2o.psf"
    if crd_file is None:
        crd_file = f"{base}/toh2/h2o.crd"
    if para_files is None:
        PARA_FILES = [
            "toppar_drude_master_protein_2013f_lj025_modhpts_chelpg.str",
            "h2o_d.str",
            "h3o_d.str",
            "oh_d.str",
            "cl_d.str",
            "na_d.str",
        ]
        para_files = [f"{base}/toh2/toppar/{para_files}" for para_files in PARA_FILES]

    if restart_file is None:
        restart_file = f"{base}/toh2/h2o_npt_7.rst"

    psf, crd, params = load_charmm_files(
        psf_file=psf_file,
        crd_file=crd_file,
        para_files=para_files,
        boxl=boxl,
    )
    system = setup_system(
        psf,
        params,
        constraints=constraints,
        dummy_atom_type=dummy_atom_type,
        # why is this different here? (probably because of the 10A small test box from Flo)
        # cutoff=3,
        # switch=2,
    )

    simulation = setup_simulation(
        psf,
        crd,
        system,
        restart_file=restart_file,
        coll_freq=coll_freq,
        drude_coll_freq=drude_coll_freq,
        dummies=dummies,
        use_plugin=use_plugin,
        platformname=platformname,
        cuda_precision=cuda_precision,
    )

    return simulation

def generate_tfa_system(
    psf_file: str = None,
    crd_file: str = None,
    restart_file: str = None,
    constraints: str = None,
    boxl: float = 48.0,
    para_files: list[str] = None,
    coll_freq: int = 10,
    drude_coll_freq: int = 100,
    dummy_atom_type: str = "DUMH",
    dummies: list[tuple[str, str]] = [("IM1", "H7"), ("OAC", "H")],
    use_plugin: bool = True,
    tfa_percent: int = 10,
):
    """Set up a solvated and parametrized system for IM1H/OAC with tfa."""
    base = f"{protex.__path__[0]}/forcefield/"
    if psf_file is None:
        psf_file = f"{base}/tfa/tfa_{tfa_percent}.psf"
    if crd_file is None:
        crd_file = f"{base}/tfa/tfa_{tfa_percent}.crd"
    if para_files is None:
        PARA_FILES = [
            "toppar_drude_master_protein_2013f_lj04.str",
            "hoac_d.str",
            "im1h_d.str",
            "im1_d_dummy.str",
            "oac_d_dummy.str",
            "tfa_d.str",
        ]
        para_files = [f"{base}/toppar/{para_files}" for para_files in PARA_FILES]

    psf, crd, params = load_charmm_files(
        psf_file=psf_file,
        crd_file=crd_file,
        para_files=para_files,
        boxl=boxl,
    )
    system = setup_system(
        psf, params, constraints=constraints, dummy_atom_type=dummy_atom_type
    )

    if restart_file is None:
        restart_file = f"{base}/tfa/tfa_{tfa_percent}_npt_7.rst"

    simulation = setup_simulation(
        psf,
        crd,
        system,
        restart_file=restart_file,
        coll_freq=coll_freq,
        drude_coll_freq=drude_coll_freq,
        dummies=dummies,
        use_plugin=use_plugin,
    )

    return simulation


# used for faster tests, not for production!
def generate_small_box(
    psf_file: str = None,
    crd_file: str = None,
    restart_file: str = None,
    constraints: str = None,
    boxl: float = 22.0,
    para_files: list[str] = None,
    coll_freq: int = 10,
    drude_coll_freq: int = 100,
    dummy_atom_type: str = "DUMH",
    dummies: list[tuple[str, str]] = [("IM1", "H7"), ("OAC", "H")],
    use_plugin: bool = True,
    platformname="CUDA",
    cuda_precision="single",
):
    """Set up a solvated and parametrized system for IM1H/OAC."""
    print(
        "This function should not be used for production. It uses a small system intended for testing"
    )
    base = f"{protex.__path__[0]}/forcefield"
    if psf_file is None:
        psf_file = f"{base}/small_box/small_box.psf"
    if crd_file is None:
        crd_file = f"{base}/small_box/small_box.crd"
    if para_files is None:
        PARA_FILES = [
            "toppar_drude_master_protein_2013f_lj025.str",
            "hoac_d.str",
            "im1h_d.str",
            "im1_d_dummy.str",
            "oac_d_dummy.str",
        ]
        para_files = [f"{base}/toppar/{para_files}" for para_files in PARA_FILES]

    psf, crd, params = load_charmm_files(
        psf_file=psf_file,
        crd_file=crd_file,
        para_files=para_files,
        boxl=boxl,
    )
    system = setup_system(
        psf, params, constraints=constraints, dummy_atom_type=dummy_atom_type
    )

    simulation = setup_simulation(
        psf,
        crd,
        system,
        restart_file=restart_file,
        coll_freq=coll_freq,
        drude_coll_freq=drude_coll_freq,
        dummies=dummies,
        use_plugin=use_plugin,
        platformname=platformname,
        cuda_precision=cuda_precision,
    )

    return simulation


def generate_single_im1h_oac_system(  # does not have dummies
    psf_file: str = None,
    crd_file: str = None,
    restart_file: str = None,
    constraints: str = None,
    boxl: float = 22.0,
    para_files: list[str] = None,
    coll_freq: int = 10,
    drude_coll_freq: int = 100,
    dummy_atom_type: str = None,  # "DUMH",
    dummies: list[tuple[str, str]] = None,  # [("IM1", "H7"), ("OAC", "H")]
    use_plugin: bool = True,
    platformname="CUDA",
    cuda_precision="single",
):
    """Set up a system with 1 IM1H, 1OAC, 1IM1 and 1 HOAC.

    Was for testing the deformation of the imidazole ring -> solved by adding the nonbonded exception to the updates.
    """
    print(
        "This function should not be used for production. It uses a small system intended for testing"
    )
    base = f"{protex.__path__[0]}/forcefield/single_pairs"
    if psf_file is None:
        psf_file = f"{base}/im1h_oac_im1_hoac_1_secondtry.psf"
    if crd_file is None:
        crd_file = f"{base}/im1h_oac_im1_hoac_1_secondtry.crd"
    if para_files is None:
        PARA_FILES = [
            "toppar_drude_master_protein_2013f_lj02.str",
            "hoac_d.str",
            "im1h_d_fm_lj_chelpg.str",
            "im1_d_fm_lj_chelpg_withlp.str",
            "oac_d_lj.str",
        ]
        para_files = [f"{base}/toppar/{para_files}" for para_files in PARA_FILES]

    psf, crd, params = load_charmm_files(
        psf_file=psf_file,
        crd_file=crd_file,
        para_files=para_files,
        boxl=boxl,
    )
    system = setup_system(
        psf, params, constraints=constraints, dummy_atom_type=dummy_atom_type
    )

    simulation = setup_simulation(
        psf,
        crd,
        system,
        restart_file=restart_file,
        coll_freq=coll_freq,
        drude_coll_freq=drude_coll_freq,
        dummies=dummies,
        use_plugin=use_plugin,
        platformname=platformname,
        cuda_precision=cuda_precision,
    )

    return simulation


def generate_im1h_oac_dummy_system(
    psf_file: str = None,
    crd_file: str = None,
    restart_file: str = None,
    constraints: str = None,
    boxl: float = 48.0,
    para_files: list[str] = None,
    coll_freq=10,
    drude_coll_freq=100,
    dummy_atom_type: str = "DUMH",
    dummies: list[tuple[str, str]] = [("IM1", "H7"), ("OAC", "H")],
    use_plugin: bool = True,
    ensemble = "nVT"
):
    """Set up a solvated and parametrized system for IM1H/OAC."""
    base = f"{protex.__path__[0]}/forcefield"
    if psf_file is None:
        psf_file = f"{base}/dummy/im1h_oac_im1_hoac_1.psf"
    if crd_file is None:
        crd_file = f"{base}/dummy/im1h_oac_im1_hoac_1.crd"
    if para_files is None:
        PARA_FILES = [
            "toppar_drude_master_protein_2013f_lj025.str",
            "hoac_d.str",
            "im1h_d.str",
            "im1_d_dummy.str",
            "oac_d_dummy.str",
        ]
        para_files = [f"{base}/toppar/{para_files}" for para_files in PARA_FILES]

    psf, crd, params = load_charmm_files(
        psf_file=psf_file,
        crd_file=crd_file,
        para_files=para_files,
        boxl=boxl,
    )
    system = setup_system(
        psf, params, constraints=constraints, dummy_atom_type=dummy_atom_type, ensemble=ensemble
    )

    simulation = setup_simulation(
        psf,
        crd,
        system,
        restart_file=restart_file,
        coll_freq=coll_freq,
        drude_coll_freq=drude_coll_freq,
        dummies=dummies,
        use_plugin=use_plugin,
    )

    return simulation


### new generate
def generate_hpts_system(  # not in use? -> delete?
    psf_file: str = None,
    crd_file: str = None,
    restart_file: str = None,
    constraints: str = None,
    boxl: float = 48.0,
    para_files: list[str] = None,
    coll_freq: int = 10,
    drude_coll_freq: int = 100,
    dummy_atom_type: str = "DUMH",
    dummies: list[tuple[str, str]] = [("IM1", "H7"), ("OAC", "H"), ("HPTS", "H7")],
    use_plugin: bool = True,
):
    """Set up a solvated and parametrized system for IM1H/OAC/HPTS."""
    base = f"{protex.__path__[0]}/forcefield/"
    if psf_file is None:
        psf_file = f"{base}/hpts.psf"
    if crd_file is None:
        crd_file = f"{base}/hpts.crd"
    if para_files is None:
        PARA_FILES = [
            "toppar_drude_master_protein_2013f_lj025_modhpts.str",
            "hoac_d.str",
            "im1h_d.str",
            "im1_d.str",
            "oac_d.str",
            "hpts_d.str",
            "hptsh_d.str",
        ]
        para_files = [f"{base}/toppar/{para_files}" for para_files in PARA_FILES]

    psf, crd, params = load_charmm_files(
        psf_file=psf_file,
        crd_file=crd_file,
        para_files=para_files,
        boxl=boxl,
    )
    system = setup_system(
        psf, params, constraints=constraints, dummy_atom_type=dummy_atom_type
    )

    if restart_file is None:
        base = f"{protex.__path__[0]}/forcefield"
        restart_file = f"{base}/traj/hpts_npt_7.rst"

    simulation = setup_simulation(
        psf,
        crd,
        system,
        restart_file=restart_file,
        coll_freq=coll_freq,
        drude_coll_freq=drude_coll_freq,
        dummies=dummies,
        use_plugin=use_plugin,
    )

    return simulation


def generate_hpts_meoh_system(
    psf_file: str = None,
    crd_file: str = None,
    restart_file: str = None,
    constraints: str = None,
    boxl: float = 70.0,
    para_files: list[str] = None,
    coll_freq: int = 10,
    drude_coll_freq: int = 100,
    dummy_atom_type: str = "DUMH",
    dummies: list[tuple[str, str]] = [
        ("IM1", "H7"),
        ("OAC", "H"),
        ("HPTS", "H7"),
        ("MEOH", "HO2"),
    ],
    use_plugin: bool = True,
):
    """Set up a solvated and parametrized system for IM1H/OAC/HPTS/MEOH."""
    base = f"{protex.__path__[0]}/forcefield/"
    if psf_file is None:
        psf_file = f"{base}/hpts.psf"
    if crd_file is None:
        crd_file = f"{base}/hpts.crd"
    if para_files is None:
        PARA_FILES = [
            "toppar_drude_master_protein_2013f_lj025_modhpts_chelpg.str",
            "hoac_d.str",
            "im1h_d.str",
            "im1_dummy_d.str",
            "oac_dummy_d.str",
            "hpts_dummy_d_chelpg.str",
            "hptsh_d_chelpg.str",
            "meoh_dummy.str",
            "meoh2_unscaled.str",
        ]
        para_files = [f"{base}/toppar/{para_files}" for para_files in PARA_FILES]

    psf, crd, params = load_charmm_files(
        psf_file=psf_file,
        crd_file=crd_file,
        para_files=para_files,
        boxl=boxl,
    )
    system = setup_system(
        psf, params, constraints=constraints, dummy_atom_type=dummy_atom_type
    )

    simulation = setup_simulation(
        psf,
        crd,
        system,
        restart_file=restart_file,
        coll_freq=coll_freq,
        drude_coll_freq=drude_coll_freq,
        dummies=dummies,
        use_plugin=use_plugin,
    )

    return simulation


def generate_hpts_meoh_lj04_system(
    psf_file: str = None,
    crd_file: str = None,
    restart_file: str = None,
    constraints: str = None,
    boxl: float = 70.0,
    para_files: list[str] = None,
    coll_freq=10,
    drude_coll_freq=100,
    dummy_atom_type: str = "DUMH",
    dummies: list[tuple[str, str]] = [
        ("IM1", "H7"),
        ("OAC", "H"),
        ("HPTS", "H7"),
        ("MEOH", "HO2"),
    ],
    use_plugin: bool = True,
):
    """Set up a solvated and parametrized system for IM1H/OAC/HPTS/MEOH."""
    base = f"{protex.__path__[0]}/forcefield/"
    if psf_file is None:
        psf_file = f"{base}/hpts.psf"
    if crd_file is None:
        crd_file = f"{base}/hpts.crd"
    if para_files is None:
        PARA_FILES = [
            "toppar_drude_master_protein_2013f_lj04_modhpts_chelpg.str",
            "hoac_d.str",
            "im1h_d.str",
            "im1_dummy_d.str",
            "oac_dummy_d.str",
            "hpts_dummy_d_chelpg.str",
            "hptsh_d_chelpg.str",
            "meoh_dummy.str",
            "meoh2_unscaled.str",
        ]
        para_files = [f"{base}/toppar/{para_files}" for para_files in PARA_FILES]

    psf, crd, params = load_charmm_files(
        psf_file=psf_file,
        crd_file=crd_file,
        para_files=para_files,
        boxl=boxl,
    )
    system = setup_system(
        psf, params, constraints=constraints, dummy_atom_type=dummy_atom_type
    )

    simulation = setup_simulation(
        psf,
        crd,
        system,
        restart_file=restart_file,
        coll_freq=coll_freq,
        drude_coll_freq=drude_coll_freq,
        dummies=dummies,
        use_plugin=use_plugin,
    )

    return simulation


# used for faster tests, not for production!
def generate_single_hpts_meoh_system(
    psf_file: str = None,
    crd_file: str = None,
    restart_file: str = None,
    constraints: str = None,
    boxl: float = 22.0,
    para_files: list[str] = None,
    coll_freq=10,
    drude_coll_freq=100,
    dummy_atom_type="DUMH",
    dummies: list[tuple[str, str]] = [
        ("IM1", "H7"),
        ("OAC", "H"),
        ("HPTS", "H7"),
        ("MEOH", "HO2"),
    ],
    use_plugin=True,
):
    """Set up a solvated and parametrized system for IM1H/OAC."""
    print(
        "This function should not be used for production. It uses a small system intended for testing"
    )
    base = f"{protex.__path__[0]}/forcefield/"
    if psf_file is None:
        psf_file = f"{base}/hpts_single/hpts_single.psf"
    if crd_file is None:
        crd_file = f"{base}/hpts_single/hpts_single.crd"
    if para_files is None:
        PARA_FILES = [
            "toppar_drude_master_protein_2013f_lj025_modhpts.str",
            "hoac_d.str",
            "im1h_d.str",
            "im1_dummy_d.str",
            "oac_dummy_d.str",
            "hpts_dummy_d_chelpg.str",
            "hptsh_d_chelpg.str",
            "meoh_dummy.str",
            "meoh2_unscaled.str",
        ]
        para_files = [f"{base}/toppar/{para_files}" for para_files in PARA_FILES]

    psf, crd, params = load_charmm_files(
        psf_file=psf_file,
        crd_file=crd_file,
        para_files=para_files,
        boxl=boxl,
    )
    system = setup_system(
        psf, params, constraints=constraints, dummy_atom_type=dummy_atom_type
    )

    simulation = setup_simulation(
        psf,
        crd,
        system,
        restart_file=restart_file,
        coll_freq=coll_freq,
        drude_coll_freq=drude_coll_freq,
        dummies=dummies,
        use_plugin=use_plugin,
    )

    return simulation


IM1H_IM1 = {
    "IM1H": {
        "starting_donors": ["H7"], "starting_acceptors": [], "possible_modes": ("donor",)
    },
    "IM1": {
        "starting_donors": [], "starting_acceptors": ["H7"], "possible_modes": ("acceptor",)
    },
}

OAC_HOAC = {
    "OAC": {
        "starting_donors": [], "starting_acceptors": ["H"], "possible_modes": ("acceptor",)
    },
    "HOAC": {
        "starting_donors": ["H"], "starting_acceptors": [], "possible_modes": ("donor",)
    },
}

HPTSH_HPTS = {
    "HPTSH": {
        "starting_donors": ["H7"], "starting_acceptors": [], "possible_modes": ("donor",)
        },
    "HPTS": {
        "starting_donors": [], "starting_acceptors": ["H7"], "possible_modes": ("acceptor",)
    },
}

MEOH_MEOH2 = {
    "MEOH": {
        "starting_donors": ["HO1"], "starting_acceptors": ["HO2"], "possible_modes": ("acceptor",)
    },
    "MEOH2": {
        "starting_donors": ["HO1","HO2"], "starting_acceptors": [], "possible_modes": ("donor",)
    },
}

# NOTE: take care whether we want to use H2O or SWM4, SPCE etc. for residue name
# TODO: at the moment fixed atom names, will revert to this at the beginning of each run -> reformulate so that donors and acceptors are filled based on atom type (should be ok with pickling residues)

OH_H2O_H3O =  {
    "OH":  {"starting_donors" : ["H1"], "starting_acceptors" : ["H2", "H3", "H4"], "possible_modes" : ("acceptor")},
    "H2O": {"starting_donors" : ["H1", "H2"], "starting_acceptors" : ["H3", "H4"], "possible_modes" : ("acceptor", "donor")},
    "H3O": {"starting_donors" : ["H1", "H2", "H3"], "starting_acceptors" : ["H4"], "possible_modes" : ("donor")},
}

# CLA = {"CLA": {"starting_donors" : [], "starting_acceptors" : [], "modes" : ()}}

# SOD = {"SOD": {"starting_donors" : [], "starting_acceptors" : [], "modes" : ()}}

# TODO:
# keep track of which H is real at the moment: like "donors" : (), "acceptors" : ()
    # caution: will have to change how we write the psf as well
        # new idea: write pasf as usual, save donors and acceptors, adjust at setup
# get atom index or something like that in the update step and swap H to D and vice versa
# switch parameters around to get the parameter sets for each possible state
    # at the moment only nonbonded parameters change, all possible donors and acceptors in a single molecule are equivalent (i.e. no two different acidic side chains)
        # update like now
        # set parameters for H and D that were used extra
            # D: force.setParticleParameters(atomidx, 0, 0, 0]
            # H: extract nonbonded parameters from template first
