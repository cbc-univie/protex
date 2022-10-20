# Import package, test suite, and other packages as needed
# import json
import copy
import io
import logging
import os
from collections import defaultdict
from sys import stdout

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
        DCDReporter,
        HBonds,
        PDBReporter,
        Simulation,
        StateDataReporter,
    )
    from openmm.unit import (
        angstroms,
        kelvin,
        kilocalories_per_mole,
        md_kilocalories,
        picoseconds,
    )
except ImportError:
    import simtk.openmm as mm
    from simtk.openmm import (
        OpenMMException,
        Platform,
        Context,
        DrudeLangevinIntegrator,
        DrudeNoseHooverIntegrator,
        XmlSerializer,
    )
    from simtk.openmm.app import DCDReporter, PDBReporter, StateDataReporter
    from simtk.openmm.app import CharmmCrdFile, CharmmParameterSet, CharmmPsfFile
    from simtk.openmm.app import PME, HBonds
    from simtk.openmm.app import Simulation
    from simtk.unit import angstroms, kelvin, picoseconds

import pytest

import protex

from ..reporter import ChargeReporter, EnergyReporter
from ..system import IonicLiquidSystem, IonicLiquidTemplates
from ..testsystems import (
    IM1H_IM1,
    OAC_HOAC,
    generate_im1h_oac_system,
    generate_single_im1h_oac_system,
)
from ..update import NaiveMCUpdate, StateUpdate


def test_available_platforms():
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

    system = psf.createSystem(
        params,
        nonbondedMethod=PME,
        nonbondedCutoff=11.0 * angstroms,
        switchDistance=10 * angstroms,
        constraints=HBonds,
    )

    # plugin
    # https://github.com/z-gong/openmm-velocityVerlet

    coll_freq = 10
    drude_coll_freq = 80

    try:
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
        context = Context(system, integrator)
        del context
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

        simulation = Simulation(
            psf.topology, system, integrator, platform=platform, platformProperties=prop
        )
    except mm.OpenMMException:
        platform = Platform.getPlatformByName("CPU")
        prop = dict()
        simulation = Simulation(
            psf.topology, system, integrator, platform=platform, platformProperties=prop
        )
    print(platform.getName())

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


def test_setup_simulation():
    simulation = generate_im1h_oac_system()
    print("Minimizing...")
    simulation.minimizeEnergy(maxIterations=100)
    system = simulation.system

    nr_of_particles = system.getNumParticles()
    assert nr_of_particles == 17500 + 500  # +lps for im1 im1h


def test_run_simulation():

    simulation = generate_im1h_oac_system()
    print("Minimizing...")
    simulation.minimizeEnergy(maxIterations=50)
    simulation.reporters.append(PDBReporter("output.pdb", 50))
    simulation.reporters.append(DCDReporter("output.dcd", 50))

    simulation.reporters.append(
        StateDataReporter(
            stdout,
            50,
            step=True,
            potentialEnergy=True,
            temperature=True,
            time=True,
            volume=True,
            density=False,
        )
    )
    print("Running dynmamics...")
    simulation.step(200)
    # If simulation aborts with Nan error, try smaller timestep (e.g. 0.0001 ps) and then extract new crd from dcd using "scripts/crdfromdcd.inp"
    os.remove("output.pdb")
    os.remove("output.dcd")


def test_create_IonicLiquidTemplate():
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 2.33}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": -2.33}

    templates = IonicLiquidTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))

    r = templates.get_residue_name_for_coupled_state("OAC")
    assert r == "HOAC"
    r = templates.get_residue_name_for_coupled_state("HOAC")
    assert r == "OAC"
    r = templates.get_residue_name_for_coupled_state("IM1H")
    assert r == "IM1"

    print("###################")
    assert templates.pairs == [["OAC", "HOAC"], ["IM1H", "IM1"]]
    assert templates.states["IM1H"]["atom_name"] == "H7"
    assert templates.states["IM1"]["atom_name"] == "N2"

    assert sorted(templates.names) == sorted(["OAC", "HOAC", "IM1H", "IM1"])
    print(templates.allowed_updates)
    assert templates.overall_max_distance == 0.16

    neutral_prob = templates.get_update_value_for(frozenset(["IM1", "HOAC"]), "prob")
    assert neutral_prob == -2.33
    ionic_prob = templates.get_update_value_for(frozenset(["IM1H", "OAC"]), "prob")
    assert ionic_prob == 2.33


def test_create_IonicLiquid():

    simulation = generate_im1h_oac_system()
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 2.33}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": -2.33}

    templates = IonicLiquidTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))

    count = defaultdict(int)
    ionic_liquid = IonicLiquidSystem(simulation, templates)

    assert len(ionic_liquid.residues) == 1000
    for idx, residue in enumerate(ionic_liquid.residues):
        # print(f"{idx} : {residue.original_name}")
        count[residue.original_name] += 1

    assert count["IM1H"] == 150
    assert count["OAC"] == 150
    assert count["IM1"] == 350
    assert count["HOAC"] == 350


def test_save_load_allowedupdates():

    simulation = generate_im1h_oac_system()
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 0.9}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 0.3}

    templates = IonicLiquidTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    ionic_liquid = IonicLiquidSystem(simulation, templates)

    assert allowed_updates == ionic_liquid.templates.allowed_updates

    ionic_liquid.save_updates("updates.yaml")
    ionic_liquid.load_updates("updates.yaml")
    os.remove("updates.yaml")

    print(ionic_liquid.templates.allowed_updates)

    assert allowed_updates == ionic_liquid.templates.allowed_updates


def test_residues():
    simulation = generate_im1h_oac_system()
    topology = simulation.topology
    for idx, r in enumerate(topology.residues()):
        if r.name == "IM1H" and idx == 0:
            atom_idxs = [atom.index for atom in r.atoms()]
            atom_names = [atom.name for atom in r.atoms()]
            print(atom_idxs)
            print(atom_names)
            assert atom_idxs == [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
            ]
            assert atom_names == [
                "C1",
                "DC1",
                "H1",
                "H2",
                "H3",
                "N1",
                "DN1",
                "C2",
                "DC2",
                "H4",
                "C3",
                "DC3",
                "H5",
                "C4",
                "DC4",
                "H6",
                "N2",
                "DN2",
                "H7",
                "LPN21",
            ]
        if r.name == "HOAC" and idx == 650:
            atom_idxs = [atom.index for atom in r.atoms()]
            atom_names = [atom.name for atom in r.atoms()]
            print(atom_idxs)
            print(atom_names)
            print(idx)
            assert atom_idxs == [
                12400,
                12401,
                12402,
                12403,
                12404,
                12405,
                12406,
                12407,
                12408,
                12409,
                12410,
                12411,
                12412,
                12413,
                12414,
                12415,
            ]

            assert atom_names == [
                "C1",
                "DC1",
                "C2",
                "DC2",
                "H1",
                "H2",
                "H3",
                "O1",
                "DO1",
                "O2",
                "DO2",
                "H",
                "LPO11",
                "LPO12",
                "LPO21",
                "LPO22",
            ]


def test_forces():

    simulation = generate_im1h_oac_system()
    system = simulation.system
    topology = simulation.topology
    force_state = defaultdict(list)  # store bond force
    atom_idxs = {}  # store atom_idxs
    atom_names = {}  # store atom_names
    names = []  # store residue names

    # iterate over residues, select the first residue for HOAC and OAC and save the individual bonded forces
    for ridx, r in enumerate(topology.residues()):
        if r.name == "HOAC" and ridx == 650:  # match first HOAC residue
            names.append(r.name)
            atom_idxs[r.name] = [atom.index for atom in r.atoms()]
            atom_names[r.name] = [atom.name for atom in r.atoms()]
            for force in system.getForces():
                if type(force).__name__ == "HarmonicBondForce":
                    for bond_id in range(force.getNumBonds()):  # iterate over all bonds
                        f = force.getBondParameters(bond_id)
                        idx1 = f[0]
                        idx2 = f[1]
                        if (
                            idx1 in atom_idxs[r.name] and idx2 in atom_idxs[r.name]
                        ):  # atom index of bond force needs to be in atom_idxs
                            force_state[r.name].append(f)

        if r.name == "OAC" and ridx == 150:
            names.append(r.name)
            atom_idxs[r.name] = [atom.index for atom in r.atoms()]
            atom_names[r.name] = [atom.name for atom in r.atoms()]
            for force in system.getForces():
                # print(type(force).__name__)
                if type(force).__name__ == "HarmonicBondForce":
                    for bond_id in range(force.getNumBonds()):
                        f = force.getBondParameters(bond_id)
                        idx1 = f[0]
                        idx2 = f[1]
                        if idx1 in atom_idxs[r.name] and idx2 in atom_idxs[r.name]:
                            force_state[r.name].append(f)

    if len(force_state[names[0]]) != len(
        force_state[names[1]]
    ):  # check the number of entries in the forces
        print(f"{names[0]}: {len(force_state[names[0]])}")
        print(f"{names[1]}: {len(force_state[names[1]])}")

        print(f"{names[0]}:Atom indicies and atom names")
        for idx, name in zip(atom_idxs[names[0]], atom_names[names[0]]):
            print(f"{idx}:{name}")
        print(f"{names[1]}:Atom indicies and atom names")
        for idx, name in zip(atom_idxs[names[1]], atom_names[names[1]]):
            print(f"{idx}:{name}")

        # print forces for the two residues
        print("########################")
        print(names[0])
        for f in force_state[names[0]]:
            print(f)

        print("########################")
        print(names[1])
        for f in force_state[names[1]]:
            print(f)

        raise AssertionError("ohoh")


def test_torsion_forces():

    simulation = generate_im1h_oac_system()
    system = simulation.system
    topology = simulation.topology
    force_state = defaultdict(list)  # store bond force
    atom_idxs = {}  # store atom_idxs
    atom_names = {}  # store atom_names
    names = []  # store residue names

    # iterate over residues, select the first residue for HOAC and OAC and save the individual bonded forces
    for ridx, r in enumerate(topology.residues()):
        if r.name == "HOAC" and ridx == 650:  # match first HOAC residue
            names.append(r.name)
            atom_idxs[r.name] = [atom.index for atom in r.atoms()]
            atom_names[r.name] = [atom.name for atom in r.atoms()]
            for force in system.getForces():
                if type(force).__name__ == "PeriodicTorsionForce":
                    for torsion_id in range(
                        force.getNumTorsions()
                    ):  # iterate over all bonds
                        f = force.getTorsionParameters(torsion_id)
                        idx1 = f[0]
                        idx2 = f[1]
                        idx3 = f[2]
                        idx4 = f[3]
                        if (
                            idx1 in atom_idxs[r.name]
                            and idx2 in atom_idxs[r.name]
                            and idx3 in atom_idxs[r.name]
                            and idx4 in atom_idxs[r.name]
                        ):  # atom index of bond force needs to be in atom_idxs
                            force_state[r.name].append(f)
                            print("hoac", f)

        if r.name == "OAC" and ridx == 150:
            names.append(r.name)
            atom_idxs[r.name] = [atom.index for atom in r.atoms()]
            atom_names[r.name] = [atom.name for atom in r.atoms()]
            for force in system.getForces():
                # print(type(force).__name__)
                if type(force).__name__ == "PeriodicTorsionForce":
                    for torsion_id in range(force.getNumTorsions()):
                        f = force.getTorsionParameters(torsion_id)
                        idx1 = f[0]
                        idx2 = f[1]
                        idx3 = f[2]
                        idx4 = f[3]
                        if (
                            idx1 in atom_idxs[r.name]
                            and idx2 in atom_idxs[r.name]
                            and idx3 in atom_idxs[r.name]
                            and idx4 in atom_idxs[r.name]
                        ):
                            force_state[r.name].append(f)
                            print("oac", f)

    if len(force_state[names[0]]) != len(
        force_state[names[1]]
    ):  # check the number of entries in the forces
        print(f"{names[0]}: {len(force_state[names[0]])}")
        print(f"{names[1]}: {len(force_state[names[1]])}")

        print(f"{names[0]}:Atom indicies and atom names")
        for idx, name in zip(atom_idxs[names[0]], atom_names[names[0]]):
            print(f"{idx}:{name}")
        print(f"{names[1]}:Atom indicies and atom names")
        for idx, name in zip(atom_idxs[names[1]], atom_names[names[1]]):
            print(f"{idx}:{name}")

        # print forces for the two residues
        print("########################")
        print(names[0])
        for f in force_state[names[0]]:
            print(f)

        print("########################")
        print(names[1])
        for f in force_state[names[1]]:
            print(f)

    # iterate over residues, select the first residue for HOAC and OAC and save the individual bonded forces
    for ridx, r in enumerate(topology.residues()):
        if r.name == "HOAC" and ridx == 650:  # match first HOAC residue
            names.append(r.name)
            atom_idxs[r.name] = [atom.index for atom in r.atoms()]
            atom_names[r.name] = [atom.name for atom in r.atoms()]
            for force in system.getForces():
                if type(force).__name__ == "CustomTorsionForce":
                    for torsion_id in range(
                        force.getNumTorsions()
                    ):  # iterate over all bonds
                        f = force.getTorsionParameters(torsion_id)
                        idx1 = f[0]
                        idx2 = f[1]
                        idx3 = f[2]
                        idx4 = f[3]
                        if (
                            idx1 in atom_idxs[r.name]
                            and idx2 in atom_idxs[r.name]
                            and idx3 in atom_idxs[r.name]
                            and idx4 in atom_idxs[r.name]
                        ):  # atom index of bond force needs to be in atom_idxs
                            force_state[r.name].append(f)
                            print("hoac", f)

        if r.name == "OAC" and ridx == 150:
            names.append(r.name)
            atom_idxs[r.name] = [atom.index for atom in r.atoms()]
            atom_names[r.name] = [atom.name for atom in r.atoms()]
            for force in system.getForces():
                # print(type(force).__name__)
                if type(force).__name__ == "CustomTorsionForce":
                    for torsion_id in range(force.getNumTorsions()):
                        f = force.getTorsionParameters(torsion_id)
                        idx1 = f[0]
                        idx2 = f[1]
                        idx3 = f[2]
                        idx4 = f[3]
                        if (
                            idx1 in atom_idxs[r.name]
                            and idx2 in atom_idxs[r.name]
                            and idx3 in atom_idxs[r.name]
                            and idx4 in atom_idxs[r.name]
                        ):
                            force_state[r.name].append(f)
                            print("oac", f)

    if len(force_state[names[0]]) != len(
        force_state[names[1]]
    ):  # check the number of entries in the forces
        print(f"{names[0]}: {len(force_state[names[0]])}")
        print(f"{names[1]}: {len(force_state[names[1]])}")

        print(f"{names[0]}:Atom indicies and atom names")
        for idx, name in zip(atom_idxs[names[0]], atom_names[names[0]]):
            print(f"{idx}:{name}")
        print(f"{names[1]}:Atom indicies and atom names")
        for idx, name in zip(atom_idxs[names[1]], atom_names[names[1]]):
            print(f"{idx}:{name}")

        # print forces for the two residues
        print("########################")
        print(names[0])
        for f in force_state[names[0]]:
            print(f)

        print("########################")
        print(names[1])
        for f in force_state[names[1]]:
            print(f)

        raise AssertionError("ohoh")


def test_drude_forces():

    simulation = generate_im1h_oac_system()
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 2.33}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": -2.33}

    templates = IonicLiquidTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))

    ionic_liquid = IonicLiquidSystem(simulation, templates)
    system = simulation.system
    topology = simulation.topology
    force_state = defaultdict(list)  # store drude force
    force_state_thole = defaultdict(list)  # store drude force
    atom_idxs = defaultdict(list)  # store atom_idxs
    atom_names = defaultdict(list)  # store atom_names
    names = []  # store names
    pair_12_13_list = ionic_liquid._build_exclusion_list(ionic_liquid.topology)

    # iterate over residues, select the first residue for HOAC and OAC and save the individual bonded forces
    for ridx, r in enumerate(topology.residues()):
        if r.name == "HOAC" and ridx == 650:  # match first HOAC residue
            names.append(r.name)
            atom_idxs[r.name] = [atom.index for atom in r.atoms()]
            atom_names[r.name] = [atom.name for atom in r.atoms()]
            drude_force = [
                f for f in system.getForces() if isinstance(f, mm.DrudeForce)
            ][0]
            # harmonic_force = system.getForces()[0]
            # harmonic_force2 = system.getForces()[2]
            print(f"{r.name=}")
            print("drude")
            print(drude_force.getNumParticles())
            for drude_id in range(drude_force.getNumParticles()):
                f = drude_force.getParticleParameters(drude_id)
                idx1, idx2 = f[0], f[1]
                if idx1 in atom_idxs[r.name] and idx2 in atom_idxs[r.name]:
                    print(f)
                    force_state[r.name].append(f)
            print("thole")
            print(drude_force.getNumScreenedPairs())
            for drude_id in range(drude_force.getNumScreenedPairs()):
                f = drude_force.getScreenedPairParameters(drude_id)
                parent1, parent2 = pair_12_13_list[drude_id]
                drude1, drude2 = parent1 + 1, parent2 + 1
                # print(f"thole {idx1=}, {idx2=}")
                # print(f"{drude_id=}, {f=}")
                if drude1 in atom_idxs[r.name] and drude2 in atom_idxs[r.name]:
                    # idx1, idx2 = f[0], f[1]
                    # if ( idx1 in atom_idxs[r.name] and idx2 in atom_idxs[r.name] ):
                    print(f)
                    force_state_thole[r.name].append(f)

        if r.name == "OAC" and ridx == 150:
            names.append(r.name)
            atom_idxs[r.name] = [atom.index for atom in r.atoms()]
            atom_names[r.name] = [atom.name for atom in r.atoms()]
            drude_force = [
                f for f in system.getForces() if isinstance(f, mm.DrudeForce)
            ][0]
            print(f"{r.name=}")
            print("drude")
            print(drude_force.getNumParticles())
            for drude_id in range(drude_force.getNumParticles()):
                f = drude_force.getParticleParameters(drude_id)
                idx1, idx2 = f[0], f[1]
                if idx1 in atom_idxs[r.name] and idx2 in atom_idxs[r.name]:
                    print(f)
                    force_state[r.name].append(f)

            print("thole")
            print(drude_force.getNumScreenedPairs())
            for drude_id in range(drude_force.getNumScreenedPairs()):
                f = drude_force.getScreenedPairParameters(drude_id)
                parent1, parent2 = pair_12_13_list[drude_id]
                drude1, drude2 = parent1 + 1, parent2 + 1
                # print(f"thole {idx1=}, {idx2=}")
                # print(f"{drude_id=}, {f=}")
                if drude1 in atom_idxs[r.name] and drude2 in atom_idxs[r.name]:
                    print(f)
                    force_state_thole[r.name].append(f)

    if len(force_state[names[0]]) != len(
        force_state[names[1]]
    ):  # check the number of entries in the forces
        print(f"{names[0]}: {len(force_state[names[0]])}")
        print(f"{names[1]}: {len(force_state[names[1]])}")

        print(f"{names[0]}:Atom indicies and atom names")
        for idx, name in zip(atom_idxs[names[0]], atom_names[names[0]]):
            print(f"{idx}:{name}")
        print(f"{names[1]}:Atom indicies and atom names")
        for idx, name in zip(atom_idxs[names[1]], atom_names[names[1]]):
            print(f"{idx}:{name}")

        # print forces for the two residues
        print("########################")
        print(names[0])
        for f in force_state[names[0]]:
            print(f)

        print("########################")
        print(names[1])
        for f in force_state[names[1]]:
            print(f)

        raise AssertionError("ohoh")


def test_create_IonicLiquid_residue():
    simulation = generate_im1h_oac_system()
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 2.33}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": -2.33}

    templates = IonicLiquidTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))

    ionic_liquid = IonicLiquidSystem(simulation, templates)
    assert len(ionic_liquid.residues) == 1000

    residue = ionic_liquid.residues[0]
    charge = residue.endstate_charge

    assert charge == 1
    print(residue.atom_names)
    assert (residue.get_idx_for_atom_name("H7")) == 18

    residue = ionic_liquid.residues[1]

    assert (residue.get_idx_for_atom_name("H7")) == 38

    # check name of first residue
    assert ionic_liquid.residues[0].current_name == "IM1H"
    assert ionic_liquid.residues[0].original_name == "IM1H"


def test_tosion_parameters_single():
    from pprint import pprint

    simulation = generate_single_im1h_oac_system()
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 1}

    templates = IonicLiquidTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))

    ionic_liquid = IonicLiquidSystem(simulation, templates)

    for residue in ionic_liquid.residues:
        print(residue.current_name)
        if residue.current_name == "IM1H":
            p0_im1h = residue._get_PeriodicTorsionForce_parameters_at_lambda(0)
            p1_im1 = residue._get_PeriodicTorsionForce_parameters_at_lambda(1)
            residue._set_PeriodicTorsionForce_parameters(p1_im1)
        if residue.current_name == "IM1":
            p0_im1 = residue._get_PeriodicTorsionForce_parameters_at_lambda(0)
            p1_im1h = residue._get_PeriodicTorsionForce_parameters_at_lambda(1)
            residue._set_PeriodicTorsionForce_parameters(p1_im1h)
        if residue.current_name == "OAC":
            p0_oac = residue._get_PeriodicTorsionForce_parameters_at_lambda(0)
            p1_hoac = residue._get_PeriodicTorsionForce_parameters_at_lambda(1)
            residue._set_PeriodicTorsionForce_parameters(p1_hoac)
        if residue.current_name == "HOAC":
            p0_hoac = residue._get_PeriodicTorsionForce_parameters_at_lambda(0)
            p1_oac = residue._get_PeriodicTorsionForce_parameters_at_lambda(1)
            residue._set_PeriodicTorsionForce_parameters(p1_oac)
    for i, j in zip(p0_im1h, p1_im1h):
        assert i == j
    for i, j in zip(p1_im1, p0_im1):
        assert i == j
    for i, j in zip(p0_oac, p1_oac):
        assert i == j
    for i, j in zip(p1_hoac, p0_hoac):
        assert i == j
    # for old, new in zip(p0_oac, p1_hoac):
    #     print(
    #         f"{old[2].in_units_of(kilocalories_per_mole)}, {new[2].in_units_of(kilocalories_per_mole)}"
    #     )


def test_bond_parameters_single():
    from pprint import pprint

    simulation = generate_single_im1h_oac_system()
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 1}

    templates = IonicLiquidTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))

    ionic_liquid = IonicLiquidSystem(simulation, templates)

    for residue in ionic_liquid.residues:
        print(residue.current_name)
        if residue.current_name == "IM1H":
            p0_im1h = residue._get_HarmonicBondForce_parameters_at_lambda(0)
            p1_im1 = residue._get_HarmonicBondForce_parameters_at_lambda(1)
            residue._set_HarmonicBondForce_parameters(p1_im1)
        if residue.current_name == "IM1":
            p0_im1 = residue._get_HarmonicBondForce_parameters_at_lambda(0)
            p1_im1h = residue._get_HarmonicBondForce_parameters_at_lambda(1)
            residue._set_HarmonicBondForce_parameters(p1_im1h)
        if residue.current_name == "OAC":
            p0_oac = residue._get_HarmonicBondForce_parameters_at_lambda(0)
            p1_hoac = residue._get_HarmonicBondForce_parameters_at_lambda(1)
            residue._set_HarmonicBondForce_parameters(p1_hoac)
        if residue.current_name == "HOAC":
            p0_hoac = residue._get_HarmonicBondForce_parameters_at_lambda(0)
            p1_oac = residue._get_HarmonicBondForce_parameters_at_lambda(1)
            residue._set_HarmonicBondForce_parameters(p1_oac)
    for i, j in zip(p0_im1h, p1_im1h):
        assert i == j
    for i, j in zip(p1_im1, p0_im1):
        assert i == j
    for i, j in zip(p0_oac, p1_oac):
        assert i == j
    for i, j in zip(p1_hoac, p0_hoac):
        assert i == j
    # for old, new in zip(p0_oac, p1_hoac):
    #     print(f"{old[1]}, {new[1]}")


# def test_save_load_residue_names():
#     # obtain simulation object
#     simulation = generate_im1h_oac_system()
#     # get ionic liquid templates
#     allowed_updates = {}
#     allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 2.33}
#     allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": -2.33}

#     templates = IonicLiquidTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
#     # wrap system in IonicLiquidSystem
#     ionic_liquid = IonicLiquidSystem(simulation, templates)
#     # initialize update method
#     update = NaiveMCUpdate(ionic_liquid)
#     # initialize state update class
#     state_update = StateUpdate(update)

#     ionic_liquid.simulation.step(50)
#     state_update.update(2)

#     residue_names_1 = [residue.current_name for residue in ionic_liquid.residues]
#     residue_parameters_1 = [residue.parameters for residue in ionic_liquid.residues]

#     ionic_liquid.save_current_names("current_names.txt")

#     ionic_liquid.load_current_names("current_names.txt")

#     residue_names_2 = [residue.current_name for residue in ionic_liquid.residues]
#     residue_parameters_2 = [residue.parameters for residue in ionic_liquid.residues]

#     assert (
#         residue_names_1 == residue_names_2
#     ), "Names should have been loaded into ionic_liquid..."

#     assert residue_parameters_1 == residue_parameters_2

#     # obtain simulation object
#     simulation = generate_im1h_oac_system()
#     # get ionic liquid templates
#     allowed_updates = {}
#     allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 2.33}
#     allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": -2.33}

#     templates = IonicLiquidTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
#     # wrap system in IonicLiquidSystem
#     ionic_liquid = IonicLiquidSystem(simulation, templates)
#     ionic_liquid.load_current_names("current_names.txt")

#     residue_names_2 = [residue.current_name for residue in ionic_liquid.residues]
#     residue_parameters_2 = [residue.parameters for residue in ionic_liquid.residues]

#     assert (
#         residue_names_1 == residue_names_2
#     ), "Names should have been loaded into ionic_liquid..."

#     assert residue_parameters_1 == residue_parameters_2


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Skipping tests that cannot pass in github actions",
)
def test_reporter_class():
    # obtain simulation object
    simulation = generate_im1h_oac_system()
    # get ionic liquid templates
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 2.33}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": -2.33}

    templates = IonicLiquidTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)
    # initialize update method
    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)

    report_interval = 5
    charge_info = {"dcd_save_freq": 500}
    charge_reporter = ChargeReporter(stdout, 20, ionic_liquid, header_data=charge_info)
    ionic_liquid.simulation.reporters.append(charge_reporter)
    ionic_liquid.simulation.reporters.append(
        StateDataReporter(
            stdout,
            report_interval,
            step=True,
            time=True,
            totalEnergy=True,
        )
    )
    ionic_liquid.simulation.reporters.append(EnergyReporter("energy_1.out", 20))

    ionic_liquid.simulation.step(19)
    state_update.update(2)
    ionic_liquid.simulation.step(18)
    state_update.update(2)
    ionic_liquid.simulation.step(18)
    state_update.update(2)
    ionic_liquid.simulation.step(18)
    ionic_liquid.simulation.step(1)


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Will fail sporadicaly.",
)
def test_write_psf_save_load():
    simulation = generate_im1h_oac_system()
    # get ionic liquid templates
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.165, "prob": 1}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.165, "prob": 1}

    templates = IonicLiquidTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)
    # initialize update method
    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)

    old_psf_file = f"{protex.__path__[0]}/forcefield/im1h_oac_150_im1_hoac_350.psf"
    ionic_liquid.write_psf(old_psf_file, "test1.psf")

    # ionic_liquid.simulation.step(50)
    state_update.update(2)

    ionic_liquid.write_psf(old_psf_file, "test2.psf")

    ionic_liquid.simulation.step(10)
    state_update.update(2)

    ionic_liquid.write_psf(old_psf_file, "test3.psf")

    ionic_liquid.saveState("state.rst")
    ionic_liquid.saveCheckpoint("checkpoint.rst")

    ionic_liquid2 = ionic_liquid  # copy.deepcopy(ionic_liquid)
    ionic_liquid.loadState("state.rst")
    ionic_liquid2.loadCheckpoint("checkpoint.rst")

    os.remove("test1.psf")
    os.remove("test2.psf")
    os.remove("test3.psf")
    os.remove("state.rst")
    os.remove("checkpoint.rst")


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Will fail sporadicaly.",
)
def test_write_psf_save_load_single():
    def get_time_energy(simulation, print=False):
        time = simulation.context.getState().getTime()
        e_pot = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        if print:
            print(f"time: {time}, e_pot: {e_pot}")
        return time, e_pot

    def save_il(ionic_liquid, number):
        ionic_liquid.write_psf(
            f"protex/forcefield/single_pairs/im1h_oac_im1_hoac_1_secondtry.psf",
            f"test_{number}.psf",
        )
        ionic_liquid.saveCheckpoint(f"test_{number}.rst")

    def load_sim(psf, rst):
        sim = generate_single_im1h_oac_system(psf_file=psf)
        sim.loadCheckpoint(rst)
        return sim

    def load_il(psf, rst, templates):
        sim = generate_single_im1h_oac_system(psf_file=psf)
        il = IonicLiquidSystem(sim, templates)
        il.loadCheckpoint(rst)
        return il

    def print_force_contrib(simulation):
        for i, f in enumerate(simulation.system.getForces()):
            group = f.getForceGroup()
            state = simulation.context.getState(getEnergy=True, groups={group})
            print(f.getName(), state.getPotentialEnergy())

    simulation = generate_single_im1h_oac_system()
    # get ionic liquid templates
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.165, "prob": 1}
    # allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.165, "prob": 1}

    templates = IonicLiquidTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)
    # initialize update method
    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)

    save_il(ionic_liquid, 0)

    state_update.update(2)

    save_il(ionic_liquid, 1)

    # sim2_1 = load_sim("protex/forcefield/single_pairs/im1_hoac_2.psf", "test_2.rst")
    # sim_2_oldcoord = load_sim(
    #    "protex/forcefield/single_pairs/im1_hoac_2.psf", "test_1.rst"
    # )


def test_single_harmonic_force(caplog):
    caplog.set_level(logging.DEBUG)

    sim0 = generate_single_im1h_oac_system()
    allowed_updates = {}
    # allowed updates according to simple protonation scheme
    allowed_updates[frozenset(["IM1H", "OAC"])] = {
        "r_max": 0.17,
        "prob": 1,
    }
    templates = IonicLiquidTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    ionic_liquid = IonicLiquidSystem(sim0, templates)
    update = NaiveMCUpdate(ionic_liquid, all_forces=True)
    state_update = StateUpdate(update)

    for force in ionic_liquid.system.getForces():
        if type(force).__name__ == "HarmonicBondForce" and force.getForceGroup() == 0:
            print(force.getNumBonds())
        if type(force).__name__ == "HarmonicBondForce" and force.getForceGroup() == 3:
            print(force.getNumBonds())

    for residue in ionic_liquid.residues:
        print(residue.current_name)
        params = residue.parameters[residue.current_name]["HarmonicBondForce"]
        ub_forces = []
        for force in ionic_liquid.system.getForces():
            if force.getForceGroup() == 3:  # UreyBradley
                for idx in range(force.getNumBonds()):
                    f = force.getBondParameters(idx)
                    for p in params:
                        if f[0:2] == p[0:2]:
                            ub_forces.append(f)
                            print(f[3] / 2 / md_kilocalories * angstroms**2)
        print(len(ub_forces))


# def test_write_psf_long():
#     simulation = generate_im1h_oac_system()
#     # get ionic liquid templates
#     allowed_updates = {}
#     allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.165, "prob": 1}
#     allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.165, "prob": 1}

#     templates = IonicLiquidTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
#     # wrap system in IonicLiquidSystem
#     ionic_liquid = IonicLiquidSystem(simulation, templates)
#     # initialize update method
#     update = NaiveMCUpdate(ionic_liquid)
#     # initialize state update class
#     state_update = StateUpdate(update)

#     old_psf_file = f"{protex.__path__[0]}/forcefield/im1h_oac_150_im1_hoac_350.psf"
#     ionic_liquid.write_psf(old_psf_file, "test01.psf")
#     # os.remove("test01.psf")

#     # ionic_liquid.simulation.step(50)
#     state_update.update(2)

#     ionic_liquid.write_psf(old_psf_file, "test02.psf")
#     ionic_liquid.save_updates("updates01.txt")
#     ionic_liquid.saveCheckpoint("nvt01.rst")
#     # os.remove("test02.psf")
#     del simulation
#     del ionic_liquid
#     del templates
#     del update
#     del state_update
#     simulation = generate_im1h_oac_system(psf_file="test02.psf")
#     templates = IonicLiquidTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
#     ionic_liquid = IonicLiquidSystem(simulation, templates)
#     ionic_liquid.load_updates("updates01.txt")
#     ionic_liquid.loadCheckpoint("nvt01.rst")
#     update = NaiveMCUpdate(ionic_liquid)
#     state_update = StateUpdate(update)
#     ionic_liquid.write_psf(old_psf_file, "test03.psf")
#     # os.remove("test03.psf")

#     ionic_liquid.simulation.step(50)
#     state_update.update(2)

#     ionic_liquid.write_psf(old_psf_file, "test04.psf")
#     ionic_liquid.save_updates("updates02.txt")
#     ionic_liquid.saveCheckpoint("nvt02.rst")

#     del simulation
#     del ionic_liquid
#     del templates
#     del update
#     del state_update
#     simulation = generate_im1h_oac_system(psf_file="test04.psf")
#     templates = IonicLiquidTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
#     ionic_liquid = IonicLiquidSystem(simulation, templates)
#     ionic_liquid.load_updates("updates02.txt")
#     ionic_liquid.loadCheckpoint("nvt02.rst")
#     update = NaiveMCUpdate(ionic_liquid)
#     state_update = StateUpdate(update)
#     ionic_liquid.write_psf(old_psf_file, "test05.psf")
#     # os.remove("test03.psf")

#     ionic_liquid.simulation.step(50)
#     state_update.update(2)

#     ionic_liquid.write_psf(old_psf_file, "test06.psf")
#     ionic_liquid.save_updates("updates03.txt")
#     ionic_liquid.saveCheckpoint("nvt03.rst")
