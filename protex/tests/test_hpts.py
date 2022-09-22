# Import package, test suite, and other packages as needed
# import json
import copy
import logging
import os
import pwd
import re
from collections import defaultdict
from sys import stdout

import numpy as np
import pandas as pd
import parmed

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
    from openmm.unit import angstroms, kelvin, picoseconds
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

import protex
import pytest
from scipy.spatial import distance_matrix

from ..reporter import ChargeReporter
from ..residue import Residue
from ..system import IonicLiquidSystem, IonicLiquidTemplates
from ..testsystems import (  # generate_single_hpts_system,
    HPTSH_HPTS,
    IM1H_IM1,
    OAC_HOAC,
    generate_hpts_system,
)
from ..update import NaiveMCUpdate, StateUpdate

############################
# TEST SYSTEM
############################


def test_available_platforms():
    # =======================================================================
    # Force field
    # =======================================================================
    # Loading CHARMM files
    print("Loading CHARMM files...")
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

    psf = CharmmPsfFile(f"{base}/hpts.psf")
    xtl = 48.0 * angstroms
    psf.setBox(xtl, xtl, xtl)
    # cooridnates can be provieded by CharmmCrdFile, CharmmRstFile or PDBFile classes
    crd = CharmmCrdFile(f"{base}/hpts.crd")

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
    if os.path.exists(f"{base}/traj/hpts_npt_7.rst"):
        with open(f"{base}/traj/hpts_npt_7.rst") as f:
            print(f"Opening restart file {f}")
            simulation.context.setState(XmlSerializer.deserialize(f.read()))
        simulation.context.computeVirtualSites()
    else:
        print(f"No restart file found. Using initial coordinate file.")
        simulation.context.computeVirtualSites()
        simulation.context.setVelocitiesToTemperature(300 * kelvin)


def test_setup_simulation():
    simulation = generate_hpts_system()
    print("Minimizing...")
    simulation.minimizeEnergy(maxIterations=100)
    system = simulation.system

    nr_of_particles = system.getNumParticles()
    assert nr_of_particles == 18310


def test_run_simulation():

    simulation = generate_hpts_system()
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
    # If simulation aborts with Nan error, try smaller timestep (e.g. 0.0001 ps) and then extract new crd from dcd using "protex/forcefield/crdfromdcd.inp"


def test_create_IonicLiquidTemplate():
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["IM1H", "IM1"])] = {"r_max": 0.16, "prob": 0.201}  # 1+2
    allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.15, "prob": 0.684}  # 3+4
    allowed_updates[frozenset(["HPTSH", "OAC"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HPTSH", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HPTSH", "IM1"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HOAC", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["IM1H", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}

    templates = IonicLiquidTemplates(
        [OAC_HOAC, IM1H_IM1, HPTSH_HPTS], (allowed_updates)
    )

    r = templates.get_residue_name_for_coupled_state("OAC")
    assert r == "HOAC"
    r = templates.get_residue_name_for_coupled_state("HOAC")
    assert r == "OAC"
    r = templates.get_residue_name_for_coupled_state("IM1H")
    assert r == "IM1"
    r = templates.get_residue_name_for_coupled_state("HPTS")
    assert r == "HPTSH"

    print("###################")
    assert templates.pairs == [["OAC", "HOAC"], ["IM1H", "IM1"], ["HPTSH", "HPTS"]]
    assert templates.states["IM1H"]["atom_name"] == "H7"
    assert templates.states["IM1"]["atom_name"] == "N2"
    assert templates.states["HPTS"]["atom_name"] == "O7"

    assert sorted(templates.names) == sorted(
        ["OAC", "HOAC", "IM1H", "IM1", "HPTS", "HPTSH"]
    )
    print(templates.allowed_updates)
    assert templates.overall_max_distance == 0.16

    neutral_prob = templates.get_update_value_for(frozenset(["IM1", "HOAC"]), "prob")
    assert neutral_prob == 1
    ionic_prob = templates.get_update_value_for(frozenset(["IM1H", "OAC"]), "prob")
    assert ionic_prob == 1
    hpts_prob = templates.get_update_value_for(frozenset(["HPTSH", "IM1"]), "prob")
    assert hpts_prob == 1


def test_create_IonicLiquid():

    simulation = generate_hpts_system()
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 2.33}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": -2.33}
    # allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.15, "prob": 0.098} #1+4
    allowed_updates[frozenset(["IM1H", "IM1"])] = {"r_max": 0.16, "prob": 0.201}  # 1+2
    allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.15, "prob": 0.684}  # 3+4
    allowed_updates[frozenset(["HPTSH", "OAC"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HPTSH", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HPTSH", "IM1"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HOAC", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["IM1H", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}

    templates = IonicLiquidTemplates(
        [OAC_HOAC, IM1H_IM1, HPTSH_HPTS], (allowed_updates)
    )

    count = defaultdict(int)
    ionic_liquid = IonicLiquidSystem(simulation, templates)

    assert len(ionic_liquid.residues) == 1009
    for idx, residue in enumerate(ionic_liquid.residues):
        # print(f"{idx} : {residue.original_name}")
        count[residue.original_name] += 1

    assert count["IM1H"] == 157
    assert count["OAC"] == 150
    assert count["IM1"] == 350
    assert count["HOAC"] == 350
    assert count["HPTS"] == 1
    assert count["HPTSH"] == 1


def test_residues():
    simulation = generate_hpts_system()
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
        if r.name == "HPTSH" and idx == 1008:
            atom_idxs = [atom.index for atom in r.atoms()]
            atom_names = [atom.name for atom in r.atoms()]
            print(atom_idxs)
            print(atom_names)
            print(idx)
            assert atom_idxs == [
                18225,
                18226,
                18227,
                18228,
                18229,
                18230,
                18231,
                18232,
                18233,
                18234,
                18235,
                18236,
                18237,
                18238,
                18239,
                18240,
                18241,
                18242,
                18243,
                18244,
                18245,
                18246,
                18247,
                18248,
                18249,
                18250,
                18251,
                18252,
                18253,
                18254,
                18255,
                18256,
                18257,
                18258,
                18259,
                18260,
                18261,
                18262,
                18263,
                18264,
                18265,
                18266,
                18267,
                18268,
                18269,
                18270,
                18271,
                18272,
                18273,
                18274,
                18275,
                18276,
                18277,
                18278,
                18279,
                18280,
                18281,
                18282,
                18283,
                18284,
                18285,
                18286,
                18287,
                18288,
                18289,
                18290,
                18291,
                18292,
                18293,
                18294,
                18295,
                18296,
                18297,
                18298,
                18299,
                18300,
                18301,
                18302,
                18303,
                18304,
                18305,
                18306,
                18307,
                18308,
                18309,
            ]

            assert atom_names == [
                "C1",
                "DC1",
                "C2",
                "DC2",
                "C3",
                "DC3",
                "C4",
                "DC4",
                "C5",
                "DC5",
                "C6",
                "DC6",
                "C7",
                "DC7",
                "C8",
                "DC8",
                "S1",
                "DS1",
                "O1",
                "DO1",
                "O2",
                "DO2",
                "O3",
                "DO3",
                "S2",
                "DS2",
                "O4",
                "DO4",
                "O5",
                "DO5",
                "O6",
                "DO6",
                "C9",
                "DC9",
                "C10",
                "DC10",
                "C11",
                "DC11",
                "C12",
                "DC12",
                "C13",
                "DC13",
                "C14",
                "DC14",
                "C15",
                "DC15",
                "C16",
                "DC16",
                "O7",
                "DO7",
                "S3",
                "DS3",
                "O8",
                "DO8",
                "O9",
                "DO9",
                "O10",
                "DO10",
                "H1",
                "H2",
                "H3",
                "H4",
                "H5",
                "H6",
                "H7",
                "LPO11",
                "LPO12",
                "LPO21",
                "LPO22",
                "LPO31",
                "LPO32",
                "LPO41",
                "LPO42",
                "LPO51",
                "LPO52",
                "LPO61",
                "LPO62",
                "LPO71",
                "LPO72",
                "LPO81",
                "LPO82",
                "LPO91",
                "LPO92",
                "LPO101",
                "LPO102",
            ]


def test_forces():

    simulation = generate_hpts_system()
    system = simulation.system
    topology = simulation.topology
    force_state = defaultdict(list)  # store bond force
    atom_idxs = {}  # store atom_idxs
    atom_names = {}  # store atom_names
    names = []  # store residue names

    # iterate over residues, select the first residue for HPTS and HPTSH and save the individual bonded forces
    for ridx, r in enumerate(topology.residues()):
        if r.name == "HPTS" and ridx == 1007:  # match first HPTS residue
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

        if r.name == "HPTSH" and ridx == 1008:
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

    simulation = generate_hpts_system()
    system = simulation.system
    topology = simulation.topology
    force_state = defaultdict(list)  # store bond force
    atom_idxs = {}  # store atom_idxs
    atom_names = {}  # store atom_names
    names = []  # store residue names

    # iterate over residues, select the first residue for HPTS and HPTSH and save the individual bonded forces
    for ridx, r in enumerate(topology.residues()):
        if r.name == "HPTS" and ridx == 1007:  # match first HPTS residue
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
                            print("hpts", f)

        if r.name == "HPTSH" and ridx == 1008:
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
                            print("hpts", f)

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

    # iterate over residues, select the first residue for HPTS and HPTSH and save the individual bonded forces
    for ridx, r in enumerate(topology.residues()):
        if r.name == "HPTS" and ridx == 1007:  # match first HPTS residue
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
                            print("hpts", f)

        if r.name == "HPTSH" and ridx == 1008:
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
                            print("hptsh", f)

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

    simulation = generate_hpts_system()
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["IM1H", "IM1"])] = {"r_max": 0.16, "prob": 0.201}  # 1+2
    allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.15, "prob": 0.684}  # 3+4
    allowed_updates[frozenset(["HPTSH", "OAC"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HPTSH", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HPTSH", "IM1"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HOAC", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["IM1H", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}

    templates = IonicLiquidTemplates(
        [OAC_HOAC, IM1H_IM1, HPTSH_HPTS], (allowed_updates)
    )

    ionic_liquid = IonicLiquidSystem(simulation, templates)
    system = simulation.system
    topology = simulation.topology
    force_state = defaultdict(list)  # store drude force
    force_state_thole = defaultdict(list)  # store drude force
    atom_idxs = defaultdict(list)  # store atom_idxs
    atom_names = defaultdict(list)  # store atom_names
    names = []  # store names

    # iterate over residues, select the first residue for HOAC and OAC and save the individual bonded forces
    for ridx, r in enumerate(topology.residues()):
        if r.name == "HOAC" and ridx == 657:  # match first HOAC residue
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
                parent1, parent2 = ionic_liquid.pair_12_13_list[drude_id]
                drude1, drude2 = parent1 + 1, parent2 + 1
                # print(f"thole {idx1=}, {idx2=}")
                # print(f"{drude_id=}, {f=}")
                if drude1 in atom_idxs[r.name] and drude2 in atom_idxs[r.name]:
                    # idx1, idx2 = f[0], f[1]
                    # if ( idx1 in atom_idxs[r.name] and idx2 in atom_idxs[r.name] ):
                    print(f)
                    force_state_thole[r.name].append(f)

        if r.name == "OAC" and ridx == 157:
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
                parent1, parent2 = ionic_liquid.pair_12_13_list[drude_id]
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
    simulation = generate_hpts_system()
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["IM1H", "IM1"])] = {"r_max": 0.16, "prob": 0.201}  # 1+2
    allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.15, "prob": 0.684}  # 3+4
    allowed_updates[frozenset(["HPTSH", "OAC"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HPTSH", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HPTSH", "IM1"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HOAC", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["IM1H", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}

    templates = IonicLiquidTemplates(
        [OAC_HOAC, IM1H_IM1, HPTSH_HPTS], (allowed_updates)
    )

    ionic_liquid = IonicLiquidSystem(simulation, templates)
    assert len(ionic_liquid.residues) == 1009

    residue = ionic_liquid.residues[0]
    charge = residue.endstate_charge

    assert charge == 1
    print(residue.atom_names)
    assert (residue.get_idx_for_atom_name("H7")) == 18

    residue = ionic_liquid.residues[1]

    assert (residue.get_idx_for_atom_name("H7")) == 38

    residue = ionic_liquid.residues[1008]

    assert (residue.get_idx_for_atom_name("H7")) == 18289

    # check name of first residue
    assert ionic_liquid.residues[0].current_name == "IM1H"
    assert ionic_liquid.residues[0].original_name == "IM1H"


# def test_save_load_residue_names():
#     # obtain simulation object
#     simulation = generate_hpts_system()
#     # get ionic liquid templates
#     allowed_updates = {}
#     allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 1}
#     allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 1}
#     allowed_updates[frozenset(["IM1H", "IM1"])] = {"r_max": 0.16, "prob": 0.201}  # 1+2
#     allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.15, "prob": 0.684}  # 3+4
#     allowed_updates[frozenset(["HPTSH", "OAC"])] = {"r_max": 0.15, "prob": 1.000}
#     allowed_updates[frozenset(["HPTSH", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
#     allowed_updates[frozenset(["HPTSH", "IM1"])] = {"r_max": 0.15, "prob": 1.000}
#     allowed_updates[frozenset(["HOAC", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
#     allowed_updates[frozenset(["IM1H", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}

#     templates = IonicLiquidTemplates(
#         [OAC_HOAC, IM1H_IM1, HPTSH_HPTS], (allowed_updates)
#     )
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
#     simulation = generate_hpts_system()
#     # get ionic liquid templates
#     allowed_updates = {}
#     allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 1}
#     allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 1}
#     allowed_updates[frozenset(["IM1H", "IM1"])] = {"r_max": 0.16, "prob": 0.201}  # 1+2
#     allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.15, "prob": 0.684}  # 3+4
#     allowed_updates[frozenset(["HPTSH", "OAC"])] = {"r_max": 0.15, "prob": 1.000}
#     allowed_updates[frozenset(["HPTSH", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
#     allowed_updates[frozenset(["HPTSH", "IM1"])] = {"r_max": 0.15, "prob": 1.000}
#     allowed_updates[frozenset(["HOAC", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
#     allowed_updates[frozenset(["IM1H", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}

#     templates = IonicLiquidTemplates(
#         [OAC_HOAC, IM1H_IM1, HPTSH_HPTS], (allowed_updates)
#     )
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
    simulation = generate_hpts_system()
    # get ionic liquid templates
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["IM1H", "IM1"])] = {"r_max": 0.16, "prob": 0.201}  # 1+2
    allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.15, "prob": 0.684}  # 3+4
    allowed_updates[frozenset(["HPTSH", "OAC"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HPTSH", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HPTSH", "IM1"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HOAC", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["IM1H", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}

    templates = IonicLiquidTemplates(
        [OAC_HOAC, IM1H_IM1, HPTSH_HPTS], (allowed_updates)
    )
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
    psf_for_parameters = f"{protex.__path__[0]}/forcefield/hpts.psf"
    psf_file = f"{protex.__path__[0]}/forcefield/hpts.psf"

    simulation = generate_hpts_system(psf_file=psf_file)
    simulation_for_parameters = generate_hpts_system(psf_file=psf_for_parameters)

    # get ionic liquid templates
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 0.994}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 0.098}
    allowed_updates[frozenset(["IM1H", "IM1"])] = {"r_max": 0.16, "prob": 0.201}  # 1+2
    allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.15, "prob": 0.684}  # 3+4
    allowed_updates[frozenset(["HPTSH", "OAC"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HPTSH", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HPTSH", "IM1"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HOAC", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["IM1H", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}

    templates = IonicLiquidTemplates(
        [OAC_HOAC, IM1H_IM1, HPTSH_HPTS], (allowed_updates)
    )
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates, simulation_for_parameters)
    # initialize update method
    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)

    old_psf_file = f"{protex.__path__[0]}/forcefield/hpts.psf"
    ionic_liquid.write_psf(old_psf_file, "test.psf")

    # ionic_liquid.simulation.step(50)
    state_update.update(2)

    ionic_liquid.write_psf(old_psf_file, "test.psf")
    ionic_liquid.saveState("state.rst")
    ionic_liquid.saveCheckpoint("checkpoint.rst")

    ionic_liquid2 = ionic_liquid  # copy.deepcopy(ionic_liquid)
    ionic_liquid.loadState("state.rst")
    ionic_liquid2.loadCheckpoint("checkpoint.rst")


#####################
# TEST UODATE
#######################


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Will fail sporadicaly.",
)
def test_updates(caplog):
    caplog.set_level(logging.DEBUG)

    simulation = generate_hpts_system()
    allowed_updates = {}
    # allowed updates according to simple protonation scheme
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.15, "prob": 1}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.15, "prob": 1}
    allowed_updates[frozenset(["IM1H", "IM1"])] = {"r_max": 0.15, "prob": 1}  # 1+2
    allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.15, "prob": 1}  # 3+4
    allowed_updates[frozenset(["HPTSH", "OAC"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HPTSH", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HPTSH", "IM1"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HOAC", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["IM1H", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}

    print(allowed_updates.keys())
    # get ionic liquid templates
    templates = IonicLiquidTemplates(
        [OAC_HOAC, IM1H_IM1, HPTSH_HPTS], (allowed_updates)
    )
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)
    pars = []
    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)
    # ionic_liquid.simulation.minimizeEnergy(maxIterations=200)
    ionic_liquid.simulation.step(50)

    for _ in range(5):
        ionic_liquid.simulation.step(100)
        pars.append(state_update.get_charges())
        candidate_pairs = state_update.update(2)

        print(candidate_pairs)

    # test whether the update changed the psf
    old_psf_file = f"{protex.__path__[0]}/forcefield/hpts.psf"
    ionic_liquid.write_psf(old_psf_file, "hpts_new.psf")

    os.remove("hpts_new.psf")


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Will fail sporadicaly.",
)
def test_pbc():

    simulation = generate_hpts_system()
    # get ionic liquid templates
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["IM1H", "IM1"])] = {"r_max": 0.16, "prob": 1}  # 1+2
    allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.15, "prob": 1}  # 3+4
    allowed_updates[frozenset(["HPTSH", "OAC"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HPTSH", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HPTSH", "IM1"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HOAC", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["IM1H", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}

    templates = IonicLiquidTemplates(
        [OAC_HOAC, IM1H_IM1, HPTSH_HPTS], (allowed_updates)
    )
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)

    boxl = ionic_liquid.boxlength
    print(f"{boxl=}")

    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)

    pos_list, res_list = state_update._get_positions_for_mutation_sites()

    # calculate distance matrix between the two molecules
    distance = distance_matrix(pos_list, pos_list)
    # print(f"{distance[0]=}")

    from scipy.spatial.distance import cdist

    def _rPBC(coor1, coor2, boxl=boxl):
        dx = abs(coor1[0] - coor2[0])
        if dx > boxl / 2:
            dx = boxl - dx
        dy = abs(coor1[1] - coor2[1])
        if dy > boxl / 2:
            dy = boxl - dy
        dz = abs(coor1[2] - coor2[2])
        if dz > boxl / 2:
            dz = boxl - dz
        return np.sqrt(dx * dx + dy * dy + dz * dz)

    distance_pbc = cdist(pos_list, pos_list, _rPBC)
    # print(f"{distance_pbc[0]=}")
    # print(f"{distance_pbc[distance_pbc>boxl]=}")
    assert (
        len(distance_pbc[distance_pbc > boxl]) == 0
    ), "After correcting for PBC no distance should be larger than the boxlength"


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Will fail sporadicaly.",
)
def test_residue_forces():
    simulation = generate_hpts_system()
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.15, "prob": 1}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.15, "prob": 1}
    allowed_updates[frozenset(["IM1H", "IM1"])] = {"r_max": 0.15, "prob": 1}  # 1+2
    allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.15, "prob": 1}  # 3+4
    allowed_updates[frozenset(["HPTSH", "OAC"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HPTSH", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HPTSH", "IM1"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HOAC", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["IM1H", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
    templates = IonicLiquidTemplates(
        [OAC_HOAC, IM1H_IM1, HPTSH_HPTS], (allowed_updates)
    )
    ionic_liquid = IonicLiquidSystem(simulation, templates)

    oac = ionic_liquid.residues[157]
    hoac = ionic_liquid.residues[657]
    im1h = ionic_liquid.residues[0]
    im1 = ionic_liquid.residues[307]
    hpts = ionic_liquid.residues[1007]
    hptsh = ionic_liquid.residues[1008]

    offset_oac = oac._get_offset("OAC")
    offset_hoac = hoac._get_offset("HOAC")
    offset_im1h = im1h._get_offset("IM1H")
    offset_im1 = im1._get_offset("IM1")
    offset_hpts = hpts._get_offset("HPTS")
    offset_hptsh = hptsh._get_offset("HPTSH")

    ### test if number of forces are equal
    # for force in ("NonbondedForce", "HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce", "CustomTorsionForce", "DrudeForce"):
    #     print(f"{force=}")
    #     par1 = hpts.parameters["HPTS"][force]
    #     par2 = hptsh.parameters["HPTSH"][force]
    #     assert len(par1) == len(par2)

    ### test indices

    # print("force=HarmonicBondForce")
    # for old_idx, old_parm in enumerate(oac.parameters["OAC"]["HarmonicBondForce"]):
    #     idx1, idx2 = old_parm[0], old_parm[1]
    #     for new_idx, new_parm in enumerate(hoac.parameters["HOAC"]["HarmonicBondForce"]):
    #         if set(
    #             [new_parm[0] - offset2, new_parm[1] - offset2]
    #         ) == set([idx1 - offset1, idx2 - offset1]):
    #             if old_idx != new_idx:
    #                 print(old_idx, new_idx)
    #                 raise RuntimeError(
    #                         "Odering is different between the two topologies."
    #                 )

    # print("force=DrudeForce")
    # for old_idx, old_parm in enumerate(oac.parameters["OAC"]["DrudeForce"]):
    #     idx1, idx2 = old_parm[0], old_parm[1]
    #     for new_idx, new_parm in enumerate(hoac.parameters["HOAC"]["DrudeForce"]):
    #         if set(
    #             [new_parm[0] - offset2, new_parm[1] - offset2]
    #         ) == set([idx1 - offset1, idx2 - offset1]):
    #             if old_idx != new_idx:
    #                 raise RuntimeError(
    #                         "Odering is different between the two topologies."
    #                 )

    # print("force=HarmonicAngleForce")
    # for old_idx, old_parm in enumerate(hpts.parameters["HPTS"]["HarmonicAngleForce"]):
    #         idx1, idx2, idx3 = old_parm[0], old_parm[1], old_parm[2]
    #         for new_idx, new_parm in enumerate(hptsh.parameters["HPTSH"]["HarmonicAngleForce"]):
    #             if set(
    #                 [new_parm[0] - offset2, new_parm[1] - offset2, new_parm[2] - offset2]
    #             ) == set([idx1 - offset1, idx2 - offset1, idx3 - offset1]):
    #                 if old_idx != new_idx:
    #                     raise RuntimeError(
    #                          "Odering is different between the two topologies."
    #                     )

    # print("force=PeriodicTorsionForce")
    # for old_idx, old_parm in enumerate(hpts.parameters["HPTS"]["PeriodicTorsionForce"]):
    #     idx1, idx2, idx3, idx4, idx5 = old_parm[0], old_parm[1], old_parm[2], old_parm[3], old_parm[4]
    #     for new_idx, new_parm in enumerate(hptsh.parameters["HPTSH"]["PeriodicTorsionForce"]):
    #         if set(
    #             [new_parm[0] - offset2, new_parm[1] - offset2, new_parm[2] - offset2, new_parm[3] - offset2, new_parm[4]]
    #         ) == set([idx1 - offset1, idx2 - offset1, idx3 - offset1, idx4 - offset1, idx5]):
    #             if old_idx != new_idx:
    #                 raise RuntimeError(
    #                         "Odering is different between the two topologies."
    #                 )

    # print("force=CustomTorsionForce")
    # for old_idx, old_parm in enumerate(hpts.parameters["HPTS"]["CustomTorsionForce"]):
    #     idx1, idx2, idx3, idx4 = old_parm[0], old_parm[1], old_parm[2], old_parm[3]
    #     for new_idx, new_parm in enumerate(hptsh.parameters["HPTSH"]["CustomTorsionForce"]):
    #         if set(
    #             [new_parm[0] - offset2, new_parm[1] - offset2, new_parm[2] - offset2, new_parm[3] - offset2]
    #         ) == set([idx1 - offset1, idx2 - offset1, idx3 - offset1, idx4 - offset1]):
    #             if old_idx != new_idx:
    #                 raise RuntimeError(
    #                         "Odering is different between the two topologies."
    #                 )


def test_list_torsionforce():
    simulation = generate_hpts_system()
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.15, "prob": 1}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.15, "prob": 1}
    allowed_updates[frozenset(["IM1H", "IM1"])] = {"r_max": 0.15, "prob": 1}  # 1+2
    allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.15, "prob": 1}  # 3+4
    allowed_updates[frozenset(["HPTSH", "OAC"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HPTSH", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HPTSH", "IM1"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HOAC", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["IM1H", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
    templates = IonicLiquidTemplates(
        [OAC_HOAC, IM1H_IM1, HPTSH_HPTS], (allowed_updates)
    )
    ionic_liquid = IonicLiquidSystem(simulation, templates)

    oac = ionic_liquid.residues[157]
    hoac = ionic_liquid.residues[657]

    offset1 = oac._get_offset("OAC")
    offset2 = hoac._get_offset("HOAC")

    for old_idx, old_parm in enumerate(oac.parameters["OAC"]["PeriodicTorsionForce"]):
        idx1, idx2, idx3, idx4, idx5, idx6, idx7 = (
            old_parm[0] - offset1,
            old_parm[1] - offset1,
            old_parm[2] - offset1,
            old_parm[3] - offset1,
            old_parm[4],
            old_parm[5],
            old_parm[6],
        )
        f = open("oac_torsion.txt", "a")
        f.write(
            str(old_idx)
            + "\t"
            + str(idx1)
            + "\t"
            + str(idx2)
            + "\t"
            + str(idx3)
            + "\t"
            + str(idx4)
            + "\t"
            + str(idx5)
            + "\t"
            + str(idx6)
            + "\t"
            + str(idx7)
            + "\n"
        )
        f.close()

    for new_idx, new_parm in enumerate(hoac.parameters["HOAC"]["PeriodicTorsionForce"]):
        id1, id2, id3, id4, id5, id6, id7 = (
            new_parm[0] - offset2,
            new_parm[1] - offset2,
            new_parm[2] - offset2,
            new_parm[3] - offset2,
            new_parm[4],
            new_parm[5],
            new_parm[6],
        )
        g = open("hoac_torsion.txt", "a")
        g.write(
            str(new_idx)
            + "\t"
            + str(id1)
            + "\t"
            + str(id2)
            + "\t"
            + str(id3)
            + "\t"
            + str(id4)
            + "\t"
            + str(id5)
            + "\t"
            + str(id6)
            + "\t"
            + str(id7)
            + "\n"
        )
        g.close()

    os.remove("oac_torsion.txt")
    os.remove("hoac_torsion.txt")


def test_count_forces():
    simulation = generate_hpts_system()
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.15, "prob": 1}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.15, "prob": 1}
    allowed_updates[frozenset(["IM1H", "IM1"])] = {"r_max": 0.15, "prob": 1}  # 1+2
    allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.15, "prob": 1}  # 3+4
    allowed_updates[frozenset(["HPTSH", "OAC"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HPTSH", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HPTSH", "IM1"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HOAC", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["IM1H", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
    templates = IonicLiquidTemplates(
        [OAC_HOAC, IM1H_IM1, HPTSH_HPTS], (allowed_updates)
    )
    ionic_liquid = IonicLiquidSystem(simulation, templates)

    oac = ionic_liquid.residues[157]
    hoac = ionic_liquid.residues[657]
    im1h = ionic_liquid.residues[0]
    im1 = ionic_liquid.residues[307]
    hpts = ionic_liquid.residues[1007]
    hptsh = ionic_liquid.residues[1008]

    ### change res and pair (alternative of res) to test different molecules
    res = oac
    pair = hoac

    offset = res._get_offset(res.current_name)
    offset_pair = pair._get_offset(pair.current_name)

    ### make list of atoms in dihedrals, check if there are duplicates
    torsions = []

    for old_idx, old_parm in enumerate(
        res.parameters[res.current_name]["PeriodicTorsionForce"]
    ):
        idx1, idx2, idx3, idx4 = (
            old_parm[0] - offset,
            old_parm[1] - offset,
            old_parm[2] - offset,
            old_parm[3] - offset,
        )
        ids = [idx1, idx2, idx3, idx4]
        torsions.append(ids)

    # print(torsions)

    for old_idx, old_parm in enumerate(
        res.parameters[res.current_name]["PeriodicTorsionForce"]
    ):
        idx1, idx2, idx3, idx4, idx5 = (
            old_parm[0] - offset,
            old_parm[1] - offset,
            old_parm[2] - offset,
            old_parm[3] - offset,
            old_parm[4],
        )
        ids = [idx1, idx2, idx3, idx4]
        print(ids)
        print(torsions.count(ids))
        if torsions.count(ids) == 1:
            print("count id 1, in branch without multiplicity")
            for new_idx, new_parm in enumerate(
                pair.parameters[pair.current_name]["PeriodicTorsionForce"]
            ):
                if set(
                    [
                        new_parm[0] - offset_pair,
                        new_parm[1] - offset_pair,
                        new_parm[2] - offset_pair,
                        new_parm[3] - offset_pair,
                    ]
                ) == set([idx1, idx2, idx3, idx4]):
                    if old_idx != new_idx:
                        print(old_idx, new_idx)
                        for force in simulation.system.getForces():
                            if type(force).__name__ == "PeriodicTorsionForce":
                                for torsion_id in range(force.getNumTorsions()):
                                    f = force.getTorsionParameters(torsion_id)
                                    if (
                                        f[0] in old_parm
                                        and f[1] in old_parm
                                        and f[2] in old_parm
                                        and f[3] in old_parm
                                    ):
                                        print("old force", f)
                                    if (
                                        f[0] in new_parm
                                        and f[1] in new_parm
                                        and f[2] in new_parm
                                        and f[3] in new_parm
                                    ):
                                        print("new force", f)
                        raise RuntimeError(
                            "Odering is different between the two topologies."
                        )
                    break

            else:
                for force in simulation.system.getForces():
                    if type(force).__name__ == "PeriodicTorsionForce":
                        for torsion_id in range(force.getNumTorsions()):
                            f = force.getTorsionParameters(torsion_id)
                            if (
                                f[0] in old_parm
                                and f[1] in old_parm
                                and f[2] in old_parm
                                and f[3] in old_parm
                            ):
                                print(old_idx, "old force", f)
                            if (
                                f[0] in new_parm
                                and f[1] in new_parm
                                and f[2] in new_parm
                                and f[3] in new_parm
                            ):
                                print(new_idx, "new force", f)
                raise RuntimeError()
        else:
            print("count id != 1, in branch with multiplicities")
            for new_idx, new_parm in enumerate(
                pair.parameters[pair.current_name]["PeriodicTorsionForce"]
            ):
                if set(
                    [
                        new_parm[0] - offset_pair,
                        new_parm[1] - offset_pair,
                        new_parm[2] - offset_pair,
                        new_parm[3] - offset_pair,
                        new_parm[4],
                    ]
                ) == set([idx1, idx2, idx3, idx4, idx5]):
                    if old_idx != new_idx:
                        print(old_idx, new_idx)
                        for force in simulation.system.getForces():
                            if type(force).__name__ == "PeriodicTorsionForce":
                                for torsion_id in range(force.getNumTorsions()):
                                    f = force.getTorsionParameters(torsion_id)
                                    if (
                                        f[0] in old_parm
                                        and f[1] in old_parm
                                        and f[2] in old_parm
                                        and f[3] in old_parm
                                    ):
                                        print("old force", f)
                                    if (
                                        f[0] in new_parm
                                        and f[1] in new_parm
                                        and f[2] in new_parm
                                        and f[3] in new_parm
                                    ):
                                        print("new force", f)
                        raise RuntimeError(
                            "Odering is different between the two topologies."
                        )
                    break

            else:
                for force in simulation.system.getForces():
                    if type(force).__name__ == "PeriodicTorsionForce":
                        for torsion_id in range(force.getNumTorsions()):
                            f = force.getTorsionParameters(torsion_id)
                            if (
                                f[0] in old_parm
                                and f[1] in old_parm
                                and f[2] in old_parm
                                and f[3] in old_parm
                            ):
                                print(old_idx, "old force", f)
                            if (
                                f[0] in new_parm
                                and f[1] in new_parm
                                and f[2] in new_parm
                                and f[3] in new_parm
                            ):
                                print(new_idx, "new force", f)
                raise RuntimeError()


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Will fail sporadicaly.",
)
def test_update_write_psf():
    psf_for_parameters = f"{protex.__path__[0]}/forcefield/hpts.psf"
    psf_file = f"{protex.__path__[0]}/forcefield/hpts.psf"

    simulation = generate_hpts_system(psf_file=psf_file)
    simulation_for_parameters = generate_hpts_system(psf_file=psf_for_parameters)

    # get ionic liquid templates
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 0.994}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 0.098}
    allowed_updates[frozenset(["IM1H", "IM1"])] = {"r_max": 0.16, "prob": 0.201}  # 1+2
    allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.15, "prob": 0.684}  # 3+4
    allowed_updates[frozenset(["HPTSH", "OAC"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HPTSH", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HPTSH", "IM1"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["HOAC", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
    allowed_updates[frozenset(["IM1H", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}

    templates = IonicLiquidTemplates(
        [OAC_HOAC, IM1H_IM1, HPTSH_HPTS], (allowed_updates)
    )
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates, simulation_for_parameters)
    # initialize update method
    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)

    old_psf_file = f"{protex.__path__[0]}/forcefield/hpts.psf"
    ionic_liquid.write_psf(old_psf_file, "test.psf")

    i = 0
    while i < 20:
        sum_charge = 0
        for x in range(0, 1009):
            resi = ionic_liquid.residues[x]
            sum_charge = sum_charge + resi.current_charge
        print(sum_charge)
        if sum_charge != 0:
            raise RuntimeError("Error in run", i)

        os.rename("test.psf", "old_psf.psf")

        simulation = generate_hpts_system(psf_file="old_psf.psf")
        ionic_liquid = IonicLiquidSystem(
            simulation, templates, simulation_for_parameters
        )
        update = NaiveMCUpdate(ionic_liquid)
        state_update = StateUpdate(update)

        ionic_liquid.simulation.step(50)
        state_update.update(2)

        ionic_liquid.write_psf("old_psf.psf", "test.psf")
        ionic_liquid.saveState("state.rst")
        ionic_liquid.saveCheckpoint("checkpoint.rst")

        ionic_liquid2 = ionic_liquid  # copy.deepcopy(ionic_liquid)
        ionic_liquid.loadState("state.rst")
        ionic_liquid2.loadCheckpoint("checkpoint.rst")

        i += 1

    os.remove("old_psf.psf")
    os.remove("state.rst")
    os.remove("checkpoints.rst")
    os.remove("test.psf")
