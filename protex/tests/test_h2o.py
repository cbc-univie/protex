# Import package, test suite, and other packages as needed
# import json
import copy
import logging
import os
from collections import defaultdict
from sys import stdout

try:  # Syntax changed in OpenMM 7.6
    import openmm as mm
    from openmm import Context, DrudeNoseHooverIntegrator, OpenMMException, Platform
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
        Context,
        DrudeNoseHooverIntegrator,
        OpenMMException,
        Platform,
    )
    from simtk.openmm.app import (
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
    from simtk.unit import angstroms, kelvin, picoseconds

import pytest
from velocityverletplugin import VVIntegrator

import protex

from ..system import ProtexSystem, ProtexTemplates
from ..testsystems import (
    H2O_H3O,  #CLA, SOD,
    HSP_HSD,
    IM1H_IM1,
    OAC_HOAC,
    OH_H2O_H3O,
    generate_ac_toh2_system,
    generate_m2_toh2_system,
    generate_small_box,
    generate_toh2_system,
)
from ..update import KeepHUpdate, NaiveMCUpdate, StateUpdate

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
        "toppar_drude_master_protein_2013f_lj025_modhpts_chelpg.str",
        "h2o_d.str",
        "oh_d.str",
        "na_d.str",
        "cl_d.str",
    ]
    base = f"{protex.__path__[0]}/forcefield/toh2"  # NOTE: this points now to the installed files!
    params = CharmmParameterSet(
        *[f"{base}/toppar/{para_files}" for para_files in PARA_FILES]
    )

    psf = CharmmPsfFile(f"{base}/h2o.psf")
    xtl = 31.0 * angstroms
    psf.setBox(xtl, xtl, xtl)
    # cooridnates can be provieded by CharmmCrdFile, CharmmRstFile or PDBFile classes
    CharmmCrdFile(f"{base}/h2o.crd")

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

    print(f"{integrator=}")

    print(
        f"{coll_freq=}, {drude_coll_freq=}"
    )  # tested with 20, 40, 80, 100, 120, 140, 160: 20,40 bad; 80 - 120 good; 140, 160 crashed
    integrator.setMaxDrudeDistance(0.25 * angstroms)
    try:
        platform = Platform.getPlatformByName("CUDA")
        prop = dict(CudaPrecision="single")  # default is single

        Simulation(
            psf.topology, system, integrator, platform=platform, platformProperties=prop
        )
    except mm.OpenMMException:
        platform = Platform.getPlatformByName("CPU")
        prop = dict()
        Simulation(
            psf.topology, system, integrator, platform=platform, platformProperties=prop
        )
    print(platform.getName())


def test_run_simulation(tmp_path):

    simulation = generate_toh2_system(use_plugin=False)
    system = simulation.system
    nr_of_particles = system.getNumParticles()
    print(f"{nr_of_particles=}")

    positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
    print(f"{positions=}")

    # print("Minimizing...")
    # simulation.minimizeEnergy(maxIterations=50)
    state=simulation.context.getState(getPositions=True, getEnergy=True)
    print(state.getPotentialEnergy())

    simulation.reporters.append(PDBReporter(f"{tmp_path}/output.pdb", 1))
    simulation.reporters.append(DCDReporter(f"{tmp_path}/output.dcd", 1))

    simulation.reporters.append(
        StateDataReporter(
            f"{tmp_path}/output.out",
            1,
            step=True,
            potentialEnergy=True,
            temperature=True,
            time=True,
            volume=True,
            density=True,
        )
    )
    print("Running dynmamics...")
    simulation.step(10)              # coordinates NaN with 200, 50
    positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
    print(f"{positions}")


def test_create_ProtexTemplate():
    allowed_updates = {}
    allowed_updates[frozenset(["OH", "H2O"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["OH", "H3O"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["H3O", "H2O"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["H2O", "H2O"])] = {"r_max": 0.16, "prob": 1}

    templates = ProtexTemplates(
        [OH_H2O_H3O], (allowed_updates)
    )

    r = templates.get_ordered_names_for("H2O")
    assert r == ("OH", "H2O", "H3O")
    r = templates.get_ordered_names_for("OH")
    assert r == ("OH", "H2O", "H3O")
    r = templates.get_ordered_names_for("H3O")
    assert r == ("OH", "H2O", "H3O")

    print("###################")
    assert templates.pairs == [
        ["OH", "H2O", "H3O"],
    ]
    assert templates.states["OH"]["possible_modes"] == ("acceptor")
    assert templates.states["H2O"]["possible_modes"] == ("acceptor", "donor")
    assert templates.states["H3O"]["possible_modes"] == ("donor")

    assert templates.states["OH"]["starting_donors"] == ["H1"]
    assert templates.states["H2O"]["starting_donors"] == ["H1", "H2"]
    assert templates.states["H3O"]["starting_donors"] == ["H1", "H2","H3"]

    assert templates.states["OH"]["starting_acceptors"] == ["H2", "H3", "H4"]
    assert templates.states["H2O"]["starting_acceptors"] == ["H3", "H4"]
    assert templates.states["H3O"]["starting_acceptors"] == ["H4"]

    assert sorted(templates.names) == sorted(
        ["H2O", "OH", "H3O"]
    )
    print(templates.allowed_updates)
    assert templates.overall_max_distance == 0.16

    neutral_prob = templates.get_update_value_for(frozenset(["H2O", "H2O"]), "prob")
    assert neutral_prob == 1
    ionic_prob = templates.get_update_value_for(frozenset(["OH", "H3O"]), "prob")
    assert ionic_prob == 1
    ionic_prob = templates.get_update_value_for(frozenset(["H3O", "OH"]), "prob")
    assert ionic_prob == 1


def test_create_IonicLiquid():
    psf_for_parameters = f"{protex.__path__[0]}/forcefield/toh2/h2o.psf"
    crd_for_parameters = f"{protex.__path__[0]}/forcefield/toh2/h2o.crd"

    simulation = generate_toh2_system(
        use_plugin=False
    )  # psf_file=psf_file)

    simulation_for_parameters = generate_toh2_system(
        crd_file=crd_for_parameters, psf_file=psf_for_parameters, use_plugin=False
    )

    allowed_updates = {}
    allowed_updates[frozenset(["OH", "H2O"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["OH", "H3O"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["H3O", "H2O"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["H2O", "H2O"])] = {"r_max": 0.16, "prob": 1}


    templates = ProtexTemplates(
        [OH_H2O_H3O], (allowed_updates)
    )

    count = defaultdict(int)
    ionic_liquid = ProtexSystem(simulation, templates, simulation_for_parameters)

    raise AssertionError("want to stop here")

    assert len(ionic_liquid.residues) == 942
    for idx, residue in enumerate(ionic_liquid.residues):
        # print(f"{idx} : {residue.original_name}")
        count[residue.original_name] += 1

    assert count["H2O"] == 902
    assert count["H3O"] == 10
    assert count["OH"] == 10
    assert count["CLA"] == 10
    assert count["SOD"] == 10

    for resi in ionic_liquid.residues:
        if resi.current_name == "H2O":
            assert len(resi.acceptors) == len(resi.donors) == 2
        elif resi.current_name == "H3O":
            assert len(resi.acceptors) == 1
            assert len(resi.donors) == 3
        elif resi.current_name == "OH":
            assert len(resi.acceptors) == 3
            assert len(resi.donors) == 1


def test_forces():
    # simulation = generate_hpts_meoh_system(psf_file=psf_file)
    # generate_hpts_meoh_system(crd_file=crd_for_parameters, psf_file=psf_for_parameters)
    simulation = generate_toh2_system(use_plugin=False)
    system = simulation.system
    topology = simulation.topology
    force_state = defaultdict(list)  # store bond force
    atom_idxs = {}  # store atom_idxs
    atom_names = {}  # store atom_names
    names = []  # store residue names

    # iterate over residues, select the first residue for a pair of species and save the individual bonded forces
    for ridx, r in enumerate(topology.residues()):
        if r.name == "OH":
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

        if r.name == "H3O":
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
    # simulation = generate_hpts_meoh_system(psf_file=psf_file)
    # generate_hpts_meoh_system(crd_file=crd_for_parameters, psf_file=psf_for_parameters)
    simulation = generate_toh2_system(use_plugin=False)
    system = simulation.system
    topology = simulation.topology
    force_state = defaultdict(list)  # store bond force
    atom_idxs = {}  # store atom_idxs
    atom_names = {}  # store atom_names
    names = []  # store residue names

    for ridx, r in enumerate(topology.residues()):
        if r.name == "H2O":
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
                            print("H2O", f)

        if r.name == "OH":
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
                            print("OH", f)

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

    for ridx, r in enumerate(topology.residues()):
        if r.name == "H2O":
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
                            print("H2O", f)

        if r.name == "OH":
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
                            print("OH", f)

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


def test_customnonbonded_forces():
    simulation = generate_toh2_system(use_plugin=False)
    system = simulation.system
    topology = simulation.topology
    force_state = defaultdict(list)  # store bond force
    atom_idxs = {}  # store atom_idxs
    atom_names = {}  # store atom_names
    names = []  # store residue names

    for ridx, r in enumerate(topology.residues()):
        if r.name == "H2O" and "H2O" not in names:
            names.append(r.name)
            atom_idxs[r.name] = [atom.index for atom in r.atoms()]
            atom_names[r.name] = [atom.name for atom in r.atoms()]
            forces_dict_H2O_exceptions = []
            for force in system.getForces():
                if type(force).__name__ == "CustomNonbondedForce":
                    forces_dict_H2O = [force.getParticleParameters(idx) for idx in atom_idxs["H2O"]]
                    # Also add exclusions
                    for exc_id in range(force.getNumExclusions()):
                        f = force.getExclusionParticles(exc_id)
                        idx1 = f[0]
                        idx2 = f[1]
                        if idx1 in atom_idxs["H2O"] and idx2 in atom_idxs["H2O"]:
                            forces_dict_H2O_exceptions.append(f)


        if r.name == "OH" and "OH" not in names:
            names.append(r.name)
            atom_idxs[r.name] = [atom.index for atom in r.atoms()]
            atom_names[r.name] = [atom.name for atom in r.atoms()]
            forces_dict_OH_exceptions = []
            for force in system.getForces():
                if type(force).__name__ == "CustomNonbondedForce":
                    forces_dict_OH = [force.getParticleParameters(idx) for idx in atom_idxs["OH"]]
                    # Also add exclusions
                    for exc_id in range(force.getNumExclusions()):
                        f = force.getExclusionParticles(exc_id)
                        idx1 = f[0]
                        idx2 = f[1]
                        if idx1 in atom_idxs["OH"] and idx2 in atom_idxs["OH"]:
                            forces_dict_OH_exceptions.append(f)


        if r.name == "H3O" and "H3O" not in names:
            names.append(r.name)
            atom_idxs[r.name] = [atom.index for atom in r.atoms()]
            atom_names[r.name] = [atom.name for atom in r.atoms()]
            forces_dict_H3O_exceptions = []
            for force in system.getForces():
                if type(force).__name__ == "CustomNonbondedForce":
                    forces_dict_H3O = [force.getParticleParameters(idx) for idx in atom_idxs["H3O"]]
                    # Also add exclusions
                    for exc_id in range(force.getNumExclusions()):
                        f = force.getExclusionParticles(exc_id)
                        idx1 = f[0]
                        idx2 = f[1]
                        if idx1 in atom_idxs["H3O"] and idx2 in atom_idxs["H3O"]:
                            forces_dict_H3O_exceptions.append(f)

        if r.name == "CLA" and "CLA" not in names:
            names.append(r.name)
            atom_idxs[r.name] = [atom.index for atom in r.atoms()]
            atom_names[r.name] = [atom.name for atom in r.atoms()]
            forces_dict_CLA_exceptions = []
            for force in system.getForces():
                if type(force).__name__ == "CustomNonbondedForce":
                    forces_dict_CLA = [force.getParticleParameters(idx) for idx in atom_idxs["CLA"]]
                    # Also add exclusions
                    for exc_id in range(force.getNumExclusions()):
                        f = force.getExclusionParticles(exc_id)
                        idx1 = f[0]
                        idx2 = f[1]
                        if idx1 in atom_idxs["CLA"] and idx2 in atom_idxs["CLA"]:
                            forces_dict_CLA_exceptions.append(f)

        if r.name == "SOD" and "SOD" not in names:
            names.append(r.name)
            atom_idxs[r.name] = [atom.index for atom in r.atoms()]
            atom_names[r.name] = [atom.name for atom in r.atoms()]
            forces_dict_SOD_exceptions = []
            for force in system.getForces():
                if type(force).__name__ == "CustomNonbondedForce":
                    forces_dict_SOD = [force.getParticleParameters(idx) for idx in atom_idxs["SOD"]]
                    # Also add exclusions
                    for exc_id in range(force.getNumExclusions()):
                        f = force.getExclusionParticles(exc_id)
                        idx1 = f[0]
                        idx2 = f[1]
                        if idx1 in atom_idxs["SOD"] and idx2 in atom_idxs["SOD"]:
                            forces_dict_SOD_exceptions.append(f)

    print(forces_dict_H2O_exceptions)
    print(forces_dict_OH_exceptions)
    print(forces_dict_H3O_exceptions)
    print(forces_dict_CLA_exceptions)
    print(forces_dict_SOD_exceptions)

    if len(forces_dict_OH) != len(
        forces_dict_H2O
    ):  # check the number of entries in the forces
        print(f"H2O: {len(forces_dict_H2O)}")
        print(f"OH: {len(forces_dict_OH)}")

        print(f"{names[0]}:Atom indicies and atom names")
        for idx, name in zip(atom_idxs[names[0]], atom_names[names[0]]):
            print(f"{idx}:{name}")
        print(f"{names[1]}:Atom indicies and atom names")
        for idx, name in zip(atom_idxs[names[1]], atom_names[names[1]]):
            print(f"{idx}:{name}")

        # print forces for the two residues
        print("########################")
        print(names[0])
        for f in forces_dict_H2O:
            print(f)

        print("########################")
        print(names[1])
        for f in forces_dict_OH:
            print(f)

    #raise AssertionError("ohoh")


def test_drude_forces():
    # f"{protex.__path__[0]}/forcefield/psf_for_parameters.psf"
    # f"{protex.__path__[0]}/forcefield/crd_for_parameters.crd"
    # f"{protex.__path__[0]}/forcefield/hpts.psf"

    # simulation = generate_hpts_meoh_system(psf_file=psf_file)
    # simulation_for_parameters = generate_hpts_meoh_system(
    #    crd_file=crd_for_parameters, psf_file=psf_for_parameters
    # )
    simulation = generate_toh2_system(use_plugin=False)
    simulation_for_parameters = generate_toh2_system(use_plugin=False)
    allowed_updates = {}
    allowed_updates[frozenset(["OH", "H2O"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["OH", "H3O"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["H3O", "H2O"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["H2O", "H2O"])] = {"r_max": 0.16, "prob": 1}


    templates = ProtexTemplates(
        [OH_H2O_H3O], (allowed_updates)
    )

    ionic_liquid = ProtexSystem(simulation, templates, simulation_for_parameters)
    system = simulation.system
    topology = simulation.topology
    force_state = defaultdict(list)  # store drude force
    force_state_thole = defaultdict(list)  # store drude force
    atom_idxs = defaultdict(list)  # store atom_idxs
    atom_names = defaultdict(list)  # store atom_names
    names = []  # store names

    #pair_12_13_list = ionic_liquid._build_exclusion_list(ionic_liquid.topology)

    # iterate over residues, select the first residue for H2O and OH and save the individual bonded forces
    for ridx, r in enumerate(topology.residues()):
        if r.name == "H2O":
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
            particle_map = {}
            for drude_id in range(drude_force.getNumParticles()):
                f = drude_force.getParticleParameters(drude_id)
                idx1, idx2 = f[0], f[1]
                particle_map[drude_id] = idx1
                if idx1 in atom_idxs[r.name] and idx2 in atom_idxs[r.name]:
                    print(f)
                    force_state[r.name].append(f)

            print("thole")
            print(drude_force.getNumScreenedPairs())
            for drude_id in range(drude_force.getNumScreenedPairs()):
                f = drude_force.getScreenedPairParameters(drude_id)
                idx1, idx2 = f[0],f[1]
                drude1 = particle_map[idx1]
                drude2 = particle_map[idx2]
                #parent1, parent2 = pair_12_13_list[drude_id]
                #drude1, drude2 = parent1 + 1, parent2 + 1
                # print(f"thole {idx1=}, {idx2=}")
                # print(f"{drude_id=}, {f=}")
                if drude1 in atom_idxs[r.name] and drude2 in atom_idxs[r.name]:
                    # idx1, idx2 = f[0], f[1]
                    # if ( idx1 in atom_idxs[r.name] and idx2 in atom_idxs[r.name] ):
                    print(f)
                    force_state_thole[r.name].append(f)
        raise AssertionError("print drude parameters")

        if r.name == "OH":
            names.append(r.name)
            atom_idxs[r.name] = [atom.index for atom in r.atoms()]
            atom_names[r.name] = [atom.name for atom in r.atoms()]
            drude_force = [
                f for f in system.getForces() if isinstance(f, mm.DrudeForce)
            ][0]
            print(f"{r.name=}")
            print("drude")
            print(drude_force.getNumParticles())
            particle_map = {}
            for drude_id in range(drude_force.getNumParticles()):
                f = drude_force.getParticleParameters(drude_id)
                idx1, idx2 = f[0], f[1]
                particle_map[drude_id] = idx1
                if idx1 in atom_idxs[r.name] and idx2 in atom_idxs[r.name]:
                    print(f)
                    force_state[r.name].append(f)

            print("thole")
            print(drude_force.getNumScreenedPairs())
            for drude_id in range(drude_force.getNumScreenedPairs()):
                f = drude_force.getScreenedPairParameters(drude_id)
                idx1, idx2 = f[0],f[1]
                drude1 = particle_map[idx1]
                drude2 = particle_map[idx2]
                #parent1, parent2 = pair_12_13_list[drude_id]
                #drude1, drude2 = parent1 + 1, parent2 + 1
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
    raise AssertionError("printing")


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Will fail sporadicaly.",
)
def test_write_psf_save_load(tmp_path):
    psf_for_parameters = f"{protex.__path__[0]}/forcefield/toh2/h2o.psf"

    simulation = generate_toh2_system(use_plugin=False)
    simulation_for_parameters = generate_toh2_system(use_plugin=False)

    # get ionic liquid templates
    allowed_updates = {}
    allowed_updates[frozenset(["OH", "H2O"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["OH", "H3O"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["H3O", "H2O"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["H2O", "H2O"])] = {"r_max": 0.16, "prob": 1}


    templates = ProtexTemplates(
        [OH_H2O_H3O], (allowed_updates)
    )
    # wrap system in IonicLiquidSystem
    ionic_liquid = ProtexSystem(simulation, templates, simulation_for_parameters)
    # initialize update method
    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)

    old_psf_file = f"{protex.__path__[0]}/forcefield/toh2/h2o.psf"
    ionic_liquid.write_psf(old_psf_file, f"{tmp_path}/test.psf", psf_for_parameters)

    # ionic_liquid.simulation.step(50)
    state_update.update(2)

    ionic_liquid.write_psf(old_psf_file, f"{tmp_path}/test.psf", psf_for_parameters)
    ionic_liquid.saveState(f"{tmp_path}/state.rst")
    ionic_liquid.saveCheckpoint(f"{tmp_path}/checkpoint.rst")

    ionic_liquid2 = ionic_liquid  # copy.deepcopy(ionic_liquid)
    ionic_liquid.loadState(f"{tmp_path}/state.rst")
    ionic_liquid2.loadCheckpoint(f"{tmp_path}/checkpoint.rst")


@pytest.mark.skipif(os.getenv("CI") == "true",
    reason="Will fail sporadicaly.",
)
def test_pickle_residues_save_load(tmp_path):
    psf_for_parameters = f"{protex.__path__[0]}/forcefield/toh2/h2o.psf"

    simulation = generate_toh2_system(use_plugin=False)
    simulation_for_parameters = generate_toh2_system(use_plugin=False)

    # get ionic liquid templates
    allowed_updates = {}
    allowed_updates[frozenset(["OH", "H2O"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["OH", "H3O"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["H3O", "H2O"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["H2O", "H2O"])] = {"r_max": 0.16, "prob": 1}

    templates = ProtexTemplates(
        [OH_H2O_H3O], (allowed_updates)
    )
    # wrap system in IonicLiquidSystem
    ionic_liquid = ProtexSystem(simulation, templates, simulation_for_parameters)
    # initialize update method
    update = KeepHUpdate(ionic_liquid)#, include_equivalent_atom=False, reorient=False)
    # initialize state update class
    state_update = StateUpdate(update)

    for i in range(10):
        print(ionic_liquid.residues[i].current_name, ionic_liquid.residues[i].donors)

    #ionic_liquid.simulation.step(50)
    state_update.update(2)

    # before_loading = [[ionic_liquid.residues[i].current_name, ionic_liquid.residues[i].donors, ionic_liquid.residues[i].mode_in_last_transfer] for i in range(10)]
    forces_dict = {}
    for i in range(10):
        residue = ionic_liquid.residues[i]
        atom_idxs = [atom.index for atom in residue.residue.atoms()]
        for force in simulation.system.getForces():
            forcename = type(force).__name__
            if forcename == "NonbondedForce":
                forces_dict[i] = [
                    force.getParticleParameters(idx) for idx in atom_idxs
                ]
    before_loading = forces_dict
    # for i in range(10):
    #     print(ionic_liquid.residues[i].current_name, ionic_liquid.residues[i].donors, ionic_liquid.residues[i].mode_in_last_transfer)

    ionic_liquid.saveState(f"{tmp_path}/state.rst")
    ionic_liquid.saveCheckpoint(f"{tmp_path}/checkpoint.rst")
    ionic_liquid.dump(f"{tmp_path}/system.pkl")

    simulation = generate_toh2_system(use_plugin=False)
    simulation_for_parameters = generate_toh2_system(use_plugin=False)

    ionic_liquid2 = ProtexSystem.load(f"{tmp_path}/system.pkl", simulation, simulation_for_parameters)

    # print("####### after load #########")
    # for i in range(10):
    #     print(ionic_liquid2.residues[i].current_name, ionic_liquid2.residues[i].donors, ionic_liquid2.residues[i].mode_in_last_transfer)

    ionic_liquid2.loadState(f"{tmp_path}/state.rst")
    ionic_liquid2.loadCheckpoint(f"{tmp_path}/checkpoint.rst")

    # after_loading = [[ionic_liquid.residues[i].current_name, ionic_liquid.residues[i].donors, ionic_liquid.residues[i].mode_in_last_transfer] for i in range(10)]
    forces_dict = {}
    for i in range(10):
        residue = ionic_liquid.residues[i]
        atom_idxs = [atom.index for atom in residue.residue.atoms()]
        for force in simulation.system.getForces():
            forcename = type(force).__name__
            if forcename == "NonbondedForce":
                forces_dict[i] = [
                    force.getParticleParameters(idx) for idx in atom_idxs
                ]
    after_loading = forces_dict

    # for i in range(10):
    #     if before_loading[i] != after_loading[i]:
    #         print(i)
    # for i in range(10):
    #     if before_loading[i] != after_loading[i]:
    #         print(before_loading[i])
    #         print(after_loading[i])

    assert before_loading == after_loading


@pytest.mark.skipif(os.getenv("CI") == "true",
    reason="Will fail sporadicaly.",
)
def test_ac_toh2_pickle_residues_save_load(tmp_path):
    simulation = generate_ac_toh2_system(use_plugin=False)
    psf_for_parameters = f"{protex.__path__[0]}/forcefield/toh2/psf_for_parameters.psf"
    crdfor_parameters = f"{protex.__path__[0]}/forcefield/toh2/psf_for_parameters.crd"
    simulation_for_parameters = generate_ac_toh2_system(crd_file=crdfor_parameters, psf_file=psf_for_parameters , use_plugin=False)
    # get ionic liquid templates
    allowed_updates = {}
    # allowed updates according to simple protonation scheme
    allowed_updates[frozenset(["TOH3", "OAC"])] = {"r_max": 0.2, "prob": 1}
    allowed_updates[frozenset(["HOAC", "TOH2"])] = {"r_max": 0.2, "prob": 1}
    allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.2, "prob": 1}
    allowed_updates[frozenset(["TOH3", "TOH2"])] = {"r_max": 0.2, "prob": 1}
    print(allowed_updates.keys())
    templates = ProtexTemplates(
        # [OAC_HOAC_chelpg, IM1H_IM1_chelpg], (set(["IM1H", "OAC"]), set(["IM1", "HOAC"]))
        [OAC_HOAC, H2O_H3O],
        (allowed_updates),
    )
    # wrap system in IonicLiquidSystem
    ionic_liquid = ProtexSystem(simulation, templates, simulation_for_parameters)

    # initialize update method
    update = KeepHUpdate(ionic_liquid)#, include_equivalent_atom=False, reorient=False)
    # initialize state update class
    state_update = StateUpdate(update)

    for i in range(10):
        print(ionic_liquid.residues[i].current_name, ionic_liquid.residues[i].donors)

    #ionic_liquid.simulation.step(50)
    state_update.update(2)

    # before_loading = [[ionic_liquid.residues[i].current_name, ionic_liquid.residues[i].donors, ionic_liquid.residues[i].mode_in_last_transfer] for i in range(10)]
    forces_dict = {}
    for i in range(10):
        residue = ionic_liquid.residues[i]
        atom_idxs = [atom.index for atom in residue.residue.atoms()]
        for force in simulation.system.getForces():
            forcename = type(force).__name__
            if forcename == "NonbondedForce":
                forces_dict[i] = [
                    force.getParticleParameters(idx) for idx in atom_idxs
                ]
    before_loading = forces_dict
    # for i in range(10):
    #     print(ionic_liquid.residues[i].current_name, ionic_liquid.residues[i].donors, ionic_liquid.residues[i].mode_in_last_transfer)

    ionic_liquid.saveState(f"{tmp_path}/state.rst")
    ionic_liquid.saveCheckpoint(f"{tmp_path}/checkpoint.rst")
    ionic_liquid.dump(f"{tmp_path}/system.pkl")

    simulation = generate_small_box(use_plugin=False)
    simulation_for_parameters = generate_small_box(use_plugin=False)

    ionic_liquid2 = ProtexSystem.load(f"{tmp_path}/system.pkl", simulation, simulation_for_parameters)

    # print("####### after load #########")
    # for i in range(10):
    #     print(ionic_liquid2.residues[i].current_name, ionic_liquid2.residues[i].donors, ionic_liquid2.residues[i].mode_in_last_transfer)

    ionic_liquid2.loadState(f"{tmp_path}/state.rst")
    ionic_liquid2.loadCheckpoint(f"{tmp_path}/checkpoint.rst")

    # after_loading = [[ionic_liquid.residues[i].current_name, ionic_liquid.residues[i].donors, ionic_liquid.residues[i].mode_in_last_transfer] for i in range(10)]
    forces_dict = {}
    for i in range(10):
        residue = ionic_liquid.residues[i]
        atom_idxs = [atom.index for atom in residue.residue.atoms()]
        for force in simulation.system.getForces():
            forcename = type(force).__name__
            if forcename == "NonbondedForce":
                forces_dict[i] = [
                    force.getParticleParameters(idx) for idx in atom_idxs
                ]
    after_loading = forces_dict

    # for i in range(10):
    #     if before_loading[i] != after_loading[i]:
    #         print(i)
    # for i in range(10):
    #     if before_loading[i] != after_loading[i]:
    #         print(before_loading[i])
    #         print(after_loading[i])

    assert before_loading == after_loading

@pytest.mark.skipif(os.getenv("CI") == "true",
    reason="Will fail sporadicaly.",
)
def test_m2_toh2_pickle_residues_save_load(tmp_path):

    base = "/site/raid5/marta/simulations/test_m2"
    psf_for_parameters = f"{base}/pph2o.psf"
    crdfor_parameters = f"{base}/pph2o.crd"
    npt_rst = f"{base}/pph2o_npt_7.rst"

    simulation = generate_m2_toh2_system(crd_file=crdfor_parameters, psf_file=psf_for_parameters, restart_file = npt_rst, use_plugin=False)
    simulation_for_parameters = generate_m2_toh2_system(crd_file=crdfor_parameters, psf_file=psf_for_parameters , use_plugin=False)
    # get ionic liquid templates
    allowed_updates = {}
    # allowed updates according to simple protonation scheme
    allowed_updates[frozenset(["TOH3", "TOH2"])] = {"r_min": 0.100, "r_max": 0.130, "prob": 1.000}
    allowed_updates[frozenset(["TOH3", "UDO"])] = {"r_min": 0.100, "r_max": 0.130, "prob": 1.000}
    allowed_updates[frozenset(["TOH2", "ULF"])] = {"r_min": 0.100, "r_max": 0.130, "prob": 1.000}
    print(allowed_updates.keys())
    templates = ProtexTemplates(
        # [OAC_HOAC_chelpg, IM1H_IM1_chelpg], (set(["IM1H", "OAC"]), set(["IM1", "HOAC"]))
        [H2O_H3O, HSP_HSD],
        (allowed_updates),
    )
    # wrap system in IonicLiquidSystem
    ionic_liquid = ProtexSystem(simulation, templates, simulation_for_parameters)

    # initialize update method
    update = KeepHUpdate(ionic_liquid)#, include_equivalent_atom=False, reorient=False)
    # initialize state update class
    state_update = StateUpdate(update)

    # first_done = False
    # first = []
    # second = []
    # NBex = []
    # for force in ionic_liquid.system.getForces():
    #     if type(force).__name__ == "CustomNonbondedForce":
    #     # if type(force).__name__ == "NonbondedForce":
    #         print(force)
    #         print(force.getNumParticles())
    #         print(force.getNumExclusions())
    #         #print(force.getForceGroup())
    #         #print(force.getNumTabulatedFunctions())
    #         # func = force.getTabulatedFunction(0)
    #         # print(func.getFunctionParameters())
    #         # print(len(func.getFunctionParameters()[-1]))
    #         # func = force.getTabulatedFunction(1)
    #         # print(func.getFunctionParameters())
    #         # print(len(func.getFunctionParameters()[-1]))
    #         for i in range(force.getNumExclusions()):
    #             if not first_done:
    #                 first.append(force.getExclusionParticles(i))
    #             else:
    #                 second.append(force.getExclusionParticles(i))
    #         first_done = True

    #     if type(force).__name__ == "NonbondedForce":
    #         print(force)
    #         print(force.getNumParticles())
    #         print(force.getNumExceptions())
    #         #print(force.getForceGroup())
    #         #print(force.getNumTabulatedFunctions())
    #         # func = force.getTabulatedFunction(0)
    #         # print(func.getFunctionParameters())
    #         # print(len(func.getFunctionParameters()[-1]))
    #         # func = force.getTabulatedFunction(1)
    #         # print(func.getFunctionParameters())
    #         # print(len(func.getFunctionParameters()[-1]))
    #         for i in range(force.getNumExceptions()):
    #             NBex.append(force.getExceptionParameters(i))

    # print(first == second)
    # print(first == NBex)

    # NBexparticles = [i[0:2] for i in NBex]

    # print(first == NBexparticles)

    # print(first[0:10])
    # print(NBexparticles[0:10])
    # print("#############")
    # print(NBex[0:10])

    # raise AssertionError("here")

    for i in range(10):
        print(ionic_liquid.residues[i].current_name, ionic_liquid.residues[i].donors)

    #ionic_liquid.simulation.step(50)
    state_update.update(2)

    # before_loading = [[ionic_liquid.residues[i].current_name, ionic_liquid.residues[i].donors, ionic_liquid.residues[i].mode_in_last_transfer] for i in range(10)]
    forces_dict = {}
    for i in range(10):
        residue = ionic_liquid.residues[i]
        atom_idxs = [atom.index for atom in residue.residue.atoms()]
        for force in simulation.system.getForces():
            forcename = type(force).__name__
            if forcename == "NonbondedForce":
                forces_dict[i] = [
                    force.getParticleParameters(idx) for idx in atom_idxs
                ]
    before_loading = forces_dict
    # for i in range(10):
    #     print(ionic_liquid.residues[i].current_name, ionic_liquid.residues[i].donors, ionic_liquid.residues[i].mode_in_last_transfer)

    ionic_liquid.saveState(f"{tmp_path}/state.rst")
    ionic_liquid.saveCheckpoint(f"{tmp_path}/checkpoint.rst")
    ionic_liquid.dump(f"{tmp_path}/system.pkl")

    simulation = generate_small_box(use_plugin=False)
    simulation_for_parameters = generate_small_box(use_plugin=False)

    ionic_liquid2 = ProtexSystem.load(f"{tmp_path}/system.pkl", simulation, simulation_for_parameters)

    # print("####### after load #########")
    # for i in range(10):
    #     print(ionic_liquid2.residues[i].current_name, ionic_liquid2.residues[i].donors, ionic_liquid2.residues[i].mode_in_last_transfer)

    ionic_liquid2.loadState(f"{tmp_path}/state.rst")
    ionic_liquid2.loadCheckpoint(f"{tmp_path}/checkpoint.rst")

    # after_loading = [[ionic_liquid.residues[i].current_name, ionic_liquid.residues[i].donors, ionic_liquid.residues[i].mode_in_last_transfer] for i in range(10)]
    forces_dict = {}
    for i in range(10):
        residue = ionic_liquid.residues[i]
        atom_idxs = [atom.index for atom in residue.residue.atoms()]
        for force in simulation.system.getForces():
            forcename = type(force).__name__
            if forcename == "NonbondedForce":
                forces_dict[i] = [
                    force.getParticleParameters(idx) for idx in atom_idxs
                ]
    after_loading = forces_dict

    # for i in range(10):
    #     if before_loading[i] != after_loading[i]:
    #         print(i)
    # for i in range(10):
    #     if before_loading[i] != after_loading[i]:
    #         print(before_loading[i])
    #         print(after_loading[i])

    assert before_loading == after_loading



#####################
# TEST UPDATE
#######################


# @pytest.mark.skipif(
#     os.getenv("CI") == "true",
#     reason="Will fail sporadicaly.",
# )
# def test_updates(caplog, tmp_path):
#     caplog.set_level(logging.DEBUG)

#     psf_for_parameters = f"{protex.__path__[0]}/forcefield/hpts_single/hpts_single.psf"
#     # f"{protex.__path__[0]}/forcefield/crd_for_parameters.crd"
#     # f"{protex.__path__[0]}/forcefield/hpts.psf"
#     # f"{protex.__path__[0]}/forcefield/traj/hpts_npt_7.rst"

#     # simulation = generate_hpts_meoh_system(psf_file=psf_file, restart_file=restart_file)
#     # simulation_for_parameters = generate_hpts_meoh_system(
#     #    psf_file=psf_for_parameters, crd_file=crd_for_parameters
#     # )

#     simulation = generate_single_hpts_meoh_system()
#     simulation_for_parameters = generate_single_hpts_meoh_system()

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
#     # allowed_updates[frozenset(["HPTSH", "HPTS"])] = {"r_max": 0.155, "prob": 1.000}
#     # allowed_updates[frozenset(["HOAC", "HPTS"])] = {"r_max": 0.155, "prob": 1.000}
#     # allowed_updates[frozenset(["IM1H", "HPTS"])] = {"r_max": 0.155, "prob": 1.000}
#     allowed_updates[frozenset(["HPTSH", "MEOH"])] = {"r_max": 0.155, "prob": 1.000}
#     allowed_updates[frozenset(["MEOH2", "MEOH"])] = {"r_max": 0.155, "prob": 1.000}
#     allowed_updates[frozenset(["MEOH2", "IM1"])] = {"r_max": 0.155, "prob": 1.000}
#     allowed_updates[frozenset(["MEOH2", "OAC"])] = {"r_max": 0.155, "prob": 1.000}

#     templates = ProtexTemplates(
#         [OAC_HOAC, IM1H_IM1, HPTSH_HPTS, MEOH_MEOH2], (allowed_updates)
#     )
#     # wrap system in IonicLiquidSystem
#     ionic_liquid = ProtexSystem(simulation, templates, simulation_for_parameters)
#     pars = []
#     update = NaiveMCUpdate(ionic_liquid)
#     # initialize state update class
#     state_update = StateUpdate(update)
#     # ionic_liquid.simulation.minimizeEnergy(maxIterations=200)
#     ionic_liquid.simulation.step(50)

#     for _ in range(5):
#         ionic_liquid.simulation.step(18)
#         pars.append(state_update.get_charges())
#         candidate_pairs = state_update.update(2)

#         print(candidate_pairs)

#     # test whether the update changed the psf
#     old_psf_file = f"{protex.__path__[0]}/forcefield/hpts_single/hpts_single.psf"
#     ionic_liquid.write_psf(old_psf_file, f"{tmp_path}/hpts_new.psf", psf_for_parameters)


# @pytest.mark.skipif(
#     os.getenv("CI") == "true",
#     reason="Will fail sporadicaly.",
# )
# def test_residue_forces():
#     # psf_for_parameters = f"{protex.__path__[0]}/forcefield/psf_for_parameters.psf"
#     # crd_for_parameters = f"{protex.__path__[0]}/forcefield/crd_for_parameters.crd"
#     # psf_file = f"{protex.__path__[0]}/forcefield/hpts.psf"

#     # generate_hpts_meoh_system(psf_file=psf_file)
#     # simulation_for_parameters = generate_hpts_meoh_system(
#     #    crd_file=crd_for_parameters, psf_file=psf_for_parameters
#     # )

#     simulation_for_parameters = generate_single_hpts_meoh_system()

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
#     # allowed_updates[frozenset(["HPTSH", "HPTS"])] = {"r_max": 0.155, "prob": 1.000}
#     # allowed_updates[frozenset(["HOAC", "HPTS"])] = {"r_max": 0.155, "prob": 1.000}
#     # allowed_updates[frozenset(["IM1H", "HPTS"])] = {"r_max": 0.155, "prob": 1.000}
#     allowed_updates[frozenset(["HPTSH", "MEOH"])] = {"r_max": 0.155, "prob": 1.000}
#     allowed_updates[frozenset(["MEOH2", "MEOH"])] = {"r_max": 0.155, "prob": 1.000}
#     allowed_updates[frozenset(["MEOH2", "IM1"])] = {"r_max": 0.155, "prob": 1.000}
#     allowed_updates[frozenset(["MEOH2", "OAC"])] = {"r_max": 0.155, "prob": 1.000}

#     templates = ProtexTemplates(
#         [OAC_HOAC, IM1H_IM1, HPTSH_HPTS, MEOH_MEOH2], (allowed_updates)
#     )
#     ionic_liquid = ProtexSystem(simulation_for_parameters, templates)

#     ionic_liquid.residues[0]
#     ionic_liquid.residues[1]
#     ionic_liquid.residues[2]
#     ionic_liquid.residues[3]
#     ionic_liquid.residues[4]
#     ionic_liquid.residues[5]
#     meoh = ionic_liquid.residues[6]
#     meoh2 = ionic_liquid.residues[7]

#     resi = meoh
#     pair = meoh2

#     offset = resi._get_offset(resi.current_name)
#     offset_pair = pair._get_offset(pair.current_name)

#     ### test if number of forces are equal
#     for force in (
#         "NonbondedForce",
#         "HarmonicBondForce",
#         "HarmonicAngleForce",
#         "PeriodicTorsionForce",
#         "CustomTorsionForce",
#         "DrudeForce",
#     ):
#         print(f"{force=}")
#         par1 = resi.parameters[resi.current_name][force]
#         par2 = pair.parameters[pair.current_name][force]
#         assert len(par1) == len(par2)

#     ### test indices

#     print("force=HarmonicBondForce")
#     for old_idx, old_parm in enumerate(
#         resi.parameters[resi.current_name]["HarmonicBondForce"]
#     ):
#         idx1, idx2 = old_parm[0], old_parm[1]
#         for new_idx, new_parm in enumerate(
#             pair.parameters[pair.current_name]["HarmonicBondForce"]
#         ):
#             if {new_parm[0] - offset_pair, new_parm[1] - offset_pair} == {
#                 idx1 - offset,
#                 idx2 - offset,
#             }:
#                 if old_idx != new_idx:
#                     print(old_idx, new_idx)
#                     raise RuntimeError(
#                         "Odering is different between the two topologies."
#                     )

#     print("force=DrudeForce")
#     for old_idx, old_parm in enumerate(
#         resi.parameters[resi.current_name]["DrudeForce"]
#     ):
#         idx1, idx2 = old_parm[0], old_parm[1]
#         for new_idx, new_parm in enumerate(
#             pair.parameters[pair.current_name]["DrudeForce"]
#         ):
#             if {new_parm[0] - offset_pair, new_parm[1] - offset_pair} == {
#                 idx1 - offset,
#                 idx2 - offset,
#             }:
#                 if old_idx != new_idx:
#                     raise RuntimeError(
#                         "Odering is different between the two topologies."
#                     )

#     print("force=HarmonicAngleForce")
#     for old_idx, old_parm in enumerate(
#         resi.parameters[resi.current_name]["HarmonicAngleForce"]
#     ):
#         idx1, idx2, idx3 = old_parm[0], old_parm[1], old_parm[2]
#         for new_idx, new_parm in enumerate(
#             pair.parameters[pair.current_name]["HarmonicAngleForce"]
#         ):
#             if {
#                 new_parm[0] - offset_pair,
#                 new_parm[1] - offset_pair,
#                 new_parm[2] - offset_pair,
#             } == {idx1 - offset, idx2 - offset, idx3 - offset}:
#                 if old_idx != new_idx:
#                     raise RuntimeError(
#                         "Odering is different between the two topologies."
#                     )

#     print("force=PeriodicTorsionForce")
#     for old_idx, old_parm in enumerate(
#         resi.parameters[resi.current_name]["PeriodicTorsionForce"]
#     ):
#         idx1, idx2, idx3, idx4, idx5 = (
#             old_parm[0],
#             old_parm[1],
#             old_parm[2],
#             old_parm[3],
#             old_parm[4],
#         )
#         for new_idx, new_parm in enumerate(
#             pair.parameters[pair.current_name]["PeriodicTorsionForce"]
#         ):
#             if {
#                 new_parm[0] - offset_pair,
#                 new_parm[1] - offset_pair,
#                 new_parm[2] - offset_pair,
#                 new_parm[3] - offset_pair,
#                 new_parm[4],
#             } == {idx1 - offset, idx2 - offset, idx3 - offset, idx4 - offset, idx5}:
#                 if old_idx != new_idx:
#                     raise RuntimeError(
#                         "Odering is different between the two topologies."
#                     )

#     print("force=CustomTorsionForce")
#     for old_idx, old_parm in enumerate(
#         resi.parameters[resi.current_name]["CustomTorsionForce"]
#     ):
#         idx1, idx2, idx3, idx4 = old_parm[0], old_parm[1], old_parm[2], old_parm[3]
#         for new_idx, new_parm in enumerate(
#             pair.parameters[pair.current_name]["CustomTorsionForce"]
#         ):
#             if {
#                 new_parm[0] - offset_pair,
#                 new_parm[1] - offset_pair,
#                 new_parm[2] - offset_pair,
#                 new_parm[3] - offset_pair,
#             } == {idx1 - offset, idx2 - offset, idx3 - offset, idx4 - offset}:
#                 if old_idx != new_idx:
#                     raise RuntimeError(
#                         "Odering is different between the two topologies."
#                     )


# def test_list_torsionforce(tmp_path):
#     # psf_for_parameters = f"{protex.__path__[0]}/forcefield/psf_for_parameters.psf"
#     # crd_for_parameters = f"{protex.__path__[0]}/forcefield/crd_for_parameters.crd"
#     # psf_file = f"{protex.__path__[0]}/forcefield/hpts.psf"

#     # generate_hpts_meoh_system(psf_file=psf_file)
#     # simulation_for_parameters = generate_hpts_meoh_system(
#     #    crd_file=crd_for_parameters, psf_file=psf_for_parameters
#     # )
#     simulation_for_parameters = generate_single_hpts_meoh_system(use_plugin=False)
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
#     # allowed_updates[frozenset(["HPTSH", "HPTS"])] = {"r_max": 0.155, "prob": 1.000}
#     # allowed_updates[frozenset(["HOAC", "HPTS"])] = {"r_max": 0.155, "prob": 1.000}
#     # allowed_updates[frozenset(["IM1H", "HPTS"])] = {"r_max": 0.155, "prob": 1.000}
#     allowed_updates[frozenset(["HPTSH", "MEOH"])] = {"r_max": 0.155, "prob": 1.000}
#     allowed_updates[frozenset(["MEOH2", "MEOH"])] = {"r_max": 0.155, "prob": 1.000}
#     allowed_updates[frozenset(["MEOH2", "IM1"])] = {"r_max": 0.155, "prob": 1.000}
#     allowed_updates[frozenset(["MEOH2", "OAC"])] = {"r_max": 0.155, "prob": 1.000}

#     templates = ProtexTemplates(
#         [OAC_HOAC, IM1H_IM1, HPTSH_HPTS, MEOH_MEOH2], (allowed_updates)
#     )
#     ionic_liquid = ProtexSystem(simulation_for_parameters, templates)

#     oac = ionic_liquid.residues[1]
#     hoac = ionic_liquid.residues[3]

#     offset1 = oac._get_offset("OAC")
#     offset2 = hoac._get_offset("HOAC")

#     for old_idx, old_parm in enumerate(oac.parameters["OAC"]["PeriodicTorsionForce"]):
#         idx1, idx2, idx3, idx4, idx5, idx6, idx7 = (
#             old_parm[0] - offset1,
#             old_parm[1] - offset1,
#             old_parm[2] - offset1,
#             old_parm[3] - offset1,
#             old_parm[4],
#             old_parm[5],
#             old_parm[6],
#         )
#         f = open(f"{tmp_path}/oac_torsion.txt", "a")
#         f.write(
#             str(old_idx)
#             + "\t"
#             + str(idx1)
#             + "\t"
#             + str(idx2)
#             + "\t"
#             + str(idx3)
#             + "\t"
#             + str(idx4)
#             + "\t"
#             + str(idx5)
#             + "\t"
#             + str(idx6)
#             + "\t"
#             + str(idx7)
#             + "\n"
#         )
#         f.close()

#     for new_idx, new_parm in enumerate(hoac.parameters["HOAC"]["PeriodicTorsionForce"]):
#         id1, id2, id3, id4, id5, id6, id7 = (
#             new_parm[0] - offset2,
#             new_parm[1] - offset2,
#             new_parm[2] - offset2,
#             new_parm[3] - offset2,
#             new_parm[4],
#             new_parm[5],
#             new_parm[6],
#         )
#         g = open(f"{tmp_path}/hoac_torsion.txt", "a")
#         g.write(
#             str(new_idx)
#             + "\t"
#             + str(id1)
#             + "\t"
#             + str(id2)
#             + "\t"
#             + str(id3)
#             + "\t"
#             + str(id4)
#             + "\t"
#             + str(id5)
#             + "\t"
#             + str(id6)
#             + "\t"
#             + str(id7)
#             + "\n"
#         )
#         g.close()


# def test_count_forces():
#     # psf_for_parameters = f"{protex.__path__[0]}/forcefield/psf_for_parameters.psf"
#     # crd_for_parameters = f"{protex.__path__[0]}/forcefield/crd_for_parameters.crd"
#     # psf_file = f"{protex.__path__[0]}/forcefield/hpts.psf"

#     # simulation = generate_hpts_meoh_system(psf_file=psf_file)
#     # simulation_for_parameters = generate_hpts_meoh_system(
#     #    crd_file=crd_for_parameters, psf_file=psf_for_parameters
#     # )
#     simulation = generate_single_hpts_meoh_system(use_plugin=False)
#     simulation_for_parameters = generate_single_hpts_meoh_system(use_plugin=False)

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
#     # allowed_updates[frozenset(["HPTSH", "HPTS"])] = {"r_max": 0.155, "prob": 1.000}
#     # allowed_updates[frozenset(["HOAC", "HPTS"])] = {"r_max": 0.155, "prob": 1.000}
#     # allowed_updates[frozenset(["IM1H", "HPTS"])] = {"r_max": 0.155, "prob": 1.000}
#     allowed_updates[frozenset(["HPTSH", "MEOH"])] = {"r_max": 0.155, "prob": 1.000}
#     allowed_updates[frozenset(["MEOH2", "MEOH"])] = {"r_max": 0.155, "prob": 1.000}
#     allowed_updates[frozenset(["MEOH2", "IM1"])] = {"r_max": 0.155, "prob": 1.000}
#     allowed_updates[frozenset(["MEOH2", "OAC"])] = {"r_max": 0.155, "prob": 1.000}

#     templates = ProtexTemplates(
#         [OAC_HOAC, IM1H_IM1, HPTSH_HPTS, MEOH_MEOH2], (allowed_updates)
#     )
#     ionic_liquid = ProtexSystem(simulation_for_parameters, templates)

#     ionic_liquid.residues[0]
#     ionic_liquid.residues[1]
#     ionic_liquid.residues[2]
#     ionic_liquid.residues[3]
#     ionic_liquid.residues[4]
#     ionic_liquid.residues[5]
#     meoh = ionic_liquid.residues[6]
#     meoh2 = ionic_liquid.residues[7]

#     ### change res and pair (alternative of res) to test different molecules
#     res = meoh
#     pair = meoh2

#     offset = res._get_offset(res.current_name)
#     offset_pair = pair._get_offset(pair.current_name)

#     ### make list of atoms in dihedrals, check if there are duplicates
#     torsions = []

#     for old_idx, old_parm in enumerate(
#         res.parameters[res.current_name]["PeriodicTorsionForce"]
#     ):
#         idx1, idx2, idx3, idx4 = (
#             old_parm[0] - offset,
#             old_parm[1] - offset,
#             old_parm[2] - offset,
#             old_parm[3] - offset,
#         )
#         ids = [idx1, idx2, idx3, idx4]
#         torsions.append(ids)

#     # print(torsions)

#     for old_idx, old_parm in enumerate(
#         res.parameters[res.current_name]["PeriodicTorsionForce"]
#     ):
#         idx1, idx2, idx3, idx4, idx5 = (
#             old_parm[0] - offset,
#             old_parm[1] - offset,
#             old_parm[2] - offset,
#             old_parm[3] - offset,
#             old_parm[4],
#         )
#         ids = [idx1, idx2, idx3, idx4]
#         print(ids)
#         print(torsions.count(ids))
#         if torsions.count(ids) == 1:
#             print("count id 1, in branch without multiplicity")
#             for new_idx, new_parm in enumerate(
#                 pair.parameters[pair.current_name]["PeriodicTorsionForce"]
#             ):
#                 if {
#                     new_parm[0] - offset_pair,
#                     new_parm[1] - offset_pair,
#                     new_parm[2] - offset_pair,
#                     new_parm[3] - offset_pair,
#                 } == {idx1, idx2, idx3, idx4}:
#                     if old_idx != new_idx:
#                         print(old_idx, new_idx)
#                         for force in simulation.system.getForces():
#                             if type(force).__name__ == "PeriodicTorsionForce":
#                                 for torsion_id in range(force.getNumTorsions()):
#                                     f = force.getTorsionParameters(torsion_id)
#                                     if (
#                                         f[0] in old_parm
#                                         and f[1] in old_parm
#                                         and f[2] in old_parm
#                                         and f[3] in old_parm
#                                     ):
#                                         print("old force", f)
#                                     if (
#                                         f[0] in new_parm
#                                         and f[1] in new_parm
#                                         and f[2] in new_parm
#                                         and f[3] in new_parm
#                                     ):
#                                         print("new force", f)
#                         raise RuntimeError(
#                             "Odering is different between the two topologies."
#                         )
#                     break

#             else:
#                 for force in simulation.system.getForces():
#                     if type(force).__name__ == "PeriodicTorsionForce":
#                         for torsion_id in range(force.getNumTorsions()):
#                             f = force.getTorsionParameters(torsion_id)
#                             if (
#                                 f[0] in old_parm
#                                 and f[1] in old_parm
#                                 and f[2] in old_parm
#                                 and f[3] in old_parm
#                             ):
#                                 print(old_idx, "old force", f)
#                             if (
#                                 f[0] in new_parm
#                                 and f[1] in new_parm
#                                 and f[2] in new_parm
#                                 and f[3] in new_parm
#                             ):
#                                 print(new_idx, "new force", f)
#                 raise RuntimeError()
#         else:
#             print("count id != 1, in branch with multiplicities")
#             for new_idx, new_parm in enumerate(
#                 pair.parameters[pair.current_name]["PeriodicTorsionForce"]
#             ):
#                 if {
#                     new_parm[0] - offset_pair,
#                     new_parm[1] - offset_pair,
#                     new_parm[2] - offset_pair,
#                     new_parm[3] - offset_pair,
#                     new_parm[4],
#                 } == {idx1, idx2, idx3, idx4, idx5}:
#                     if old_idx != new_idx:
#                         print(old_idx, new_idx)
#                         for force in simulation.system.getForces():
#                             if type(force).__name__ == "PeriodicTorsionForce":
#                                 for torsion_id in range(force.getNumTorsions()):
#                                     f = force.getTorsionParameters(torsion_id)
#                                     if (
#                                         f[0] in old_parm
#                                         and f[1] in old_parm
#                                         and f[2] in old_parm
#                                         and f[3] in old_parm
#                                     ):
#                                         print("old force", f)
#                                     if (
#                                         f[0] in new_parm
#                                         and f[1] in new_parm
#                                         and f[2] in new_parm
#                                         and f[3] in new_parm
#                                     ):
#                                         print("new force", f)
#                         raise RuntimeError(
#                             "Odering is different between the two topologies."
#                         )
#                     break

#             else:
#                 for force in simulation.system.getForces():
#                     if type(force).__name__ == "PeriodicTorsionForce":
#                         for torsion_id in range(force.getNumTorsions()):
#                             f = force.getTorsionParameters(torsion_id)
#                             if (
#                                 f[0] in old_parm
#                                 and f[1] in old_parm
#                                 and f[2] in old_parm
#                                 and f[3] in old_parm
#                             ):
#                                 print(old_idx, "old force", f)
#                             if (
#                                 f[0] in new_parm
#                                 and f[1] in new_parm
#                                 and f[2] in new_parm
#                                 and f[3] in new_parm
#                             ):
#                                 print(new_idx, "new force", f)
#                 raise RuntimeError()


# @pytest.mark.skipif(
#     os.getenv("CI") == "true",
#     reason="Will fail sporadicaly.",
# )
# def test_update_write_psf(tmp_path):
#     # psf_for_parameters = f"{protex.__path__[0]}/forcefield/hpts_single/hpts_single.psf"
#     psf_for_parameters = f"{protex.__path__[0]}/forcefield/psf_for_parameters.psf"
#     psf_file = f"{protex.__path__[0]}/forcefield/hpts.psf"
#     crd_for_parameters = f"{protex.__path__[0]}/forcefield/crd_for_parameters.crd"

#     simulation = generate_hpts_meoh_system(psf_file=psf_file)
#     simulation_for_parameters = generate_hpts_meoh_system(
#         psf_file=psf_for_parameters, crd_file=crd_for_parameters
#     )
#     # simulation = generate_hpts_meoh_system()
#     # simulation_for_parameters = generate_hpts_meoh_system()

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
#     # allowed_updates[frozenset(["HPTSH", "HPTS"])] = {"r_max": 0.155, "prob": 1.000}
#     # allowed_updates[frozenset(["HOAC", "HPTS"])] = {"r_max": 0.155, "prob": 1.000}
#     # allowed_updates[frozenset(["IM1H", "HPTS"])] = {"r_max": 0.155, "prob": 1.000}
#     allowed_updates[frozenset(["HPTSH", "MEOH"])] = {"r_max": 0.155, "prob": 1.000}
#     allowed_updates[frozenset(["MEOH2", "MEOH"])] = {"r_max": 0.155, "prob": 1.000}
#     allowed_updates[frozenset(["MEOH2", "IM1"])] = {"r_max": 0.155, "prob": 1.000}
#     allowed_updates[frozenset(["MEOH2", "OAC"])] = {"r_max": 0.155, "prob": 1.000}

#     templates = ProtexTemplates(
#         [OAC_HOAC, IM1H_IM1, HPTSH_HPTS, MEOH_MEOH2], (allowed_updates)
#     )
#     # wrap system in IonicLiquidSystem
#     ionic_liquid = ProtexSystem(simulation, templates, simulation_for_parameters)
#     # initialize update method
#     update = NaiveMCUpdate(ionic_liquid)
#     # initialize state update class
#     state_update = StateUpdate(update)

#     old_psf_file = f"{protex.__path__[0]}/forcefield/hpts.psf"
#     # old_psf_file = f"{protex.__path__[0]}/forcefield/hpts_single/hpts_single.psf"
#     ionic_liquid.write_psf(old_psf_file, f"{tmp_path}/test.psf", psf_for_parameters)

#     i = 0
#     while i < 5:  # eventuell change to 20
#         sum_charge = 0
#         for x in range(0, 4457):
#             resi = ionic_liquid.residues[x]
#             sum_charge = sum_charge + resi.current_charge
#         print(sum_charge)
#         if sum_charge != 0:
#             raise RuntimeError("Error in run", i)

#         os.rename(f"{tmp_path}/test.psf", f"{tmp_path}/old_psf.psf")

#         simulation = generate_hpts_meoh_system(psf_file=f"{tmp_path}/old_psf.psf")
#         #simulation = generate_single_hpts_meoh_system(psf_file=f"{tmp_path}/old_psf.psf")
#         ionic_liquid = ProtexSystem(simulation, templates, simulation_for_parameters)
#         update = NaiveMCUpdate(ionic_liquid)
#         state_update = StateUpdate(update)

#         ionic_liquid.simulation.step(50)
#         state_update.update(2)

#         # NOTE: psf_for_parameters missing
#         ionic_liquid.write_psf(
#             f"{tmp_path}/old_psf.psf", f"{tmp_path}/test.psf", psf_for_parameters
#         )
#         ionic_liquid.saveState(f"{tmp_path}/state.rst")
#         ionic_liquid.saveCheckpoint(f"{tmp_path}/checkpoint.rst")

#         ionic_liquid2 = ionic_liquid  # copy.deepcopy(ionic_liquid)
#         ionic_liquid.loadState(f"{tmp_path}/state.rst")
#         ionic_liquid2.loadCheckpoint(f"{tmp_path}/checkpoint.rst")

#         i += 1


# @pytest.mark.skipif(
#     os.getenv("CI") == "true",
#     reason="Does not really test anything. Used for debugging.",
# )
# def test_meoh2_update():
#     psf_for_parameters = f"{protex.__path__[0]}/forcefield/psf_for_parameters.psf"
#     psf_file = f"{protex.__path__[0]}/forcefield/hpts.psf"
#     crd_for_parameters = f"{protex.__path__[0]}/forcefield/crd_for_parameters.crd"

#     simulation = generate_hpts_meoh_system(psf_file=psf_file)
#     simulation_for_parameters = generate_hpts_meoh_system(
#         psf_file=psf_for_parameters, crd_file=crd_for_parameters
#     )

#     # get ionic liquid templates
#     allowed_updates = {}
#     # allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 1}
#     # allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 1}
#     # allowed_updates[frozenset(["IM1H", "IM1"])] = {"r_max": 0.16, "prob": 0.201}  # 1+2
#     # allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.15, "prob": 0.684}  # 3+4
#     # allowed_updates[frozenset(["HPTSH", "OAC"])] = {"r_max": 0.15, "prob": 1.000}
#     # allowed_updates[frozenset(["HPTSH", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
#     # allowed_updates[frozenset(["HPTSH", "IM1"])] = {"r_max": 0.15, "prob": 1.000}
#     # allowed_updates[frozenset(["HOAC", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
#     # allowed_updates[frozenset(["IM1H", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
#     ## allowed_updates[frozenset(["HPTSH", "HPTS"])] = {"r_max": 0.155, "prob": 1.000}
#     ## allowed_updates[frozenset(["HOAC", "HPTS"])] = {"r_max": 0.155, "prob": 1.000}
#     ## allowed_updates[frozenset(["IM1H", "HPTS"])] = {"r_max": 0.155, "prob": 1.000}
#     allowed_updates[frozenset(["HPTSH", "MEOH"])] = {"r_max": 0.155, "prob": 1.000}
#     allowed_updates[frozenset(["MEOH2", "MEOH"])] = {"r_max": 0.155, "prob": 1.000}
#     allowed_updates[frozenset(["MEOH2", "IM1"])] = {"r_max": 0.155, "prob": 1.000}
#     allowed_updates[frozenset(["MEOH2", "OAC"])] = {"r_max": 0.155, "prob": 1.000}

#     templates = ProtexTemplates(
#         [OAC_HOAC, IM1H_IM1, HPTSH_HPTS, MEOH_MEOH2], (allowed_updates)
#     )
#     # wrap system in IonicLiquidSystem
#     ionic_liquid = ProtexSystem(simulation, templates, simulation_for_parameters)
#     # initialize update method
#     # update = NaiveMCUpdate(ionic_liquid, meoh2=True)
#     update = KeepHUpdate(ionic_liquid, include_equivalent_atom=True, reorient=True)
#     # initialize state update class
#     StateUpdate(update)

#     # old_psf_file = f"{protex.__path__[0]}/forcefield/hpts.psf"
#     # ionic_liquid.write_psf(old_psf_file, "test.psf", psf_for_parameters)

#     # print(state_update.updateMethod.meoh2)
#     print("Finished")
#     # assert False

#     # state_update.update(2)

#     # i = 0
#     # while i < 20:
#     #     sum_charge = 0
#     #     for x in range(0, 4457):
#     #         resi = ionic_liquid.residues[x]
#     #         sum_charge = sum_charge + resi.current_charge
#     #     print(sum_charge)
#     #     if sum_charge != 0:
#     #         raise RuntimeError("Error in run", i)

#     #     os.rename("test.psf", "old_psf.psf")

#     #     simulation = generate_hpts_meoh_system(psf_file="old_psf.psf")
#     #     ionic_liquid = ProtexSystem(simulation, templates, simulation_for_parameters)
#     #     update = NaiveMCUpdate(ionic_liquid)
#     #     state_update = StateUpdate(update)

#     #     ionic_liquid.simulation.step(50)
#     #     state_update.update(2)

#     #     # NOTE: psf_for_parameters missing
#     #     ionic_liquid.write_psf("old_psf.psf", "test.psf", psf_for_parameters)
#     #     ionic_liquid.saveState("state.rst")
#     #     ionic_liquid.saveCheckpoint("checkpoint.rst")

#     #     ionic_liquid2 = ionic_liquid  # copy.deepcopy(ionic_liquid)
#     #     ionic_liquid.loadState("state.rst")
#     #     ionic_liquid2.loadCheckpoint("checkpoint.rst")

#     #     i += 1

#     # # os.remove("old_psf.psf")
#     # os.remove("state.rst")
#     # os.remove("checkpoint.rst")
#     # os.remove("test.psf")


# @pytest.mark.skipif(
#     os.getenv("CI") == "true",
#     reason="Will fail sporadicaly.",
# )
# def test_updates_with_reorient(tmp_path):
#     # caplog.set_level(logging.DEBUG)

#     psf_for_parameters = f"{protex.__path__[0]}/forcefield/psf_for_parameters.psf"
#     crd_for_parameters = f"{protex.__path__[0]}/forcefield/crd_for_parameters.crd"
#     psf_file = f"{protex.__path__[0]}/forcefield/hpts.psf"
#     restart_file = f"{protex.__path__[0]}/forcefield/traj/hpts_npt_7.rst"

#     simulation = generate_hpts_meoh_system(psf_file=psf_file, restart_file=restart_file)
#     simulation_for_parameters = generate_hpts_meoh_system(
#         psf_file=psf_for_parameters, crd_file=crd_for_parameters
#     )
#     allowed_updates = {}
#     allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 1}
#     allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 1}
#     allowed_updates[frozenset(["IM1H", "IM1"])] = {"r_max": 0.16, "prob": 0.201}  # 1+2
#     allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.15, "prob": 0.684}  # 3+4
#     allowed_updates[frozenset(["HPTSH", "OAC"])] = {"r_max": 0.15, "prob": 1.000}
#     # allowed_updates[frozenset(["HPTSH", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
#     allowed_updates[frozenset(["HPTSH", "IM1"])] = {"r_max": 0.15, "prob": 1.000}
#     # allowed_updates[frozenset(["HOAC", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
#     # allowed_updates[frozenset(["IM1H", "HPTS"])] = {"r_max": 0.15, "prob": 1.000}
#     # allowed_updates[frozenset(["HPTSH", "HPTS"])] = {"r_max": 0.155, "prob": 1.000}
#     # allowed_updates[frozenset(["HOAC", "HPTS"])] = {"r_max": 0.155, "prob": 1.000}
#     # allowed_updates[frozenset(["IM1H", "HPTS"])] = {"r_max": 0.155, "prob": 1.000}
#     allowed_updates[frozenset(["HPTSH", "MEOH"])] = {"r_max": 0.155, "prob": 1.000}
#     allowed_updates[frozenset(["MEOH2", "MEOH"])] = {"r_max": 0.155, "prob": 1.000}
#     allowed_updates[frozenset(["MEOH2", "IM1"])] = {"r_max": 0.155, "prob": 1.000}
#     allowed_updates[frozenset(["MEOH2", "OAC"])] = {"r_max": 0.155, "prob": 1.000}

#     templates = ProtexTemplates(
#         [OAC_HOAC, IM1H_IM1, HPTSH_HPTS, MEOH_MEOH2], (allowed_updates)
#     )
#     # wrap system in IonicLiquidSystem
#     ionic_liquid = ProtexSystem(simulation, templates, simulation_for_parameters)
#     pars = []
#     # update = NaiveMCUpdate(ionic_liquid, meoh2=True)
#     update = KeepHUpdate(ionic_liquid, include_equivalent_atom=True, reorient=True)
#     # initialize state update class
#     state_update = StateUpdate(update)
#     # ionic_liquid.simulation.minimizeEnergy(maxIterations=200)
#     ionic_liquid.simulation.step(50)

#     for _ in range(2):
#         ionic_liquid.simulation.step(18)
#         pars.append(state_update.get_charges())
#         candidate_pairs = state_update.update(2)

#         print(candidate_pairs)

#     # test whether the update changed the psf
#     old_psf_file = f"{protex.__path__[0]}/forcefield/hpts.psf"
#     ionic_liquid.write_psf(old_psf_file, f"{tmp_path}/hpts_new.psf", psf_for_parameters)


# @pytest.mark.skipif(
#     os.getenv("CI") == "true",
#     reason="Does not really test anything. Used for debugging.",
# )
# def test_manipulating_coordinates():
#     psf_for_parameters = f"{protex.__path__[0]}/forcefield/psf_for_parameters.psf"
#     crd_for_parameters = f"{protex.__path__[0]}/forcefield/crd_for_parameters.crd"
#     psf_file = f"{protex.__path__[0]}/forcefield/hpts.psf"
#     restart_file = f"{protex.__path__[0]}/forcefield/traj/hpts_npt_7.rst"

#     simulation = generate_hpts_meoh_system(psf_file=psf_file, restart_file=restart_file)
#     simulation_for_parameters = generate_hpts_meoh_system(
#         psf_file=psf_for_parameters, crd_file=crd_for_parameters
#     )
#     allowed_updates = {}
#     allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 1}
#     allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 1}
#     allowed_updates[frozenset(["IM1H", "IM1"])] = {"r_max": 0.16, "prob": 0.201}  # 1+2
#     allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.15, "prob": 0.684}  # 3+4
#     allowed_updates[frozenset(["HPTSH", "OAC"])] = {"r_max": 0.15, "prob": 1.000}
#     allowed_updates[frozenset(["HPTSH", "IM1"])] = {"r_max": 0.15, "prob": 1.000}
#     allowed_updates[frozenset(["HPTSH", "MEOH"])] = {"r_max": 0.155, "prob": 1.000}
#     allowed_updates[frozenset(["MEOH2", "MEOH"])] = {"r_max": 0.155, "prob": 1.000}
#     allowed_updates[frozenset(["MEOH2", "IM1"])] = {"r_max": 0.155, "prob": 1.000}
#     allowed_updates[frozenset(["MEOH2", "OAC"])] = {"r_max": 0.155, "prob": 1.000}

#     templates = ProtexTemplates(
#         [OAC_HOAC, IM1H_IM1, HPTSH_HPTS, MEOH_MEOH2], (allowed_updates)
#     )
#     # wrap system in IonicLiquidSystem
#     ionic_liquid = ProtexSystem(simulation, templates, simulation_for_parameters)

#     positions = ionic_liquid.simulation.context.getState(
#         getPositions=True
#     ).getPositions(asNumpy=False)

#     positions_copy = copy.deepcopy(positions)
#     x1 = positions_copy[0]
#     x2 = positions_copy[10]

#     print(f"{x1=}, {x2=}")
#     positions[0] = x2
#     positions[10] = x1

#     ionic_liquid.simulation.context.setPositions(positions)

#     print(f"{positions[0]=}, {positions[10]=}")
