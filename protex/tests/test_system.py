# Import package, test suite, and other packages as needed
# import json
import os
from collections import defaultdict
from sys import stdout

import pytest

import protex

try:  # Syntax changed in OpenMM 7.6
    import openmm as mm
    from openmm import Context, DrudeNoseHooverIntegrator, OpenMMException, Platform
    from openmm.app import (
        PME,
        CharmmCrdFile,
        CharmmParameterSet,
        CharmmPsfFile,
        HBonds,
        Simulation,
        StateDataReporter,
    )
    from openmm.unit import angstroms, kelvin, nanometers, picoseconds
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
        HBonds,
        Simulation,
        StateDataReporter,
    )
    from simtk.unit import angstroms, kelvin, nanometers, picoseconds

from ..reporter import ChargeReporter, EnergyReporter
from ..system import ProtexSystem, ProtexTemplates
from ..testsystems import (
    IM1H_IM1,
    OAC_HOAC,
    generate_im1h_oac_dummy_system,
    generate_im1h_oac_system,
    generate_single_im1h_oac_system,
    generate_small_box,
    generate_tfa_system,
)
from ..update import NaiveMCUpdate, StateUpdate


#################
# single
#############
@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="No need to run it in CI, just here to check when ParmEd support our problem",
)
def test_parmed_hack(tmp_path):
    psf_file = "protex/forcefield/dummy/nonumaniso.psf"
    simulation = generate_im1h_oac_system(psf_file=psf_file)
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.165, "prob": 1}
    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    ionic_liquid = ProtexSystem(simulation, templates)
    ionic_liquid.write_psf(
        psf_file,
        f"{tmp_path}/test.psf",
    )

def test_parmed_hack_tfa(tmp_path):
    psf_file = "protex/forcefield/tfa/tfa_10.psf"
    simulation = generate_tfa_system(use_plugin=False)
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.165, "prob": 1}
    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    ionic_liquid = ProtexSystem(simulation, templates)
    ionic_liquid.write_psf(
        psf_file,
        f"{tmp_path}/test.psf",
    )


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Will fail sporadicaly.",
)
def test_write_psf_save_load_single(tmp_path):
    def get_time_energy(simulation, print=False):
        time = simulation.context.getState().getTime()
        e_pot = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        if print:
            print(f"time: {time}, e_pot: {e_pot}")
        return time, e_pot

    def save_il(ionic_liquid, number):
        ionic_liquid.write_psf(
            "protex/forcefield/single_pairs/im1h_oac_im1_hoac_1_secondtry.psf",
            f"{tmp_path}/test_{number}.psf",
        )
        ionic_liquid.saveCheckpoint(f"{tmp_path}/test_{number}.rst")

    def load_sim(psf, rst):
        sim = generate_single_im1h_oac_system(psf_file=psf)
        sim.loadCheckpoint(rst)
        return sim

    def load_il(psf, rst, templates):
        sim = generate_single_im1h_oac_system(psf_file=psf)
        il = ProtexSystem(sim, templates)
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

    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    # wrap system in IonicLiquidSystem
    ionic_liquid = ProtexSystem(simulation, templates)
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


def test_forces():
    simulation = generate_single_im1h_oac_system(use_plugin=False)
    system = simulation.system
    topology = simulation.topology
    force_state = defaultdict(list)  # store bond force
    atom_idxs = {}  # store atom_idxs
    atom_names = {}  # store atom_names
    names = []  # store residue names
    # for force in system.getForces():
    #     print(force)
    #     if type(force).__name__ == "CMAPTorsionForce":
    #         print(force.getNumTorsions())

    # quit()

    # iterate over residues, select the first residue for HOAC and OAC and save the individual bonded forces
    for ridx, r in enumerate(topology.residues()):
        if r.name == "HOAC":  # and ridx == 650:  # match first HOAC residue
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

        if r.name == "OAC":  # and ridx == 150:
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
    # simulation = generate_im1h_oac_system()
    simulation = generate_single_im1h_oac_system(use_plugin=False)
    system = simulation.system
    topology = simulation.topology
    force_state = defaultdict(list)  # store bond force
    atom_idxs = {}  # store atom_idxs
    atom_names = {}  # store atom_names
    names = []  # store residue names

    # iterate over residues, select the first residue for HOAC and OAC and save the individual bonded forces
    for ridx, r in enumerate(topology.residues()):
        if r.name == "HOAC":  # and ridx == 650:  # match first HOAC residue
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

        if r.name == "OAC":  # and ridx == 150:
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
        if r.name == "HOAC":  # and ridx == 650:  # match first HOAC residue
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

        if r.name == "OAC":  # and ridx == 150:
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
    simulation = generate_single_im1h_oac_system(use_plugin=False)
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 2.33}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": -2.33}

    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))

    ionic_liquid = ProtexSystem(simulation, templates)
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
        if r.name == "HOAC":  # and ridx == 650:  # match first HOAC residue
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

        if r.name == "OAC":  # and ridx == 150:
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


###################################
# Basic tests with only a small box
###################################
def test_setup_simulation():
    simulation = generate_small_box(use_plugin=False)
    print("Minimizing...")
    simulation.minimizeEnergy(maxIterations=100)
    system = simulation.system

    nr_of_particles = system.getNumParticles()
    assert nr_of_particles == 1750 + 50  # +lps for im1 im1h


def test_create_IonicLiquidTemplate():
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 2.33}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": -2.33}

    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))

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
    simulation = generate_small_box(use_plugin=False)
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 2.33}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": -2.33}

    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))

    count = defaultdict(int)
    ionic_liquid = ProtexSystem(simulation, templates)
    print(ionic_liquid.boxlength.value_in_unit(nanometers))

    assert len(ionic_liquid.residues) == 100
    for idx, residue in enumerate(ionic_liquid.residues):
        # print(f"{idx} : {residue.original_name}")
        count[residue.original_name] += 1

    assert count["IM1H"] == 15
    assert count["OAC"] == 15
    assert count["IM1"] == 35
    assert count["HOAC"] == 35


def test_save_load_allowedupdates(tmp_path):
    simulation = generate_small_box(use_plugin=False)
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 0.9}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 0.3}

    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    ionic_liquid = ProtexSystem(simulation, templates)

    assert allowed_updates == ionic_liquid.templates.allowed_updates

    ionic_liquid.save_updates(f"{tmp_path}/updates.yaml")
    ionic_liquid.load_updates(f"{tmp_path}/updates.yaml")

    print(ionic_liquid.templates.allowed_updates)

    assert allowed_updates == ionic_liquid.templates.allowed_updates


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Skipping tests that cannot pass in github actions",
)
def test_reporter_class(tmp_path):
    # obtain simulation object
    simulation = generate_small_box()
    # get ionic liquid templates
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 2.33}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": -2.33}

    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    # wrap system in IonicLiquidSystem
    ionic_liquid = ProtexSystem(simulation, templates)
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
    ionic_liquid.simulation.reporters.append(
        EnergyReporter(f"{tmp_path}/energy_1.out", 20)
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
    reason="Skipping tests that cannot pass in github actions",
)
def test_pickle(tmp_path):
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 2.33}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": -2.33}

    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))

    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.1, "prob": 2.33}
    templates.allowed_updates = allowed_updates
    templates.set_update_value_for(frozenset(["IM1H", "OAC"]), "prob", 100)
    templates.overall_max_distance = 2

    templates.dump(f"{tmp_path}/templates.pkl")

    del templates

    templates = ProtexTemplates.load(f"{tmp_path}/templates.pkl")
    assert templates.allowed_updates == allowed_updates
    assert templates.get_update_value_for(frozenset(["IM1H", "OAC"]), "prob") == 100
    assert templates.overall_max_distance == 2

    simulation = generate_small_box()
    ionic_liquid = ProtexSystem(simulation, templates)
    boxl = ionic_liquid.boxlength

    ionic_liquid.dump(f"{tmp_path}/ionicliquid.pkl")

    del ionic_liquid

    ionic_liquid = ProtexSystem.load(f"{tmp_path}/ionicliquid.pkl", simulation)
    assert ionic_liquid.boxlength == boxl


#########################
# Tests with large system
#########################


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
        "im1_d_dummy.str",
        "oac_d_dummy.str",
    ]
    base = f"{protex.__path__[0]}/forcefield"  # NOTE: this points now to the installed files!
    params = CharmmParameterSet(
        *[f"{base}/toppar/{para_files}" for para_files in PARA_FILES]
    )

    psf = CharmmPsfFile(f"{base}/small_box/small_box.psf")
    xtl = 48.0 * angstroms
    psf.setBox(xtl, xtl, xtl)
    # cooridnates can be provieded by CharmmCrdFile, CharmmRstFile or PDBFile classes
    CharmmCrdFile(f"{base}/small_box/small_box.crd")

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


# @pytest.mark.skipif(
#     os.getenv("CI") == "true",
#     reason="Will fail sporadicaly.",
# )
# def test_write_psf_save_load(tmp_path):
#     simulation = generate_im1h_oac_system()
#     # get ionic liquid templates
#     allowed_updates = {}
#     allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.165, "prob": 1}
#     allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.165, "prob": 1}

#     templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
#     # wrap system in IonicLiquidSystem
#     ionic_liquid = ProtexSystem(simulation, templates)
#     # initialize update method
#     update = NaiveMCUpdate(ionic_liquid)
#     # initialize state update class
#     state_update = StateUpdate(update)

#     old_psf_file = f"{protex.__path__[0]}/forcefield/im1h_oac_150_im1_hoac_350.psf"
#     ionic_liquid.write_psf(old_psf_file, f"{tmp_path}/test1.psf")

#     # ionic_liquid.simulation.step(50)
#     state_update.update(2)

#     ionic_liquid.write_psf(old_psf_file, f"{tmp_path}/test2.psf")

#     ionic_liquid.simulation.step(10)
#     state_update.update(2)

#     ionic_liquid.write_psf(old_psf_file, f"{tmp_path}/test3.psf")

#     ionic_liquid.saveState(f"{tmp_path}/state.rst")
#     ionic_liquid.saveCheckpoint(f"{tmp_path}/checkpoint.rst")

#     ionic_liquid2 = ionic_liquid  # copy.deepcopy(ionic_liquid)
#     ionic_liquid.loadState(f"{tmp_path}/state.rst")
#     ionic_liquid2.loadCheckpoint(f"{tmp_path}/checkpoint.rst")


# @pytest.mark.skipif(
#     os.getenv("CI") == "true",
#     reason="Needs local files",
# )
# def test_write_psf_save_load_clap(tmp_path):
#     psf = "/site/raid3/florian/clap/b3lyp/im1h_oac_150_im1_hoac_350.psf"
#     crd = "/site/raid3/florian/clap/b3lyp/im1h_oac_150_im1_hoac_350.crd"
#     PARA_FILES = [
#         "polarizable_flo_dummy_nolj.rtf",
#         "polarizable_flo_dummy_nolj.prm",
#     ]
#     para_files = [
#         f"/site/raid3/florian/clap/toppar/{para_file}" for para_file in PARA_FILES
#     ]

#     simulation = generate_im1h_oac_system_clap(
#         psf_file=psf, crd_file=crd, para_files=para_files, dummy_atom_type="DUM"
#     )
#     # get ionic liquid templates
#     allowed_updates = {}
#     allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.145, "prob": 1}
#     allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.145, "prob": 1}

#     IM1H_IM1 = {"IM1H": {"atom_name": "HN1"}, "IM1": {"atom_name": "NA1"}}
#     OAC_HOAC = {"OAC": {"atom_name": "O1"}, "HOAC": {"atom_name": "H3"}}

#     templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
#     # wrap system in IonicLiquidSystem
#     ionic_liquid = ProtexSystem(simulation, templates)
#     # initialize update method
#     update = NaiveMCUpdate(ionic_liquid)
#     # initialize state update class
#     state_update = StateUpdate(update)

#     old_psf_file = psf
#     ionic_liquid.write_psf(old_psf_file, f"{tmp_path}/test1c.psf")

#     # ionic_liquid.simulation.step(50)
#     state_update.update(2)

#     ionic_liquid.write_psf(old_psf_file, f"{tmp_path}/test2c.psf")

#     ionic_liquid.simulation.step(10)
#     state_update.update(2)

#     ionic_liquid.write_psf(old_psf_file, f"{tmp_path}/test3c.psf")

#     ionic_liquid.saveState(f"{tmp_path}/statec.rst")
#     ionic_liquid.saveCheckpoint(f"{tmp_path}/checkpointc.rst")

#     ionic_liquid2 = ionic_liquid  # copy.deepcopy(ionic_liquid)
#     ionic_liquid.loadState(f"{tmp_path}/statec.rst")
#     ionic_liquid2.loadCheckpoint(f"{tmp_path}/checkpointc.rst")


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Will fail sporadicaly.",
)
def test_dummy(tmp_path):
    def get_time_energy(simulation, print=False):
        time = simulation.context.getState().getTime()
        e_pot = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        if print:
            print(f"time: {time}, e_pot: {e_pot}")
        return time, e_pot

    def save_il(ionic_liquid, number):
        ionic_liquid.write_psf(
            "protex/forcefield/dummy/im1h_oac_im1_hoac_1.psf",
            f"{tmp_path}/test_{number}.psf",
        )
        ionic_liquid.saveCheckpoint(f"{tmp_path}/test_{number}.rst")

    def load_sim(psf, rst):
        sim = generate_single_im1h_oac_system(psf_file=psf)
        sim.loadCheckpoint(rst)
        return sim

    def load_il(psf, rst, templates):
        sim = generate_single_im1h_oac_system(psf_file=psf)
        il = ProtexSystem(sim, templates)
        il.loadCheckpoint(rst)
        return il

    def print_force_contrib(simulation):
        for i, f in enumerate(simulation.system.getForces()):
            group = f.getForceGroup()
            state = simulation.context.getState(getEnergy=True, groups={group})
            print(f.getName(), state.getPotentialEnergy())

    simulation = generate_im1h_oac_dummy_system()
    nonbonded_force = [
        f for f in simulation.system.getForces() if isinstance(f, mm.NonbondedForce)
    ][0]
    dummy_atoms = []
    for atom in simulation.topology.atoms():
        if atom.residue.name == "IM1" and atom.name == "H7":
            dummy_atoms.append(atom.index)
            print(atom)
            print(nonbonded_force.getParticleParameters(atom.index))
            # nonbonded_force.setParticleParameters(atom.index, 0.0, 0.0, 0.0)
            # print(nonbonded_force.getParticleParameters(atom.index))
        if atom.residue.name == "OAC" and atom.name == "H":
            dummy_atoms.append(atom.index)
            print(nonbonded_force.getParticleParameters(atom.index))
            # nonbonded_force.setParticleParameters(atom.index, 0.0, 0.0, 0.0)
    for exc_id in range(nonbonded_force.getNumExceptions()):
        f = nonbonded_force.getExceptionParameters(exc_id)
        idx1 = f[0]
        idx2 = f[1]
        chargeProd, sigma, epsilon = f[2:]
        if idx1 in dummy_atoms or idx2 in dummy_atoms:
            # nonbonded_force.setExceptionParameters(exc_id, idx1, idx2, 0.0, sigma, 0.0)
            print(f)

    # get ionic liquid templates
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.785, "prob": 1}
    # allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.165, "prob": 1}

    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    # wrap system in IonicLiquidSystem
    ionic_liquid = ProtexSystem(simulation, templates)
    # initialize update method
    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)

    save_il(ionic_liquid, 0)
    # for i in range(5):
    #     state_update.update(2)
    #     ionic_liquid.simulation.step(10)
    #     print(ionic_liquid.simulation.context.getPlatform().getName())

    state_update.update(2)

    save_il(ionic_liquid, 1)

    # sim2_1 = load_sim("protex/forcefield/single_pairs/im1_hoac_2.psf", "test_2.rst")
    # sim_2_oldcoord = load_sim(
    #    "protex/forcefield/single_pairs/im1_hoac_2.psf", "test_1.rst"
    # )
