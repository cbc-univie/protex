# Import package, test suite, and other packages as needed
from sys import stdout
from ..testsystems import generate_im1h_oac_system
from ..testsystems import generate_im1h_oac_system_chelpg
from ..system import IonicLiquidSystem, IonicLiquidTemplates
import numpy as np


def test_setup_simulation():
    simulation = generate_im1h_oac_system()
    print("Minimizing...")
    simulation.minimizeEnergy(maxIterations=100)
    system = simulation.system

    nr_of_particles = system.getNumParticles()
    assert nr_of_particles == 17500 + 500  # +lps for im1 im1h


def test_setup_simulation_chelpg():
    simulation = generate_im1h_oac_system_chelpg()
    # print(simulation.context.getPeriodicBoxVectors())
    print("Minimizing...")
    simulation.minimizeEnergy(maxIterations=100)
    system = simulation.system

    nr_of_particles = system.getNumParticles()
    assert nr_of_particles == 17500 + 500  # +lps for im1 im1h


def test_run_simulation():
    from simtk.openmm.app import StateDataReporter, PDBReporter, DCDReporter

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
    # If simulation aborts with Nan error, try smaller timestep (e.g. 0.0001 ps) and then extract new crd from dcd using "protex/charmm_ff/crdfromdcd.inp"


def test_create_IonicLiquidTemplate():
    from ..testsystems import generate_im1h_oac_system, OAC_HOAC, IM1H_IM1

    templates = IonicLiquidTemplates(
        [OAC_HOAC, IM1H_IM1], (set(["IM1H", "OAC"]), set(["IM1", "HOAC"]))
    )

    print(templates.states)
    r = templates.get_residue_name_for_coupled_state("OAC")
    assert r == "HOAC"
    r = templates.get_residue_name_for_coupled_state("HOAC")
    assert r == "OAC"
    r = templates.get_residue_name_for_coupled_state("IM1H")
    assert r == "IM1"


def test_create_IonicLiquid():
    from ..testsystems import generate_im1h_oac_system, OAC_HOAC, IM1H_IM1
    from collections import defaultdict

    simulation = generate_im1h_oac_system()
    templates = IonicLiquidTemplates(
        [OAC_HOAC, IM1H_IM1], (set(["IM1H", "OAC"]), set(["IM1", "HOAC"]))
    )
    count = defaultdict(int)
    ionic_liquid = IonicLiquidSystem(simulation, templates)
    assert len(ionic_liquid.residues) == 1000
    for idx, residue in enumerate(ionic_liquid.residues):
        print(f"{idx} : {residue.original_name}")
        count[residue.original_name] += 1
    print(count)


def test_residues():
    from ..testsystems import generate_im1h_oac_system, OAC_HOAC, IM1H_IM1

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
    from ..testsystems import generate_im1h_oac_system, OAC_HOAC, IM1H_IM1
    from collections import defaultdict

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


def test_drude_forces():
    from ..testsystems import generate_im1h_oac_system, OAC_HOAC, IM1H_IM1
    from collections import defaultdict
    import simtk.openmm as mm

    simulation = generate_im1h_oac_system()
    templates = IonicLiquidTemplates(
        [OAC_HOAC, IM1H_IM1], (set(["IM1H", "OAC"]), set(["IM1", "HOAC"]))
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
                parent1, parent2 = ionic_liquid.pair_12_13_list[drude_id]
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
    from ..testsystems import generate_im1h_oac_system, OAC_HOAC, IM1H_IM1

    simulation = generate_im1h_oac_system()
    templates = IonicLiquidTemplates(
        [OAC_HOAC, IM1H_IM1], (set(["IM1H", "OAC"]), set(["IM1", "HOAC"]))
    )

    ionic_liquid = IonicLiquidSystem(simulation, templates)
    assert len(ionic_liquid.residues) == 1000

    residue = ionic_liquid.residues[0]
    charge = residue.current_charge

    assert charge == 1
    print(residue.atom_names)
    assert (residue.get_idx_for_atom_name("H7")) == 18

    residue = ionic_liquid.residues[1]

    assert (residue.get_idx_for_atom_name("H7")) == 38

    # check name of first residue
    assert ionic_liquid.residues[0].current_name == "IM1H"
    assert ionic_liquid.residues[0].original_name == "IM1H"


def test_report_charge_changes():
    import json
    from ..testsystems import generate_im1h_oac_system, OAC_HOAC, IM1H_IM1
    from ..update import NaiveMCUpdate, StateUpdate

    # obtain simulation object
    simulation = generate_im1h_oac_system()
    # get ionic liquid templates
    templates = IonicLiquidTemplates(
        [OAC_HOAC, IM1H_IM1], (set(["IM1H", "OAC"]), set(["IM1", "HOAC"]))
    )
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)
    # initialize update method
    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)

    n_steps = 2
    ionic_liquid.report_charge_changes(
        filename="test_charge_changes.dat", step=0, n_steps=n_steps
    )
    state_update.update()
    ionic_liquid.simulation.step(1000)
    ionic_liquid.report_charge_changes(
        filename="test_charge_changes.dat", step=1, n_steps=n_steps
    )

    # ionic_liquid.charge_changes_to_json("test.json", append=False)

    with open("test_charge_changes.dat", "r") as json_file:
        data = json.load(json_file)
    print(data)
    # test if dict after writing and reading json stays same
    assert len(data["charges_at_step"]["0"]) == 1000
    assert len(data["charges_at_step"]["1"]) == 1000
