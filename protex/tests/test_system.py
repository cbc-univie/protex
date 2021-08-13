# Import package, test suite, and other packages as needed
from sys import stdout
from ..testsystems import generate_im1h_oac_system
from ..system import IonicLiquidSystem, IonicLiquidTemplates
import numpy as np


def test_setup_simulation():
    simulation = generate_im1h_oac_system()
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
    simulation.reporters.append(PDBReporter("output.pdb", 1))
    simulation.reporters.append(DCDReporter("output.dcd", 1))

    simulation.reporters.append(
        StateDataReporter(
            stdout,
            5,
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

    simulation = generate_im1h_oac_system()
    templates = IonicLiquidTemplates(
        simulation, [OAC_HOAC, IM1H_IM1], (set(["IM1H", "OAC"]), set(["IM1", "HOAC"]))
    )

    print(templates.states)
    r = templates.get_charge_template_for("OAC")
    assert r == [
        3.1817,
        -2.4737,
        2.9879,
        -3.1819,
        0.004,
        0.004,
        0.004,
        2.0548,
        -2.0518,
        2.0548,
        -2.0518,
        0,
        -0.383,
        -0.383,
        -0.383,
        -0.383,
    ]
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
    templates = IonicLiquidTemplates(
        [OAC_HOAC, IM1H_IM1], (set(["IM1H", "OAC"]), set(["IM1", "HOAC"]))
    )

    system = simulation.system
    topology = simulation.topology
    for idx, r in enumerate(topology.residues()):
        if r.name == "IM1H" and idx == 0:
            atom_idxs = [atom.index for atom in r.atoms()]
            atom_names = [atom.name for atom in r.atoms()]
            print(atom_idxs)
            print(atom_names)
            for force in system.getForces():
                # print(type(force).__name__)
                if type(force).__name__ == "HarmonicBondForce":
                    for bond_id in range(force.getNumBonds()):
                        f = force.getBondParameters(bond_id)
                        idx1 = f[0]
                        idx2 = f[1]
                        if idx1 in atom_idxs and idx2 in atom_idxs:
                            print(f)
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
            for force in system.getForces():
                # print(type(force).__name__)
                if type(force).__name__ == "HarmonicBondForce":
                    for bond_id in range(force.getNumBonds()):
                        f = force.getBondParameters(bond_id)
                        idx1 = f[0]
                        idx2 = f[1]
                        if idx1 in atom_idxs and idx2 in atom_idxs:
                            print(f)
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

    simulation = generate_im1h_oac_system()
    templates = IonicLiquidTemplates(
        [OAC_HOAC, IM1H_IM1], (set(["IM1H", "OAC"]), set(["IM1", "HOAC"]))
    )

    system = simulation.system
    topology = simulation.topology
    for idx, r in enumerate(topology.residues()):
        if r.name == "HOAC" and idx == 650:
            atom_idxs = [atom.index for atom in r.atoms()]
            atom_names = [atom.name for atom in r.atoms()]
            print(atom_idxs)
            print(atom_names)
            bf = []
            for force in system.getForces():
                # print(type(force).__name__)
                if type(force).__name__ == "HarmonicBondForce":
                    for bond_id in range(force.getNumBonds()):
                        f = force.getBondParameters(bond_id)
                        idx1 = f[0]
                        idx2 = f[1]
                        if idx1 in atom_idxs and idx2 in atom_idxs:
                            bf.append(f)
            assert len(bf) == 17
        if r.name == "OAC" and idx == 150:
            atom_idxs = [atom.index for atom in r.atoms()]
            atom_names = [atom.name for atom in r.atoms()]
            print(atom_idxs)
            print(atom_names)
            bf = []
            for force in system.getForces():
                # print(type(force).__name__)
                if type(force).__name__ == "HarmonicBondForce":
                    for bond_id in range(force.getNumBonds()):
                        f = force.getBondParameters(bond_id)
                        idx1 = f[0]
                        idx2 = f[1]
                        if idx1 in atom_idxs and idx2 in atom_idxs:
                            bf.append(f)
            assert len(bf) == 17


def test_create_IonicLiquid_residue():
    from ..testsystems import generate_im1h_oac_system, OAC_HOAC, IM1H_IM1

    simulation = generate_im1h_oac_system()
    templates = IonicLiquidTemplates(
        [OAC_HOAC, IM1H_IM1], (set(["IM1H", "OAC"]), set(["IM1", "HOAC"]))
    )

    ionic_liquid = IonicLiquidSystem(simulation, templates)
    assert False
    assert len(ionic_liquid.residues) == 1000

    residue = ionic_liquid.residues[0]
    charge = residue.current_charge
    charges = residue.get_current_charges()
    inactive_charges = residue.get_inactive_charges()

    assert charge == 1
    assert charges != inactive_charges
    assert len(charges) == len(inactive_charges)
    assert np.isclose(charge, np.sum(charges))
    assert np.isclose(0.0, np.sum(inactive_charges))

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
    # initialize state report
    ionic_liquid.report_charge_changes(step=0)
    # initialize update method
    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)

    state_update.update()
    ionic_liquid.report_charge_changes(step=1)
    ionic_liquid.simulation.step(1000)
    ionic_liquid.report_charge_changes(step=1000)

    # check correct amount of charge entries per step
    assert len(ionic_liquid.charge_changes["charges_at_step"]["0"]) == 1000
    assert len(ionic_liquid.charge_changes["charges_at_step"]["1"]) == 1000
    assert len(ionic_liquid.charge_changes["charges_at_step"]["1000"]) == 1000

    ionic_liquid.charge_changes_to_json("test.json", append=False)

    with open("test.json", "r") as json_file:
        data = json.load(json_file)

    # test if dict after writing and reading json stays same
    assert data == ionic_liquid.charge_changes
