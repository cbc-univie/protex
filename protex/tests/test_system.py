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
        [OAC_HOAC, IM1H_IM1], (set(["IM1H", "OAC"]), set(["IM1", "HOAC"]))
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


def test_create_IonicLiquid_residue():
    from ..testsystems import generate_im1h_oac_system, OAC_HOAC, IM1H_IM1

    simulation = generate_im1h_oac_system()
    templates = IonicLiquidTemplates(
        [OAC_HOAC, IM1H_IM1], (set(["IM1H", "OAC"]), set(["IM1", "HOAC"]))
    )

    ionic_liquid = IonicLiquidSystem(simulation, templates)
    assert len(ionic_liquid.residues) == 1000

    residue = ionic_liquid.residues[0]
    charge = residue.get_current_charge()
    charges = residue.get_current_charges()
    inactive_charges = residue.get_inactive_charges()

    assert charge == 1
    assert charges != inactive_charges
    assert len(charges) == len(inactive_charges)
    assert np.isclose(charge, np.sum(charges))
    assert np.isclose(0.0, np.sum(inactive_charges))

    print(residue.atom_names)
    assert (residue.get_idx_for_name("H7")) == 18

    residue = ionic_liquid.residues[1]

    assert (residue.get_idx_for_name("H7")) == 38


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
