# Import package, test suite, and other packages as needed
from sys import stdout
from ..testsystems import generate_im1h_oac_system
from ..system import IonicLiquidSystem, IonicLiqudTemplates
import numpy as np


def test_setup_simulation():
    simulation = generate_im1h_oac_system()
    print("Minimizing...")
    simulation.minimizeEnergy(maxIterations=100)
    system = simulation.system

    nr_of_particles = system.getNumParticles()
    assert nr_of_particles == 17500


def test_run_simulation():
    from simtk.openmm.app import StateDataReporter, PDBReporter, DCDReporter

    simulation = generate_im1h_oac_system()
    print("Minimizing...")
    simulation.minimizeEnergy(maxIterations=100)
    simulation.reporters.append(PDBReporter("output.pdb", 1))
    simulation.reporters.append(DCDReporter("output.dcd", 1))

    simulation.reporters.append(
        StateDataReporter(
            stdout,
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
    simulation.step(200)
    # If simulation aborts with Nan error, try smaller timestep (e.g. 0.0001) and then extract new crd from dcd using "protex/charm_ff/crdfromdcd.inp"


def test_create_IonicLiquidTemplate():
    from ..testsystems import generate_im1h_oac_system, OAC_HOAC, IM1H_IM1

    simulation = generate_im1h_oac_system()
    templates = IonicLiqudTemplates([OAC_HOAC, IM1H_IM1], "charged")

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
    templates = IonicLiqudTemplates([OAC_HOAC, IM1H_IM1], 'charged')
    count = defaultdict(int)
    ionic_liquid = IonicLiquidSystem(simulation, templates)
    assert len(ionic_liquid.residues) == 1000
    for idx, residue in enumerate(ionic_liquid.residues):
        print(f"{idx} : {residue.name}")
        count[residue.name] += 1
    print(count)


def test_create_IonicLiquid_residue():
    from ..testsystems import generate_im1h_oac_system, OAC_HOAC, IM1H_IM1

    simulation = generate_im1h_oac_system()
    templates = IonicLiqudTemplates([OAC_HOAC, IM1H_IM1](set(["IM1H", "OAC"]), set(["IM1", "HOAC"]))

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

    assert (residue.get_idx_for_name("H7")) == 37
