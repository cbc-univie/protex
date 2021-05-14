# Import package, test suite, and other packages as needed
import protex
import pytest
import sys
from sys import stdout
from protex.testsystems import generate_im1h_oac_system


def test_setup_simulation():
    simulation = generate_im1h_oac_system()
    print("Minimizing...")
    simulation.minimizeEnergy(maxIterations=100)
    system = simulation.system

    nr_of_particles = system.getNumParticles()
    assert nr_of_particles == 17350


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
    simulation.step(10)
