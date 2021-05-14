from sys import stdout
from protex.testsystems import generate_im1h_oac_system
from simtk import unit
import numpy as np


def test_perform_charge_muatation():
    simulation = generate_im1h_oac_system()
    system = simulation.system

    # iterating over forces
    forces = []
    for idx in range(system.getNumForces()):
        forces.append(type(system.getForce(idx)).__name__)

    assert set(forces) == set(
        [
            "HarmonicBondForce",
            "HarmonicAngleForce",
            "PeriodicTorsionForce",
            "CustomTorsionForce",
            "CMAPTorsionForce",
            "NonbondedForce",
            "DrudeForce",
            "CMMotionRemover",
        ]
    )

    # start with manually performing a protonation state update
    # update charge at particle idx=1
    update_atom_idx = 1
    for idx in range(system.getNumForces()):
        # extract nonbonded force
        if (type(system.getForce(idx)).__name__) != "NonbondedForce":
            continue

        nonbonded_force = system.getForce(idx)
        # extracting current charge at atom_idx
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(update_atom_idx)
        assert np.isclose(charge.value_in_unit(unit.elementary_charge), -3.1819)
        # updating to new charge at atom_idx
        

    assert False
