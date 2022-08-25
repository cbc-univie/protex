import os
from sys import stdout

import pytest

from ..system import IonicLiquidSystem, IonicLiquidTemplates
from ..reporter import ChargeReporter
from ..testsystems import (
    IM1H_IM1,
    OAC_HOAC,
    generate_im1h_oac_system,
)
from ..update import NaiveMCUpdate, StateUpdate

try:
    from openmm.app import DCDReporter, StateDataReporter
except ImportError:
    from simtk.openmm.app import DCDReporter, StateDataReporter


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Skipping tests that cannot pass in github actions",
)
def test_outline():

    from ..scripts.ommhelper import DrudeTemperatureReporter

    # obtain simulation object
    simulation = generate_im1h_oac_system(coll_freq=10, drude_coll_freq=120)
    allowed_updates = {}
    # allowed updates according to simple protonation scheme
    allowed_updates[frozenset(["IM1H", "OAC"])] = {
        "r_max": 0.16,
        "prob": 1,
    }  # r_max in nanometer, prob between 0 and 1
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 1}
    # allowed_updates[set(["IM1H", "IM1"])] = {"r_max": 0.2, "prob": 1}
    # allowed_updates[set(["HOAC", "OAC"])] = {"r_max": 0.2, "prob": 1}
    # get ionic liquid templates
    templates = IonicLiquidTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)
    ionic_liquid.report_states()
    # initialize update method
    update = NaiveMCUpdate(ionic_liquid, all_forces=True)
    # initialize state update class
    state_update = StateUpdate(update)
    # ionic_liquid.simulation.minimizeEnergy(maxIterations=200)
    # adding reporter
    ionic_liquid.simulation.reporters.append(DCDReporter("outline1.dcd", 500))

    ionic_liquid.simulation.reporters.append(
        StateDataReporter(
            stdout,
            500,
            step=True,
            potentialEnergy=True,
            temperature=True,
            time=True,
            volume=True,
            density=False,
        )
    )
    ionic_liquid.simulation.reporters.append(
        DrudeTemperatureReporter("drude_temp1.out", 500)
    )

    charge_info = {"dcd_save_freq": 500}
    charge_reporter = ChargeReporter(
        stdout, 1000, ionic_liquid, header_data=charge_info
    )
    ionic_liquid.simulation.reporters.append(charge_reporter)

    n_steps = 10
    update_steps = 2
    sim_steps = 1000
    print(
        f"Simulation {n_steps} proton transfers with {update_steps} update steps and {sim_steps} simulation steps."
    )
    ionic_liquid.simulation.step(int(sim_steps - update_steps / 2))
    for step in range(1, n_steps):
        print(step)
        state_update.update(update_steps)
        ionic_liquid.simulation.step(int(sim_steps - update_steps))
    ionic_liquid.simulation.step(int(update_steps / 2))

    os.remove("outline1.dcd")
    os.remove("drude_temp1.out")
