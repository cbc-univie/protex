# from ..testsystems import generate_im1h_oac_system, OAC_HOAC, IM1H_IM1
import os
from sys import stdout

import pytest

from ..system import IonicLiquidSystem, IonicLiquidTemplates
from ..testsystems import (IM1H_IM1_chelpg, OAC_HOAC_chelpg,
                           generate_im1h_oac_system_chelpg)
from ..update import NaiveMCUpdate, StateUpdate


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Skipping tests that cannot pass in github actions",
)
def test_outline_chelpg():
    from simtk.openmm.app import DCDReporter, StateDataReporter

    from ..scripts.ommhelper import DrudeTemperatureReporter

    # obtain simulation object
    simulation = generate_im1h_oac_system_chelpg(coll_freq=10, drude_coll_freq=120)
    allowed_updates = {}
    # allowed updates according to simple protonation scheme
    allowed_updates[frozenset(["IM1H", "OAC"])] = {
        "r_max": 0.16,
        "delta_e": 2.33,
    }  # r_max in nanometer, delta_e in kcal/mol
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "delta_e": -2.33}
    # allowed_updates[set(["IM1H", "IM1"])] = {"r_max": 0.2, "delta_e": 1.78}
    # allowed_updates[set(["HOAC", "OAC"])] = {"r_max": 0.2, "delta_e": 0.68}
    # get ionic liquid templates
    templates = IonicLiquidTemplates(
        [OAC_HOAC_chelpg, IM1H_IM1_chelpg], (allowed_updates)
    )
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)
    ionic_liquid.report_states()
    # initialize update method
    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)
    # ionic_liquid.simulation.minimizeEnergy(maxIterations=200)
    # adding reporter
    ionic_liquid.simulation.reporters.append(DCDReporter("outline_chelpg.dcd", 500))

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
        DrudeTemperatureReporter("drude_temp.out", 500)
    )

    n_steps = 10
    update_steps = 1
    sim_steps = 1000
    print(
        f"Simulation {n_steps} proton transfers with {update_steps} update steps and {sim_steps} simulation steps."
    )
    for _ in range(n_steps):
        print(_)
        ionic_liquid.report_charge_changes(
            "charge_changes_chelpg.json", step=_, n_steps=n_steps
        )
        ionic_liquid.simulation.step(sim_steps)
        state_update.update(update_steps)
        ionic_liquid.report_states()
