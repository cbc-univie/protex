from ..testsystems import generate_im1h_oac_system, OAC_HOAC, IM1H_IM1
from ..system import IonicLiquidSystem, IonicLiquidTemplates
from ..update import NaiveMCUpdate, StateUpdate
from sys import stdout
import logging


def test_outline(caplog):
    from simtk.openmm.app import StateDataReporter, PDBReporter, DCDReporter

    caplog.set_level(logging.DEBUG)

    # obtain simulation object
    simulation = generate_im1h_oac_system()
    # get ionic liquid templates
    templates = IonicLiquidTemplates(
        [OAC_HOAC, IM1H_IM1], (set(["IM1H", "OAC"]), set(["IM1", "HOAC"]))
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
    ionic_liquid.simulation.reporters.append(PDBReporter("output.pdb", 100))

    ionic_liquid.simulation.reporters.append(
        StateDataReporter(
            stdout,
            100,
            step=True,
            potentialEnergy=True,
            temperature=True,
            time=False,
            volume=True,
            density=False,
        )
    )
    for _ in range(10):
        print(_)
        ionic_liquid.simulation.step(1000)
        state_update.update(101)
        ionic_liquid.report_states()
