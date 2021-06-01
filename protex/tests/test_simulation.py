from ..testsystems import generate_im1h_oac_system, OAC_HOAC, IM1H_IM1
from ..system import IonicLiquidSystem, IonicLiqudTemplates
from ..update import NaiveMCUpdate, StateUpdate
from sys import stdout
from simtk.openmm.app import StateDataReporter


def test_outline():

    # obtain simulation object
    simulation = generate_im1h_oac_system()
    # get ionic liquid templates
    templates = IonicLiqudTemplates([OAC_HOAC, IM1H_IM1])
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)
    ionic_liquid.report_states()
    # initialize update method
    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)
    ionic_liquid.simulation.minimizeEnergy(maxIterations=100)
    # adding reporter
    ionic_liquid.simulation.reporters.append(
        StateDataReporter(
            stdout,
            10,
            step=True,
            potentialEnergy=True,
            temperature=True,
            time=False,
            volume=True,
            density=False,
        )
    )
    for _ in range(5):
        print(_)
        ionic_liquid.simulation.step(100)
        state_update.update()
        ionic_liquid.report_states()
