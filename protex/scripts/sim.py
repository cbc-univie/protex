from protex.testsystems import generate_im1h_oac_system, OAC_HOAC, IM1H_IM1
from protex.system import IonicLiquidSystem, IonicLiquidTemplates
from protex.update import NaiveMCUpdate, StateUpdate
from sys import stdout


def test_outline():
    from simtk.openmm.app import StateDataReporter, PDBReporter, DCDReporter

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
    ionic_liquid.simulation.minimizeEnergy(maxIterations=1000)
    # adding reporter
    ionic_liquid.simulation.reporters.append(DCDReporter("output.dcd", 500))

    ionic_liquid.simulation.reporters.append(
        StateDataReporter(
            stdout,
            200,
            step=True,
            potentialEnergy=True,
            temperature=True,
            time=False,
            volume=True,
            density=False,
        )
    )
    for _ in range(100):
        print(_)
        ionic_liquid.simulation.step(5000)
        state_update.update(1001)
        ionic_liquid.report_states()


test_outline()
