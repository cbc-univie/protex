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
    ionic_liquid.simulation.minimizeEnergy(maxIterations=100)
    # adding reporter
    ionic_liquid.simulation.reporters.append(DCDReporter("output.dcd", 200))

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
    n_steps = 500
    for _ in range(n_steps):
        print(_)
        ionic_liquid.report_charge_changes(
            "charge_changes.json", step=_, n_steps=n_steps
        )
        ionic_liquid.simulation.step(10000)
        state_update.update(101)


test_outline()
