from sys import stdout

from protex.system import IonicLiquidSystem, IonicLiquidTemplates
from protex.testsystems import IM1H_IM1, OAC_HOAC, generate_im1h_oac_system
from protex.update import NaiveMCUpdate, StateUpdate


def test_outline():
    from simtk.openmm.app import DCDReporter, PDBReporter, StateDataReporter

    from protex.reporter import DrudeTemperatureReporter

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
    ionic_liquid.simulation.reporters.append(
        DCDReporter("output_chelpg_1_d80.dcd", 500)
    )

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
        DrudeTemperatureReporter("drude_temp_1_d80.out", 500)
    )

    n_steps = 10000

    for _ in range(n_steps):
        print(_)
        ionic_liquid.report_charge_changes(
            "charge_changes_chelpg_1_d80.json", step=_, tot_steps=n_steps
        )
        ionic_liquid.simulation.step(10000)
        state_update.update(1)
        #ionic_liquid.report_states()

    # ionic_liquid.charge_changes_to_json("charge_changes_chelpg_1.json", append=False)


test_outline()
