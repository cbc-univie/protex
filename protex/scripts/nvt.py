from ..testsystems import generate_im1h_oac_system, OAC_HOAC, IM1H_IM1
from ..system import IonicLiquidSystem, IonicLiqudTemplates
from ..update import NaiveMCUpdate, StateUpdate
from sys import stdout


def main():
    from simtk.openmm.app import StateDataReporter, PDBReporter, DCDReporter

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
    ionic_liquid.simulation.minimizeEnergy(maxIterations=5000)
    # adding reporter
    #ionic_liquid.simulation.reporters.append(PDBReporter("output.pdb", 100))
    ionic_liquid.simulation.reporter.append(DCDReporter("traj/output.dcd", 100))

    ionic_liquid.simulation.reporters.append(
        StateDataReporter(
            "out/output.out",
            100,
            step=True,
            time=False,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
            density=False,
        )
    )
    for _ in range(20):
        print(_)
        ionic_liquid.simulation.step(10000) #5ps
        state_update.update()
        #ionic_liquid.report_states()
        ionic_liquid.report_charge_changes()
    
    ionic_liquid.charge_changes_to_json("charge_changes.json", append=False)

if __name__ == "main":
    main()
