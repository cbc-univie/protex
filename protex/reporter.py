from protex.system import IonicLiquidSystem


class ChargeReporter:
    """Charge Reporter reports the charges after an update intervals
    Used to reoprt the charge for each molecule in the system.
    Ideally call in conjunction with an update and report just when charge changed.
    i.e.
    ChargeReporter(file, 10)
    simulation.step(9)
    state_update.update(2)
    simulation.step(8)
    state_update.update(2)
    simulation.step(8)
    simulation.step(1)
    then the reporter is invoked directyl in the update, after the one step with the previous charge, but before any charge changes take effect.

    """

    def __init__(
        self,
        file: str,
        reportInterval: int,
        ionic_liquid: IonicLiquidSystem,
        append: bool = False,
        header_data: dict = None,
    ):
        # allow for a header line with information
        self._openedFile = isinstance(file, str)
        if self._openedFile:
            self._out = open(file, "a" if append else "w")
        else:
            self._out = file
        self._hasInitialized = False
        self.header_data = header_data
        self.ionic_liquid = ionic_liquid
        self._reportInterval = reportInterval

    def describeNextReport(self, simulation):
        """Get information about the next report this object will generate.
        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        Returns
        -------
        tuple
            A five element tuple. The first element is the number of steps
            until the next report. The remaining elements specify whether
            that report will require positions, velocities, forces, and
            energies respectively.
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, False, False)

    def report(self, simulation, state):
        if not self._hasInitialized:
            self._hasInitialized = True
            if isinstance(self.header_data, dict):
                print(
                    ", ".join(
                        [f"{key}: {value}" for key, value in self.header_data.items()]
                    ),
                    file=self._out,
                )
            print("Step\tCharges", file=self._out)

        print(
            f"{self.ionic_liquid.simulation.currentStep}\t{[residue.endstate_charge for residue in self.ionic_liquid.residues]}",
            file=self._out,
        )

        if hasattr(self._out, "flush") and callable(self._out.flush):
            self._out.flush()

    def __del__(self):
        self._out.close()
