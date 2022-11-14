import numpy as np

try:
    import openmm
    import openmm.unit as unit
except ImportError:
    import simtk.openmm as openmm
    import simtk.unit as unit

from protex.system import ProtexSystem


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
        ionic_liquid: ProtexSystem,
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


class EnergyReporter:
    """Energy reporter reports the energy contributions to the potential energy"""

    def __init__(
        self,
        file: str,
        reportInterval: int,
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
        return (steps, False, False, False, True)

    def report(self, simulation, state):
        tab = "\t"  # becuase: SyntaxError: f-string expression part cannot include a backslash
        names = []
        values = []
        for f in simulation.system.getForces():
            group = f.getForceGroup()
            state = simulation.context.getState(getEnergy=True, groups={group})
            names.append(f"{f.getName()} (kJ/mole)")
            values.append(
                state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            )

        if not self._hasInitialized:
            self._hasInitialized = True
            if isinstance(self.header_data, dict):
                print(
                    ", ".join(
                        [f"{key}: {value}" for key, value in self.header_data.items()]
                    ),
                    file=self._out,
                )
            print(f"Step\t{tab.join(n for n in names)}", file=self._out)

        print(
            f"{simulation.currentStep}\t{tab.join(str(v) for v in values)}",
            file=self._out,
        )

        if hasattr(self._out, "flush") and callable(self._out.flush):
            self._out.flush()

    def __del__(self):
        self._out.close()


# Reporter from https://github.com/z-gong/openmm-velocityVerlet/tree/master/examples/ommhelper/reporter
class DrudeTemperatureReporter(object):
    """
    DrudeTemperatureReporter reports the temperatures of different DOFs in a Drude simulation system.

    The temperatures for three sets of degrees of freedom are reported
    -- molecular center of mass, internal atomic and Drude temperature.
    It's better to set the reportInterval larger than 10000 to avoid performance penalty

    Parameters
    ----------
    file : string
        The file to write to
    reportInterval : int
        The interval (in time steps) at which to write frames
    append : bool
        Whether or not append to existing file
    """

    def __init__(self, file, reportInterval, append=False):
        self._reportInterval = reportInterval
        if append:
            self._out = open(file, "a")
        else:
            self._out = open(file, "w")
        self._hasInitialized = False

    def describeNextReport(self, simulation):
        """Get information about the next report this object will generate.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for

        Returns
        -------
        tuple
            A six element tuple. The first element is the number of steps
            until the next report. The next four elements specify whether
            that report will require positions, velocities, forces, and
            energies respectively.  The final element specifies whether
            positions should be wrapped to lie in a single periodic box.
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, True, False, False)

    def report(self, simulation, state):
        """Generate a report.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """
        system: openmm.System = simulation.system
        if not self._hasInitialized:
            self.n_atom = system.getNumParticles()
            self.molecules = [
                list(atoms) for atoms in simulation.context.getMolecules()
            ]
            self.n_mol = len(self.molecules)
            self.mol_atoms = np.zeros(
                self.n_atom, dtype=int
            )  # record which molecule the atoms are in
            self.mass_molecules = np.zeros(self.n_mol)  # record the mass of molecules
            for i, atoms in enumerate(self.molecules):
                for atom in atoms:
                    self.mol_atoms[atom] = i
                    self.mass_molecules[i] += system.getParticleMass(
                        atom
                    ).value_in_unit(unit.dalton)

            self.dof_com = np.count_nonzero(self.mass_molecules) * 3
            self.dof_atom = self.dof_drude = 0
            for i in range(self.n_atom):
                if system.getParticleMass(i) > 0 * unit.dalton:
                    self.dof_atom += 3
            self.dof_atom -= self.dof_com + system.getNumConstraints()

            if any(type(f) == openmm.CMMotionRemover for f in system.getForces()):
                self.dof_com -= 3

            drude_set = set()
            self.pair_set = set()
            force = next(f for f in system.getForces() if type(f) == openmm.DrudeForce)
            self.dof_atom -= 3 * force.getNumParticles()
            self.dof_drude += 3 * force.getNumParticles()
            for i in range(force.getNumParticles()):
                i_drude, i_core = force.getParticleParameters(i)[:2]
                drude_set.add(i_drude)
                self.pair_set.add((i_drude, i_core))
            self.drude_array = np.array(list(drude_set))
            self.atom_array = np.array(list(set(range(self.n_atom)) - drude_set))

            self._hasInitialized = True
            print(
                '#"Step"\t"T_COM"\t"T_Atom"\t"T_Drude"\t"KE_COM"\t"KE_Atom"\t"KE_Drude"',
                file=self._out,
            )

        velocities = state.getVelocities(asNumpy=True).value_in_unit(
            unit.nanometer / unit.picosecond
        )
        masses = np.array(
            [
                system.getParticleMass(i).value_in_unit(unit.dalton)
                for i in range(self.n_atom)
            ]
        )

        vel_mol = np.zeros([self.n_mol, 3])
        for i, atoms in enumerate(self.molecules):
            if self.mass_molecules[i] == 0:
                continue
            mv = masses[atoms][:, np.newaxis] * velocities[atoms]
            vel_mol[i] = np.sum(mv, axis=0) / self.mass_molecules[i]
        mvv_com = self.mass_molecules * np.sum(vel_mol**2, axis=1)
        ke_com = (
            mvv_com.sum() / 2 * (unit.nanometer / unit.picosecond) ** 2 * unit.dalton
        )
        t_com = 2 * ke_com / (self.dof_com * unit.MOLAR_GAS_CONSTANT_R)

        velocities -= np.array([vel_mol[self.mol_atoms[i]] for i in range(self.n_atom)])
        for i_drude, i_core in self.pair_set:
            v_drude = velocities[i_drude]
            v_core = velocities[i_core]
            m_drude = masses[i_drude]
            m_core = masses[i_core]
            m_com = m_drude + m_core
            m_rel = m_drude * m_core / m_com
            v_com = (m_drude * v_drude + m_core * v_core) / m_com
            v_rel = v_drude - v_core
            velocities[i_drude] = v_rel
            velocities[i_core] = v_com
            masses[i_drude] = m_rel
            masses[i_core] = m_com
        mvv = masses * np.sum(velocities**2, axis=1)
        ke = (
            mvv[self.atom_array].sum()
            / 2
            * (unit.nanometer / unit.picosecond) ** 2
            * unit.dalton
        )
        ke_drude = (
            mvv[self.drude_array].sum()
            / 2
            * (unit.nanometer / unit.picosecond) ** 2
            * unit.dalton
        )
        t = 2 * ke / (self.dof_atom * unit.MOLAR_GAS_CONSTANT_R)
        t_drude = 2 * ke_drude / (self.dof_drude * unit.MOLAR_GAS_CONSTANT_R)
        print(
            simulation.currentStep,
            t_com.value_in_unit(unit.kelvin),
            t.value_in_unit(unit.kelvin),
            t_drude.value_in_unit(unit.kelvin),
            ke_com.value_in_unit(unit.kilojoule_per_mole),
            ke.value_in_unit(unit.kilojoule_per_mole),
            ke_drude.value_in_unit(unit.kilojoule_per_mole),
            sep="\t",
            file=self._out,
        )

        if hasattr(self._out, "flush") and callable(self._out.flush):
            self._out.flush()

    def __del__(self):
        self._out.close()
