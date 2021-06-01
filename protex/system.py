import sys
import itertools
import numpy as np
import logging

logger = logging.getLogger(__name__)


class IonicLiqudTemplates:
    def __init__(self, states: list) -> None:
        self.states = states  #
        self.names = list(itertools.chain(*[list(k.keys()) for k in states]))
        self.pairs = [
            (self.names[0], self.names[2]),
            (self.names[1], self.names[3]),
        ]  # NOTE: this needs a specific order of the states

    def get_residue_name_for_other_charged_state(self, name: str):
        """
        get_residue_name_of_paired_ion returns the paired residue name given a reisue name

        Parameters
        ----------
        name : str
            residue name

        Returns
        -------
        str

        Raises
        ------
        RuntimeError
            is raised if no paired residue name can be found
        """
        assert name in self.names

        for template in self.states:
            if name in template.keys():
                state_1, state_2 = template.keys()
                if state_1 == name:
                    return state_2
                else:
                    return state_1
        else:
            raise RuntimeError("something went wrong")

    # def is_ionic_pair(self, name1, name2) -> bool:
    #     assert name1 in self.names and name2 in self.names

    #     for template in self.states:
    #         if set([name1, name2]) == set(template.keys()):
    #             return True
    #     else:
    #         return False

    def get_charge_template_for(self, name: str):
        """
        get_template_for returns the charge template for a residue

        Parameters
        ----------
        name : str
            Name of the residue

        Returns
        -------
        list
            charge state of residue

        Raises
        ------
        RuntimeError
            [description]
        """
        assert name in self.names

        for template in self.states:
            for state in template:
                if state == name:
                    return template[state]
        else:
            raise RuntimeError("something went wrong")


class Residue:
    def __init__(
        self,
        residue,
        nonbonded_force,
        initial_charge: list,
        alternative_charge: list,
    ) -> None:

        assert len(initial_charge) == len(alternative_charge)

        self.residue = residue
        self.name = residue.name
        self.atom_idxs = [atom.index for atom in residue.atoms()]
        self.nonbonded_force = nonbonded_force
        self.record_charge_state = []

        # generate charge set as dict with charge as key {0 : charges, -1: charges}
        self.charge_sets = {
            int(np.round(np.sum(initial_charge), 2)): initial_charge,
            int(np.round(np.sum(alternative_charge), 2)): alternative_charge,
        }

        assert len(set(list(self.charge_sets.keys()))) == 2
        # calculate initial charge
        intial_charge = int(
            np.round(
                sum(
                    [
                        self.nonbonded_force.getParticleParameters(idx)[0]._value
                        for idx in self.atom_idxs
                    ]
                ),
                4,
            )
        )
        assert intial_charge in list(self.charge_sets.keys())
        self.record_charge_state.append(intial_charge)

    def get_current_charge(self) -> int:
        return self.record_charge_state[-1]

    def get_current_charges(self) -> list:
        return self.charge_sets[self.record_charge_state[-1]]

    def get_inactive_charges(self) -> list:
        """
        get_inactive_charges retrieves the charges that are not active in the residue

        Returns
        -------
        list
            list of charges

        Raises
        ------
        RuntimeError
            if none found
        """
        current_charge = self.get_current_charge()
        for charge in self.charge_sets:
            if charge != current_charge:
                return self.charge_sets[charge]

        else:
            raise RuntimeError()

    def set_new_charge(self, new_charges: list) -> None:
        from simtk import unit

        for new_charge, idx in zip(new_charges, self.atom_idxs):
            (
                _,
                old_sigma,
                old_epsilon,
            ) = self.nonbonded_force.getParticleParameters(idx)
            self.nonbonded_force.setParticleParameters(
                idx, new_charge * unit.elementary_charge, old_sigma, old_epsilon
            )


class IonicLiquidSystem:
    """
    This class defines the full system, performs the MD steps and offers an
    interface for protonation state updates.
    """

    def __init__(self, simulation, templates: IonicLiqudTemplates) -> None:
        self.system = simulation.system
        self.topology = simulation.topology
        self.simulation = simulation
        self.templates = templates
        for force in self.system.getForces():
            if (type(force).__name__) == "NonbondedForce":
                self.nonbonded_force = force

        self.residues = self._set_initial_states()

    def _set_initial_states(self) -> list:
        """
        set_initial_states For each ionic liquid residue in the system the protonation state
        is interfered from the provided openMM system object and the protonation site is defined.
        """
        residues = []

        for r in self.topology.residues():
            name = r.name
            if name in self.templates.names:
                name_of_paired_ion = (
                    self.templates.get_residue_name_for_other_charged_state(name)
                )
                residues.append(
                    Residue(
                        r,
                        self.nonbonded_force,
                        self.templates.get_charge_template_for(name),
                        self.templates.get_charge_template_for(name_of_paired_ion),
                    )
                )

            else:
                raise RuntimeError("Found resiude not present in Templates: {r.name}")

        return residues

    def report_states(self) -> None:
        """
        report_states prints out a summary of the current protonation state of the ionic liquid
        """
        pass
