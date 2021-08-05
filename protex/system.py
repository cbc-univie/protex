import itertools
import numpy as np
import logging
from collections import ChainMap

logger = logging.getLogger(__name__)


class IonicLiquidTemplates:
    def __init__(self, states: list, allowed_updates: tuple) -> None:

        self.pairs = [list(i.keys()) for i in states]
        self.states = dict(ChainMap(*states))  #
        self.names = list(itertools.chain(*self.pairs))
        self.allowed_updates = allowed_updates

    def get_canonical_name(self, name: str) -> str:
        assert name in self.names
        for state in self.states:
            if name in state:
                return self.states[name]["canonical_name"]

    def get_residue_name_for_coupled_state(self, name: str):
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

        for pair in self.pairs:
            if name in pair:
                state_1, state_2 = pair
                if state_1 == name:
                    return state_2
                else:
                    return state_1
        else:
            raise RuntimeError("something went wrong")

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

        return self.states[name]["charge"]


class Residue:
    def __init__(
        self,
        residue,
        nonbonded_force,
        initial_partial_charges: list,
        alternative_partial_charges: list,
        canonical_name: str,
    ) -> None:

        assert len(initial_partial_charges) == len(alternative_partial_charges)

        self.residue = residue
        self.original_name = residue.name
        self.current_name = ""
        self.atom_idxs = [atom.index for atom in residue.atoms()]
        self.atom_names = [atom.name for atom in residue.atoms()]
        self.nonbonded_force = nonbonded_force
        self.record_charge_state = []
        self.canonical_name = canonical_name

        # generate charge set as dict with charge as key {0 : charges, -1: charges}
        self.charge_sets = {
            int(np.round(np.sum(initial_partial_charges), 2)): initial_partial_charges,
            int(
                np.round(np.sum(alternative_partial_charges), 2)
            ): alternative_partial_charges,
        }

        assert len(set(list(self.charge_sets.keys()))) == 2
        # calculate initial charge
        intial_charge = self._get_current_charge()
        self.record_charge_state.append(intial_charge)

    # NOTE: this is a bug!
    def get_idx_for_name(self, name: str):
        for idx, atom_name in zip(self.atom_idxs, self.atom_names):
            if name == atom_name:
                return idx
        else:
            raise RuntimeError()

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

    def _get_current_charge(self) -> int:
        charge = int(
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
        assert charge in list(self.charge_sets.keys())
        return charge

    def set_new_state(self, new_charges: list, new_res_name: str) -> None:
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
        self.current_name = new_res_name
        self.record_charge_state.append(self._get_current_charge())


class IonicLiquidSystem:
    """
    This class defines the full system, performs the MD steps and offers an
    interface for protonation state updates.
    """

    def __init__(self, simulation, templates: IonicLiquidTemplates) -> None:
        self.system = simulation.system
        self.topology = simulation.topology
        self.simulation = simulation
        self.templates = templates
        for force in self.system.getForces():
            if (type(force).__name__) == "NonbondedForce":
                self.nonbonded_force = force

        self.residues = self._set_initial_states()

        # Should this be here or somewhere else? (needed for report_charge_changes)
        self.charge_changes = {}
        self.charge_changes[
            "dcd_save_freq"
        ] = 100  # this number should automatically be fetched from input somehow form dcdreporter
        self.charge_changes["charges_at_step"] = {}

    def _set_initial_states(self) -> list:
        """
        set_initial_states For each ionic liquid residue in the system the protonation state
        is interfered from the provided openMM system object and the protonation site is defined.
        """
        residues = []

        for r in self.topology.residues():
            name = r.name
            if name in self.templates.names:
                name_of_paired_ion = self.templates.get_residue_name_for_coupled_state(
                    name
                )
                residues.append(
                    Residue(
                        r,
                        self.nonbonded_force,
                        self.templates.get_charge_template_for(name),
                        self.templates.get_charge_template_for(name_of_paired_ion),
                        self.templates.get_canonical_name(name),
                    )
                )
                residues[-1].current_name = name

            else:
                raise RuntimeError("Found resiude not present in Templates: {r.name}")
        return residues

    def report_states(self) -> None:
        """
        report_states prints out a summary of the current protonation state of the ionic liquid
        """
        pass

    def report_charge_changes(self, step=0):
        """
        report_charge_changes reports the current charges after each update step in a dictionary format:
        {"step": [residue_charges]}
        additional header data is the dcd save frequency needed for later reconstruction of the charges at different steps
        """
        self.charge_changes["charges_at_step"][str(step)] = [
            residue.get_current_charge() for residue in self.residues
        ]

    def charge_changes_to_json(self, filename, append=False):
        """
        charge_changes_to_json writes the charge_chages dictionary constructed with report_charge_changes to a json file
        """
        import json

        if append:
            mode = "r+"
        else:
            mode = "w"

        with open(filename, mode) as f:
            json.dump(self.charge_changes, f)
