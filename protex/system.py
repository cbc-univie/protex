import itertools
import numpy as np
import logging
from collections import ChainMap, defaultdict
from simtk.openmm.app import Simulation

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
        alternativ_name,
        inital_forces,
        alternativ_forces,
        canonical_name: str,
    ) -> None:

        self.residue = residue
        self.original_name = residue.name
        self.current_name = self.original_name
        self.atom_idxs = [atom.index for atom in residue.atoms()]
        self.atom_names = [atom.name for atom in residue.atoms()]
        self.forces = {
            self.original_name: inital_forces,
            alternativ_name: alternativ_forces,
        }
        self.record_charge_state = []
        self.canonical_name = canonical_name

        self.record_charge_state.append(self.current_charge)

    # NOTE: this is a bug!
    def get_idx_for_atom_name(self, query_atom_name: str):
        for idx, atom_name in zip(self.atom_idxs, self.atom_names):
            if query_atom_name == atom_name:
                return idx
        else:
            raise RuntimeError()

    @property
    def current_charge(self) -> int:
        charge = int(
            np.round(
                sum(
                    [
                        parm[0]._value
                        for parm in self.forces[self.current_name]["NonbondedForce"]
                    ]
                ),
                4,
            )
        )

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
        self.record_charge_state.append(self._current_charge)


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

        self.residues = self._set_initial_states()

        # Should this be here or somewhere else? (needed for report_charge_changes)
        self.charge_changes = {}
        self.charge_changes[
            "dcd_save_freq"
        ] = 100  # this number should automatically be fetched from input somehow form dcdreporter
        self.charge_changes["charges_at_step"] = {}

    def _extract_templates(self, query_name: str) -> defaultdict:

        forces_dict = defaultdict(list)
        # HarmonicBondForce
        # HarmonicAngleForce
        # HarmonicBondForce
        # PeriodicTorsionForce
        # CustomTorsionForce
        # CMAPTorsionForce      # this is currently excluded since no force is defined
        # NonbondedForce
        # DrudeForce            # TODO
        # CMMotionRemover       # this can be ignored

        for residue in self.topology.residues():
            if query_name == residue.name:
                atom_idxs = [atom.index for atom in residue.atoms()]
                atom_names = [atom.name for atom in residue.atoms()]
                logger.debug(atom_idxs)
                logger.debug(atom_names)

                for force in self.system.getForces():
                    # print(type(force).__name__)
                    if type(force).__name__ == "NonbondedForce":
                        forces_dict[type(force).__name__] = [
                            force.getParticleParameters(idx) for idx in atom_idxs
                        ]

                    if type(force).__name__ == "HarmonicBondForce":
                        for bond_id in range(force.getNumBonds()):
                            f = force.getBondParameters(bond_id)
                            idx1 = f[0]
                            idx2 = f[1]
                            if idx1 in atom_idxs and idx2 in atom_idxs:
                                forces_dict[type(force).__name__].append(f)

                    if type(force).__name__ == "HarmonicAngleForce":
                        for angle_id in range(force.getNumAngles()):
                            f = force.getAngleParameters(angle_id)
                            if (
                                f[0] in atom_idxs
                                and f[1] in atom_idxs
                                and f[2] in atom_idxs
                            ):
                                forces_dict[type(force).__name__].append(f)

                    if type(force).__name__ == "PeriodicTorsionForce":
                        for torsion_id in range(force.getNumTorsions()):
                            f = force.getTorsionParameters(torsion_id)
                            if (
                                f[0] in atom_idxs
                                and f[1] in atom_idxs
                                and f[2] in atom_idxs
                                and f[3] in atom_idxs
                            ):
                                forces_dict[type(force).__name__].append(f)

                    if type(force).__name__ == "CustomTorsionForce":
                        for torsion_id in range(force.getNumTorsions()):
                            f = force.getTorsionParameters(torsion_id)

                            if (
                                f[0] in atom_idxs
                                and f[1] in atom_idxs
                                and f[2] in atom_idxs
                                and f[3] in atom_idxs
                            ):
                                forces_dict[type(force).__name__].append(f)
                                # print(force.getNumTorsions())
                                # print(query_name)

                    if type(force).__name__ == "CMAPTorsionForce":
                        pass
                        # print(dir(force))
                        # print(force.getNumTorsions())

                break

        return forces_dict

    def _set_initial_states(self) -> list:
        """
        set_initial_states For each ionic liquid residue in the system the protonation state
        is interfered from the provided openMM system object and the protonation site is defined.
        """
        residues = []
        templates = dict()
        for r in self.topology.residues():
            name = r.name
            if name in templates:
                continue
            name_of_paired_ion = self.templates.get_residue_name_for_coupled_state(name)
            templates[name] = self._extract_templates(name)
            templates[name_of_paired_ion] = self._extract_templates(name_of_paired_ion)

        for r in self.topology.residues():
            name = r.name
            if name in self.templates.names:
                name_of_paired_ion = self.templates.get_residue_name_for_coupled_state(
                    name
                )
                forces_state1 = templates[name]
                forces_state2 = templates[name_of_paired_ion]
                assert len(forces_state1) == len(
                    forces_state2
                )  # check the number of forces
                for force_name in forces_state1:
                    if len(forces_state1[force_name]) != len(
                        forces_state2[
                            force_name
                        ]  # check the number of entries in the forces
                    ):
                        logger.critical(force_name)
                        logger.critical(name)
                        logger.critical(len(forces_state1[force_name]))
                        logger.critical(len(forces_state2[force_name]))

                        for b1, b2, in zip(
                            forces_state1[force_name],
                            forces_state2[force_name],
                        ):
                            logger.critical(f"{name}:{b1}")
                            logger.critical(f"{name_of_paired_ion}:{b2}")

                        logger.critical(f"{name}:{forces_state1[force_name][-1]}")
                        logger.critical(
                            f"{name_of_paired_ion}:{forces_state2[force_name][-1]}"
                        )

                        raise AssertionError("ohoh")

                residues.append(
                    Residue(
                        r,
                        name_of_paired_ion,
                        forces_state1,
                        forces_state2,
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
            residue.current_charge for residue in self.residues
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
