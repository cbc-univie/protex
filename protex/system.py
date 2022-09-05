import itertools
import logging
from collections import ChainMap, defaultdict, deque
from pdb import pm
import numpy as np
import parmed
import yaml, json
from copy import deepcopy

try:
    import openmm
except ImportError:
    import simtk.openmm

from protex.residue import Residue

logger = logging.getLogger(__name__)


class IonicLiquidTemplates:
    """
    Creates the basic foundation for the Ionic Liquid System.

    Parameters
    -----------
    states:
        A list of dictionary depicting the residue name and the atom name which should be changed, i.e.:

        .. code-block:: python

            IM1H_IM1 = { "IM1H": {"atom_name": "H7", "canonical_name": "IM1"},
                        "IM1": {"atom_name": "N2", "canonical_name": "IM1"} }
            OAC_HOAC = { "OAC": {"atom_name": "O2", "canonical_name": "OAC"},
                        "HOAC": {"atom_name": "H", "canonical_name": "OAC"} }
            states = [IM1H_IM1, OAC_HOAC]

    allowed_updates:
        A dictionary specifiying which updates are possile.
        Key is a frozenset with the two residue names for the update.
        The values is a dictionary which specifies the maximum distance ("r_max") and the probability for this update ("prob")
        r_max is in nanometer and the prob between 0 and 1

        .. code-block:: python

            allowed_updates = {}
            allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.155, "prob": 1}
            allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.155, "prob": 1}
            allowed_updates[frozenset(["IM1H", "IM1"])] = {"r_max": 0.155, "prob": 0.201}
            allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.155, "prob": 0.684}

    Attributes
    -----------
    pairs:
        A list with the pairs where the hydrogen can be transfered
    states:
        The passed states list as a joined dictionary
    names:
        A list with the different residue names
    allowed_updates:
        the allowed_updates passed to the class
    overall_max_distance:
        the longest allowed distance for a possible transfer between two updates
    """

    def __init__(
        self,
        states: list[dict[str, dict[str, str]]],
        allowed_updates: dict[frozenset[str], dict[str, float]],
    ) -> None:

        self.pairs: list[list[str]] = [list(i.keys()) for i in states]
        self.states: dict[str, dict[str, str]] = dict(ChainMap(*states))
        self.names: list[str] = list(itertools.chain(*self.pairs))
        self.allowed_updates: dict[frozenset[str], dict[str, float]] = allowed_updates
        self.overall_max_distance: float = max(
            [value["r_max"] for value in self.allowed_updates.values()]
        )

    def get_update_value_for(self, residue_set: frozenset[str], property: str) -> float:
        """
        returns the value in the allowed updates dictionary

        Parameters
        ----------
        residue_set:
            dictionary key for residue_set, i.e ["IM1H", "OAC"]
        property:
            dictionary key for the property defined for the residue key, i.e. prob

        Returns
        -------
        float
            the value of the property

        Raises
        ------
        RuntimeError
            if keys do not exist
        """
        if (
            residue_set in self.allowed_updates
            and property in self.allowed_updates[residue_set]
        ):
            return self.allowed_updates[residue_set][property]
        else:
            raise RuntimeError(
                "You tried to access a residue_set or property key which is not defined"
            )

    def set_update_value_for(
        self, residue_set: frozenset[str], property: str, value: float
    ):
        """
        Updates a value in the allowed updates dictionary

        Parameters
        ----------
        residue:
            dictionary key for residue_set, i.e ["IM1H","OAC"]
        property:
            dictionary key for the property defined for the residue key, i.e. prob
        value:
            the value the property should be set to

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            is raised if new residue_set or new property is trying to be inserted
        """

        if (
            residue_set in self.allowed_updates
            and property in self.allowed_updates[residue_set]
        ):
            self.allowed_updates[residue_set][property] = value

        # should we check for existance or also allow creation of new properties?
        # if residue in self.allowed_updates:
        #    self.allowed_updates[residue][property] = value

        else:
            raise RuntimeError(
                "You tried to create a new residue_set or property key! This is only allowed at startup!"
            )

    # Not used
    def set_allowed_updates(
        self, allowed_updates: dict[frozenset[str], dict[str, float]]
    ) -> None:
        self.allowed_updates = allowed_updates

    # Not used(?)
    # def get_canonical_name(self, name: str) -> str:
    #     assert name in self.names
    #     for state in self.states:
    #         if name in state:
    #             return self.states[name]["canonical_name"]

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

    # Not used
    def get_charge_template_for(self, name: str):
        """
        get_charge_template_for returns the charge template for a residue

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


class IonicLiquidSystem:
    """
    This class defines the full system, performs the MD steps and offers an
    interface for protonation state updates.

    Parameters
    ----------
    simulation:
        the OpenMM simulation object
    templates:
        An instance of the IonicLiquidTemplates class

    """

    def __init__(
        self,
        simulation: openmm.app.simulation.Simulation,
        templates: IonicLiquidTemplates,
    ) -> None:
        self.system: openmm.openmm.System = simulation.system
        self.topology: openmm.app.topology.Topology = simulation.topology
        self.simulation: openmm.app.simulation.Simulation = simulation
        self.templates: IonicLiquidTemplates = templates
        self.residues: list[Residue] = self._set_initial_states()
        self.boxlength: float = (
            simulation.context.getState().getPeriodicBoxVectors()[0][0]._value
        )  # NOTE: supports only cubic boxes

    def get_current_number_of_each_residue_type(self) -> dict[str, int]:
        current_number_of_each_residue_type = defaultdict(int)
        for residue in self.residues:
            current_number_of_each_residue_type[residue.current_name] += 1
        return current_number_of_each_residue_type

    def update_context(self, name: str):
        for force in self.system.getForces():
            if type(force).__name__ == name:
                force.updateParametersInContext(self.simulation.context)
                break

    def _build_exclusion_list(self):
        pair_12_set = set()
        pair_13_set = set()
        for bond in self.topology.bonds():
            a1, a2 = bond.atom1, bond.atom2
            if "H" not in a1.name and "H" not in a2.name:
                pair = (
                    min(a1.index, a2.index),
                    max(a1.index, a2.index),
                )
                pair_12_set.add(pair)
        for a in pair_12_set:
            for b in pair_12_set:
                shared = set(a).intersection(set(b))
                if len(shared) == 1:
                    pair = tuple(set(list(a) + list(b)) - shared)
                    pair_13_set.add(pair)

        self.pair_12_list = list(sorted(pair_12_set))
        self.pair_13_list = list(sorted(pair_13_set - pair_12_set))
        self.pair_12_13_list = self.pair_12_list + self.pair_13_list
        # change to return the list and set the parameters in the init method?

    def _extract_templates(self, query_name: str) -> defaultdict:
        # returns the forces for the residue name
        forces_dict = defaultdict(list)

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
                        # Also add exceptions
                        for exc_id in range(force.getNumExceptions()):
                            f = force.getExceptionParameters(exc_id)
                            idx1 = f[0]
                            idx2 = f[1]
                            if idx1 in atom_idxs and idx2 in atom_idxs:
                                forces_dict[type(force).__name__ + "Exceptions"].append(
                                    f
                                )

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

                    if type(force).__name__ == "CMAPTorsionForce":
                        pass

                    # DrudeForce stores charge and polarizability in ParticleParameters and Thole values in ScreenedPairParameters
                    # Number of these two is not the same -> i did two loops, and called the thole parameters DrudeForceThole.
                    # Not ideal but i could not think of anything better, pay attention to the set and get methods for drudes.
                    if type(force).__name__ == "DrudeForce":
                        for drude_id in range(force.getNumParticles()):
                            f = force.getParticleParameters(drude_id)
                            idx1 = f[0]  # drude
                            idx2 = f[1]  # parentatom
                            if idx1 in atom_idxs and idx2 in atom_idxs:
                                forces_dict[type(force).__name__].append(f)
                        assert len(self.pair_12_13_list) == force.getNumScreenedPairs()
                        for drude_id in range(force.getNumScreenedPairs()):
                            f = force.getScreenedPairParameters(drude_id)
                            # idx1 = f[0]
                            # idx2 = f[1]
                            parent1, parent2 = self.pair_12_13_list[drude_id]
                            drude1, drude2 = parent1 + 1, parent2 + 1
                            # print(f"thole {idx1=}, {idx2=}")
                            # print(f"{drude_id=}, {f=}")
                            if drude1 in atom_idxs and drude2 in atom_idxs:
                                # print(f"Thole {query_name=}")
                                # print(f"{drude1=}, {drude2=}")
                                forces_dict[type(force).__name__ + "Thole"].append(f)
                break  # do this only for the relevant amino acid once
        return forces_dict

    @staticmethod
    def _check_nr_of_forces(forces_state1, forces_state2, name, name_of_paired_ion):
        # check if two forces lists have the same number of forces
        assert len(forces_state1) == len(forces_state2)  # check the number of forces
        for force_name in forces_state1:
            if len(forces_state1[force_name]) != len(
                forces_state2[force_name]  # check the number of entries in the forces
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
                logger.critical(f"{name_of_paired_ion}:{forces_state2[force_name][-1]}")

                raise AssertionError("ohoh")

    def _set_initial_states(self) -> list:
        """
        set_initial_states For each ionic liquid residue in the system the protonation state
        is interfered from the provided openMM system object and the protonation site is defined.
        """

        self._build_exclusion_list()

        residues = []
        templates = dict()

        # for each residue type get forces
        for r in self.topology.residues():
            name = r.name
            name_of_paired_ion = self.templates.get_residue_name_for_coupled_state(name)

            if name in templates or name_of_paired_ion in templates:
                continue

            templates[name] = self._extract_templates(name)
            templates[name_of_paired_ion] = self._extract_templates(name_of_paired_ion)

        for r in self.topology.residues():
            name = r.name
            if name in self.templates.names:
                name_of_paired_ion = self.templates.get_residue_name_for_coupled_state(
                    name
                )

                parameters_state1 = templates[name]
                parameters_state2 = templates[name_of_paired_ion]
                # check that we have the same number of parameters
                self._check_nr_of_forces(
                    parameters_state1, parameters_state2, name, name_of_paired_ion
                )

                residues.append(
                    Residue(
                        r,
                        name_of_paired_ion,
                        self.system,
                        parameters_state1,
                        parameters_state2,
                        # self.templates.get_canonical_name(name),
                        self.pair_12_13_list,
                    )
                )
                residues[
                    -1
                ].current_name = (
                    name  # Why, isnt it done in the initializer of Residue?
                )

            else:
                raise RuntimeError("Found resiude not present in Templates: {r.name}")
        return residues

    # def save_current_names(self, file: str) -> None:
    #     """
    #     Save a file with the current residue names.
    #     Can be used with load_current_names to set the residues in the IonicLiquidSystem
    #     in the state of these names and also adapt corresponding charges, parameters,...
    #     """
    #     with open(file, "w") as f:
    #         for residue in self.residues:
    #             print(residue.current_name, file=f)

    # def load_current_names(self, file: str) -> None:
    #     """
    #     Load the names of the residues (order important!)
    #     Update the current_name of all residues to the given one
    #     """
    #     residue_names = []
    #     with open(file, "r") as f:
    #         for line in f.readlines():
    #             residue_names.append(line.strip())
    #     assert (
    #         len(residue_names) == self.topology.getNumResidues()
    #     ), "Number of residues not matching"
    #     for residue, name in zip(self.residues, residue_names):
    #         residue.current_name = name

    # not used
    # def report_states(self) -> None:
    #     """
    #     report_states prints out a summary of the current protonation state of the ionic liquid
    #     """
    #     pass

    def _adapt_parmed_psf_file(
        self, psf: parmed.charmm.CharmmPsfFile, psf_copy: parmed.charmm.CharmmPsfFile
    ) -> parmed.charmm.CharmmPsfFile:
        """
        Helper function to adapt the psf
        """
        assert len(self.residues) == len(psf.residues)

        # make a dict with parmed representations of each residue, use it to assign the opposite one if a transfer occured
        pm_unique_residues: dict[str, parmed.Residue] = {}
        # incremented by one each time it is used to track the current residue number
        residue_counts: dict[str, int] = {}
        for pm_residue in psf_copy.residues:
            if pm_residue.name in pm_unique_residues:
                continue
            else:
                pm_unique_residues[pm_residue.name] = pm_residue
                residue_counts[pm_residue.name] = 1

        for residue, pm_residue in zip(self.residues, psf.residues):
            # if the new residue (residue.current_name) is different than the original one from the old psf (pm_residue.name)
            # a proton transfer occured and we want to change this in the new psf, which means overwriting the parmed residue instance
            # with the new information
            # if residue.current_name != pm_residue.name:
            # do changes
            name = residue.current_name
            pm_residue.name = name
            pm_residue.chain = name
            pm_residue.segid = name
            pm_residue.number = residue_counts[name]
            for unique_atom, pm_atom in zip(
                pm_unique_residues[name].atoms, pm_residue.atoms
            ):
                pm_atom._charge = unique_atom._charge
                pm_atom.type = unique_atom.type
                pm_atom.props = unique_atom.props
            residue_counts[name] += 1

        return psf

    def write_psf(self, old_psf_infname: str, new_psf_outfname: str) -> None:
        """
        write a new psf file, which reflects the occured transfer events and changed residues
        to load the written psf create a new ionic_liquid instance and load the new psf via OpenMM
        """
        import parmed

        pm_old_psf = parmed.charmm.CharmmPsfFile(old_psf_infname)
        # copying parmed structure did not work
        pm_old_psf_copy = parmed.charmm.CharmmPsfFile(old_psf_infname)
        pm_new_psf = self._adapt_parmed_psf_file(pm_old_psf, pm_old_psf_copy)
        pm_new_psf.write_psf(new_psf_outfname)

    # possibly in future when parmed and openmm drude connection is working
    # def write_psf_notworking(
    #     self, fname: str, format=None, overwrite=False, **kwargs
    # ) -> None:
    #     """
    #     Write a psf file from the current topology.
    #     In principle any file that parmeds struct.save method supports can be written.
    #     """
    #     import parmed

    #     struct = parmed.openmm.load_topology(self.topology, self.system)
    #     struct.save(fname, format=None, overwrite=False, **kwargs)

    def saveCheckpoint(self, file) -> None:
        """
        Wrapper method which just calls the underlying same function on the simulation object of the ionic liquid object

        Parameters
        ----------
        file: string or file
            a File-like object to write the checkpoint to, or alternatively a
            filename
        """
        self.simulation.saveCheckpoint(file)

    def loadCheckpoint(self, file) -> None:
        """Wrapper method which just calls the underlying same function on the simulation object of the ionic liquid object

        Parameters
        ----------
        file : string or file
            a File-like object to load the checkpoint from, or alternatively a
            filename
        """
        self.simulation.loadCheckpoint(file)

    def saveState(self, file) -> None:
        """Wrapper method which just calls the underlying same function on the simulation object of the ionic liquid object

        Parameters
        ----------
        file : string or file
            a File-like object to write the state to, or alternatively a
            filename
        """
        self.simulation.saveState(file)

    def loadState(self, file) -> None:
        """Wrapper method which just calls the underlying same function on the simulation object of the ionic liquid object

        Parameters
        ----------
        file : string or file
            a File-like object to load the state from, or alternatively a
            filename
        """
        self.simulation.loadState(file)

    def save_updates(self, file) -> None:
        """
        Save the current update values into a yaml file. Used to have the current probability values.

        Parameters
        ----------
        file: string or file
        """
        # TODO
        # there should be a better way to get the frozen set into and back from a yaml file...
        data = {
            str(key): value for key, value in self.templates.allowed_updates.items()
        }
        with open(file, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def load_updates(self, file) -> None:
        """
        Load the current update values from a yaml file, which was generated using "save_updates".

        Parameters
        ----------
        file: string or file
        """
        with open(file, "r") as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print("Error")
                print(exc)
        # TODO
        # bad coding here, to get the frozenset back from the yaml
        final_data = {}
        for key, value in data.items():
            key = key.split("'")
            final_data[frozenset([key[1], key[3]])] = value
        self.templates.allowed_updates = final_data
