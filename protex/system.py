from __future__ import annotations

import itertools
import logging
import pickle
import warnings
from collections import ChainMap, defaultdict

# import parmed
# NOTE trying to get away from parmed
# use pickled system, don't write any more psfs
import yaml

try:
    import openmm
except ImportError:
    from simtk import openmm

# from protex.helpers import ProtexException  # , CustomPSFFile
from protex import ProtexException
from protex.residue import Residue

# Use custom write function for now, remove with new parmed version
# https://github.com/ParmEd/ParmEd/pull/1274
# as in parmed/formats/__init__.py:
# parmed.Structure.write_psf = CustomPSFFile.write

logger = logging.getLogger(__name__)


class ProtexTemplates:
    """Creates the basic foundation for the Protex System.

    Parameters
    ----------
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
    ----------
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

    @staticmethod
    def load(fname: str) -> ProtexTemplates:
        """Load a pickled ProtexTemplates instance.

        Parameters
        ----------
        fname : str
            The file name

        Returns
        -------
        ProtexTemplates
            An instance of the ProtexTemplates
        """
        with open(fname, "rb") as inp:
            templates = pickle.load(inp)
        return templates

    def __init__(
        self,
        states: list[dict[str, dict[str, str]]],
        allowed_updates: dict[frozenset[str], dict[str, float]],
    ) -> None:
        self.__states = states
        self.pairs: list[list[str]] = [list(i.keys()) for i in states]
        self.states: dict[str, dict[str, str]] = dict(ChainMap(*states))
        self.names: list[str] = list(itertools.chain(*self.pairs))
        self.allowed_updates: dict[frozenset[str], dict[str, float]] = allowed_updates
        self.overall_max_distance: float = max(
            [value["r_max"] for value in self.allowed_updates.values()]
        )
        # store names in variables, in case syntax for states dict changes
        self._atom_name: str = "atom_name"
        self._equivalent_atom: str = "equivalent_atom"

    def dump(self, fname: str) -> None:
        """Pickle the current ProtexTemplates object.

        Parameters
        ----------
        fname : str
            The file name of the object
        """
        with open(fname, "wb") as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def get_atom_name_for(self, resname: str) -> str:
        """Get the atom name for a specific residue.

        Parameters
        ----------
        resname : str
            The residue name

        Returns
        -------
        str
            The atom name
        """
        return self.states[resname][self._atom_name]

    def has_equivalent_atom(self, resname: str) -> bool:
        """Check if a given residue has an equivalent atom defined.

        Parameters
        ----------
        resname : str
            The residue name

        Returns
        -------
        bool
            True if this residue has an equivalent atom defined, false otherwise
        """
        return self._equivalent_atom in self.states[resname]

    def get_equivalent_atom_for(self, resname: str) -> str:
        """Get the name of the equivalent atom for a given residue name.

        Parameters
        ----------
        resname : str
            The residue name

        Returns
        -------
        str
            The atom name
        """
        return self.states[resname][self._equivalent_atom]

    def get_update_value_for(self, residue_set: frozenset[str], property: str) -> float:
        """Returns the value in the allowed updates dictionary.

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
        """Update a value in the allowed updates dictionary.

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

    def get_residue_name_for_coupled_state(self, name: str) -> str:
        """get_residue_name_of_paired_ion returns the paired residue name given a reisue name.

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
        raise RuntimeError("something went wrong")

    # Not used
    def get_charge_template_for(self, name: str) -> list:
        """get_charge_template_for returns the charge template for a residue.

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


class ProtexSystem:
    """Defines the full system, performs the MD steps and offers an interface for protonation state updates.

    Parameters
    ----------
    simulation:
        the OpenMM simulation object
    templates:
        An instance of the IonicLiquidTemplates class
    simulation_for_parameters:
        another OpenMM simulation object, set up with a psf that contains all possible forms (protonated and deprotonated) of all species,
        needed for finding the parameters of the other form during an update
    fast:
        If True indices list for all forces for each residue will be made once at initilaizaiton of ProtexSystem, and then used for finding the indices during an update.
        If False, the whole forces.getNumExceptions and similar calls will be used for each update and each residue, which is way slower

    """

    # These lists contain the forces generated by OpenMM as they are given when calling: type(force).__name__
    # forces which are ignored on purpose, because we do not care about them
    IGNORED_FORCES = ["CMMotionRemover", "MonteCarloBarostat"]
    ######################
    # To include a new force to the covered forces, you need to add them to the following places:
    # ProtexSystem._extract_templates
    # ProtexSystem._create_force_idx_dict
    # Residue.update (and add the correspoding get and set methods!)
    # Update.allowed_forces #probably to the part where all_forces is True
    #####################
    COVERED_FORCES = [
        "NonbondedForce",
        "CustomNonbondedForce",
        "HarmonicBondForce",
        "HarmonicAngleForce",
        "PeriodicTorsionForce",
        "CustomTorsionForce",
        "DrudeForce",
    ]
    # add force here to ignore them, but raise a warning, because they might be important in the future
    # until now it was there but empty, but who knows for later
    UNCOVERED_FORCES = ["CMAPTorsionForce"]

    @classmethod
    def force_is_valid(cls, name: str) -> bool:
        """Check if the given force name is covered."""
        if name in cls.COVERED_FORCES or name in cls.IGNORED_FORCES:
            return True
        elif name in cls.UNCOVERED_FORCES:
            warnings.warn(
                f"{name} is not processed by protex. Until now this was not a problem. But you should double check with your system, if this force is used."
            )
            return True
        else:
            raise ProtexException(
                f"{name} is not yet covered in Protex. Please write an issue on Github."
            )

    @staticmethod
    def load(
        fname: str,
        simulation: openmm.app.simulation.Simulation,
        simulation_for_parameters: openmm.app.simulation.Simulation = None,
    ) -> ProtexSystem:
        """Load pickled protex system.

        Parameters
        ----------
        fname : str
            The file name to load
        simulation : openmm.app.simulation.Simulation
            An already generated OpenMM simulation object, needed to initialize the ProtesSystem instance
        simulation_for_parameters : openmm.app.simulation.Simulation, optional
            An OpenMM simulation object, which contains all possible residues
            needed if the simulation object does not contain all possible residues, by default None

        Returns
        -------
        ProtexSystem
            A new instance of the ProtexSystem
        """
        with open(fname, "rb") as inp:
            from_pickle = pickle.load(inp)  # ensure correct order of arguments
        protex_system = ProtexSystem(
            simulation, from_pickle[0], simulation_for_parameters
        )
        residues = from_pickle[1]
        # update parameters for residues where name in psf doesn't match name in pickled residues
        # TODO test
        
        for resi_pickled, resi_current in zip(residues, protex_system.residues):
            if resi_current.current_name != resi_pickled.current_name: # resi_current was set up from original psf -> has original (potentially wrong parameters)
                for force_to_be_updated in ProtexSystem.COVERED_FORCES:
                    resi_current.update(force_to_be_updated, 1)
                resi_current.current_name = resi_current.alternativ_name

        for force_to_be_updated in ProtexSystem.COVERED_FORCES:
            protex_system.update_context(force_to_be_updated)

        # protex_system.residues = residues # do we need this if everything is done previously?

        return protex_system

    def __init__(
        self,
        simulation: openmm.app.simulation.Simulation,
        templates: ProtexTemplates,
        simulation_for_parameters: openmm.app.simulation.Simulation = None,
        fast: bool = True,
    ) -> None:
        self.system: openmm.openmm.System = simulation.system
        self.topology: openmm.app.topology.Topology = simulation.topology
        self.simulation: openmm.app.simulation.Simulation = simulation
        self.templates: ProtexTemplates = templates
        self.simulation_for_parameters = simulation_for_parameters
        self.ensemble = "nVT"
        self._check_forces()
        self.detected_forces: set[str] = self._detect_forces()
        self.fast: bool = fast
        self.residues: list[Residue] = self._set_initial_states()
        self.boxlength: openmm.Quantity = (
            simulation.context.getState().getPeriodicBoxVectors()[0][0] # NOTE: supports only cubic boxes and nVT -> changed in update.py to get at every step for npT
        )

    def dump(self, fname: str) -> None:
        """Pickle the current ProtexSystem object.

        Parameters
        ----------
        fname : str
            The file name to store the object
        """
        to_pickle = [self.templates, self.residues]  # ensure correct order of arguments
        with open(fname, "wb") as outp:
            pickle.dump(to_pickle, outp, pickle.HIGHEST_PROTOCOL)

    def _detect_forces(self) -> set[str]:
        def _is_populated(force):
            if type(force).__name__ in self.IGNORED_FORCES:
                return False
            if isinstance(force, openmm.NonbondedForce):
                return force.getNumParticles() > 0
            elif isinstance(force, openmm.HarmonicBondForce):
                return force.getNumBonds() > 0
            elif isinstance(force, openmm.HarmonicAngleForce):
                return force.getNumAngles() > 0
            elif isinstance(force, openmm.PeriodicTorsionForce):
                return force.getNumTorsions() > 0
            elif isinstance(force, openmm.CustomTorsionForce):
                return force.getNumTorsions() > 0
            elif isinstance(force, openmm.DrudeForce):
                return force.getNumParticles() > 0
            elif isinstance(force, openmm.CustomNonbondedForce):
                return force.getNumParticles() > 0
            elif isinstance(force, openmm.CMAPTorsionForce):
                return force.getNumTorsions() > 0
            else:
                raise ProtexException(f"This force should be covered here: {type(force).__name__}: {force}")

        detected_forces: list = []
        for force in self.system.getForces():
            if _is_populated(force):
                detected_forces.append(type(force).__name__)
        if self.simulation_for_parameters is not None:
            for force in self.simulation_for_parameters.system.getForces():
                if _is_populated(force):
                    detected_forces.append(type(force).__name__)
        return set(detected_forces)

    def _check_forces(self) -> None:
        """Will fail if a force is not covered."""
        for force in self.system.getForces():
            self.force_is_valid(type(force).__name__)
            if type(force).__name__ == "MonteCarloBarostat": # find out if we have a barostat (maybe not the best place to do it?)
                self.ensemble = "npT"
        if self.simulation_for_parameters is not None:
            for force in self.simulation_for_parameters.system.getForces():
                self.force_is_valid(type(force).__name__)

    def get_current_number_of_each_residue_type(self) -> dict[str, int]:
        """Get a dictionary with the resname and the current number of residues belonging to that name.

        Returns
        -------
        dict[str, int]
            resname: number of residues
        """
        current_number_of_each_residue_type: dict[str, int] = defaultdict(int)
        for residue in self.residues:
            current_number_of_each_residue_type[residue.current_name] += 1
        return current_number_of_each_residue_type

    def update_context(self, name: str):
        """Update the context for the given force.

        Parameters
        ----------
        name : str
            The name of the force to update
        """
        for force in self.system.getForces():
            if type(force).__name__ == name:
                force.updateParametersInContext(self.simulation.context)

    def _extract_templates(self, query_name: str) -> defaultdict:
        # returns the forces for the residue name
        forces_dict = defaultdict(list)

        # if there is an additional parameter file with all possible residues,
        # use this for getting the templates
        if self.simulation_for_parameters is not None:
            sim = self.simulation_for_parameters
        else:
            sim = self.simulation

        for residue in sim.topology.residues():
            if query_name == residue.name:
                atom_idxs = [atom.index for atom in residue.atoms()]
                atom_names = [atom.name for atom in residue.atoms()]
                logger.debug(atom_idxs)
                logger.debug(atom_names)

                for force in sim.system.getForces():
                    forcename = type(force).__name__
                    if forcename == "NonbondedForce":
                        forces_dict[forcename] = [
                            force.getParticleParameters(idx) for idx in atom_idxs
                        ]
                        # Also add exceptions
                        for exc_id in range(force.getNumExceptions()):
                            f = force.getExceptionParameters(exc_id)
                            idx1 = f[0]
                            idx2 = f[1]
                            if idx1 in atom_idxs and idx2 in atom_idxs:
                                forces_dict[forcename + "Exceptions"].append(f)

                    elif forcename == "CustomNonbondedForce":
                        # lookup indices of the tabulated table (?)
                        #index is position, but what about previous ones?
                        forces_dict[forcename] = [force.getParticleParameters(idx) for idx in atom_idxs]
                        # Also add exclusions
                        for exc_id in range(force.getNumExclusions()):
                            f = force.getExclusionParticles(exc_id)
                            idx1 = f[0]
                            idx2 = f[1]
                            if idx1 in atom_idxs and idx2 in atom_idxs:
                                forces_dict[forcename + "Exclusions"].append(f)

                    elif forcename == "HarmonicBondForce":
                        # logger.debug(residue)
                        for bond_id in range(force.getNumBonds()):
                            f = force.getBondParameters(bond_id)
                            idx1 = f[0]
                            idx2 = f[1]
                            if idx1 in atom_idxs and idx2 in atom_idxs:
                                forces_dict[forcename].append(f)
                                # logger.debug(f)

                    elif forcename == "HarmonicAngleForce":
                        for angle_id in range(force.getNumAngles()):
                            f = force.getAngleParameters(angle_id)
                            if (
                                f[0] in atom_idxs
                                and f[1] in atom_idxs
                                and f[2] in atom_idxs
                            ):
                                forces_dict[forcename].append(f)

                    elif forcename == "PeriodicTorsionForce":
                        for torsion_id in range(force.getNumTorsions()):
                            f = force.getTorsionParameters(torsion_id)
                            if (
                                f[0] in atom_idxs
                                and f[1] in atom_idxs
                                and f[2] in atom_idxs
                                and f[3] in atom_idxs
                            ):
                                forces_dict[forcename].append(f)

                    elif forcename == "CustomTorsionForce":
                        for torsion_id in range(force.getNumTorsions()):
                            f = force.getTorsionParameters(torsion_id)

                            if (
                                f[0] in atom_idxs
                                and f[1] in atom_idxs
                                and f[2] in atom_idxs
                                and f[3] in atom_idxs
                            ):
                                forces_dict[forcename].append(f)

                    # DrudeForce stores charge and polarizability in ParticleParameters and Thole values in ScreenedPairParameters
                    # Number of these two is not the same -> i did two loops, and called the thole parameters DrudeForceThole.
                    # Not ideal but i could not think of anything better, pay attention to the set and get methods for drudes.
                    elif forcename == "DrudeForce":
                        particle_map = {}
                        for drude_id in range(force.getNumParticles()):
                            f = force.getParticleParameters(drude_id)
                            idx1 = f[0]  # drude
                            idx2 = f[1]  # parentatom
                            if idx1 in atom_idxs and idx2 in atom_idxs:
                                forces_dict[forcename].append(f)
                            # store the drude idx as they are in the system
                            particle_map[drude_id] = idx1
                        for drude_id in range(force.getNumScreenedPairs()):
                            f = force.getScreenedPairParameters(drude_id)
                            # yields the id within this force == drude_id from getNumParticles
                            idx1 = f[0]
                            idx2 = f[1]
                            # get the drude idxs in the system
                            drude1 = particle_map[idx1]
                            drude2 = particle_map[idx2]
                            if drude1 in atom_idxs and drude2 in atom_idxs:
                                forces_dict[forcename + "Thole"].append(f)
                break  # do this only for the relevant residue once
        else:
            raise RuntimeError("residue not found")
        return forces_dict

    @staticmethod
    def _check_nr_of_forces(
        forces_state1, forces_state2, name: str, name_of_paired_ion: str
    ) -> None:
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

                for b1 in forces_state1[force_name]:
                    logger.critical(f"{name}:{b1}")

                for b2 in forces_state2[force_name]:
                    logger.critical(f"{name_of_paired_ion}:{b2}")

                logger.critical(f"{name}:{forces_state1[force_name][-1]}")
                logger.critical(f"{name_of_paired_ion}:{forces_state2[force_name][-1]}")

                raise AssertionError(
                    "There are not the same number of forces or a wrong order. Possible Problems: Bond/Angle/... is missing. Urey_Bradley Term is not defined for both states, ...)"
                )

    def _add_force(
        self, fgroup: int, forcename: str, max_value: int, insert_values: tuple[int]
    ) -> None:
        """Add the values of a force to the per_residue_forces dict.

        Parameters
        ----------
        fgroup : int
            The integer of the force group
        forcename : str
            The name of the force
        max_value : int
            The maximum value of the atom indices in the force to add
        insert_values : tuple[int]
            The values of the force which should be added, probably the index in the force and the atom indices
        """
        for tuple_item in self.per_residue_forces.keys():
            start, end = tuple_item
            if start <= max_value <= end:
                try:
                    self.per_residue_forces[tuple_item][fgroup][forcename].append(
                        insert_values
                    )
                except KeyError:
                    self.per_residue_forces[tuple_item][fgroup] = {}
                    self.per_residue_forces[tuple_item][fgroup][forcename] = [
                        insert_values
                    ]

    def _create_force_idx_dict(self) -> None:
        """Create a dictionary containig all the indices to the forces.

        It is used to specify which force belong to which residue.
        It filles the self.per_residue_forces variable.
        """
        for force in self.system.getForces():
            forcename = type(force).__name__
            fgroup = force.getForceGroup()
            if forcename == "NonbondedForce":
                # only treat exceptions
                for exc_idx in range(force.getNumExceptions()):
                    f = force.getExceptionParameters(exc_idx)
                    idx1, idx2 = f[0], f[1]
                    value = (exc_idx, idx1, idx2)
                    maxi = max(idx1, idx2)
                    self._add_force(fgroup, "NonbondedForceExceptions", maxi, value)
            elif forcename == "CustomNonbondedForce":
                for excl_idx in range(force.getNumExclusions()):
                    f = force.getExclusionParticles(excl_idx)
                    idx1, idx2 = f[0], f[1]
                    value = (exc_idx, idx1, idx2)
                    maxi = max(idx1, idx2)
                    self._add_force(fgroup, "CustomNonbondedForceExclusions", maxi, value)
            elif forcename == "DrudeForce":
                particle_map = {}
                for drude_idx in range(force.getNumParticles()):
                    f = force.getParticleParameters(drude_idx)
                    idx1, idx2, idx3, idx4, idx5 = f[0], f[1], f[2], f[3], f[4]
                    value = (drude_idx, idx1, idx2, idx3, idx4, idx5)
                    maxi = max(
                        idx1, idx2
                    )  # drude and parent is enough, we do not need the anisotropy stuff for this here
                    self._add_force(fgroup, "DrudeForce", maxi, value)
                    particle_map[drude_idx] = idx1
                for drude_idx in range(force.getNumScreenedPairs()):
                    f = force.getScreenedPairParameters(drude_idx)
                    idx1, idx2 = f[0], f[1]
                    drude1 = particle_map[idx1]
                    drude2 = particle_map[idx2]
                    value = (drude_idx, idx1, idx2)
                    maxi = max(drude1, drude2)
                    self._add_force(fgroup, "DrudeForceThole", maxi, value)
            elif forcename == "HarmonicBondForce":
                for bond_idx in range(force.getNumBonds()):
                    f = force.getBondParameters(bond_idx)
                    idx1, idx2 = f[0], f[1]
                    value = (bond_idx, idx1, idx2)
                    maxi = max(idx1, idx2)
                    self._add_force(fgroup, "HarmonicBondForce", maxi, value)
            elif forcename == "HarmonicAngleForce":
                for angle_idx in range(force.getNumAngles()):
                    f = force.getAngleParameters(angle_idx)
                    idx1, idx2, idx3 = f[0], f[1], f[2]
                    value = (angle_idx, idx1, idx2, idx3)
                    maxi = max(idx1, idx2, idx3)
                    self._add_force(fgroup, "HarmonicAngleForce", maxi, value)
            elif forcename == "PeriodicTorsionForce":
                for torsion_idx in range(force.getNumTorsions()):
                    f = force.getTorsionParameters(torsion_idx)
                    idx1, idx2, idx3, idx4 = f[0], f[1], f[2], f[3]
                    value = (torsion_idx, idx1, idx2, idx3, idx4)
                    maxi = max(idx1, idx2, idx3, idx4)
                    self._add_force(fgroup, "PeriodicTorsionForce", maxi, value)
            elif forcename == "CustomTorsionForce":
                for ctorsion_idx in range(force.getNumTorsions()):
                    f = force.getTorsionParameters(ctorsion_idx)
                    idx1, idx2, idx3, idx4 = f[0], f[1], f[2], f[3]
                    value = (ctorsion_idx, idx1, idx2, idx3, idx4)
                    maxi = max(idx1, idx2, idx3, idx4)
                    self._add_force(fgroup, "CustomTorsionForce", maxi, value)

    def _fill_residue_templates(self, name):
        if name in self.templates.names:
            name_of_paired_ion = self.templates.get_residue_name_for_coupled_state(name)
            if (
                name in self.residue_templates
                or name_of_paired_ion in self.residue_templates
            ):
                return
            self.residue_templates[name] = self._extract_templates(name)
            self.residue_templates[name_of_paired_ion] = self._extract_templates(
                name_of_paired_ion
            )
        else:
            if name in self.residue_templates:  # or name_of_paired_ion in templates:
                return
            self.residue_templates[name] = self._extract_templates(name)


    def _set_initial_states(self) -> list:
        """set_initial_states.

        For each ionic liquid residue in the system the protonation state
        is interfered from the provided openMM system object and the protonation site is defined.
        """
        residues = []
        self.residue_templates = dict()
        # this will become a dict of the form:
        # self.per_residue_forces[(min,max tuple of the atom idxs for each residue)][forcegroup][forcename]: list of the idxs of this force for this residue
        self.per_residue_forces = {}
        # for each residue type get forces
        for r in self.topology.residues():
            atom_idxs = [atom.index for atom in r.atoms()]
            mini, maxi = min(atom_idxs), max(atom_idxs)
            self.per_residue_forces[(mini, maxi)] = {}
            name = r.name
            self._fill_residue_templates(name)

        if self.fast:
            # this takes some time, but the update calls on the residues are then much faster
            self._force_idx_dict = self._create_force_idx_dict()

        for r in self.topology.residues():
            name = r.name
            if name in self.templates.names:
                name_of_paired_ion = self.templates.get_residue_name_for_coupled_state(
                    name
                )

                parameters_state1 = self.residue_templates[name]
                parameters_state2 = self.residue_templates[name_of_paired_ion]
                # check that we have the same number of parameters
                self._check_nr_of_forces(
                    parameters_state1, parameters_state2, name, name_of_paired_ion
                )
                atom_idxs = [
                    atom.index for atom in r.atoms()
                ]  # also give them to initilaizer, not inside residue?
                minmax = (min(atom_idxs), max(atom_idxs))
                if self.fast:
                    new_res = Residue(
                        r,
                        name_of_paired_ion,
                        self.system,
                        parameters_state1,
                        parameters_state2,
                        (
                            self.templates.has_equivalent_atom(name),
                            self.templates.has_equivalent_atom(name_of_paired_ion),
                        ),
                        force_idxs=self.per_residue_forces[minmax],
                    )
                else:
                    new_res = Residue(
                        r,
                        name_of_paired_ion,
                        self.system,
                        parameters_state1,
                        parameters_state2,
                        (
                            self.templates.has_equivalent_atom(name),
                            self.templates.has_equivalent_atom(name_of_paired_ion),
                        ),
                    )
                residues.append(new_res)
                residues[
                    -1
                ].current_name = (
                    name  # Why, isnt it done in the initializer of Residue?
                )
            else:
                parameters_state1 = self.residue_templates[name]
                r = Residue(r, None, self.system, parameters_state1, None, None)
                # the residue is not part of any proton transfers,
                # we still need it in the residue list for the parmed hack...
                # there we need the current_name attribute, hence give it to the residue
                # r.current_name = r.name
                residues.append(r)
            # else: #if there are residues on purpose not with protex we want to just ignore them
            #    raise RuntimeError(
            #        f"Found resiude not present in Templates: {r.name}"
            #    )  # we want to ignore meoh, doesn't work the way it actually is
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

    # should be deprecated with saving system
    # def _adapt_parmed_psf_file(
    #     self,
    #     psf: parmed.charmm.CharmmPsfFile,
    #     parameters: parmed.charmm.CharmmPsfFile,
    # ) -> parmed.charmm.CharmmPsfFile:
    #     """Helper function to adapt the psf."""
    #     print(len(self.residues), len(psf.residues))
    #     assert len(self.residues) == len(psf.residues)

    #     # make a dict with parmed representations of each residue, use it to assign the opposite one if a transfer occured
    #     pm_unique_residues: dict[str, parmed.Residue] = {}
    #     # incremented by one each time it is used to track the current residue number
    #     residue_counts: dict[str, int] = {}

    #     for pm_residue in parameters.residues:
    #         if pm_residue.name in pm_unique_residues:
    #             continue
    #         else:
    #             pm_unique_residues[pm_residue.name] = pm_residue
    #             residue_counts[pm_residue.name] = 1

    #     # get offset for lonepairs which are defined differently.
    #     # dict with lists for each LocalCoordinates frame, there is one for each LP in the residue
    #     differences_dict: dict[tuple[str], list[list[int]]] = {}
    #     for pair in self.templates.pairs:
    #         name1, name2 = pair
    #         pm_res1 = pm_unique_residues[name1]
    #         pm_res2 = pm_unique_residues[name2]
    #         # get first index to get relative difference
    #         at1_idx1 = pm_res1.atoms[0].idx
    #         at2_idx1 = pm_res2.atoms[0].idx

    #         differences_dict[tuple([name1, name2])] = []
    #         differences_dict[tuple([name2, name1])] = []

    #         for at1, at2 in zip(pm_res1, pm_res2):
    #             if isinstance(at1, parmed.topologyobjects.ExtraPoint) and isinstance(
    #                 at1.frame_type, parmed.topologyobjects.LocalCoordinatesFrame
    #             ):
    #                 frame_hosts1 = [
    #                     fatom
    #                     for fatom in at1.frame_type.get_atoms()[
    #                         : at1.frame_type.frame_size
    #                     ]
    #                 ]
    #                 frame_hosts2 = [
    #                     fatom
    #                     for fatom in at2.frame_type.get_atoms()[
    #                         : at2.frame_type.frame_size
    #                     ]
    #                 ]
    #                 relative_pos1 = [fatom.idx - at1_idx1 for fatom in frame_hosts1]
    #                 relative_pos2 = [fatom.idx - at2_idx1 for fatom in frame_hosts2]
    #                 differences = [
    #                     p1 - p2 for p1, p2 in zip(relative_pos1, relative_pos2)
    #                 ]
    #                 # Depending on the direction of the tranfer we either need the positive or negative value
    #                 differences_dict[tuple([name1, name2])].append(differences)
    #                 differences_dict[tuple([name2, name1])].append(
    #                     [-d for d in differences]
    #                 )

    #     # either it is the same or just one group will be assumed. -> Not best for proteins, but hopefully Parmed will release new version soon, so that we do not need all this hacks.
    #     n_residue_is_n_groups = len(psf.groups) == len(psf.residues)
    #     group_iter = iter(psf.groups)
    #     for residue, pm_residue in zip(self.residues, psf.residues):
    #         # if the new residue (residue.current_name) is different than the original one from the old psf (pm_residue.name)
    #         # a proton transfer occured and we want to change this in the new psf, which means overwriting the parmed residue instance
    #         # with the new information
    #         # if residue.current_name != pm_residue.name:
    #         # do changes
    #         name = residue.current_name
    #         name_change: tuple[str] = tuple([name, pm_residue.name])
    #         # get the differences in LP position, make an iterator to yiled the next as needed
    #         diff_iter = None
    #         if name_change in differences_dict.keys():
    #             differences_list = differences_dict[name_change]
    #             diff_iter = iter(differences_list)
    #         pm_residue.name = name
    #         pm_residue.chain = name
    #         pm_residue.segid = name
    #         pm_residue.number = residue_counts[name]
    #         for unique_atom, pm_atom in zip(
    #             pm_unique_residues[name].atoms, pm_residue.atoms
    #         ):
    #             pm_atom._charge = unique_atom._charge
    #             pm_atom.type = unique_atom.type
    #             pm_atom.props = unique_atom.props
    #             # NUMLP NUMLPH lonepair section update
    #             if isinstance(
    #                 unique_atom, parmed.topologyobjects.ExtraPoint
    #             ) and isinstance(
    #                 unique_atom.frame_type, parmed.topologyobjects.LocalCoordinatesFrame
    #             ):
    #                 pm_atom.frame_type.distance = unique_atom.frame_type.distance
    #                 pm_atom.frame_type.angle = unique_atom.frame_type.angle
    #                 pm_atom.frame_type.dihedral = unique_atom.frame_type.dihedral
    #                 # if the positioning of the lonepair changes, update the corresponding atoms in the LocalCoordinateFrame
    #                 if diff_iter:
    #                     differences = next(diff_iter)
    #                     first_idx = pm_residue.atoms[0].idx

    #                     # normalize to the first idx in the list of atoms for one residue
    #                     current_idx1 = pm_atom.frame_type.atom1.idx
    #                     # print(f"{current_idx1=}")
    #                     # print(f"{differences[0]=}")
    #                     pm_atom.frame_type.atom1 = pm_residue.atoms[
    #                         current_idx1 - first_idx + differences[0]
    #                     ]
    #                     current_idx2 = pm_atom.frame_type.atom2.idx
    #                     # print(f"{current_idx2=}")
    #                     # print(f"{differences[1]=}")
    #                     pm_atom.frame_type.atom2 = pm_residue.atoms[
    #                         current_idx2 - first_idx + differences[1]
    #                     ]
    #                     current_idx3 = pm_atom.frame_type.atom3.idx
    #                     # print(f"{current_idx3=}")
    #                     # print(f"{differences[2]=}")
    #                     pm_atom.frame_type.atom3 = pm_residue.atoms[
    #                         current_idx3 - first_idx + differences[2]
    #                     ]

    #             # NUMANISO Section update
    #             if isinstance(unique_atom, parmed.topologyobjects.DrudeAtom):
    #                 if unique_atom.anisotropy is None:
    #                     continue
    #                 pm_atom.anisotropy.params["k11"] = unique_atom.anisotropy.params[
    #                     "k11"
    #                 ]
    #                 pm_atom.anisotropy.params["k22"] = unique_atom.anisotropy.params[
    #                     "k22"
    #                 ]
    #                 pm_atom.anisotropy.params["k33"] = unique_atom.anisotropy.params[
    #                     "k33"
    #                 ]
    #         residue_counts[name] += 1

    #         if n_residue_is_n_groups:
    #             group = next(group_iter)
    #             typ = 1 if abs(sum(a.charge for a in pm_residue.atoms)) < 1e-4 else 2
    #             group.type = typ

    #     if not n_residue_is_n_groups:
    #         # Maybe a bit hacky to get Parmed to use all atoms as 1 group
    #         # https://github.com/ParmEd/ParmEd/blob/master/parmed/formats/psf.py#L250
    #         psf.groups = []

    #     return psf

    # should be deprecated with saving system
    # def write_psf(
    #     self,
    #     old_psf_infname: str,
    #     new_psf_outfname: str,
    #     psf_for_parameters: str = None,
    # ) -> None:
    #     """Write a new psf file, which reflects the occured transfer events and changed residues
    #     to load the written psf create a new ionic_liquid instance and load the new psf via OpenMM.

    #     Parameters
    #     ----------
    #     old_psf_infname:
    #         Name of the old psf_file, which serves for the basic strucutre, same number of atoms, same bonds, angles, ...
    #     new_psf_outfname:
    #         Name of the new psf that will be written
    #     psf_for_parameters:
    #         Optional psf file which contains all possible molecules/states, if they are not represented by the old_psf_infname.
    #         I.e. one species gets protonated and is not present anymore, this file can be used to have all potential states.

    #     Returns
    #     -------
    #     None
    #     """
    #     if psf_for_parameters is None:
    #         psf_for_parameters = old_psf_infname

    #     pm_old_psf = parmed.charmm.CharmmPsfFile(old_psf_infname)
    #     # copying parmed structure did not work
    #     # pm_old_psf_copy = parmed.charmm.CharmmPsfFile(old_psf_infname)
    #     pm_parameters = parmed.charmm.CharmmPsfFile(psf_for_parameters)
    #     pm_new_psf = self._adapt_parmed_psf_file(pm_old_psf, pm_parameters)
    #     pm_new_psf.write_psf(new_psf_outfname)

    # # possibly in future when parmed and openmm drude connection is working
    # # def write_psf_notworking(
    # #     self, fname: str, format=None, overwrite=False, **kwargs
    # # ) -> None:
    # #     """
    # #     Write a psf file from the current topology.
    # #     In principle any file that parmeds struct.save method supports can be written.
    # #     """
    # #     import parmed

    # #     struct = parmed.openmm.load_topology(self.topology, self.system)
    # #     struct.save(fname, format=None, overwrite=False, **kwargs)

    def saveCheckpoint(self, file) -> None:
        """Wrapper method which just calls the underlying same function on the simulation object of the ionic liquid object.

        Parameters
        ----------
        file: string or file
            a File-like object to write the checkpoint to, or alternatively a
            filename
        """
        self.simulation.saveCheckpoint(file)

    def loadCheckpoint(self, file) -> None:
        """Wrapper method which just calls the underlying same function on the simulation object of the ionic liquid object.

        Parameters
        ----------
        file : string or file
            a File-like object to load the checkpoint from, or alternatively a
            filename
        """
        self.simulation.loadCheckpoint(file)

    def saveState(self, file) -> None:
        """Wrapper method which just calls the underlying same function on the simulation object of the ionic liquid object.

        Parameters
        ----------
        file : string or file
            a File-like object to write the state to, or alternatively a
            filename
        """
        self.simulation.saveState(file)

    def loadState(self, file) -> None:
        """Wrapper method which just calls the underlying same function on the simulation object of the ionic liquid object.

        Parameters
        ----------
        file : string or file
            a File-like object to load the state from, or alternatively a
            filename
        """
        self.simulation.loadState(file)

    def save_updates(self, file) -> None:
        """Save the current update values into a yaml file. Used to have the current probability values.

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
        """Load the current update values from a yaml file, which was generated using "save_updates".

        Parameters
        ----------
        file: string or file
        """
        with open(file) as f:
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
