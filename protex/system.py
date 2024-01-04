from __future__ import annotations

import itertools
import logging
import pickle
import warnings
from collections import ChainMap, defaultdict

import parmed
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
        states: list[dict[str, dict[str, list]]],
        allowed_updates: dict[frozenset[str], dict[str, float]],
    ) -> None:
        self.__states = states
        self.pairs: list[list[str]] = [list(i.keys()) for i in states]  # same
        self.states: dict[str, dict[str, list[dict[str, str]]]] = dict(  # different
            ChainMap(*states)
        )  # first check that names are unique?
        self.names: list[str] = list(  # same
            itertools.chain(*self.pairs)
        )  # also what about duplicates
        self.ordered_names: list[
            tuple[str,...]
        ] = self._setup_ordered_names()
        self.allowed_updates: dict[frozenset[str], dict[str, float]] = allowed_updates
        self.overall_max_distance: float = max(
            [value["r_max"] for value in self.allowed_updates.values()]
        )

        self._starting_donors: str = "starting_donors"
        self._starting_acceptors: str = "starting_acceptors"
        self._modes: str = "possible_modes"

    # TODO function to determine cutoffs from equilibration dcd (MDAnalysis, find shortest distances between atom pairs)

    def get_states(self):
        return self.__states

    def get_all_states_for(self, resname: str) -> dict[str,dict[str,list[dict[str,str]]]]:
        """Get all the state information for a specific resname and all corresponding resnames.

        Parameters
        ----------
        resname : str
            The resname

        Returns
        -------
        dict[str,dict[str,list[dict[str,str]]]]
            A dict with the resname and all information

        Raises
        ------
        RuntimeError
            If the given resname does not exist
        """
        # return all info correpsonding to one pair tuple, i.e all for oac,hoac,h2oac
        current_states = {}
        for pair in self.pairs:
            if resname in pair:
                for name in pair:
                    current_states[name] = self.states[name]
                return current_states
        raise RuntimeError(f"Resname {resname} not found. Typo?")

    def _setup_ordered_names(self) -> list[tuple[str,...]]:
        # from low H to many H
        #from most acceptor to most donor
        #i.e. oac -> -1 (one acceptor)
        #hoac -> 0 (one acceptor, one donor)
        #h2oac -> 1 (one donor)
        ordered_names = []
        for pair in self.pairs:
            count_dict = {}
            for p in pair:
                count_dict[p] = 0
                if "donor" in self.states[p]["possible_modes"]:
                    count_dict[p] +=1
                if "acceptor" in self.states[p]["possible_modes"]:
                    count_dict[p] -= 1
            sorted_count_dict = tuple(sorted(count_dict, key=lambda k:(count_dict[k],k)))
            # logger.debug(sorted_count_dict)
            ordered_names.append(sorted_count_dict)

        # logger.debug(ordered_names)
        return ordered_names
        #returns something like [("IM1", "IM1H"), ("OAC", "HOAC", "H2OAC")]

    def get_ordered_names_for(self, resname: str) -> tuple[str,...]:
        """Get the tuple of the correct resname sequences for a given resname.

        Parameters
        ----------
        resname : str
            The resname

        Returns
        -------
        tuple[str]
            A tuple with the order of resnames from low H (acceptor) to high H (donor) count
        """
        # logger.debug(self.ordered_names)
        # logger.debug(resname)
        # logger.debug(any(resname in sublist for sublist in self.ordered_names))
        if any(resname in sublist for sublist in self.ordered_names):
            for tup in self.ordered_names:
                if resname in tup:
                    return tup
        else: # not protexable resi
            return [resname]
        #raise RuntimeError(f"Resname {resname} not found in the present names.")

    def dump(self, fname: str) -> None:
        """Pickle the current ProtexTemplates object.

        Parameters
        ----------
        fname : str
            The file name of the object
        """
        with open(fname, "wb") as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def get_starting_donors_for(self, resname: str) -> tuple:
        """Get the atom names of donors for a specific residue.

        Parameters
        ----------
        resname : str
            The residue name

        Returns
        -------
        tuple
            The atom names
        """
        return self.states[resname][self._starting_donors]

    def get_starting_acceptors_for(self, resname: str) -> tuple:
        """Get the atom names of acceptors for a specific residue.

        Parameters
        ----------
        resname : str
            The residue name

        Returns
        -------
        tuple
            The atom names
        """
        return self.states[resname][self._starting_acceptors]

    def get_possible_modes_for(self, resname: str) -> tuple:
        """Get the possible modes for a specific residue.

        Parameters
        ----------
        resname : str
            The residue name

        Returns
        -------
        dictionary
            dictionary of tuples of mode(s) for each other_name
        """
        modes = {}
        onames = self.get_ordered_names_for(resname)
        for oname in onames:
            modes[oname] = self.states[oname][self._modes]
        return modes

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

    def get_other_resnames(self, resname: str) -> list[str]:
        """Return the corresponding alternative resname to the given one.

        Parameters
        ----------
        resname : str
            The resname

        Returns
        -------
        list[str]
            The other resnames connected to the current one
        """
        # species (e.g. solvent) that should not exchange protons
        if resname not in self.names:
            return [resname]

        else:
            for pair in self.pairs:
                if resname in pair:
                    pair_copy = pair[:]
                    pair_copy.remove(resname)
                    return pair_copy



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
    #####################
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

        system = from_pickle[0]
        templates = from_pickle[1]
        residues = from_pickle[2]

        protex_system = ProtexSystem(
            simulation, templates, simulation_for_parameters
        )
        protex_system.simulation.system = system
        protex_system.residues = residues

        return protex_system

    def __init__(
        self,
        simulation: openmm.app.simulation.Simulation,
        templates: ProtexTemplates,
        simulation_for_parameters: openmm.app.simulation.Simulation = None,
        real_Hs: list[tuple[str,str]] = [("TOH2", "H1"), ("TOH2", "H2"), ("H2O", "H1"), ("H2O", "H2"), ("OH", "H1"),
                                         ("TOH3", "H1"), ("TOH3", "H2"), ("TOH3", "H3"), ("H3O", "H1"), ("H3O", "H2"), ("H3O", "H3"),
                                         ("HOAC", "HO2"), ("IM1H", "H7"),
                                         ("ULF", "HD1"), ("ULF", "HE2"), ("UDO", "HD1")
                                         ],
        fast: bool = True,
    ) -> None:
        self.system: openmm.openmm.System = simulation.system
        self.topology: openmm.app.topology.Topology = simulation.topology
        self.simulation: openmm.app.simulation.Simulation = simulation
        self.templates: ProtexTemplates = templates
        self.simulation_for_parameters = simulation_for_parameters
        self.real_Hs = real_Hs
        self._check_forces()
        self.detected_forces: set[str] = self._detect_forces()
        self.fast: bool = fast
        self.residues: list[Residue] = self._set_initial_states()
        self.boxlength: openmm.Quantity = (
            simulation.context.getState().getPeriodicBoxVectors()[0][0]
        )  # NOTE: supports only cubic boxes


    def dump(self, fname: str) -> None:
        """Pickle the current ProtexSystem object.

        Parameters
        ----------
        fname : str
            The file name to store the object
        """
        to_pickle = [self.system, self.templates, self.residues]  # ensure correct order of arguments
        with open(fname, "wb") as outp:
            pickle.dump(to_pickle, outp, pickle.HIGHEST_PROTOCOL)


    def _detect_forces(self) -> set[str]:
        def _is_populated(force): # deprecated
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

        # NOTE is there a more elegant way of finding out if a force exists in a residue?
        def _is_populated_in_residue(force, residue):
            if type(force).__name__ in self.IGNORED_FORCES:
                return False
            atom_idxes  = [atom.index for atom in residue.atoms()]

            if isinstance(force, openmm.NonbondedForce):
                val = False
                for idx in atom_idxes:
                    try:
                        force_entry = force.getParticleParameters(idx)
                        val = True # force exists if we can get parameters for atoms in resi
                    except:
                        continue
                return val

            elif isinstance(force, openmm.HarmonicBondForce):
                for bond_id in range(force.getNumBonds()):  # iterate over all bonds
                    f = force.getBondParameters(bond_id)
                    for atom_idx in atom_idxes:
                        if atom_idx in f[0:2]:
                            return True # force exists if there is an entry involving the atom index
                return False

            elif isinstance(force, openmm.HarmonicAngleForce):
                for angle_id in range(force.getNumAngles()):  # iterate over all angles
                    f = force.getAngleParameters(angle_id)
                    for atom_idx in atom_idxes:
                        if atom_idx in f[0:3]:
                            return True # force exists if there is an entry involving the atom index
                return False

            elif isinstance(force, openmm.PeriodicTorsionForce):
                for torsion_id in range(force.getNumTorsions()):  # iterate over all dihedrals
                    f = force.getTorsionParameters(torsion_id)
                    for atom_idx in atom_idxes:
                        if atom_idx in f[0:4]:
                            return True # force exists if there is an entry involving the atom index
                return False

            elif isinstance(force, openmm.CustomTorsionForce):
                for torsion_id in range(force.getNumTorsions()):  # iterate over all dihedrals
                    f = force.getTorsionParameters(torsion_id)
                    for atom_idx in atom_idxes:
                        if atom_idx in f[0:4]:
                            return True # force exists if there is an entry involving the atom index
                return False

            elif isinstance(force, openmm.DrudeForce):
                for drude_id in range(force.getNumParticles()):  # iterate over all drudes TODO still need to handle Thole screening (is workaround in residue.py okay?)
                    f = force.getParticleParameters(drude_id)
                    for atom_idx in atom_idxes:
                        if atom_idx == f[0]:
                            return True # force exists if there is an entry involving the atom index of the drude
                return False

            elif isinstance(force, openmm.CustomNonbondedForce):
                val = False
                for idx in atom_idxes:
                    try:
                        force_entry = force.getParticleParameters(idx)
                        val = True # force exists if we can get parameters for atoms in resi
                    except:
                        logger.debug("no customnbforce found")
                        continue
                return val

            elif isinstance(force, openmm.CMAPTorsionForce):
                for torsion_id in range(force.getNumTorsions()):  # iterate over all dihedrals
                    f = force.getTorsionParameters(torsion_id)
                    for atom_idx in atom_idxes:
                        if atom_idx in f[1:9]:
                            return True # force exists if there is an entry involving the atom index
                return False

            else:
                raise ProtexException(f"This force should be covered here: {type(force).__name__}: {force}")

        # make a dictionary of forces present in each residue
        if self.simulation_for_parameters is not None:
            detected_forces: dict = {}
            for residue in self.simulation_for_parameters.topology.residues():
                if residue.name not in detected_forces.keys():
                    detected_forces[residue.name] = []
                    for force in self.simulation_for_parameters.system.getForces():
                        if _is_populated_in_residue(force, residue):
                            detected_forces[residue.name].append(type(force).__name__)
                    #detected_forces[residue.name] = set(detected_forces[residue.name]) # remove duplicates

        else:
            detected_forces: dict = {}
            for residue in self.topology.residues():
                if residue.name not in detected_forces.keys():
                    detected_forces[residue.name] = []
                    for force in self.system.getForces():
                        if _is_populated_in_residue(force, residue):
                            detected_forces[residue.name].append(type(force).__name__)
                    detected_forces[residue.name] = set(detected_forces[residue.name]) # remove duplicates

        # logger.debug(detected_forces)
        print(f"{detected_forces=}")
        return detected_forces

    def _check_forces(self) -> None:
        """Will fail if a force is not covered."""
        for force in self.system.getForces():
            self.force_is_valid(type(force).__name__)
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

    # deprecated
    # def _build_exclusion_list(self, topology):
    #     pair_12_set = set()
    #     pair_13_set = set()
    #     for bond in topology.bonds():
    #         a1, a2 = bond.atom1, bond.atom2
    #         if "H" not in a1.name and "H" not in a2.name:
    #             pair = (
    #                 min(a1.index, a2.index),
    #                 max(a1.index, a2.index),
    #             )
    #             pair_12_set.add(pair)
    #     for a in pair_12_set:
    #         for b in pair_12_set:
    #             shared = set(a).intersection(set(b))
    #             if len(shared) == 1:
    #                 pair = tuple(sorted(set(list(a) + list(b)) - shared))
    #                 pair_13_set.add(pair)
    #                 # there were duplicates in pair_13_set, e.g. (1,3) and (3,1), needs to be sorted

    #     # self.pair_12_list = list(sorted(pair_12_set))
    #     # self.pair_13_list = list(sorted(pair_13_set - pair_12_set))
    #     # self.pair_12_13_list = self.pair_12_list + self.pair_13_list
    #     # change to return the list and set the parameters in the init method?
    #     pair_12_list = list(sorted(pair_12_set))
    #     pair_13_list = list(sorted(pair_13_set - pair_12_set))
    #     pair_12_13_list = pair_12_list + pair_13_list
    #     return pair_12_13_list

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
                # atom_names = [atom.name for atom in residue.atoms()]
                # logger.debug(atom_idxs)
                # logger.debug(atom_names)

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
                            if idx1 in atom_idxs or idx2 in atom_idxs:
                                forces_dict[forcename + "Exceptions"].append(f)

                    elif forcename == "CustomNonbondedForce":
                        # lookup indices of the tabulated table (?)
                        #index is position, but what about previous ones?
                        # BUG problem with slow method: need every atom idx for exceptions, now only from 1 residue
                        # TODO normalize / offset etc for each residue
                        # logger.debug(force)
                        forces_dict[forcename] = [force.getParticleParameters(idx) for idx in atom_idxs]
                        # Also add exclusions
                        for exc_id in range(force.getNumExclusions()):
                            f = force.getExclusionParticles(exc_id)
                            idx1 = f[0]
                            idx2 = f[1]
                            if idx1 in atom_idxs or idx2 in atom_idxs:
                                forces_dict[forcename + "Exclusions"].append(f)

                    elif forcename == "HarmonicBondForce":
                        for bond_id in range(force.getNumBonds()):
                            f = force.getBondParameters(bond_id)
                            idx1 = f[0]
                            idx2 = f[1]
                            # changed bonds, angles, dihedrals etc. to check for at least 1 atom belinging to residue (in protein residues are connected)
                            # TODO now forces can belong to multiple residues -> what happens at update? (esp. if we update 2 neighbouring resis)
                            # TODO update force_idx_dict as well? (now maxi can be in the next residue, force is ignored for the first one)
                            if idx1 in atom_idxs or idx2 in atom_idxs: 
                                forces_dict[forcename].append(f)

                    elif forcename == "HarmonicAngleForce":
                        for angle_id in range(force.getNumAngles()):
                            f = force.getAngleParameters(angle_id)
                            if (
                                f[0] in atom_idxs
                                or f[1] in atom_idxs
                                or f[2] in atom_idxs
                            ):
                                forces_dict[forcename].append(f)

                    elif forcename == "PeriodicTorsionForce":
                        for torsion_id in range(force.getNumTorsions()):
                            f = force.getTorsionParameters(torsion_id)
                            if (
                                f[0] in atom_idxs
                                or f[1] in atom_idxs
                                or f[2] in atom_idxs
                                or f[3] in atom_idxs
                            ):
                                forces_dict[forcename].append(f)

                    elif forcename == "CustomTorsionForce":
                        for torsion_id in range(force.getNumTorsions()):
                            f = force.getTorsionParameters(torsion_id)

                            if (
                                f[0] in atom_idxs
                                or f[1] in atom_idxs
                                or f[2] in atom_idxs
                                or f[3] in atom_idxs
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
                            if drude1 in atom_idxs or drude2 in atom_idxs:
                                forces_dict[forcename + "Thole"].append(f)
                break  # do this only for the relevant residue once
        else:
            raise RuntimeError(f"residue not found: {query_name}")
        return forces_dict


    def _extract_H_templates(self, query_name: str) -> defaultdict:
        # returns the nonbonded parameters of real Hs for the residue name
        forces_dict = defaultdict(list)

        # if there is an additional parameter file with all possible residues,
        # use this for getting the templates
        if self.simulation_for_parameters is not None:
            sim = self.simulation_for_parameters
        else:
            sim = self.simulation

        for residue in sim.topology.residues():
            if query_name == residue.name:
                # logger.debug(residue.name)
                # logger.debug(self.real_Hs)
                atom_names = [self.real_Hs[i][1] for i in range(len(self.real_Hs)) if self.real_Hs[i][0] == residue.name]
                atom_idxs_all = [atom.index for atom in residue.atoms()]
                atom_names_all = [atom.name for atom in residue.atoms()]
                atom_idxs = [atom_idxs_all[i] for i in range(len(atom_idxs_all)) if atom_names_all[i] in atom_names]
                # logger.debug(atom_idxs)
                # logger.debug(atom_names)

                for force in sim.system.getForces():
                    forcename = type(force).__name__
                    # logger.debug(forcename)
                    if forcename == "NonbondedForce":
                        if len(atom_names) == 1:
                            idx = atom_idxs[0]
                            forces_dict[forcename] = force.getParticleParameters(idx)

                        # # Also add exceptions TODO: what do we do with these? they need atom idxes and can be applied to e.g H1 and H2. is there a difference with dummies?
                        # # will probably leave them out for the moment. the number of bonds is the same with H and D, nonbonded exceptions should still apply
                        # for exc_id in range(force.getNumExceptions()):
                        #     f = force.getExceptionParameters(exc_id)
                        #     idx1 = f[0]
                        #     idx2 = f[1]
                        #     if (idx1 in atom_idxs and idx2 in atom_idxs_all) or (idx2 in atom_idxs and idx1 in atom_idxs_all):
                        #         forces_dict[forcename + "Exceptions"].append(f)

                        # double-checking for molecules with multiple equivalent atoms, then use one set of parameters only (we assume here that all acidic Hs in the residue are the same, e.g. MeOH2, H2O, H2OAc)
                            # TODO expand this part to cover multiple different acidic Hs, e.g. couple parameters to atom names
                        elif len(atom_names) > 1:
                            forces_dict[forcename] = [
                                force.getParticleParameters(idx) for idx in atom_idxs
                            ]
                            # logger.debug(f"forces_dict: {forces_dict[forcename]}")
                            for i in range(len(atom_names)):
                                assert (forces_dict[forcename][0][0], forces_dict[forcename][0][1], forces_dict[forcename][0][2]) == (forces_dict[forcename][i][0], forces_dict[forcename][i][1], forces_dict[forcename][i][2])
                            forces_dict[forcename] = forces_dict[forcename][0]

                break  # do this only for the relevant residue once
        else:
            raise RuntimeError("residue not found")
        # logger.debug(forces_dict)
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
                logger.critical(name_of_paired_ion)
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
                    # different force names can belong to same fgroup (e.g. DrudeForce and DrudeForceThole)
                    # force names were overwritten with the old method
                    try:
                        self.per_residue_forces[tuple_item][fgroup][forcename] = {}
                        self.per_residue_forces[tuple_item][fgroup][forcename] = [
                            insert_values
                        ]
                    except KeyError:
                        self.per_residue_forces[tuple_item][fgroup] = {}
                        self.per_residue_forces[tuple_item][fgroup][forcename] = [
                            insert_values
                        ]

    def _create_force_idx_dict(self) -> None:
        """Create a dictionary containig all the indices to the forces.

        It is used to specify which force belongs to which residue.
        It fills the self.per_residue_forces variable.
        """
        for force in self.system.getForces():
            forcename = type(force).__name__
            fgroup = force.getForceGroup()
            if forcename == "NonbondedForce":
                # only treat exceptions # why???
                for exc_idx in range(force.getNumExceptions()):
                    f = force.getExceptionParameters(exc_idx)
                    idx1, idx2 = f[0], f[1]
                    value = (exc_idx, idx1, idx2)
                    maxi = max(idx1, idx2)
                    self._add_force(fgroup, "NonbondedForceExceptions", maxi, value)
            elif forcename == "CustomNonbondedForce":
                for exc_idx in range(force.getNumExclusions()):
                    f = force.getExclusionParticles(exc_idx)
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
                    self._add_force(fgroup, "DrudeForceThole", maxi, value) # BUG both DrudeForce and DrudeForceThole are in fgroup 7 -> overwritten -> hopefully fixed now
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

    # def _fill_residue_templates(self, name): # not used anymore?
    #     if name in self.templates.names:
    #         name_of_paired_ion = self.templates.get_residue_name_for_coupled_state(name)
    #         if (
    #             name in self.residue_templates
    #             or name_of_paired_ion in self.residue_templates
    #         ):
    #             return
    #         self.residue_templates[name] = self._extract_templates(name)
    #         self.residue_templates[name_of_paired_ion] = self._extract_templates(
    #             name_of_paired_ion
    #         )
    #     else:
    #         if name in self.residue_templates:  # or name_of_paired_ion in templates:
    #             return
    #         self.residue_templates[name] = self._extract_templates(name)

    # def _fill_H_templates(self, name): # not used anymore?
    #     if name in self.templates.names:
    #         if name in self.H_templates:
    #             return
    #         self.H_templates[name] = self._extract_H_templates_H(name)
    #     else:
    #         if name in self.H_templates:  # or name_of_paired_ion in templates:
    #             return
    #         self.H_templates[name] = self._extract_H_templates_H(name)

    def _set_initial_states(self) -> list:
        """set_initial_states.

        For each ionic liquid residue in the system the protonation state
        is interfered from the provided openMM system object and the protonation site is defined.
        """
        #pair_12_13_list = self._build_exclusion_list(self.topology) # deprecated

        residues = []
        templates = dict()
        H_templates = dict()
        # this will become a dict of the form:
        # self.per_residue_forces[(min,max tuple of the atom idxs for each residue)][forcegroup][forcename]: list of the idxs of this force for this residue
        self.per_residue_forces = {}
        # for each residue type get forces

        for r in self.topology.residues():
            name = r.name
            ordered_names = self.templates.get_ordered_names_for(name)

            # skip if name or all corresponding names are already in the template
            if name in templates and all(oname in templates for oname in ordered_names):
                continue

            for oname in ordered_names:
                templates[oname] = self._extract_templates(oname)
                H_templates[oname] = self._extract_H_templates(oname)

            # logger.debug(H_templates)

        for r in self.topology.residues():
            atom_idxs = [atom.index for atom in r.atoms()]
            mini, maxi = min(atom_idxs), max(atom_idxs)
            self.per_residue_forces[(mini, maxi)] = {}


        if self.fast:
            # this takes some time, but the update calls on the residues are then much faster
            self._force_idx_dict = self._create_force_idx_dict()
            #print(f"{self.per_residue_forces=}")

        handled_resis = [] # for debugging
        for r in self.topology.residues():
            name = r.name
            if name in self.templates.names:
                ordered_names = self.templates.get_ordered_names_for(name)
                assert name in ordered_names
                parameters = {}
                H_parameters = {}
                for oname in ordered_names:
                    parameters[oname] = templates[oname]
                    H_parameters[oname] = H_templates[oname]

                # check that we have the same number of parameters
                for name1, name2 in itertools.combinations(parameters, 2):
                    self._check_nr_of_forces(
                        parameters[name1], parameters[name2], name1, name2
                    )

                atom_idxs = [
                    atom.index for atom in r.atoms()
                ]  # also give them to initilaizer, not inside residue?
                minmax = (min(atom_idxs), max(atom_idxs))
                if self.fast:
                    new_res = Residue(
                        r,
                        ordered_names,
                        self.system,
                        parameters,
                        H_parameters,
                        self.templates.get_all_states_for(name),
                        self.templates.get_possible_modes_for(name),
                        self.templates.get_starting_donors_for(name),
                        self.templates.get_starting_acceptors_for(name),
                        self.templates.get_starting_donors_for(name),
                        self.templates.get_starting_acceptors_for(name),
                        force_idxs=self.per_residue_forces[minmax],
                    )
                    if name not in handled_resis: # for debugging
                        print(f"{name}: {self.per_residue_forces[minmax]}")
                        print(f"{name}: {len(self.per_residue_forces[minmax][0]['HarmonicBondForce'])+len(self.per_residue_forces[minmax][3]['HarmonicBondForce'])}")
                        print((new_res.parameters[name]['HarmonicBondForce']))
                        handled_resis.append(name)
                else:
                    new_res = Residue(
                        r,
                        ordered_names,
                        self.system,
                        parameters,
                        H_parameters,
                        self.templates.get_all_states_for(name),
                        self.templates.get_possible_modes_for(name),
                        self.templates.get_starting_donors_for(name),
                        self.templates.get_starting_acceptors_for(name),
                        self.templates.get_starting_donors_for(name),
                        self.templates.get_starting_acceptors_for(name),
                    )
                residues.append(new_res)
                # residues[
                #     -1
                # ].current_name = (
                #     name  # Why, isnt it done in the initializer of Residue?
                # )
            else:
                parameters = {name: templates[name]}
                # parameters_state1 = templates[name]
                r = Residue(r, None, self.system, parameters, None, None, None, None, None, None, None, None)
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

    # def save_donors_acceptors(self, file: str) -> None:
    #     """
    #     Save a file with the current Hs and Ds for each residue.
    #     Can be used with load_donors_acceptors to set the residues in the IonicLiquidSystem
    #     in the state of these names and also adapt corresponding charges, parameters,...
    #     """
    #     with open(file, "w") as f:
    #         for residue in self.residues:
    #             f.write(f"{residue.donors}; {residue.acceptors}")

    # def load_current_names(self, file: str) -> None:
    #     """
    #     Load the names of the residues (order important!)
    #     Update the donors and acceptors of all residues to the given one
    #     """
    #     residue_donors = []
    #     residue_acceptors = []
    #     with open(file, "r") as f:
    #         for line in f.readlines():
    #             residue_donors.append(line.split(";")[0])
    #             residue_acceptors.append(line.split(";")[1])
    #     assert (
    #         len(residue_donors) == len(residue_acceptors) == self.topology.getNumResidues()
    #     ), "Number of residues not matching"
    #     for residue, donor, acceptor in zip(self.residues, residue_donors, residue_acceptors):
    #         residue.donors = donor
    #         residue.acceptors = acceptor

    # not used
    # def report_states(self) -> None:
    #     """
    #     report_states prints out a summary of the current protonation state of the ionic liquid
    #     """
    #     pass


    # NOTE saving psfs is deprecated with pickling protex system
    def _adapt_parmed_psf_file(
        self,
        psf: parmed.charmm.CharmmPsfFile,
        parameters: parmed.charmm.CharmmPsfFile,
    ) -> parmed.charmm.CharmmPsfFile:
        """Helper function to adapt the psf."""
        print(len(self.residues), len(psf.residues))
        assert len(self.residues) == len(psf.residues)

        # make a dict with parmed representations of each residue, use it to assign the opposite one if a transfer occured
        pm_unique_residues: dict[str, parmed.Residue] = {}
        # incremented by one each time it is used to track the current residue number
        residue_counts: dict[str, int] = {}

        for pm_residue in parameters.residues:
            if pm_residue.name in pm_unique_residues:
                continue
            else:
                pm_unique_residues[pm_residue.name] = pm_residue
                residue_counts[pm_residue.name] = 1

        # get offset for lonepairs which are defined differently.
        # dict with lists for each LocalCoordinates frame, there is one for each LP in the residue
        differences_dict: dict[tuple[str], list[list[int]]] = {}
        for pair in self.templates.pairs:
            name1, name2 = pair
            pm_res1 = pm_unique_residues[name1]
            pm_res2 = pm_unique_residues[name2]
            # get first index to get relative difference
            at1_idx1 = pm_res1.atoms[0].idx
            at2_idx1 = pm_res2.atoms[0].idx

            differences_dict[tuple([name1, name2])] = []
            differences_dict[tuple([name2, name1])] = []

            for at1, at2 in zip(pm_res1, pm_res2):
                if isinstance(at1, parmed.topologyobjects.ExtraPoint) and isinstance(
                    at1.frame_type, parmed.topologyobjects.LocalCoordinatesFrame
                ):
                    frame_hosts1 = [
                        fatom
                        for fatom in at1.frame_type.get_atoms()[
                            : at1.frame_type.frame_size
                        ]
                    ]
                    frame_hosts2 = [
                        fatom
                        for fatom in at2.frame_type.get_atoms()[
                            : at2.frame_type.frame_size
                        ]
                    ]
                    relative_pos1 = [fatom.idx - at1_idx1 for fatom in frame_hosts1]
                    relative_pos2 = [fatom.idx - at2_idx1 for fatom in frame_hosts2]
                    differences = [
                        p1 - p2 for p1, p2 in zip(relative_pos1, relative_pos2)
                    ]
                    # Depending on the direction of the tranfer we either need the positive or negative value
                    differences_dict[tuple([name1, name2])].append(differences)
                    differences_dict[tuple([name2, name1])].append(
                        [-d for d in differences]
                    )

        # either it is the same or just one group will be assumed. -> Not best for proteins, but hopefully Parmed will release new version soon, so that we do not need all these hacks.
        n_residue_is_n_groups = len(psf.groups) == len(psf.residues)
        group_iter = iter(psf.groups)
        for residue, pm_residue in zip(self.residues, psf.residues):
            # if the new residue (residue.current_name) is different than the original one from the old psf (pm_residue.name)
            # a proton transfer occured and we want to change this in the new psf, which means overwriting the parmed residue instance
            # with the new information
            # if residue.current_name != pm_residue.name:
            # do changes
            name = residue.current_name
            name_change: tuple[str] = tuple([name, pm_residue.name])
            # get the differences in LP position, make an iterator to yiled the next as needed
            diff_iter = None
            if name_change in differences_dict.keys():
                differences_list = differences_dict[name_change]
                diff_iter = iter(differences_list)
            pm_residue.name = name
            pm_residue.chain = name
            pm_residue.segid = name
            pm_residue.number = residue_counts[name]
            for unique_atom, pm_atom in zip(
                pm_unique_residues[name].atoms, pm_residue.atoms
            ):
                # TODO this block has to be changed to account for different possible dummy-real H combinations, esp. charge and type
                pm_atom._charge = unique_atom._charge
                pm_atom.type = unique_atom.type
                pm_atom.props = unique_atom.props
                # NUMLP NUMLPH lonepair section update
                if isinstance(
                    unique_atom, parmed.topologyobjects.ExtraPoint
                ) and isinstance(
                    unique_atom.frame_type, parmed.topologyobjects.LocalCoordinatesFrame
                ):
                    pm_atom.frame_type.distance = unique_atom.frame_type.distance
                    pm_atom.frame_type.angle = unique_atom.frame_type.angle
                    pm_atom.frame_type.dihedral = unique_atom.frame_type.dihedral
                    # if the positioning of the lonepair changes, update the corresponding atoms in the LocalCoordinateFrame
                    if diff_iter:
                        differences = next(diff_iter)
                        first_idx = pm_residue.atoms[0].idx

                        # normalize to the first idx in the list of atoms for one residue
                        current_idx1 = pm_atom.frame_type.atom1.idx
                        # print(f"{current_idx1=}")
                        # print(f"{differences[0]=}")
                        pm_atom.frame_type.atom1 = pm_residue.atoms[
                            current_idx1 - first_idx + differences[0]
                        ]
                        current_idx2 = pm_atom.frame_type.atom2.idx
                        # print(f"{current_idx2=}")
                        # print(f"{differences[1]=}")
                        pm_atom.frame_type.atom2 = pm_residue.atoms[
                            current_idx2 - first_idx + differences[1]
                        ]
                        current_idx3 = pm_atom.frame_type.atom3.idx
                        # print(f"{current_idx3=}")
                        # print(f"{differences[2]=}")
                        pm_atom.frame_type.atom3 = pm_residue.atoms[
                            current_idx3 - first_idx + differences[2]
                        ]

                # NUMANISO Section update
                if isinstance(unique_atom, parmed.topologyobjects.DrudeAtom):
                    if unique_atom.anisotropy is None:
                        continue
                    pm_atom.anisotropy.params["k11"] = unique_atom.anisotropy.params[
                        "k11"
                    ]
                    pm_atom.anisotropy.params["k22"] = unique_atom.anisotropy.params[
                        "k22"
                    ]
                    pm_atom.anisotropy.params["k33"] = unique_atom.anisotropy.params[
                        "k33"
                    ]
            residue_counts[name] += 1

            if n_residue_is_n_groups:
                group = next(group_iter)
                typ = 1 if abs(sum(a.charge for a in pm_residue.atoms)) < 1e-4 else 2
                group.type = typ

        if not n_residue_is_n_groups:
            # Maybe a bit hacky to get Parmed to use all atoms as 1 group
            # https://github.com/ParmEd/ParmEd/blob/master/parmed/formats/psf.py#L250
            psf.groups = []

        return psf

    # NOTE saving psfs is deprecated with pickling protex system
    def write_psf(
        self,
        old_psf_infname: str,
        new_psf_outfname: str,
        psf_for_parameters: str = None,
    ) -> None:
        """Write a new psf file, which reflects the occured transfer events and changed residues
        to load the written psf create a new ionic_liquid instance and load the new psf via OpenMM.

        Parameters
        ----------
        old_psf_infname:
            Name of the old psf_file, which serves for the basic strucutre, same number of atoms, same bonds, angles, ...
        new_psf_outfname:
            Name of the new psf that will be written
        psf_for_parameters:
            Optional psf file which contains all possible molecules/states, if they are not represented by the old_psf_infname.
            I.e. one species gets protonated and is not present anymore, this file can be used to have all potential states.

        Returns
        -------
        None
        """
        if psf_for_parameters is None:
            psf_for_parameters = old_psf_infname

        pm_old_psf = parmed.charmm.CharmmPsfFile(old_psf_infname)
        # copying parmed structure did not work
        # pm_old_psf_copy = parmed.charmm.CharmmPsfFile(old_psf_infname)
        pm_parameters = parmed.charmm.CharmmPsfFile(psf_for_parameters)
        pm_new_psf = self._adapt_parmed_psf_file(pm_old_psf, pm_parameters)
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
