from __future__ import annotations

import copy
import logging
import pickle
import random
from abc import ABC, abstractmethod
from collections import Counter, deque

import numpy as np
from scipy.spatial import distance_matrix

try:
    from openmm.unit import nanometers
except ImportError:
    from simtk.unit import nanometers

from protex.residue import Residue
from protex.system import ProtexSystem

logger = logging.getLogger(__name__)


class Update(ABC):
    """ABC for implementing different Update Methods.

    Parameters
    ----------
    ionic_liquid: ProtexSystem
    """

    @staticmethod
    @abstractmethod
    def load(fname: str, protex_system: ProtexSystem) -> Update:
        """Load a picklesUpdate instance.

        Parameters
        ----------
        fname : str
            The file name
        protex_system : ProtexSystem
            An instance of ProtexSystem, used to create the Update instance

        Returns
        -------
        Update
            An update instance
        """
        pass

    def __init__(
        self,
        ionic_liquid: ProtexSystem,
        to_adapt: list[tuple[str, int, frozenset[str]]],
        all_forces: bool,
        include_equivalent_atom: bool,
        reorient: bool,
    ) -> None:
        self.ionic_liquid: ProtexSystem = ionic_liquid
        self.to_adapt: list[tuple[str, int, frozenset[str]]] = to_adapt
        self.all_forces: bool = all_forces
        self.include_equivalent_atom = include_equivalent_atom
        self.reorient = reorient
        allowed_forces: list[str] = [  # change charges only
            "NonbondedForce",  # BUG: Charge stored in the DrudeForce does NOT get updated, probably you want to allow DrudeForce as well!
            "CustomNonbondedForce",  # NEW
            "DrudeForce",
        ]
        if self.all_forces:
            allowed_forces.extend(
                [
                    "HarmonicBondForce",
                    "HarmonicAngleForce",
                    "PeriodicTorsionForce",
                    "CustomTorsionForce",
                ]
            )
        self.reject_length: int = (
            10  # specify the number of update steps the same residue will be rejected
        )
        self.allowed_forces = list(set(allowed_forces).intersection(self.ionic_liquid.detected_forces))
        discarded = set(allowed_forces).difference(self.ionic_liquid.detected_forces)
        if discarded:
            print(f"Discarded the following forces, becuase they are not in the system: {', '.join(discarded)}")
        available = set(self.ionic_liquid.detected_forces).difference(set(allowed_forces))
        if available:
            print(f"The following forces are available but not updated: {', '.join(available)}")

    @abstractmethod
    def dump(self, fname: str) -> None:
        """Pickle an Update instance.

        Parameters
        ----------
        fname : str
            The file name
        """
        pass

    @abstractmethod
    def _update(self, candidates: list[tuple], nr_of_steps: int) -> None:
        pass

    def _adapt_probabilities(
        self, to_adapt=list[tuple[str, int, frozenset[str]]]
    ) -> None:
        """Adapt the probability for certain events depending on the current equilibrium, in order to stay close to a given reference
        i.e. prob_neu = prob_orig + K*( x(t) - x(eq) )^3 where x(t) is the current percentage in the system of one species.

        Parameters
        ----------
        to_adapt: List of tuples with first the residue name of the species,
            the number of residues of this species in the system
            and the specific reaction in which this species occurs and the probability should be updated
        """
        # check that there are not duplicate frozen sets
        counts = Counter(s[2] for s in to_adapt)
        assert (
            len([e for e, c in counts.items() if c > 1]) == 0
        ), "No duplicates for the transfer reactions allowed!"

        K = 300
        current_numbers: dict[
            str, int
        ] = self.ionic_liquid.get_current_number_of_each_residue_type()
        for entry_tuple in to_adapt:
            res_name, initial_number, update_set = (
                entry_tuple[0],
                entry_tuple[1],
                entry_tuple[2],
            )
            assert res_name in update_set, "Resname not in update set"
            try:
                current_number = current_numbers[res_name]
            except KeyError as e:
                print(
                    "The given name is not a valid residue name. Is it present in the psf file? Typo?"
                )
                print(e)
            logger.debug(
                f"{res_name=}, {initial_number=}, {current_number=}, {update_set=}"
            )
            perc_change = current_number / initial_number
            factor = K * (perc_change - 1) ** 3
            logger.debug(f"{perc_change=}, {factor=}")
            new_prob = (
                self.ionic_liquid.templates.get_update_value_for(update_set, "prob")
                + factor
            )
            if new_prob > 1:
                logger.info(
                    f"Probability set to 1, cannot be greater. (Was: {new_prob})"
                )
                new_prob = 1
            if new_prob < 0:
                logger.info(
                    f"Probability set to 0, cannot be smaller. (Was: {new_prob})"
                )
                new_prob = 0
            self.ionic_liquid.templates.set_update_value_for(
                update_set, "prob", new_prob
            )

            print(f"New Prob for {update_set}: {new_prob}")


class KeepHUpdate(Update):
    """KeepHUpdate performs updates but uses the original H position
    keep the position of the original H when switching from dummy to real H.
    """

    @staticmethod
    def load(fname, protex_system: ProtexSystem) -> KeepHUpdate:
        with open(fname, "rb") as inp:
            from_pickle = pickle.load(inp)  # ensure correct order of arguments
        update = KeepHUpdate(protex_system, *from_pickle)
        return update

    def __init__(
        self,
        ionic_liquid: ProtexSystem,
        to_adapt: list[tuple[str, int, frozenset[str]]] or None = None,
        all_forces: bool = True,
        include_equivalent_atom: bool = False,
        reorient: bool = False,
    ) -> None:
        super().__init__(
            ionic_liquid, to_adapt, all_forces, include_equivalent_atom, reorient)

    def dump(self, fname: str) -> None:
        to_pickle = [
            self.all_forces,
            self.to_adapt,
        ]  # enusre correct order of arguments
        with open(fname, "wb") as outp:
            pickle.dump(to_pickle, outp, pickle.HIGHEST_PROTOCOL)

    def _update(self, candidates: list[tuple], nr_of_steps: int) -> None:
        logger.info("called _update")

        # get current state
        state = self.ionic_liquid.simulation.context.getState(getEnergy=True)
        # get initial energy
        initial_e = state.getPotentialEnergy()
        if np.isnan(initial_e._value):
            raise RuntimeError(f"Energy is {initial_e}")

        logger.info("Start changing states ...")
        assert nr_of_steps > 1, "Use an update step number of at least 2."
        for lamb in np.linspace(0, 1, nr_of_steps):
            logger.debug(lamb)
            for candidate in candidates:
                donor, acceptor = candidate

                print(
                    f"{lamb}: donor: {donor.current_name}; charge:{donor.current_charge}: acceptor: {acceptor.current_name}; charge:{acceptor.current_charge}"
                )

                # TODO does this and update_context work with updating used atom extra? maybe write new function here to update used atom
                for force_to_be_updated in self.allowed_forces:
                    ######################
                    # candidate1
                    ######################
                    donor.update(force_to_be_updated, lamb)

                    ######################
                    # candidate2
                    ######################
                    acceptor.update(force_to_be_updated, lamb)


            # update the context to include the new parameters (do this in every lambda step)
            for force_to_be_updated in self.allowed_forces:
                self.ionic_liquid.update_context(force_to_be_updated)

            # get new energy
            state = self.ionic_liquid.simulation.context.getState(getEnergy=True)
            new_e = state.getPotentialEnergy()
            if np.isnan(new_e._value):
                raise RuntimeError(f"Energy is {new_e}")

            self.ionic_liquid.simulation.step(1)

        # update names and involved atoms after last lambda step
        for candidate in candidates:
            donor, acceptor = candidate
            # candidate_idxs = (donor.residue.index, acceptor.residue.index)
            # logger.debug("############# BEFORE UPDATE ################")
            # logger.debug(candidate_idxs)
            # logger.debug(donor)
            # logger.debug(donor.current_name)
            # logger.debug(donor.__hash__)
            # logger.debug(donor.donors)
            # logger.debug(donor.used_atom)
            # logger.debug(acceptor)
            # logger.debug(acceptor.current_name)
            # logger.debug(acceptor.__hash__)
            # logger.debug(acceptor.donors)
            # logger.debug(acceptor.used_atom)
            # nexts = candidates[candidates.index(candidate)+2]
            # ndon, nacc = nexts
            # logger.debug(ndon.current_name)
            # logger.debug(nacc.current_name)
            # logger.debug(ndon.donors)
            # logger.debug(nacc.acceptors)
            assert donor.mode_in_last_transfer == "donor"
            assert acceptor.mode_in_last_transfer == "acceptor"
            # assert (donor.current_name == "H3O" or donor.current_name == "H2O")
            # if donor.current_name == "H2O":
            #     assert len(donor.donors) == 2
            #     assert len(donor.acceptors) == 2
            # if donor.current_name == "H3O":
            #     assert len(donor.donors) == 3
            #     assert len(donor.acceptors) == 1
            # assert (acceptor.current_name == "OH" or acceptor.current_name == "H2O")
            # if acceptor.current_name == "H2O":
            #     assert len(acceptor.donors) == 2
            #     assert len(acceptor.acceptors) == 2
            # if acceptor.current_name == "OH":
            #     assert len(acceptor.donors) == 1
            #     assert len(acceptor.acceptors) == 3
            # assert (donor.used_atom in donor.donors and acceptor.used_atom in acceptor.acceptors)

            # NOTE this works with deepcopy, why doesn't it work without copy, like the current_name?
            ddonors = copy.deepcopy(donor.donors)
            dacceptors = copy.deepcopy(donor.acceptors)
            ddonors.remove(donor.used_atom)
            dacceptors.append(donor.used_atom)
            donor.donors = ddonors
            donor.acceptors = dacceptors

            adonors = copy.deepcopy(acceptor.donors)
            aacceptors = copy.deepcopy(acceptor.acceptors)
            adonors.append(acceptor.used_atom)
            aacceptors.remove(acceptor.used_atom)
            acceptor.donors = adonors
            acceptor.acceptors = aacceptors

            donor.current_name = donor.alternativ_resname
            acceptor.current_name = acceptor.alternativ_resname

            # logger.debug("############# AFTER UPDATE ################")
            # candidate_idxs = (donor.residue.index, acceptor.residue.index)
            # logger.debug(candidate_idxs)
            # logger.debug(donor)
            # logger.debug(donor.current_name)
            # logger.debug(donor.__hash__)
            # logger.debug(donor.donors)
            # logger.debug(donor.used_atom)
            # logger.debug(acceptor)
            # logger.debug(acceptor.current_name)
            # logger.debug(acceptor.__hash__)
            # logger.debug(acceptor.donors)
            # logger.debug(acceptor.used_atom)
            # nexts = candidates[candidates.index(candidate)+2]
            # ndon, nacc = nexts
            # logger.debug(ndon.current_name)
            # logger.debug(nacc.current_name)
            # logger.debug(ndon.donors)
            # logger.debug(nacc.acceptors)

        # get new energy
        state = self.ionic_liquid.simulation.context.getState(getEnergy=True)
        new_e = state.getPotentialEnergy()
        logger.info(f"Energy before/after state change:{initial_e}/{new_e}")



class NaiveMCUpdate(Update):
    """NaiveMCUpdate Performs naive MC update on molecule pairs in close proximity.

    Parameters
    ----------
    UpdateMethod : [type]
        [description]
    """

    @staticmethod
    def load(fname: str, protex_system: ProtexSystem) -> NaiveMCUpdate:
        """Load a pickled NaiveMCUpdate instance.

        Parameters
        ----------
        fname : str
            The file name
        protex_system : ProtexSystem
            A ProtexSystem instance

        Returns
        -------
        NaiveMCUpdate
            A NaiveMCUpdate instance
        """
        with open(fname, "rb") as inp:
            from_pickle = pickle.load(inp)  # ensure correct order of arguments
        update = NaiveMCUpdate(protex_system, *from_pickle)
        return update

    def __init__(
        self,
        ionic_liquid: ProtexSystem,
        to_adapt: list[tuple[str, int, frozenset[str]]] or None = None,
        all_forces: bool = True,
        include_equivalent_atom: bool = False,
        reorient: bool = False,
    ) -> None:
        super().__init__(
            ionic_liquid, to_adapt, all_forces, include_equivalent_atom, reorient
        )
        if reorient:
            raise NotImplementedError(
                "Currently reorienting atoms if equivalent atoms are used is not implemented. Set reorient=False."
            )

    def dump(self, fname: str) -> None:
        """Pickle the NaiveMCUpdate instance.

        Parameters
        ----------
        fname : str
            The file name
        """
        to_pickle = [
            self.all_forces,
            self.to_adapt,
            self.include_equivalent_atom,
            self.reorient,
        ]  # enusre correct order of arguments
        with open(fname, "wb") as outp:
            pickle.dump(to_pickle, outp, pickle.HIGHEST_PROTOCOL)

    def _update(self, candidates: list[tuple], nr_of_steps: int) -> None:
        logger.info("called _update")
        # get current state
        state = self.ionic_liquid.simulation.context.getState(getEnergy=True)
        # get initial energy
        initial_e = state.getPotentialEnergy()
        if np.isnan(initial_e._value):
            raise RuntimeError(f"Energy is {initial_e}")

        logger.info("Start changing states ...")
        assert nr_of_steps > 1, "Use an update step number of at least 2."
        for lamb in np.linspace(0, 1, nr_of_steps):
            # for lamb in reversed(np.linspace(1, 0, nr_of_steps, endpoint=False)):
            for candidate in candidates:
                # retrive residue instances
                candidate1_residue, candidate2_residue = sorted(
                    candidate, key=lambda candidate: candidate.current_name
                )

                print(
                    f"{lamb}: candiadate_1: {candidate1_residue.current_name}; charge:{candidate1_residue.current_charge}: candiadate_2: {candidate2_residue.current_name}; charge:{candidate2_residue.current_charge}"
                )

                for force_to_be_updated in self.allowed_forces:
                    ######################
                    # candidate1
                    ######################
                    candidate1_residue.update(force_to_be_updated, lamb)

                    ######################
                    # candidate2
                    ######################
                    candidate2_residue.update(force_to_be_updated, lamb)

            # update the context to include the new parameters
            for force_to_be_updated in self.allowed_forces:
                self.ionic_liquid.update_context(force_to_be_updated)

            # get new energy
            state = self.ionic_liquid.simulation.context.getState(getEnergy=True)
            new_e = state.getPotentialEnergy()
            if np.isnan(new_e._value):
                raise RuntimeError(f"Energy is {new_e}")

            self.ionic_liquid.simulation.step(1)

        for candidate in candidates:
            candidate1_residue, candidate2_residue = candidate
            # after the update is finished the current_name attribute is updated (and since alternative_name depends on current_name it too is updated)
            candidate1_residue.current_name = candidate1_residue.alternativ_resname
            candidate2_residue.current_name = candidate2_residue.alternativ_resname

            assert candidate1_residue.current_name != candidate1_residue.alternativ_resname
            assert candidate2_residue.current_name != candidate2_residue.alternativ_resname

        # get new energy
        state = self.ionic_liquid.simulation.context.getState(getEnergy=True)
        new_e = state.getPotentialEnergy()
        logger.info(f"Energy before/after state change:{initial_e}/{new_e}")

        # self.ionic_liquid.simulation.context.setVelocitiesToTemperature(
        #    300.0 * unit.kelvin
        # )


class StateUpdate:
    """Controls the update sheme and proposes the residues that need an update."""

    @staticmethod
    def load(fname: str, updateMethod: Update) -> StateUpdate:
        """Load a pickled StateUpdate instance.

        Parameters
        ----------
        fname : str
            The file name
        updateMethod : Update
            The update method instance

        Returns
        -------
        StateUpdate
            An instance of StateUpdate
        """
        state_update = StateUpdate(updateMethod)
        with open(fname, "rb") as inp:
            from_pickle = pickle.load(inp)  # ensure correct order of arguments
        state_update.history = from_pickle[0]
        state_update.update_trial = from_pickle[1]
        return state_update

    def __init__(self, updateMethod: Update) -> None:
        self.updateMethod: Update = updateMethod
        self.ionic_liquid: ProtexSystem = self.updateMethod.ionic_liquid
        self.history: deque = deque(maxlen=10)
        self.update_trial: int = 0

    def dump(self, fname: str) -> None:
        """Pickle the StateUpdate instance.

        Parameters
        ----------
        fname : str
            The file name
        """
        to_pickle = [
            self.history,
            self.update_trial,
        ]  # enusre correct order of arguments
        with open(fname, "wb") as outp:
            pickle.dump(to_pickle, outp, pickle.HIGHEST_PROTOCOL)

    def write_charges(self, filename: str) -> None:  # deprecated?
        """Write current charges to a file.

        Parameters
        ----------
        filename : str
            The name of the file to wrtie the charges to
        """
        par = self.get_charges()
        with open(filename, "w+") as f:
            for atom_idx, atom, charge in par:
                charge = charge._value
                f.write(
                    f"{atom.residue.name:>4}:{int(atom.id): 4}:{int(atom.residue.id): 4}:{atom.name:>4}:{charge}\n"
                )

    # instead of these to functions use the ChargeReporter probably
    def get_charges(self) -> list:  # deprecated?
        """_summary_.

        Returns
        -------
        list
            atom_idxs, atom object, charge

        Raises
        ------
        RuntimeError
            If system does not contain a nonbonded force
        """
        par = []
        for force in self.ionic_liquid.system.getForces():
            if type(force).__name__ == "NonbondedForce":
                for idx, atom in zip(
                    range(force.getNumParticles()), self.ionic_liquid.topology.atoms()
                ):
                    charge, _, _ = force.getParticleParameters(idx)
                    par.append((idx, atom, charge))
                return par
        raise RuntimeError("Something went wrong. There was no NonbondedForce")

    # redundant with ProtexSystem.get_current_number_of_each_residue_type
    def get_num_residues(self) -> dict:
        """Deprecated 1.1."""
        res_dict = {
            "IM1H": 0,
            "OAC": 0,
            "IM1": 0,
            "HOAC": 0,
            "HPTS": 0,
            "HPTSH": 0,
            "MEOH": 0,
            "MEOH2": 0,
        }
        for residue in self.ionic_liquid.residues:
            res_dict[residue.current_name] += 1
        return res_dict

    def _print_start(self):
        print(
            f"""
            ##############################
            ##############################
            --- Update trial: {self.update_trial} ---
            ##############################
            ##############################
            """
        )
        # --- Nr of charged residues: ---
        # --- Nr of uncharged residues: ---

    def _print_stop(self):
        print(
            """
            ##############################
            ##############################
            """
        )

    def update(self, nr_of_steps: int = 2) -> list[tuple[Residue, Residue]]:
        r"""Updates the current state using the method defined in the UpdateMethod class.

        Parameters
        ----------
        nr_of_steps : int, optional
            The number of intermediate :math:`{\lambda}` states.
            The default of two means 1 with the initial and one with the final state,
            so no intermediate states, by default 2

        Returns
        -------
        list[tuple[Residue, Residue]]
            A list with all the updated residue tuples
        """
        # calculate the distance betwen updateable residues
        donor_atoms_list, donor_resis_list, donor_names_list, acceptor_atoms_list, acceptor_resis_list, acceptor_names_list = self._get_positions_for_mutation_sites()
        # propose the update candidates based on distances
        self._print_start()
        candidate_pairs = self._propose_candidate_pair(donor_atoms_list, donor_resis_list, donor_names_list, acceptor_atoms_list, acceptor_resis_list, acceptor_names_list)
        print(f"{len(candidate_pairs)=}")

        if len(candidate_pairs) == 0:
            print("No transfers this time")
            self.ionic_liquid.simulation.step(
                nr_of_steps
            )  # also do the simulation steps if no update occurs, to be consistent in simulation time
        elif len(candidate_pairs) > 0:
            self.updateMethod._update(candidate_pairs, nr_of_steps)

        self.update_trial += 1

        if self.updateMethod.to_adapt is not None:
            self.updateMethod._adapt_probabilities(self.updateMethod.to_adapt)

        self._print_stop()

        return candidate_pairs

    def _propose_candidate_pair(
        self, donor_atoms_list: list[float], donor_resis_list: list[Residue], donor_names_list : list[str], acceptor_atoms_list: list[float], acceptor_resis_list: list[Residue], acceptor_names_list: list[str], use_pbc: bool = True
    ) -> list[tuple[Residue, Residue]]:
        """Takes the return value of _get_positions_of_mutation_sites."""
        assert len(donor_atoms_list) == len(
            donor_resis_list
        ) == len(donor_names_list), "Should be equal length and same order, because residue is found by index of pos list"
        assert len(acceptor_atoms_list) == len(
            acceptor_resis_list
        ) == len(acceptor_names_list), "Should be equal length and same order, because residue is found by index of pos list"

        from scipy.spatial.distance import cdist

        def _rPBC(
            coor1, coor2, boxl=self.ionic_liquid.boxlength.value_in_unit(nanometers)
        ):
            dx = abs(coor1[0] - coor2[0])
            if dx > boxl / 2:
                dx = boxl - dx
            dy = abs(coor1[1] - coor2[1])
            if dy > boxl / 2:
                dy = boxl - dy
            dz = abs(coor1[2] - coor2[2])
            if dz > boxl / 2:
                dz = boxl - dz
            return np.sqrt(dx * dx + dy * dy + dz * dz)

        # calculate distance matrix between the two molecules
        if use_pbc:
            logger.debug("Using PBC correction for distance calculation")
            distance = cdist(donor_atoms_list, acceptor_atoms_list, _rPBC)
        else:
            logger.debug("No PBC correction for distance calculation")
            distance = distance_matrix(donor_atoms_list, acceptor_atoms_list)
        # # shape diagonal to not have self terms between ie same HOAC-HOAC
        # np.fill_diagonal(distance, np.inf) # not needed any more, as diagonal elements do not necessarily belong to the same residue (need to filter them out anyway)
        # # print(f"{distance=}, {distance_pbc=}")
        # # get a list of indices for elements in the distance matrix sorted by increasing distance
        # # also combinations which are not possible are in list
        # # -> the selecion is then done with the check if both residues
        # # corresponding to the distance index are an allowed update
        shape = distance.shape
        idx = np.dstack(np.unravel_index(np.argsort(distance.ravel()), shape))[0]
        # print(f"{idx=}")

        proposed_candidate_pairs = []
        proposed_candidate_pair_sets = (
            []
        )  # didn't want to make sets from proposed_candidate_pairs altogether, second list for basically same information may be superfluous
        used_residues = []
        # check if charge transfer is possible
        for candidate_idx1, candidate_idx2 in idx:
            donor = donor_resis_list[candidate_idx1]
            assert (donor.current_name == "TOH3")
            # if donor.current_name == "TOH2":
            #     assert len(donor.donors) == 2
            #     assert len(donor.acceptors) == 2
            #     assert donor.possible_modes == ("acceptor", "donor")
            if donor.current_name == "TOH3":
                assert len(donor.donors) == 3
                assert len(donor.acceptors) == 1
                assert donor.possible_modes == ("donor")

            acceptor = acceptor_resis_list[candidate_idx2]
            assert (acceptor.current_name == "TOH2")
            if acceptor.current_name == "TOH2":
                assert len(acceptor.donors) == 2
                assert len(acceptor.acceptors) == 2
                assert acceptor.possible_modes == ("acceptor")
            # if acceptor.current_name == "OH":
            #     assert len(acceptor.donors) == 1
            #     assert len(acceptor.acceptors) == 3
            #     assert acceptor.possible_modes == ("acceptor")
            # is this the same residue?
            if donor == acceptor:
                continue
            assert donor.residue.index != acceptor.residue.index
            # is this combination allowed?
            if (
                frozenset([donor.current_name, acceptor.current_name])
                in self.ionic_liquid.templates.allowed_updates.keys()
            ):
                r_max = self.ionic_liquid.templates.allowed_updates[
                    frozenset([donor.current_name, acceptor.current_name])
                ]["r_max"] # TODO: now checking distances between Hs and dummies, calculate RDFs and redefine cutoffs in sim_pt.py
                prob = self.ionic_liquid.templates.allowed_updates[
                    frozenset([donor.current_name, acceptor.current_name])
                ]["prob"]
                #logger.debug(f"{r_max=}, {prob=}")
                r = distance[candidate_idx1, candidate_idx2]
                # break for loop if no pair can fulfill distance condition
                if r > self.ionic_liquid.templates.overall_max_distance:
                    break
                elif r <= r_max and random.random() <= prob:  # random enough?
                    charge_candidate_idx1 = donor.endstate_charge
                    charge_candidate_idx2 = acceptor.endstate_charge

                    # logger.debug(
                    #     f"donor:{donor.original_name}:{donor.current_name}:{donor.residue.id}:{charge_candidate_idx1}:acceptor:{acceptor.original_name}:{acceptor.current_name}:{acceptor.residue.id}:{charge_candidate_idx2} pair suggested ..."
                    # )
                    # logger.debug(
                    #     f"Distance between pairs: {distance[candidate_idx1,candidate_idx2]}"
                    # )
                    proposed_candidate_pair = (donor, acceptor)
                    # reject if already used in this transfer call
                    # print(f"{residue1=}, {residue2=}")
                    if (
                        donor in used_residues or acceptor in used_residues
                    ):  # TODO: is this working with changing some variables in the classes?
                        # logger.debug(
                        #     f"{donor.current_name}:{donor.residue.id}:{charge_candidate_idx1}:{acceptor.current_name}:{acceptor.residue.id}:{charge_candidate_idx2} pair rejected, bc used this transfer call ..."
                        # )
                        continue
                    # reject if already in last 10 updates
                    if any(
                        set(proposed_candidate_pair) in sublist
                        for sublist in self.history
                    ):
                        logger.debug(
                            f"{donor.current_name}:{donor.residue.id}:{charge_candidate_idx1}:{acceptor.current_name}:{acceptor.residue.id}:{charge_candidate_idx2} pair rejected, bc in history ..."
                        )

                        continue
                    # accept otherwise
                    proposed_candidate_pairs.append(proposed_candidate_pair)
                    # info = f"donor: {donor.current_name}, acceptor: {acceptor.current_name}"
                    # logger.debug(info)
                    used_residues.append(donor)
                    donor.used_atom = donor_names_list[candidate_idx1]
                    donor.mode_in_last_transfer = "donor"
                    used_residues.append(acceptor)
                    acceptor.used_atom = acceptor_names_list[candidate_idx2]
                    acceptor.mode_in_last_transfer = "acceptor"
                    proposed_candidate_pair_sets.append(set(proposed_candidate_pair))

                    print(
                        f"{donor.current_name}:{donor.residue.id}:{charge_candidate_idx1}:{acceptor.current_name}:{acceptor.residue.id}:{charge_candidate_idx2} pair accepted ..."
                    )
                    # residue.index 0-based through whole topology
                    print(
                        f"UpdatePair:{donor.current_name}:{donor.residue.index}:{charge_candidate_idx1}:{acceptor.current_name}:{acceptor.residue.index}:{charge_candidate_idx2}"
                    )

        if len(proposed_candidate_pair_sets) == 0:
            self.history.append([])
        else:
            self.history.append(proposed_candidate_pair_sets)


        return proposed_candidate_pairs

    def _get_positions_for_mutation_sites(self) -> tuple[list[float], list[Residue]]:
        """_get_positions_for_mutation_sites returns."""
        pos = self.ionic_liquid.simulation.context.getState(
            getPositions=True
        ).getPositions(asNumpy=True)

        # fill in the positions for each species
        # TODO all these lists are clunky, maybe find a better syntax
        donor_atoms_list = []
        donor_resis_list = []
        donor_names_list =[]
        acceptor_atoms_list = []
        acceptor_resis_list = []
        acceptor_names_list = []

        # loop over all residues and add the positions of the atoms that can be updated to the pos_dict
        for residue in self.ionic_liquid.residues:
            #assert residue.current_name in self.ionic_liquid.templates.names
            # logger.debug(residue.current_name)
            # logger.debug(residue.possible_modes)
            if residue.current_name in self.ionic_liquid.templates.names:
                #logger.debug(residue.donors)
                #logger.debug(residue.acceptors)
                if residue.current_name == "TOH2":
                    assert len(residue.donors) == 2
                    assert len(residue.acceptors) == 2
                    assert residue.possible_modes == ("acceptor")
                if residue.current_name == "TOH3":
                    assert len(residue.donors) == 3
                    assert len(residue.acceptors) == 1
                    assert residue.possible_modes == ("donor")
               # if residue.current_name == "OH":
               #     assert len(residue.donors) == 1
               #     assert len(residue.acceptors) == 3
               #     assert residue.possible_modes == ("acceptor")

                if "donor" in residue.possible_modes:
                    for atom in residue.donors:
                        donor_atoms_list.append(pos[
                            residue.get_idx_for_atom_name(
                            atom
                            )])
                        donor_names_list.append(atom)
                        donor_resis_list.append(residue)

                if "acceptor" in residue.possible_modes:
                    for atom in residue.acceptors:
                        acceptor_atoms_list.append(pos[
                            residue.get_idx_for_atom_name(
                            atom
                            )])
                        acceptor_names_list.append(atom)
                        acceptor_resis_list.append(residue)

        # logger.debug(acceptor_resis_list)
        # logger.debug(donor_resis_list)

        current_numbers: dict[
            str, int
        ] = self.ionic_liquid.get_current_number_of_each_residue_type()
        assert len(donor_atoms_list) == len(donor_names_list) == len(donor_resis_list) == current_numbers["TOH3"]*3
        assert len(acceptor_atoms_list) == len(acceptor_names_list) == len(acceptor_resis_list) == current_numbers["TOH2"]*2

        return donor_atoms_list, donor_resis_list, donor_names_list, acceptor_atoms_list, acceptor_resis_list, acceptor_names_list
