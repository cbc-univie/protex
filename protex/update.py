import copy
import logging
import random
from collections import Counter

import numpy as np
from scipy.spatial import distance_matrix

from protex.residue import Residue
from protex.system import ProtexSystem

logger = logging.getLogger(__name__)


class Update:
    """
    ABC for implementing different Update Methods

    Parameters
    ----------
    ionic_liquid: IonicLiquidSystem
        Needs the IonicLiquidSystem
    """

    def __init__(
        self,
        ionic_liquid: ProtexSystem,
        to_adapt=None,
    ) -> None:
        self.ionic_liquid: ProtexSystem = ionic_liquid
        self.to_adapt: list[tuple[str, int, frozenset[str]]] = to_adapt


class NaiveMCUpdate(Update):
    """
    NaiveMCUpdate Performs naive MC update on molecule pairs in close proximity

    Parameters
    ----------
    UpdateMethod : [type]
        [description]
    """

    def __init__(
        self,
        ionic_liquid: ProtexSystem,
        all_forces: bool = False,
        to_adapt: list[tuple[str, int, frozenset[str]]] = None,
        meoh2: bool = False,
    ) -> None:
        super().__init__(ionic_liquid, to_adapt)
        self.allowed_forces: list[str] = [  # change charges only
            "NonbondedForce",  # BUG: Charge stored in the DrudeForce does NOT get updated, probably you want to allow DrudeForce as well!
            "DrudeForce",
        ]
        if all_forces:
            self.allowed_forces.extend(
                [
                    "HarmonicBondForce",
                    "HarmonicAngleForce",
                    "PeriodicTorsionForce",
                    "CustomTorsionForce",
                ]
            )
        self.meoh2 = meoh2

    def _reorient_atoms(self, candidate):
        # Function to reorient atoms if the equivalent atom was used for shortest distance
        # exchange positions of atom and equivalent atom

        atom_idx = candidate.get_idx_for_atom_name(
            self.ionic_liquid.templates.get_atom_name_for(candidate.current_name)
        )
        equivalent_idx = candidate.get_idx_for_atom_name(
            self.ionic_liquid.templates.get_equivalent_atom_for(candidate.current_name)
        )

        positions = self.ionic_liquid.simulation.context.getState(
            getPositions=True
        ).getPositions(asNumpy=False)

        positions_copy = copy.deepcopy(positions)

        pos_atom = positions_copy[atom_idx]
        print(f"{pos_atom=}")
        pos_equivalent = positions_copy[equivalent_idx]
        print(f"{pos_equivalent=}")

        positions[atom_idx] = pos_equivalent
        positions[equivalent_idx] = pos_atom

        self.ionic_liquid.simulation.context.setPositions(positions)
        print(
            f"setting position of {self.ionic_liquid.templates.get_atom_name_for(candidate.current_name)} to {positions[atom_idx]} and {self.ionic_liquid.templates.get_equivalent_atom_for(candidate.current_name)} to {positions[equivalent_idx]}"
        )

    def _update(self, candidates: list[tuple], nr_of_steps: int):
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

        # idea: get positions after simulation steps in update
        positions = self.ionic_liquid.simulation.context.getState(
            getPositions=True
        ).getPositions(asNumpy=True)

        positions_copy = copy.deepcopy(positions)

        donors = []  # determine which candidate was the H-donor / acceptor
        acceptors = []
        pos_donated_Hs = []  # collect the positions of the real Hs at transfer
        pos_accepted_Hs = (
            []
        )  # can't put new H on top of dummy H -> put it a little bit closer to the acceptor
        pos_acceptor_atoms = []
        # all these lists are not needed, if contents used in the same loop (as it is currently)

        for candidate in candidates:
            candidate1_residue, candidate2_residue = candidate

            if "H" in self.ionic_liquid.templates.get_atom_name_for(
                candidate1_residue.current_name
            ) or (
                self.ionic_liquid.templates.has_equivalent_atom(candidate1_residue.current_name) == True
                and "H"
                in self.ionic_liquid.templates.get_equivalent_atom_for(
                    candidate1_residue.current_name
                )
            ):
                donor = candidate1_residue
                acceptor = candidate2_residue

            else:
                donor = candidate2_residue
                acceptor = candidate1_residue

            donors.append(donor)
            acceptors.append(acceptor)
            print(f'{donor.current_name=}, {self.ionic_liquid.templates.has_equivalent_atom(donor.current_name)=}, {acceptor.current_name=}, {self.ionic_liquid.templates.has_equivalent_atom(acceptor.current_name)=}')
            print(f'{donor.used_equivalent_atom=}, {acceptor.used_equivalent_atom=}')

            if donor.used_equivalent_atom == True:
                pos_donated_H = positions_copy[
                    donor.get_idx_for_atom_name(
                        self.ionic_liquid.templates.get_equivalent_atom_for(
                            donor.current_name
                        )
                    )
                ]
            else:
                pos_donated_H = positions_copy[
                    donor.get_idx_for_atom_name(
                        self.ionic_liquid.templates.get_atom_name_for(
                            donor.current_name
                        )
                    )
                ]

            if acceptor.used_equivalent_atom == True:
                pos_acceptor_atom = positions_copy[
                    acceptor.get_idx_for_atom_name(
                        self.ionic_liquid.templates.get_equivalent_atom_for(
                            acceptor.current_name
                        )
                    )
                ]

            else:
                pos_acceptor_atom = positions_copy[
                    acceptor.get_idx_for_atom_name(
                        self.ionic_liquid.templates.get_atom_name_for(
                            acceptor.current_name
                        )
                    )
                ]

            pos_acceptor_atoms.append(pos_acceptor_atom)

            # account for PBC
            boxl_vec = (
                self.ionic_liquid.boxlength * pos_donated_H[0] / pos_donated_H[0]._value
            )  # very stupid workaround to get a quantity with unit from a number

            for i in range(0, 3):
                if (
                    abs(pos_acceptor_atom[i] - pos_donated_H[i]) > boxl_vec / 2
                ):  # could also be some other value
                    if pos_acceptor_atom[i] > pos_donated_H[i]:
                        pos_donated_H[i] = pos_donated_H[i] + boxl_vec
                    else:
                        pos_donated_H[i] = pos_donated_H[i] - boxl_vec

            pos_donated_Hs.append(pos_donated_H)
            pos_accepted_H = pos_donated_H - 0.33 * (
                pos_donated_H - pos_acceptor_atom
            )  # set position at ca. 1 angstrom from acceptor -> maybe more stable?
            pos_accepted_Hs.append(pos_accepted_H)
            ###

            # reorient equivalent atoms if needed
            if candidate1_residue.used_equivalent_atom:
                candidate1_residue.used_equivalent_atom = (
                    False  # reset for next update round
                )
                self._reorient_atoms(candidate1_residue)
                # # troubleshooting reorient
                # atom_idx = candidate1_residue.get_idx_for_atom_name(
                #     self.ionic_liquid.templates.get_atom_name_for(
                #         candidate1_residue.current_name
                #     )
                # )
                # equivalent_idx = candidate1_residue.get_idx_for_atom_name(
                #     self.ionic_liquid.templates.get_equivalent_atom_for(
                #         candidate1_residue.current_name
                #     )
                # )

                # positions = self.ionic_liquid.simulation.context.getState(
                #     getPositions=True
                # ).getPositions(asNumpy=False)

                # print(
                #     f"position of {self.ionic_liquid.templates.get_atom_name_for(candidate1_residue.current_name)} : {positions[atom_idx]} and {self.ionic_liquid.templates.get_equivalent_atom_for(candidate1_residue.current_name)} : {positions[equivalent_idx]}"
                # )
                # # troubleshoot end

            if candidate2_residue.used_equivalent_atom:
                candidate2_residue.used_equivalent_atom = False
                self._reorient_atoms(candidate2_residue)
                # # troubleshooting reorient
                # atom_idx = candidate2_residue.get_idx_for_atom_name(
                #     self.ionic_liquid.templates.get_atom_name_for(
                #         candidate2_residue.current_name
                #     )
                # )
                # equivalent_idx = candidate2_residue.get_idx_for_atom_name(
                #     self.ionic_liquid.templates.get_equivalent_atom_for(
                #         candidate2_residue.current_name
                #     )
                # )

                # positions = self.ionic_liquid.simulation.context.getState(
                #     getPositions=True
                # ).getPositions(asNumpy=False)

                # print(
                #     f"position of {self.ionic_liquid.templates.get_atom_name_for(candidate2_residue.current_name)} : {positions[atom_idx]} and {self.ionic_liquid.templates.get_equivalent_atom_for(candidate2_residue.current_name)} : {positions[equivalent_idx]}"
                # )
                # # troubleshoot end

            # # also update has_equivalent_atom (not needed if has_equivalent_atom updated via self.ionic_liquid.templates.has_equivalent_atom(residue.current_name))
            # if candidate1_residue.current_name in ("MEOH2", "MEOH", "HOAC", "OAC"):
            #     candidate1_residue.has_equivalent_atom = (
            #         not candidate1_residue.has_equivalent_atom
            #     )

            # if candidate2_residue.current_name in ("MEOH2", "MEOH", "HOAC", "OAC"):
            #     candidate2_residue.has_equivalent_atom = (
            #         not candidate2_residue.has_equivalent_atom
            #    )

            # after the update is finished the current_name attribute is updated (and since alternative_name depends on current_name it too is updated)
            candidate1_residue.current_name = candidate1_residue.alternativ_name
            candidate2_residue.current_name = candidate2_residue.alternativ_name

            assert candidate1_residue.current_name != candidate1_residue.alternativ_name
            assert candidate2_residue.current_name != candidate2_residue.alternativ_name

            # acceptor current name refers now to a future donor -> atom name is the H that can be given on -> this used to be the dummy H
            acceptor = acceptors[candidates.index(candidate)]
            idx_accepted_H = acceptor.get_idx_for_atom_name(
                self.ionic_liquid.templates.get_atom_name_for(acceptor.current_name)
            )

            # update position of the once-dummy H to that of the donated H (a bit closer to the acceptor to avoid LJ collusion with the now dummy H)
            positions[idx_accepted_H] = pos_accepted_Hs[candidates.index(candidate)]
            # # troubleshooting
            # print(
            #     f"acceptor: {pos_acceptor_atoms[candidates.index(candidate)]}, donor_H: {pos_donated_Hs[candidates.index(candidate)]}"
            # )
            # print(
            #     f"setting position of {self.ionic_liquid.templates.get_atom_name_for(acceptor.current_name)} of {acceptor.current_name}:{acceptor.residue.index} to {pos_accepted_Hs[candidates.index(candidate)]}"
            # )
            # # troubleshooting end

        self.ionic_liquid.simulation.context.setPositions(positions)
        # NOTE: should this happen directly after the simulation steps if there are multiple steps within the update?
        # NOTE: should this happen before or after forces are updated?

        # get new energy
        state = self.ionic_liquid.simulation.context.getState(getEnergy=True)
        new_e = state.getPotentialEnergy()
        logger.info(f"Energy before/after state change:{initial_e}/{new_e}")

        # self.ionic_liquid.simulation.context.setVelocitiesToTemperature(
        #    300.0 * unit.kelvin
        # )

    def _adapt_probabilities(
        self, to_adapt=list[tuple[str, int, frozenset[str]]]
    ) -> None:
        """
        Adapt the probability for certain events depending on the current equilibrium, in order to stay close to a given reference
        i.e. prob_neu = prob_orig + K*( x(t) - x(eq) )^3 where x(t) is the current percentage in the system of one species

        Parameters:
        -----------
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


class StateUpdate:
    """
    Controls the update sheme and proposes the residues that need an update
    """

    def __init__(self, updateMethod: Update) -> None:
        self.updateMethod: Update = updateMethod
        self.ionic_liquid: ProtexSystem = self.updateMethod.ionic_liquid
        self.history: list = []
        self.update_trial: int = 0

    def write_charges(self, filename: str) -> None:

        par = self.get_charges()
        with open(filename, "w+") as f:
            for atom_idx, atom, charge in par:
                charge = charge._value
                f.write(
                    f"{atom.residue.name:>4}:{int(atom.id): 4}:{int(atom.residue.id): 4}:{atom.name:>4}:{charge}\n"
                )

    def get_charges(self) -> list:

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

    def get_num_residues(self) -> dict:
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
            f"""
            ##############################
            ##############################
            """
        )

    def update(self, nr_of_steps: int = 2) -> list[tuple[Residue, Residue]]:
        """
        updates the current state using the method defined in the UpdateMethod class
        """
        # calculate the distance betwen updateable residues
        pos_list, res_list = self._get_positions_for_mutation_sites()
        # propose the update candidates based on distances
        self._print_start()
        candidate_pairs = self._propose_candidate_pair(pos_list, res_list)
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
        self, pos_list: list[float], res_list: list[Residue], use_pbc: bool = True
    ) -> list[tuple[Residue, Residue]]:
        """
        Takes the return value of _get_positions_of_mutation_sites

        """
        assert len(pos_list) == len(
            res_list
        ), "Should be equal length and same order, because residue is found by index of pos list"

        from scipy.spatial.distance import cdist

        def _rPBC(coor1, coor2, boxl=self.ionic_liquid.boxlength):
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
            distance = cdist(pos_list, pos_list, _rPBC)
        else:
            logger.debug("No PBC correction for distance calculation")
            distance = distance_matrix(pos_list, pos_list)
        # shape diagonal to not have self terms between ie same HOAC-HOAC
        np.fill_diagonal(distance, np.inf)
        # print(f"{distance=}, {distance_pbc=}")
        # get a list of indices for elements in the distance matrix sorted by increasing distance
        # also combinations which are not psossible are in list
        # -> the selecion is then done with the check if both residues
        # corresponiding to the distance index are an allowed update
        shape = distance.shape
        idx = np.dstack(np.unravel_index(np.argsort(distance.ravel()), shape))[0]
        # print(f"{idx=}")

        proposed_candidate_pairs = []
        used_residues = []
        # check if charge transfer is possible
        for candidate_idx1, candidate_idx2 in idx:
            residue1: Residue = res_list[candidate_idx1]
            residue2: Residue = res_list[candidate_idx2]
            # is this combination allowed?
            if (
                frozenset([residue1.current_name, residue2.current_name])
                in self.ionic_liquid.templates.allowed_updates.keys()
            ):
                r_max = self.ionic_liquid.templates.allowed_updates[
                    frozenset([residue1.current_name, residue2.current_name])
                ]["r_max"]
                prob = self.ionic_liquid.templates.allowed_updates[
                    frozenset([residue1.current_name, residue2.current_name])
                ]["prob"]
                logger.debug(f"{r_max=}, {prob=}")
                r = distance[candidate_idx1, candidate_idx2]
                # break for loop if no pair can fulfill distance condition
                if r > self.ionic_liquid.templates.overall_max_distance:
                    break
                elif r <= r_max and random.random() <= prob:  # random enough?
                    charge_candidate_idx1 = residue1.endstate_charge
                    charge_candidate_idx2 = residue2.endstate_charge

                    logger.debug(
                        f"{residue1.original_name}:{residue1.current_name}:{residue1.residue.id}:{charge_candidate_idx1}-{residue2.original_name}:{residue2.current_name}:{residue2.residue.id}:{charge_candidate_idx2} pair suggested ..."
                    )
                    logger.debug(
                        f"Distance between pairs: {distance[candidate_idx1,candidate_idx2]}"
                    )
                    proposed_candidate_pair = (residue1, residue2)
                    # reject if already used in this transfer call
                    # print(f"{residue1=}, {residue2=}")
                    if (
                        residue1 in used_residues or residue2 in used_residues
                    ):  # TODO: is this working with changing some variables in the classes?
                        logger.debug(
                            f"{residue1.current_name}:{residue1.residue.id}:{charge_candidate_idx1}-{residue2.current_name}:{residue2.residue.id}:{charge_candidate_idx2} pair rejected, bc used this transfer call ..."
                        )
                        continue
                    # reject if already in last 10 updates
                    if set(proposed_candidate_pair) in self.history[-10:]:
                        logger.debug(
                            f"{residue1.current_name}:{residue1.residue.id}:{charge_candidate_idx1}-{residue2.current_name}:{residue2.residue.id}:{charge_candidate_idx2} pair rejected, bc in history ..."
                        )

                        continue
                    # accept otherwise
                    proposed_candidate_pairs.append(proposed_candidate_pair)
                    if (
                        candidate_idx1 == residue1.equivalent_atom_pos_in_list
                    ):  # check if we used the actual atom or onyl the equivalent
                        residue1.used_equivalent_atom = True
                    if candidate_idx2 == residue2.equivalent_atom_pos_in_list:
                        residue2.used_equivalent_atom = True
                    used_residues.append(residue1)
                    used_residues.append(residue2)
                    self.history.append(set(proposed_candidate_pair))
                    print(
                        f"{residue1.current_name}:{residue1.residue.id}:{charge_candidate_idx1}-{residue2.current_name}:{residue2.residue.id}:{charge_candidate_idx2} pair accepted ..."
                    )
                    # residue.index 0-based through whole topology
                    print(
                        f"UpdatePair:{residue1.current_name}:{residue1.residue.index}:{charge_candidate_idx1}:{residue2.current_name}:{residue2.residue.index}:{charge_candidate_idx2}"
                    )
                # return proposed_candidate_pair
        return proposed_candidate_pairs

    def _get_positions_for_mutation_sites(self) -> tuple[list[float], list[Residue]]:
        """
        _get_positions_for_mutation_sites returns
        """
        pos = self.ionic_liquid.simulation.context.getState(
            getPositions=True
        ).getPositions(asNumpy=True)

        # fill in the positions for each species
        pos_list = []
        res_list = []

        # loop over all residues and add the positions of the atoms that can be updated to the pos_dict
        for residue in self.ionic_liquid.residues:
            residue.equivalent_atom_pos_in_list = None
            assert residue.current_name in self.ionic_liquid.templates.names
            # get the position of the atom (Hydrogen or the possible acceptor)
            # new idea: just make one list with all positions and then calc distances of everything with everything... -> not so fast, but i need i.e. IM1H-IM1
            pos_list.append(
                pos[
                    residue.get_idx_for_atom_name(
                        self.ionic_liquid.templates.get_atom_name_for(
                            residue.current_name
                        )
                    )
                    # this needs the atom idx to be the same for both topologies
                    # TODO: maybe get rid of this caveat
                    # maybe some mapping between possible residue states and corresponding atom positions
                ]
            )
            res_list.append(residue)

            if self.ionic_liquid.templates.has_equivalent_atom(residue.current_name):
                pos_list.append(
                    pos[
                        residue.get_idx_for_atom_name(
                            self.ionic_liquid.templates.get_equivalent_atom_for(
                                residue.current_name
                            )
                        )
                    ]
                )
                residue.equivalent_atom_pos_in_list = len(
                    res_list
                )  # store idx to know which coordinates where used for distance
                
                res_list.append(
                    residue
                )  # add second time the residue to have same length of pos_list and res_list

        return pos_list, res_list
