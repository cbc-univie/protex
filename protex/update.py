import logging
from collections import defaultdict
from typing import List, Tuple
import random

import numpy as np
from scipy.spatial import distance_matrix

from protex.system import IonicLiquidSystem

# try:
#     from openmm import unit
# except ImportError:
#     from simtk import unit

logger = logging.getLogger(__name__)


class Update:
    def __init__(self, ionic_liquid: IonicLiquidSystem) -> None:
        self.ionic_liquid = ionic_liquid


class NaiveMCUpdate(Update):
    """
    NaiveMCUpdate Performs naive MC update on molecule pairs in close proximity

    Parameters
    ----------
    UpdateMethod : [type]
        [description]
    """

    def __init__(
        self, ionic_liquid: IonicLiquidSystem, all_forces: bool = False
    ) -> None:
        super().__init__(ionic_liquid)
        self.allowed_forces = [  # change charges only
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

    def _update(self, candidates: List[Tuple], nr_of_steps: int):
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
            candidate1_residue.current_name = candidate1_residue.alternativ_name
            candidate2_residue.current_name = candidate2_residue.alternativ_name

            assert candidate1_residue.current_name != candidate1_residue.alternativ_name
            assert candidate2_residue.current_name != candidate2_residue.alternativ_name

        # get new energy
        state = self.ionic_liquid.simulation.context.getState(getEnergy=True)
        new_e = state.getPotentialEnergy()
        logger.info(f"Energy before/after state change:{initial_e}/{new_e}")

        # self.ionic_liquid.simulation.context.setVelocitiesToTemperature(
        #    300.0 * unit.kelvin
        # )


class StateUpdate:
    """
    Controls the update sheme and proposes the residues that need an update
    """

    def __init__(self, updateMethod: Update) -> None:
        self.updateMethod = updateMethod
        self.ionic_liquid = self.updateMethod.ionic_liquid
        self.history = []
        self.update_trial = 0

    def write_charges(self, filename: str):

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

    def get_num_residues(self) -> dict:
        res_dict = {"IM1H": 0, "OAC": 0, "IM1": 0, "HOAC": 0}
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

    def update(self, nr_of_steps: int = 101) -> tuple:
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
        self._print_stop()

        return candidate_pairs

    def _propose_candidate_pair(
        self, pos_list: list, res_list: list, use_pbc: bool = True
    ) -> list[tuple]:
        """
        Takes the return value of _get_positions_of_mutation_sites

        """

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
        # np.fill_diagonal(distance, np.inf)
        distance[
            np.tril_indices_from(distance)
        ] = np.inf  # its a symmetric qudratic matrix, we do not care about one half

        # print(f"{distance=}, {distance_pbc=}")
        # get a list of indices for elements in the distance matrix sorted by increasing distance
        # NOTE: This always accepts a move!
        # shape = distance.shape
        # idx = np.dstack(np.unravel_index(np.argsort(distance.ravel()), shape))[0]
        # already get only indices until the max distance-> then randomly probe from them, so that there is no bias that only the clostest transfer and than the latter one are dismissed -> does it make a difference?
        idx = np.argwhere(distance <= self.ionic_liquid.templates.overall_max_distance)
        rng = np.random.default_rng()
        # randomize in case of residues being present more than one time
        rng.shuffle(idx)
        # print(f"{idx=}")

        proposed_candidate_pairs = []
        used_residues = []
        # check if charge transfer is possible
        for candidate_idx1, candidate_idx2 in idx:
            residue1 = res_list[candidate_idx1]
            residue2 = res_list[candidate_idx2]
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
                # print(f"{r_max=}, {prob=}")
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
                    if residue1 in used_residues or residue2 in used_residues:
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
                    used_residues.append(residue1)
                    used_residues.append(residue2)
                    self.history.append(set(proposed_candidate_pair))
                    print(
                        f"{residue1.current_name}:{residue1.residue.id}:{charge_candidate_idx1}-{residue2.current_name}:{residue2.residue.id}:{charge_candidate_idx2} pair accepted ..."
                    )
                # return proposed_candidate_pair
        return proposed_candidate_pairs

    def _get_positions_for_mutation_sites(self) -> tuple[list, listt]:
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
            assert residue.current_name in self.ionic_liquid.templates.names
            # get the position of the atom (Hydrogen or the possible acceptor)
            # new idea: just make one list with all positions and then calc distances of everything with everything... -> not so fast, but i need i.e. IM1H-IM1
            pos_list.append(
                pos[
                    residue.get_idx_for_atom_name(
                        self.ionic_liquid.templates.states[residue.original_name][
                            "atom_name"
                        ]
                    )
                    # this needs the atom idx to be the same for both topologies
                    # TODO: maybe get rid of this caveat
                    # maybe some mapping between possible residue states and corresponding atom positions
                ]
            )
            res_list.append(residue)

        return pos_list, res_list
