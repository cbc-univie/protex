from protex.system import IonicLiquidSystem
from typing import List, Tuple
import logging
import numpy as np

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

    def __init__(self, ionic_liquid: IonicLiquidSystem) -> None:
        super().__init__(ionic_liquid)

    def _update(self, candidate: tuple):
        logger.info("called _update")
        # get current state
        state = self.ionic_liquid.simulation.context.getState(getEnergy=True)
        # get initial energy
        initial_e = state.getPotentialEnergy()
        # retrive residue instances
        candidate1_residue, candidate2_residue = candidate

        logger.debug(
            f"candiadate_1: {candidate1_residue.name}; charge:{candidate1_residue.get_current_charge()}"
        )
        logger.debug(
            f"candiadate_2: {candidate2_residue.name}; charge:{candidate2_residue.get_current_charge()}"
        )

        # update charge set for residue 1
        new_charge = candidate1_residue.get_inactive_charges()
        candidate1_residue.set_new_charges(new_charge)

        # update charge set for residue 2
        new_charge = candidate2_residue.get_inactive_charges()
        candidate2_residue.set_new_charges(new_charge)
        # update the context to include the new parameters
        self.ionic_liquid.nonbonded_force.updateParametersInContext(
            self.ionic_liquid.simulation.context
        )
        # get new energy
        state = self.ionic_liquid.simulation.context.getState(getEnergy=True)
        new_e = state.getPotentialEnergy()
        print(f"Energy before/after state change:{initial_e}/{new_e}")


class StateUpdate:
    """
    Controls the update sheme and proposes the residues that need an update
    """

    def __init__(self, updateMethod: Update) -> None:
        self.updateMethod = updateMethod
        self.ionic_liquid = self.updateMethod.ionic_liquid

    def write_parameters(self, filename: str):

        nonbonded_force = self.ionic_liquid.nonbonded_force
        atom_list = list(self.ionic_liquid.topology.atoms())
        with open(filename, "w+") as f:
            for i in range(nonbonded_force.getNumParticles()):
                p = nonbonded_force.getParticleParameters(i)
                a = atom_list[i]
                f.write(
                    f"{a.residue.name:>4}:{int(a.id): 4}:{int(a.residue.id): 4}:{a.name:>4}:{p[0]._value}\n"
                )

    def update(self):
        """
        updates the current state using the method defined in the UpdateMethod class
        """
        # calculate the distance betwen updateable residues
        distance_matrix = self._calculate_distance_between_all_residues()
        # propose the update candidates based on distances
        candidate_pairs = self._propose_candidate_pair()
        assert len(candidate_pairs) == 2
        self.updateMethod._update(candidate_pairs)

    def _propose_candidate_pair(self) -> tuple:

        # TODO: select the relevant residue pairs based on the distance matrix that
        # is given as argument
        # for now we hardcode candidates
        trial_proposed_candidate_pair = (651, 649)

        # from the residue_idx we select the residue instances
        # NOTE: the logic that checks for correct pairing should be moved
        # to calculate_distance_between_all_residues since we calculate
        # distance matrices for each pair
        idx1, idx2 = trial_proposed_candidate_pair
        residue1, residue2 = (
            self.ionic_liquid.residues[idx1],
            self.ionic_liquid.residues[idx2],
        )

        print(f"{residue1.name}-{residue2.name} pair suggested ...")
        proposed_candidate_pair = (residue1, residue2)
        logger.info(f"{residue1.name}-{residue2.name} pair suggested ...")

        return proposed_candidate_pair

    def _calculate_distance_between_all_residues(self) -> np.ndarray:
        """
        calculate_distance_between_all_residues returns distance matrix
        """
        # TODO: distance matrix needs to be calculated for each IonicLiquid species seperatly
        state = self.ionic_liquid.simulation.context.getState(getPositions=True)
        pos = state.getPositions()
        for atom in self.ionic_liquid.simulation.topology.atoms():
            assert atom.residue.name in self.ionic_liquid.templates.names
            # print(atom.id)
            # print(atom)
            # print(atom.residue.name)
            # print(atom.residue.id)

        for idx, r in enumerate(self.ionic_liquid.simulation.topology.residues()):
            name = r.name

        return np.ndarray([0, 0])
