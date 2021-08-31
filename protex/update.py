from collections import defaultdict

from scipy.spatial import distance_matrix
from protex.system import IonicLiquidSystem
import logging
import numpy as np
from simtk import unit

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

    def _update(self, candidate: tuple, nr_of_steps: int):
        logger.info("called _update")
        # get current state
        state = self.ionic_liquid.simulation.context.getState(getEnergy=True)
        # get initial energy
        initial_e = state.getPotentialEnergy()
        # retrive residue instances
        candidate1_residue, candidate2_residue = candidate

        print(
            f"candiadate_1: {candidate1_residue.current_name}; charge:{candidate1_residue.current_charge}"
        )
        print(
            f"candiadate_2: {candidate2_residue.current_name}; charge:{candidate2_residue.current_charge}"
        )

        logger.info("Start changing states ...")
        for lamb in np.linspace(0, 1, nr_of_steps):
            ######################
            # nonbonded parameters
            ######################
            candidate1_residue.update(
                "NonbondedForce", self.ionic_liquid.simulation.context, lamb
            )
            candidate1_residue.update(
                "HarmonicBondedForce", self.ionic_liquid.simulation.context, lamb
            )
            candidate1_residue.update(
                "HarmonicAngleForce", self.ionic_liquid.simulation.context, lamb
            )
            candidate1_residue.update(
                "DrudeForce", self.ionic_liquid.simulation.context, lamb
            )

            # update the context to include the new parameters
            # self.ionic_liquid.nonbonded_force.updateParametersInContext(
            #     self.ionic_liquid.simulation.context
            # )
            # get new energy
            state = self.ionic_liquid.simulation.context.getState(getEnergy=True)
            new_e = state.getPotentialEnergy()

            self.ionic_liquid.simulation.step(1)
        # after the update is finished the current_name attribute is updated (and since alternative_name depends on current_name it too is updated)
        candidate1_residue.current_name = candidate1_residue.alternativ_name
        candidate2_residue.current_name = candidate2_residue.alternativ_name
        # get new energy
        state = self.ionic_liquid.simulation.context.getState(getEnergy=True)
        new_e = state.getPotentialEnergy()
        logger.info(f"Energy before/after state change:{initial_e}/{new_e}")
        self.ionic_liquid.simulation.context.setVelocitiesToTemperature(
            300.0 * unit.kelvin
        )


class StateUpdate:
    """
    Controls the update sheme and proposes the residues that need an update
    """

    def __init__(self, updateMethod: Update) -> None:
        self.updateMethod = updateMethod
        self.ionic_liquid = self.updateMethod.ionic_liquid
        self.history = []

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
                    charge, sigma, epsiolon = force.getParticleParameters(idx)
                    par.append((idx, atom, charge))
                return par

    def _print_start(self):
        print(
            f"""
##############################
##############################
--- Update trial: {len(self.history)} ---
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
        distance_dict, res_dict = self._get_positions_for_mutation_sites()
        # propose the update candidates based on distances
        self._print_start()
        candidate_pairs = self._propose_candidate_pair(distance_dict, res_dict)
        # add candidate pairs to history
        assert len(candidate_pairs) == 2
        self.history.append(set(candidate_pairs))
        # print details
        # update
        self.updateMethod._update(candidate_pairs, nr_of_steps)
        self._print_stop()

        return candidate_pairs

    def _propose_candidate_pair(self, distance_dict: dict, res_dict: dict) -> tuple:

        canonical_names = list(
            set([residue.canonical_name for residue in self.ionic_liquid.residues])
        )
        logger.debug(canonical_names)
        # calculate distance matrix between the two molecules
        distance = distance_matrix(
            distance_dict[canonical_names[0]], distance_dict[canonical_names[1]]
        )
        # get a list of indices for elements in the distance matrix sorted by increasing distance
        # NOTE: This always accepts a move!
        shape = distance.shape
        idx = np.dstack(np.unravel_index(np.argsort(distance.ravel()), shape))[0]
        # TODO: PBC need to be enforced
        # check if charge transfer is possible
        for candidate_idx1, candidate_idx2 in idx:
            residue1 = res_dict[canonical_names[0]][candidate_idx1]
            residue2 = res_dict[canonical_names[1]][candidate_idx2]
            # is this combination allowed?
            if (
                set([residue1.current_name, residue2.current_name])
                in self.ionic_liquid.templates.allowed_updates
            ):
                charge_candidate_idx1 = residue1.current_charge
                charge_candidate_idx2 = residue2.current_charge

                print(
                    f"{residue1.original_name}:{residue1.current_name}:{residue1.residue.id}:{charge_candidate_idx1}-{residue2.original_name}:{residue2.current_name}:{residue2.residue.id}:{charge_candidate_idx2} pair suggested ..."
                )
                print(
                    f"Distance between pairs: {distance[candidate_idx1,candidate_idx2]}"
                )
                proposed_candidate_pair = (residue1, residue2)
                # reject if already in last 10 updates
                if set(proposed_candidate_pair) in self.history[-10:]:
                    print(
                        f"{residue1.current_name}:{residue1.residue.id}:{charge_candidate_idx1}-{residue2.current_name}:{residue2.residue.id}:{charge_candidate_idx2} pair rejected ..."
                    )

                    continue
                # accept otherwise
                print(
                    f"{residue1.current_name}:{residue1.residue.id}:{charge_candidate_idx1}-{residue2.current_name}:{residue2.residue.id}:{charge_candidate_idx2} pair accepted ..."
                )
                return proposed_candidate_pair

    def _get_positions_for_mutation_sites(self) -> tuple[dict, dict]:
        """
        _get_positions_for_mutation_sites returns
        """
        pos = self.ionic_liquid.simulation.context.getState(
            getPositions=True
        ).getPositions(asNumpy=True)

        # fill in the positions
        pos_dict = defaultdict(list)
        res_dict = defaultdict(list)

        # loop over all residues and add the positions of the atoms that can be updated to the pos_dict
        for residue in self.ionic_liquid.residues:
            assert residue.current_name in self.ionic_liquid.templates.names
            pos_dict[residue.canonical_name].append(
                pos[
                    residue.get_idx_for_atom_name(
                        self.ionic_liquid.templates.states[residue.original_name][
                            "atom_name"
                        ]
                    )  # this needs the atom idx to be the same for both topologies
                ]
            )
            res_dict[residue.canonical_name].append(residue)

        return pos_dict, res_dict
