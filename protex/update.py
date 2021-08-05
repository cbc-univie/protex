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
            f"candiadate_1: {candidate1_residue.current_name}; charge:{candidate1_residue.get_current_charge()}"
        )
        print(
            f"candiadate_2: {candidate2_residue.current_name}; charge:{candidate2_residue.get_current_charge()}"
        )

        # get charge set for residue 1
        new_charge_candidate1 = candidate1_residue.get_inactive_charges()
        old_charge_candidate1 = candidate1_residue.get_current_charges()
        new_res_name_candidate1 = (
            self.ionic_liquid.templates.get_residue_name_for_coupled_state(
                candidate1_residue.current_name
            )
        )

        # get charge set for residue 2
        new_charge_candidate2 = candidate2_residue.get_inactive_charges()
        old_charge_candidate2 = candidate2_residue.get_current_charges()
        new_res_name_candidate2 = (
            self.ionic_liquid.templates.get_residue_name_for_coupled_state(
                candidate2_residue.current_name
            )
        )

        logger.info("Start changing states ...")
        for lamb in np.linspace(0, 1, nr_of_steps):
            charge_candidate1 = (1.0 - lamb) * np.array(
                old_charge_candidate1
            ) + lamb * np.array(new_charge_candidate1)
            charge_candidate2 = (1.0 - lamb) * np.array(
                old_charge_candidate2
            ) + lamb * np.array(new_charge_candidate2)

            # set new charges
            candidate1_residue.set_new_state(
                list(charge_candidate1), new_res_name_candidate1
            )
            candidate2_residue.set_new_state(
                list(charge_candidate2), new_res_name_candidate2
            )
            # update the context to include the new parameters
            self.ionic_liquid.nonbonded_force.updateParametersInContext(
                self.ionic_liquid.simulation.context
            )
            # get new energy
            state = self.ionic_liquid.simulation.context.getState(getEnergy=True)
            new_e = state.getPotentialEnergy()

            self.ionic_liquid.simulation.step(10)

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

    def write_parameters(self, filename: str):

        par = self.get_parameters()
        with open(filename, "w+") as f:
            for p, atom in par:
                charge = p[0]._value
                f.write(
                    f"{atom.residue.name:>4}:{int(atom.id): 4}:{int(atom.residue.id): 4}:{atom.name:>4}:{charge}\n"
                )

    def get_parameters(self) -> list:

        nonbonded_force = self.ionic_liquid.nonbonded_force
        atom_list = list(self.ionic_liquid.topology.atoms())
        par = []
        for i in range(nonbonded_force.getNumParticles()):
            p = nonbonded_force.getParticleParameters(i)
            a = atom_list[i]
            par.append((p, a))
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
        self.history.append(candidate_pairs)
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
                charge_candidate_idx1 = residue1.get_current_charge()
                charge_candidate_idx2 = residue2.get_current_charge()

                print(
                    f"{residue1.original_name}:{residue1.current_name}:{residue1.residue.id}:{charge_candidate_idx1}-{residue2.original_name}:{residue2.current_name}:{residue2.residue.id}:{charge_candidate_idx2} pair suggested ..."
                )
                print(
                    f"Distance between pairs: {distance[candidate_idx1,candidate_idx2]}"
                )
                proposed_candidate_pair = (residue1, residue2)
                # reject if already in last 10 updates
                if proposed_candidate_pair in self.history[-10:]:
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
                    residue.get_idx_for_name(
                        self.ionic_liquid.templates.states[residue.current_name][
                            "atom_name"
                        ]
                    )  # this needs the atom idx to be the same for both topologies
                ]
            )
            res_dict[residue.canonical_name].append(residue)

        return pos_dict, res_dict
