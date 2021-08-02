from protex.system import IonicLiquidSystem
from typing import List, Tuple
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

        # get charge set for residue 1
        new_charge_candidate1 = candidate1_residue.get_inactive_charges()
        old_charge_candidate1 = candidate1_residue.get_current_charges()

        # get charge set for residue 2
        new_charge_candidate2 = candidate2_residue.get_inactive_charges()
        old_charge_candidate2 = candidate2_residue.get_current_charges()

        logger.info("Start changing states ...")
        for lamb in np.linspace(0, 1, 101):
            charge_candidate1 = (1.0 - lamb) * np.array(
                old_charge_candidate1
            ) + lamb * np.array(new_charge_candidate1)
            charge_candidate2 = (1.0 - lamb) * np.array(
                old_charge_candidate2
            ) + lamb * np.array(new_charge_candidate2)

            # set new charges
            candidate1_residue.set_new_charges(list(charge_candidate1))
            candidate2_residue.set_new_charges(list(charge_candidate2))
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

    def update(self):
        """
        updates the current state using the method defined in the UpdateMethod class
        """
        # calculate the distance betwen updateable residues
        distance_matrix = self._calculate_distance_between_all_residues()
        # propose the update candidates based on distances
        candidate_pairs = self._propose_candidate_pair(distance_matrix)
        assert len(candidate_pairs) == 2
        self.updateMethod._update(candidate_pairs)

    def _propose_candidate_pair(self, distance_matrix) -> tuple:

        # TODO: select the relevant residue pairs based on the distance matrix that
        # is given as argument
        # for now we hardcode candidates
        #trial_proposed_candidate_pair = (651, 649)

        import random

        random_choice = random.randint(0,1)
        active_matrix = distance_matrix[random_choice] #select from protonation or deprotonation matrix
        
        distance_based = 1.0 # choose distane criterion in 70% of the cases otherwise change randomly

        if random.random() < distance_based:
            #print(np.min(active_matrix))
            if np.min(active_matrix) <= 0.15: # add distance criterion in nm (eg. 0.15nm)
                idx1,idx2 = np.where(active_matrix == np.min(active_matrix)) # muss dann jeweils noch anzahl der vorigen species dauzaddieren...
                if random_choice == 0: # im1h,oac matrix
                    idx1, idx2 = int(idx1), int(int(idx2)/2)+150 # +150 IM1h
                else: # im1,hoac matrix
                    idx1, idx2 = int(idx1)+300, int(idx2)+650 # +300 (IM1h, OAC), +650 (im1h, oac, im1)
        else:
            idx1 = random.randint(0,149) if random.choice == 0 else random.randint(300,649)
            idx2 = random.randint(150,299) if random.choice == 0 else random.randint(650,999)
        #print("Residues", idx1, idx2)

        # from the residue_idx we select the residue instances
        # NOTE: the logic that checks for correct pairing should be moved
        # to calculate_distance_between_all_residues since we calculate
        # distance matrices for each pair
        #idx1, idx2 = trial_proposed_candidate_pair
        residue1, residue2 = (
            self.ionic_liquid.residues[idx1],
            self.ionic_liquid.residues[idx2],
        )

        logger.info(f"{residue1.name}-{residue2.name} pair suggested ...")
        proposed_candidate_pair = (residue1, residue2)
        logger.info(f"{residue1.name}-{residue2.name} pair suggested ...")

        return proposed_candidate_pair

    def _calculate_distance_between_all_residues(self) -> np.ndarray:
        """
        calculate_distance_between_all_residues returns distance matrix
        """
        # TODO: distance matrix needs to be calculated for each IonicLiquid species seperatly
        from scipy.spatial import distance_matrix
        state = self.ionic_liquid.simulation.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True)
        im1h_h_pos = []
        oac_o_pos = []
        im1_n_pos = []
        #hoac_o_pos = []
        hoac_h_pos = []
        #print(f"{self.ionic_liquid.simulation.topology.getPeriodicBoxVectors()=}")
        #print(f"{self.ionic_liquid.simulation.topology.getNumResidues()=}")
        #print(f"{self.ionic_liquid.simulation.topology.getNumChains()=}")
        #for chain in self.ionic_liquid.simulation.topology.chains():
        #    print(f"{len(chain)}")
        #print(f"{self.ionic_liquid.simulation.topology.getNumAtoms()=}")
        #print(f"{self.ionic_liquid.simulation.topology.getNumBonds()=}")
        atoms_im1_h = 20
        for atom in self.ionic_liquid.simulation.topology.atoms():
            #print(len(self.ionic_liquid.system.getForces()[6]))
            #   break
            assert atom.residue.name in self.ionic_liquid.templates.names
            atom_id_zero_based = int(atom.id) -1
            if atom.name == "H7":
                if  atom_id_zero_based > atoms_im1_h*150:
                    id = int((atom_id_zero_based - atoms_im1_h*150-16*150)/atoms_im1_h)+300
                else:
                    id = int(atom_id_zero_based/atoms_im1_h)
                if self.ionic_liquid.residues[id].get_current_charge() == 1: #->IM1H
                    im1h_h_pos.append(pos[atom_id_zero_based])
            if (atom.name == "O1" or atom.name == "O2"):
                if atom_id_zero_based > atoms_im1_h*150+16*150:
                    id = int((atom_id_zero_based - atoms_im1_h*150-16*150-atoms_im1_h*350)/16)+650
                else:
                    id = int((atom_id_zero_based -atoms_im1_h*150)/16)+150
                if self.ionic_liquid.residues[id].get_current_charge() == -1: #->OAC
                    oac_o_pos.append(pos[atom_id_zero_based])
            if atom.name == "N2":
                if atom_id_zero_based > atoms_im1_h*150:
                    id = int((atom_id_zero_based - atoms_im1_h*150-16*150)/atoms_im1_h)+300
                else: 
                    id = int(atom_id_zero_based/atoms_im1_h)
                if self.ionic_liquid.residues[id].get_current_charge() == 0: #->IM1
                    im1_n_pos.append(pos[atom_id_zero_based])
            if atom.name == "H":
                if atom_id_zero_based > atoms_im1_h*150+16*150:
                    id = int((atom_id_zero_based - atoms_im1_h*150-16*150-atoms_im1_h*350)/16)+650
                else: 
                    id = int((atom_id_zero_based - atoms_im1_h*150)/16)+150 
                if self.ionic_liquid.residues[id].get_current_charge() == 0: #->HOAC
                    hoac_h_pos.append(pos[atom_id_zero_based-1])

            #print(atom.id)
            #print(atom)
            #print(atom.residue.name)
            #print(atom.residue.id)
        #print(len(im1h_h_pos))
        #print(len(oac_o_pos))
        #print(len(im1_n_pos))
        #print(len(hoac_h_pos))
        dm_h7_o = distance_matrix(im1h_h_pos,oac_o_pos) #deprot 
        dm_h_n = distance_matrix(im1_n_pos,hoac_h_pos) # prot
        boxl=np.round(self.ionic_liquid.system.getDefaultPeriodicBoxVectors()[0]._value[0],4) #Attention: x=y=z required! unit: nm
        # take PBC into account
        dm_h7_o[dm_h7_o > boxl] -= boxl
        dm_h_n[dm_h_n > boxl] -= boxl
        print(dm_h7_o)
       
        # for idx, r in enumerate(self.ionic_liquid.simulation.topology.residues()):
        #     name = r.name
        
        # im1h_h_pos1 = []
        # oac_o_pos1 = []
        # im1_n_pos1 = []
        # #hoac_o_pos = []
        # hoac_h_pos1 = []

        # for idx, residue in enumerate(self.ionic_liquid.residues):
        #     if residue.name == "IM1H" or residue.name == "IM1":
        #         #IM1H
        #         if residue.get_current_charge() == 1:
        #             for atom in residue.atoms():
        #                 if atom.name == "H7":
        #                     im1h_h_pos1.append(pos[atom.index])
        #         #IM1
        #         if residue.get_current_charge() == 0:
        #             for atom in residue.atoms():
        #                 if atom.name == "N2":
        #                     im1_n_pos1.append(pos[atom.index])

        #     if residue.name == "OAC" or residue.name == "HOAC":
        #         #OAC
        #         if residue.get_current_charge() == -1:
        #             for atom in residue.atoms():
        #                 if atom.name == "O1" or atom.name == "O2":
        #                     oac_o_pos1.append(pos[atom.index])
        #         #HOAC
        #         if residue.get_current_charge() == 0:
        #             for atom in residue.atoms():
        #                 if atom.name == "H":
        #                     hoac_h_pos1.append(pos[atom.index])

        #     #print(r.name)
        # dm_h7_o1 = distance_matrix(im1h_h_pos1,oac_o_pos1) #deprot 
        # dm_h_n1 = distance_matrix(im1_n_pos1,hoac_h_pos1) # prot

        #print(dm_h7_o, dm_h7_o1)

        #return np.ndarray([0, 0])
        return [dm_h7_o, dm_h_n]