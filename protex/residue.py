from __future__ import annotations

import copy
import itertools
import logging
from collections import deque

try:
    import openmm.unit as unit
except ImportError:
    import simtk.unit as unit

import numpy as np

logger = logging.getLogger(__name__)

def is_allowed_combination(residue1: Residue, atom_name1: str, residue2: Residue, atom_name2: str) -> bool:
    """Check if the two residues are different residues and if the mode combination is allowed.

    Parameters
    ----------
    residue1 : Residue
        The first residue in the update
    atom_name1 : str
        The atom used of residue 1
    residue2 : Residue
        the second residue in the update
    atom_name2 : str
        The atom used of residue 2

    Returns
    -------
    bool
        True if the combination is allowed, false otherwise
    """
    mode1 = residue1.mode_in_last_transfer
    mode2 = residue2.mode_in_last_transfer
    idx1 = residue1.residue.index
    idx2 = residue2.residue.index

    return idx1 != idx2 and is_allowed_mode_combination(mode1, mode2)

def is_allowed_mode_combination(mode1: str, mode2: str) -> bool:
    """Determine if the two modes are one acceptor and one donor.

    Parameters
    ----------
    mode1 : str
        first mode, either 'donor' or 'acceptor'
    mode2 : str
        second mode, either 'donor' or 'acceptor'

    Returns
    -------
    bool
        True if one acceptor one donor, false otherwise
    """
    if mode1 == "acceptor" and mode2 == "donor":
        return True
    elif mode1 == "donor" and mode2 == "acceptor":
        return True
    else:
        return False

class Residue:
    """Residue extends the OpenMM Residue Class by important features needed for the proton transfer.

    Parameters
    ----------
    residue: openmm.app.topology.Residue
        The residue from  an OpenMM Topology
    orderes_names: list(str)
        The name(s) of the corresponding protonated/deprotonated form(s) (eg. OAC and H2OAc for HOAC)
    system: openmm.openmm.System
        The system generated with openMM, where all residues are in
    initial_parameters: dict[list]
        The parameters for the residue
    alternativ_parameters: dict[list]
        The parameters for the alternativ (protonated/deprotonated) state
    force_idxs:
    has_equivalent_atoms:  tuple[bool,bool]
        if original name and alternative name have equivalent atoms

    Attributes
    ----------
    residue: openmm.app.topology.Residue
        The residue from  an OpenMM Topology
    original_name: str
        The name of the residue given by the OpenMM Residue, will not change throughout the simulation(?)
    current_name: str
        The current name of the residue, depending on the protonation state
    atom_idxs: list[int]
        List of all atom indices belonging to that residue
    atom_names; list[str]
        List of all atom names belonging to that residue
    parameters: dict[str: dict[list]]
        Dictionary containnig the parameters for ``original_name`` and ``alternativ_names``
    record_charge_state: list
        deprecated 1.1?
        Records the charge state of that residue
    system: openmm.openmm.System
        The system generated with openMM, where all residues are in
    equivalent_atoms: dict[str, bool]
        if orignal_name and alternative name have equivalent atoms
    force_idxs:

    donors: list(str)
        atom names (NOTE: maybe use indices within molecule?) that are currently real Hs (NOTE: at the moment residues with only acceptor mode may also have donor Hs, e.g. OH)
    acceptors: list(str)
        atom names (NOTE: maybe use indices within molecule?) that are currently dummies
    """

    def __init__(
        self,
        residue,
        ordered_names,
        system,
        parameters,
        H_parameters,
        D_parameters,
        # pair_12_13_exclusion_list, # deprecated
        states, # do we need this?
        modes_dict,
        starting_donors, # is this still needed?
        starting_acceptors,
        donors,
        acceptors,
        force_idxs=dict(),
    ) -> None:
        self.residue = residue
        self.original_name = residue.name
        self.current_name = self.original_name
        self.system = system
        self.ordered_names = ordered_names
        self.atom_idxs = [atom.index for atom in residue.atoms()]
        self.atom_names = [atom.name for atom in residue.atoms()]
        self.parameters = parameters
        self.H_parameters = H_parameters
        self.D_parameters = D_parameters
        self.record_charge_state = []
        self.record_charge_state.append(self.endstate_charge)
        self.modes_dict = modes_dict
        self.starting_donors = starting_donors
        self.starting_acceptors = starting_acceptors
        self.donors = donors
        self.acceptors = acceptors
        self.force_idxs = force_idxs
        # print(self.current_name) # for debugging
        # print(f"{self.force_idxs=}")
        # self.pair_12_13_list = pair_12_13_exclusion_list
        self.states = states # do we need this?
        self.used_atom = None
        self.mode_in_last_transfer = None
        #self._setup_donors_acceptors()

    def __str__(self) -> str:
        return f"Residue {self.current_name}, {self.residue}"

    def __eq__(self, other) -> bool:
        return self.residue.index == other.residue.index

    def __hash__(self):
        return hash(self.residue.index)

    def _get_shift(self, mode):
            if mode == "acceptor":
                return 1
            if mode == "donor":
                return -1

    ## not needed any more when pickling whole system
    # def _setup_donors_acceptors(self):
    #     #print(self.current_name, self.donors, self.starting_donors, self.force_idxs, self.states)
    #     if self.starting_acceptors is not None and self.starting_donors is not None:
    #         if set(self.donors) != set(self.starting_donors) or set(self.acceptors) != set(self.starting_acceptors):
    #             H_parms = self._get_H_D_NonbondedForce_parameters_at_setup("donors")
    #             D_parms = self._get_H_D_NonbondedForce_parameters_at_setup("acceptors")
    #             self._setup_donor_acceptor_parms(H_parms, D_parms)

    # def _get_H_D_NonbondedForce_parameters_at_setup(self, mode) -> list[float]:
    #     if mode == "donors": # want to have parameters for real H
    #         nonbonded_parm_new = self.H_parameters[self.current_name]["NonbondedForce"]
    #     elif mode == "acceptors": # want to have parameters for D, TODO change to syntax for H_parameters if we want to use combined lone pair - dummy Hs
    #         nonbonded_parm_new = [unit.quantity.Quantity(value=0.0, unit=unit.elementary_charge), unit.quantity.Quantity(value=0.0, unit=unit.nanometer), unit.quantity.Quantity(value=0.0, unit=unit.kilojoule_per_mole)]

    #     return nonbonded_parm_new

    # def _setup_donor_acceptor_parms(self, H_parms, D_parms):
    #         for force in self.system.getForces():
    #             if type(force).__name__ == "NonbondedForce":
    #                 for atom in self.residue.atoms():
    #                     # print(atom)
    #                     # print(H_parms)
    #                     # print(D_parms)
    #                     idx = atom.index
    #                     if atom.name  in self.donors:
    #                         charge, sigma, epsilon = H_parms
    #                     elif atom.name in self.acceptors:
    #                         charge, sigma, epsilon = D_parms
    #                     else: # atom not a protonatable H or dummy
    #                         continue

    #                     force.setParticleParameters(idx, charge, sigma, epsilon)

    # @property
    # def update_resname(self):
    #     """Updates the current name to the new name.

    #     This is based on the current residue name and the used_atom for this update.
    #     The used_atom is reset afterwards, and has to be set again before using this funciton again.
    #     This is on purpose, to only allow name changes once per update.
    #     """
    #     # do we need this? we only allow one update per residue anyways
    #     new_name = self.alternativ_resname
    #     # should be used only once per update
    #     self.current_name = new_name

    @property
    def possible_modes(self) -> bool:
        """Determines which modes the current residue can have.

        It depends on if the residue is currently OAC (acceptor) or HOAC (donor).

        Returns
        -------
        tuple
            the possible modes
        """
        if self.modes_dict is not None:
            return self.modes_dict[self.current_name]
        else:
            return None

    @property
    def alternativ_resname(self) -> str:
        """Alternative name for the residue, e.g. the corresponding name for the protonated/deprotonated form.

        Returns
        -------
        str
            The alternative name
        """
        if self.mode_in_last_transfer is None:
            logger.critical(f"{self.original_name=}, {self.current_name=}, {self.residue.index=}, {self.residue=}")
            raise RuntimeError("Residue was not used in any transfers yet.")
        # check position in ordered names and then decide if go to left (= less H -> donated), or right ->more H
        current_pos = self.ordered_names.index(self.current_name)
        mode = self.mode_in_last_transfer
        # logger.debug(self.current_name)
        # logger.debug(mode)
        # logger.debug(self.ordered_names)
        # logger.debug(self._get_shift(mode))
        try:
            new_name = self.ordered_names[current_pos + self._get_shift(mode)]
        except(IndexError):
            logger.debug(self.current_name)
            logger.debug(mode)
            logger.debug(self.ordered_names)
            logger.debug(self._get_shift(mode))
        return new_name

    def get_mode_in_last_transfer_for(self) -> str:
        # TODO do we need this? (generally, when do we have getters and setters vs direct access vs properties)
        """Return the mode of the current resname and atom_name.

        Parameters
        ----------
        atom_name: str
            Name of the atom

        Returns
        -------
        str
            The mode
        """
        if self.used_atom is None:
            raise RuntimeError("self.used_atom not set")

        # # original idea from Flo:
        # atom_name = self.used_atom
        # for atom in self.states[self.current_name]["atoms"]:
        #     # also check if it is an equivalent atom, then the transfer is also fine
        #     possible_atom_names = [atom["name"], atom.get("equivalent_atom", None)]
        #     if atom_name in possible_atom_names:
        #         return atom["mode"]
        # # now trying to make it more universal (mode is property of residue template, atoms are classified as donors or acceptors, but this can change)
        logger.debug(self.current_name)
        logger.debug(self.possible_modes)
        logger.debug(self.used_atom)
        logger.debug(self.acceptors)
        logger.debug(self.donors)
        logger.debug(self.mode_in_last_transfer)

        return self.mode_in_last_transfer

        # # original idea from Marta
        # if atom_name in self.donors and "donor" in self.possible_modes:
        #     mode = "donor"
        # elif atom_name in self.acceptors and "acceptor" in self.possible_modes:
        #     mode = "acceptor"
        # else:
        #     raise RuntimeError("Not allowed combination of atoms or modes.")
        # return mode


    def update(
        self, force_name: str, lamb: float
    ) -> (
        None
    ):  # we don't need to call update in context since we are doing this in NaiveMCUpdate
        """Update the requested force in that residue.

        Parameters
        ----------
        force_name: str
            Name of the force to update
        lamb: float
            lambda state at which to get corresponding values (between 0 and 1)

        Returns
        -------
        None
        """
        if force_name == "NonbondedForce":
            parms = self._get_NonbondedForce_parameters_at_lambda(lamb)
            #H_D_parms = self._get_H_D_NonbondedForce_parameters_at_lambda(lamb)
            self._set_NonbondedForce_parameters(parms)
        elif force_name == "CustomNonbondedForce":
            parms = self._get_CustomNonbondedForce_parameters_at_lambda(lamb)
            self._set_CustomNonbondedForce_parameters(parms)
        elif force_name == "HarmonicBondForce":
            parms = self._get_HarmonicBondForce_parameters_at_lambda(lamb)
            self._set_HarmonicBondForce_parameters(parms)
        elif force_name == "HarmonicAngleForce":
            parms = self._get_HarmonicAngleForce_parameters_at_lambda(lamb)
            self._set_HarmonicAngleForce_parameters(parms)
        elif force_name == "PeriodicTorsionForce":
            parms = self._get_PeriodicTorsionForce_parameters_at_lambda(lamb)
            self._set_PeriodicTorsionForce_parameters(parms)
        elif force_name == "CustomTorsionForce":
            parms = self._get_CustomTorsionForce_parameters_at_lambda(lamb)
            self._set_CustomTorsionForce_parameters(parms)
        elif force_name == "DrudeForce":
            parms = self._get_DrudeForce_parameters_at_lambda(lamb)
            self._set_DrudeForce_parameters(parms)
        else:
            raise RuntimeWarning(
                f"Force name {force_name} is not covered, no updates will happen on this one!"
            )

    # getting H/D parameters correctly should be handled by _get_NonbondedForce_parameters_at_lambda now
        # can set them here without making distinctions
    def _set_NonbondedForce_parameters(self, parms) -> None:  # noqa: N802
        parms_nonb = deque(parms[0])
        parms_exceptions = deque(parms[1])
        for force in self.system.getForces():
            fgroup = force.getForceGroup()
            if type(force).__name__ == "NonbondedForce":
                for parms_nonbonded, idx in zip(parms_nonb, self.atom_idxs):
                    charge, sigma, epsilon = parms_nonbonded
                    force.setParticleParameters(idx, charge, sigma, epsilon)
                try:  # use the fast way
                    lst = self.force_idxs[fgroup]["NonbondedForceExceptions"]
                    for exc_idx, idx1, idx2 in lst:
                        chargeprod, sigma, epsilon = parms_exceptions.popleft()
                        force.setExceptionParameters(
                            exc_idx, idx1, idx2, chargeprod, sigma, epsilon
                        )
                except KeyError:  # use the old slow way
                    for exc_idx in range(force.getNumExceptions()):
                        f = force.getExceptionParameters(exc_idx)
                        idx1 = f[0]
                        idx2 = f[1]
                        if idx1 in self.atom_idxs and idx2 in self.atom_idxs:
                            chargeprod, sigma, epsilon = parms_exceptions.popleft()
                            force.setExceptionParameters(
                                exc_idx, idx1, idx2, chargeprod, sigma, epsilon
                            )

    def _set_CustomNonbondedForce_parameters(self, parms) -> None:  # noqa: N802
        #print(f"{parms=}")
        parms_nonb = deque(parms[0])
        #print(f"{parms_nonb=}")
        parms_exclusions = deque(parms[1])
        parms_nonb_thole = deque(parms[2])
        parms_exclusions_thole = deque(parms[3])
        # NOTE take care with order of parameters
        # now including 2 CustomNonbondedForces (from NBFIX and NBTHOLE)
        # TODO what happens if there are more? can there be more?
        print(f"{len(parms_exclusions)=}")
        for force in self.system.getForces():
            fgroup = force.getForceGroup()
            if type(force).__name__ == "CustomNonbondedForce" and len(force.getParticleParameters(0)) == 1: # from NBFIX, tabulated
                for parms_nonbonded, idx in zip(parms_nonb, self.atom_idxs):
                    force.setParticleParameters(idx, parms_nonbonded)
                try:  # use the fast way
                    lst = self.force_idxs[fgroup]["CustomNonbondedForceExclusions"]
                    print(f"{len(lst)=}")
                    for exc_idx, idx1, idx2 in lst:
                        excl_idx1, excl_idx2 = parms_exclusions.popleft()
                        force.setExclusionParticles(
                            exc_idx, excl_idx1, excl_idx2
                        )
                except KeyError:  # use the old slow way # NOTE probably doesn't work anymore
                    #logger.debug(force.getNumExclusions())
                    #logger.debug(len(parms_exclusions))
                    for exc_idx in range(force.getNumExclusions()):
                        f = force.getExclusionParticles(exc_idx)
                        #logger.debug(f)
                        idx1 = f[0]
                        idx2 = f[1]
                        if idx1 in self.atom_idxs and idx2 in self.atom_idxs:
                            idxs = (idx1, idx2)
                            #logger.debug(idxs)
                            excl1, excl2 = parms_exclusions.popleft()
                            force.setExclusionParticles(
                                exc_idx, excl1,excl2
                            )

            elif type(force).__name__ == "CustomNonbondedForce" and len(force.getParticleParameters(0)) == 3: # from NBTHOLE
                for parms_nonbonded, idx in zip(parms_nonb_thole, self.atom_idxs):
                    force.setParticleParameters(idx, parms_nonbonded)
                try:  # use the fast way
                    lst = self.force_idxs[fgroup]["CustomNonbondedForceTholeExclusions"]
                    print(f"{len(lst)=}")
                    for exc_idx, idx1, idx2 in lst:
                        excl_idx1, excl_idx2 = parms_exclusions_thole.popleft()
                        force.setExclusionParticles(
                            exc_idx, excl_idx1, excl_idx2
                        )
                except KeyError:  # use the old slow way # NOTE probably doesn't work anymore
                    #logger.debug(force.getNumExclusions())
                    #logger.debug(len(parms_exclusions))
                    for exc_idx in range(force.getNumExclusions()):
                        f = force.getExclusionParticles(exc_idx)
                        #logger.debug(f)
                        idx1 = f[0]
                        idx2 = f[1]
                        if idx1 in self.atom_idxs and idx2 in self.atom_idxs:
                            idxs = (idx1, idx2)
                            #logger.debug(idxs)
                            excl1, excl2 = parms_exclusions.popleft()
                            force.setExclusionParticles(
                                exc_idx, excl1,excl2
                            )

    def _set_HarmonicBondForce_parameters(self, parms) -> None:  # noqa: N802
        parms = deque(parms)
        for force in self.system.getForces():
            fgroup = force.getForceGroup()
            if type(force).__name__ == "HarmonicBondForce":
                try:  # use the fast way
                    lst = self.force_idxs[fgroup]["HarmonicBondForce"]
                    for bond_idx, idx1, idx2 in lst:
                        r, k = parms.popleft()
                        force.setBondParameters(bond_idx, idx1, idx2, r, k)
                except KeyError:
                    for bond_idx in range(force.getNumBonds()):
                        f = force.getBondParameters(bond_idx)
                        idx1, idx2 = f[0], f[1]
                        if idx1 in self.atom_idxs and idx2 in self.atom_idxs:
                            r, k = parms.popleft()
                            force.setBondParameters(bond_idx, idx1, idx2, r, k)

    def _set_HarmonicAngleForce_parameters(self, parms) -> None:  # noqa: N802
        parms = deque(parms)

        for force in self.system.getForces():
            if type(force).__name__ == "HarmonicAngleForce":
                fgroup = force.getForceGroup()
                try:
                    lst = self.force_idxs[fgroup]["HarmonicAngleForce"]
                    for angle_idx, idx1, idx2, idx3 in lst:
                        thetha, k = parms.popleft()
                        force.setAngleParameters(angle_idx, idx1, idx2, idx3, thetha, k)
                except KeyError:
                    # else:
                    for angle_idx in range(force.getNumAngles()):
                        f = force.getAngleParameters(angle_idx)
                        idx1, idx2, idx3 = f[0], f[1], f[2]
                        if (
                            idx1 in self.atom_idxs
                            and idx2 in self.atom_idxs
                            and idx3 in self.atom_idxs
                        ):
                            thetha, k = parms.popleft()
                            force.setAngleParameters(
                                angle_idx, idx1, idx2, idx3, thetha, k
                            )

    def _set_PeriodicTorsionForce_parameters(self, parms) -> None:  # noqa: N802
        parms = deque(parms)

        for force in self.system.getForces():
            if type(force).__name__ == "PeriodicTorsionForce":
                fgroup = force.getForceGroup()
                try:
                    lst = self.force_idxs[fgroup]["PeriodicTorsionForce"]
                    for (
                        torsion_idx,
                        idx1,
                        idx2,
                        idx3,
                        idx4,
                    ) in lst:
                        per, phase, k = parms.popleft()
                        force.setTorsionParameters(
                            torsion_idx, idx1, idx2, idx3, idx4, per, phase, k
                        )
                except KeyError:
                    for torsion_idx in range(force.getNumTorsions()):
                        f = force.getTorsionParameters(torsion_idx)
                        idx1, idx2, idx3, idx4 = f[0], f[1], f[2], f[3]
                        if (
                            idx1 in self.atom_idxs
                            and idx2 in self.atom_idxs
                            and idx3 in self.atom_idxs
                            and idx4 in self.atom_idxs
                        ):
                            per, phase, k = parms.popleft()
                            force.setTorsionParameters(
                                torsion_idx, idx1, idx2, idx3, idx4, per, phase, k
                            )

    def _set_CustomTorsionForce_parameters(self, parms) -> None:  # noqa: N802
        parms = deque(parms)

        for force in self.system.getForces():
            if type(force).__name__ == "CustomTorsionForce":
                fgroup = force.getForceGroup()
                try:
                    lst = self.force_idxs[fgroup]["CustomTorsionForce"]
                    for (
                        ctorsion_idx,
                        idx1,
                        idx2,
                        idx3,
                        idx4,
                    ) in lst:
                        k, psi0 = parms.popleft()  # tuple with (k,psi0)
                        force.setTorsionParameters(
                            ctorsion_idx, idx1, idx2, idx3, idx4, (k, psi0)
                        )
                except KeyError:
                    for torsion_idx in range(force.getNumTorsions()):
                        f = force.getTorsionParameters(torsion_idx)
                        idx1, idx2, idx3, idx4 = f[0], f[1], f[2], f[3]
                        if (
                            idx1 in self.atom_idxs
                            and idx2 in self.atom_idxs
                            and idx3 in self.atom_idxs
                            and idx4 in self.atom_idxs
                        ):
                            k, psi0 = parms.popleft()  # tuple with (k,psi0)
                            force.setTorsionParameters(
                                torsion_idx, idx1, idx2, idx3, idx4, (k, psi0)
                            )

    def _set_DrudeForce_parameters(self, parms) -> None:  # noqa: N802 # NOTE maybe split DrudeForce and DrudeForceThole? (thole not always present)
        #print(f"setting drudes for {self.current_name}")
        parms_pol = deque(parms[0])
        parms_thole = deque(parms[1])
        particle_map = {}
        for force in self.system.getForces():
            if type(force).__name__ == "DrudeForce":
                #print(f"{force.getForceGroup()=}, {force.getNumParticles()=}")
                fgroup = force.getForceGroup()
                try:
                    lst = self.force_idxs[fgroup]["DrudeForce"]
                    for (
                        drude_idx,
                        idx1,
                        idx2,
                        idx3,
                        idx4,
                        idx5,
                    ) in lst:
                        charge, pol, aniso12, aniso14 = parms_pol.popleft()
                        force.setParticleParameters(
                            drude_idx,
                            idx1,
                            idx2,
                            idx3,
                            idx4,
                            idx5,
                            charge,
                            pol,
                            aniso12,
                            aniso14,
                        )
                        particle_map[drude_idx] = idx1
                        #print(f"added {drude_idx} to map normally")
                except KeyError:
                    for drude_idx in range(force.getNumParticles()):
                        f = force.getParticleParameters(drude_idx)
                        idx1, idx2, idx3, idx4, idx5 = f[0], f[1], f[2], f[3], f[4]
                        if idx1 in self.atom_idxs and idx2 in self.atom_idxs:
                            charge, pol, aniso12, aniso14 = parms_pol.popleft()
                            force.setParticleParameters(
                                drude_idx,
                                idx1,
                                idx2,
                                idx3,
                                idx4,
                                idx5,
                                charge,
                                pol,
                                aniso12,
                                aniso14,
                            )
                        particle_map[drude_idx] = idx1
                        #print(f"added {drude_idx} to map via exception")
                try:
                    lst = self.force_idxs[fgroup]["DrudeForceThole"]
                    #print(f"{lst=}")
                    for thole_idx, idx1, idx2 in lst:
                        thole = parms_thole.popleft()
                        force.setScreenedPairParameters(thole_idx, idx1, idx2, thole)
                        #print(f"set thole exception parameters for {idx1}, {idx2} normally")
                except KeyError:
                    try:
                        for drude_idx in range(force.getNumScreenedPairs()):
                            f = force.getScreenedPairParameters(drude_idx)
                            idx1, idx2 = f[0], f[1]
                            drude1, drude2 = particle_map[idx1], particle_map[idx2]
                            # parent1, parent2 = self.pair_12_13_list[drude_idx]
                            # drude1, drude2 = parent1 + 1, parent2 + 1
                            if drude1 in self.atom_idxs and drude2 in self.atom_idxs:
                                thole = parms_thole.popleft()
                                force.setScreenedPairParameters(
                                    drude_idx, idx1, idx2, thole
                                )
                                #print(f"set thole exception parameters for {idx1}, {idx2} via exception")
                    except:
                        continue
                        # hopefully skipping residues without thole screening only
                        # TODO test

    def _get_NonbondedForce_parameters_at_lambda(  # noqa: N802
        self, lamb: float
    ) -> list[list[float]]:
        # returns interpolated sorted nonbonded Forces.
        assert lamb >= 0 and lamb <= 1
        current_name = self.current_name
        new_name = self.alternativ_resname

        nonbonded_parm_old = [
            parm for parm in self.parameters[current_name]["NonbondedForce"]
        ]
        nonbonded_parm_new = [
            parm for parm in self.parameters[new_name]["NonbondedForce"]
        ]
        # NOTE idea: instead of getting H/D parms extra, replace parameters of Hs and Ds here?
            # could do the same with CNBF, take care with extra entries
        ##################### like this:
        for atom in self.atom_names:
            if atom in self.acceptors:
                nonbonded_parm_new[self.atom_names.index(atom)] = self.D_parameters[new_name]["NonbondedForce"]
            elif atom in self.donors:
                nonbonded_parm_new[self.atom_names.index(atom)] = self.H_parameters[new_name]["NonbondedForce"]
        ###############


        assert len(nonbonded_parm_old) == len(nonbonded_parm_new)
        parm_interpolated = []

        for parm_old, parm_new in zip(nonbonded_parm_old, nonbonded_parm_new):
            charge_old, sigma_old, epsilon_old = parm_old
            charge_new, sigma_new, epsilon_new = parm_new

            charge_interpolated = (1 - lamb) * charge_old + lamb * charge_new
            sigma_interpolated = (1 - lamb) * sigma_old + lamb * sigma_new
            epsilon_interpolated = (1 - lamb) * epsilon_old + lamb * epsilon_new

            # test only charge transfer, no sigma, epsilon:
            # sigma_interpolated = sigma_old
            # epsilon_interpolated = epsilon_old
            # charge_interpolated = charge_old

            parm_interpolated.append(
                [charge_interpolated, sigma_interpolated, epsilon_interpolated]
            )

        # Exceptions
        force_name = "NonbondedForceExceptions"
        new_parms_offset = self._get_offset(new_name)
        old_parms_offset = self._get_offset(current_name)

        # match parameters
        parms_old = []
        parms_new = []
        for old_idx, old_parm in enumerate(self.parameters[current_name][force_name]):
            idx1, idx2 = old_parm[0], old_parm[1]
            for new_idx, new_parm in enumerate(self.parameters[new_name][force_name]):
                if {new_parm[0] - new_parms_offset, new_parm[1] - new_parms_offset} == {
                    idx1 - old_parms_offset,
                    idx2 - old_parms_offset,
                }:
                    if old_idx != new_idx:
                        raise RuntimeError(
                            "Odering of Nonbonded Exception parameters is different between the two topologies."
                        )
                    parms_old.append(old_parm)
                    parms_new.append(new_parm)
                    break
            else:
                raise RuntimeError()

        # update parameters for Hs and Ds extra
            # something like this
            # BUG still won't work, have to juggle around atom indices more (keep parameters and index that is not of the H/D, exchange idx of the H/D)
        for atom in self.atom_names:
            idx = self.get_idx_for_atom_name(atom)
            used_parms_old = [parm for parm in parms_old if (parm[0] - old_parms_offset == idx or parm[1] - old_parms_offset == idx) ]
            for parm in used_parms_old:
                if atom in self.acceptors:
                    parms_new[parms_old.index(parm)] = self.D_parameters["NonbondedForceExceptions"][parms_old.index(parm)]
                elif atom in self.donors:
                    parms_new[parms_old.index(parm)] = self.H_parameters["NonbondedForceExceptions"][parms_old.index(parm)]

        # interpolate parameters
        exceptions_interpolated = []
        for parm_old_i, parm_new_i in zip(parms_old, parms_new):
            chargeprod_old, sigma_old, epsilon_old = parm_old_i[-3:]
            chargeprod_new, sigma_new, epsilon_new = parm_new_i[-3:]
            chargeprod_interpolated = (
                1 - lamb
            ) * chargeprod_old + lamb * chargeprod_new
            sigma_interpolated = (1 - lamb) * sigma_old + lamb * sigma_new
            epsilon_interpolated = (1 - lamb) * epsilon_old + lamb * epsilon_new

            exceptions_interpolated.append(
                [chargeprod_interpolated, sigma_interpolated, epsilon_interpolated]
            )

        return [parm_interpolated, exceptions_interpolated]

    # deprecated?
    # def _get_H_D_NonbondedForce_parameters_at_lambda(self,  lamb: float
    # ) -> list[list[float]]:

    #     # logger.debug(self.H_parameters)

    #     assert lamb >= 0 and lamb <= 1
    #     current_name = self.current_name
    #     new_name = self.alternativ_resname

    #     parms_interpolated = {}

    #     for atom in self.atom_names:
    #         if atom not in self.acceptors and atom not in self.donors:
    #             continue
    #         else:
    #             idx = self.atom_names.index(atom)
    #             nonbonded_parm_old = self.parameters[current_name]["NonbondedForce"][idx]
    #             if atom in self.acceptors:
    #                 nonbonded_parm_new = [unit.quantity.Quantity(value=0.0, unit=unit.elementary_charge), unit.quantity.Quantity(value=0.0, unit=unit.nanometer), unit.quantity.Quantity(value=0.0, unit=unit.kilojoule_per_mole)]
    #             elif atom in self.donors:
    #                 nonbonded_parm_new = self.H_parameters[new_name]["NonbondedForce"]

    #             charge_old, sigma_old, epsilon_old = nonbonded_parm_old
    #             charge_new, sigma_new, epsilon_new = nonbonded_parm_new
    #             charge_interpolated = (1 - lamb) * charge_old + lamb * charge_new
    #             sigma_interpolated = (1 - lamb) * sigma_old + lamb * sigma_new
    #             epsilon_interpolated = (1 - lamb) * epsilon_old + lamb * epsilon_new

    #             parms_interpolated[atom] = [charge_interpolated, sigma_interpolated, epsilon_interpolated]


    #     # # leave exceptions to be handled by the general nonbonded force update for the moment
    #     # force_name = "NonbondedForceExceptions"
    #     # new_parms_offset = self._get_offset(new_name)
    #     # old_parms_offset = self._get_offset(current_name)

    #     # # match parameters
    #     # parms_old = []
    #     # parms_new = []
    #     # for old_idx, old_parm in enumerate(self.parameters[current_name][force_name]):
    #     #     idx1, idx2 = old_parm[0], old_parm[1]
    #     #     for new_idx, new_parm in enumerate(self.parameters[new_name][force_name]):
    #     #         if {new_parm[0] - new_parms_offset, new_parm[1] - new_parms_offset} == {
    #     #             idx1 - old_parms_offset,
    #     #             idx2 - old_parms_offset,
    #     #         }:
    #     #             if old_idx != new_idx:
    #     #                 raise RuntimeError(
    #     #                     "Odering of Nonbonded Exception parameters is different between the two topologies."
    #     #                 )
    #     #             parms_old.append(old_parm)
    #     #             parms_new.append(new_parm)
    #     #             break
    #     #     else:
    #     #         raise RuntimeError()

    #     # # interpolate parameters
    #     # exceptions_interpolated = []
    #     # for parm_old_i, parm_new_i in zip(parms_old, parms_new):
    #     #     chargeprod_old, sigma_old, epsilon_old = parm_old_i[-3:]
    #     #     chargeprod_new, sigma_new, epsilon_new = parm_new_i[-3:]
    #     #     chargeprod_interpolated = (
    #     #         1 - lamb
    #     #     ) * chargeprod_old + lamb * chargeprod_new
    #     #     sigma_interpolated = (1 - lamb) * sigma_old + lamb * sigma_new
    #     #     epsilon_interpolated = (1 - lamb) * epsilon_old + lamb * epsilon_new

    #     #     exceptions_interpolated.append(
    #     #         [chargeprod_interpolated, sigma_interpolated, epsilon_interpolated]
    #     #     )

    #     # return [parm_interpolated, exceptions_interpolated]
    #     return parms_interpolated

    def _get_CustomNonbondedForce_parameters_at_lambda(  # noqa: N802
        self, lamb: float
    ) -> list[list[int]]:
        # we cover the Customnonbonded force which depends on atom type, this is not interpolatable, hence it is just switched after lamb > 0.5
        # TODO CNBF can hold LJ parameters if there are NBFIX present -> should we interpolate them?
            # there are multiple types of CNBF (e.g. from NBFIX, NBTHOLE)
        assert lamb >= 0 and lamb <= 1
        #what we need to set are the types and exclusions

        if lamb < 0.5:
            #old
            current_name = self.current_name
            cnb_parm = [
                parm for parm in self.parameters[current_name]["CustomNonbondedForce"]
            ]
            cnb_exclusions = [
                parm for parm in self.parameters[current_name]["CustomNonbondedForceExclusions"]
            ]
            cnb_parm_thole = [
                parm for parm in self.parameters[current_name]["CustomNonbondedForceThole"]
            ]
            cnb_exclusions_thole = [
                parm for parm in self.parameters[current_name]["CustomNonbondedForceTholeExclusions"]
            ]
        else: #lamb >= 0.5
            #new
            new_name = self.alternativ_resname
            cnb_parm = [
                parm for parm in self.parameters[new_name]["CustomNonbondedForce"]
            ]
            cnb_exclusions = [
                parm for parm in self.parameters[new_name]["CustomNonbondedForceExclusions"]
            ]
            cnb_parm_thole = [
                parm for parm in self.parameters[new_name]["CustomNonbondedForceThole"]
            ]
            cnb_exclusions_thole = [
                parm for parm in self.parameters[new_name]["CustomNonbondedForceTholeExclusions"]
            ]
            # need to set CNB parameters extra for Hs and Ds, as they can contain LJ parameters
            # FIXME leaving exclusions to be done by the main updater (not correct, esp. for NBF) -> need to calculate new chargeprod and sigma
            # NOTE can't be interpolated at the moment
            # NOTE have to take care with different CNBFs, call the one from NBTHOLE CNBFThole (this one comes first in the list, at least up until now...)
            for atom in self.atom_names:
                if atom in self.acceptors:
                    cnb_parm_thole[self.atom_names.index(atom)] = self.D_parameters[new_name]["CustomNonbondedForce"]
                    cnb_parm[self.atom_names.index(atom)] = self.D_parameters[new_name]["CustomNonbondedForceThole"]
                elif atom in self.donors:
                    cnb_parm[self.atom_names.index(atom)] = self.H_parameters[new_name]["CustomNonbondedForce"]
                    cnb_parm_thole[self.atom_names.index(atom)] = self.H_parameters[new_name]["CustomNonbondedForceThole"]

        return [cnb_parm, cnb_exclusions, cnb_parm_thole, cnb_exclusions_thole]

    def _get_offset(self, name, force_name=None):
        # get offset for atom idx
        if force_name is None:
            force_name = "HarmonicBondForce"
        return min(
            itertools.chain(
                *[query_parm[0:2] for query_parm in self.parameters[name][force_name]]
            )
        )

    def _get_offset_thole(self, name):
        # get offset for atom idx for thole parameters
        force_name = "DrudeForceThole"
        #print(self.parameters[name])
        return min(
            itertools.chain(
                *[query_parm[0:2] for query_parm in self.parameters[name][force_name]]
            )
        )

    def _get_HarmonicBondForce_parameters_at_lambda(self, lamb):  # noqa: N802
        # returns nonbonded Forces ordered.
        assert lamb >= 0 and lamb <= 1
        # get the names of new and current state
        old_name = self.current_name
        new_name = self.alternativ_resname
        parm_interpolated = []
        force_name = "HarmonicBondForce"
        new_parms_offset = self._get_offset(new_name)
        old_parms_offset = self._get_offset(old_name)
        # print(f"{old_name=}, {new_name=}")
        # print(f"{new_parms_offset=}, {old_parms_offset=}")
        # print(f"{self.parameters[old_name][force_name]=}")
        # print(f"{self.parameters[new_name][force_name]=}")

        # match parameters
        parms_old = []
        parms_new = []
        for old_idx, old_parm in enumerate(self.parameters[old_name][force_name]):
            idx1, idx2 = old_parm[0], old_parm[1]
            for new_idx, new_parm in enumerate(self.parameters[new_name][force_name]):
                if {new_parm[0] - new_parms_offset, new_parm[1] - new_parms_offset} == {
                    idx1 - old_parms_offset,
                    idx2 - old_parms_offset,
                }:
                    if old_idx != new_idx:
                        raise RuntimeError(
                            "Odering of bond parameters is different between the two topologies.\n"
                            f"{old_name=}, {old_idx=}, {old_parms_offset=}\n"
                            f"{new_name=}, {new_idx=}, {new_parms_offset=}\n"
                            f"{old_parm=}, {new_parm=}\n"
                        )
                    parms_old.append(old_parm)
                    parms_new.append(new_parm)
                    break
            else:
                raise RuntimeError()

        # interpolate parameters
        for parm_old_i, parm_new_i in zip(parms_old, parms_new):
            r_old, k_old = parm_old_i[-2:]
            r_new, k_new = parm_new_i[-2:]
            r_interpolated = (1 - lamb) * r_old + lamb * r_new
            k_interpolated = (1 - lamb) * k_old + lamb * k_new

            parm_interpolated.append([r_interpolated, k_interpolated])

        return parm_interpolated

    def _get_HarmonicAngleForce_parameters_at_lambda(self, lamb):  # noqa: N802
        # returns HarmonicAngleForce Forces ordered.
        assert lamb >= 0 and lamb <= 1
        # get the names of new and current state
        old_name = self.current_name
        new_name = self.alternativ_resname
        parm_interpolated = []
        force_name = "HarmonicAngleForce"
        new_parms_offset = self._get_offset(new_name)
        old_parms_offset = self._get_offset(old_name)

        # match parameters
        parms_old = []
        parms_new = []

        for old_idx, old_parm in enumerate(self.parameters[old_name][force_name]):
            idx1, idx2, idx3 = old_parm[0], old_parm[1], old_parm[2]
            for new_idx, new_parm in enumerate(self.parameters[new_name][force_name]):
                if {
                    new_parm[0] - new_parms_offset,
                    new_parm[1] - new_parms_offset,
                    new_parm[2] - new_parms_offset,
                } == {
                    idx1 - old_parms_offset,
                    idx2 - old_parms_offset,
                    idx3 - old_parms_offset,
                }:
                    if old_idx != new_idx:
                        raise RuntimeError(
                            "Odering of angle parameters is different between the two topologies."
                        )

                    parms_old.append(old_parm)
                    parms_new.append(new_parm)
                    break
            else:
                raise RuntimeError()

        # interpolate parameters
        for parm_old_i, parm_new_i in zip(parms_old, parms_new):
            r_old, k_old = parm_old_i[-2:]
            r_new, k_new = parm_new_i[-2:]
            theta_interpolated = (1 - lamb) * r_old + lamb * r_new
            k_interpolated = (1 - lamb) * k_old + lamb * k_new

            parm_interpolated.append([theta_interpolated, k_interpolated])

        return parm_interpolated

    def _get_PeriodicTorsionForce_parameters_at_lambda(self, lamb):  # noqa: N802
        # returns PeriodicTorsionForce Forces ordered.
        assert lamb >= 0 and lamb <= 1
        # get the names of new and current state
        old_name = self.current_name
        new_name = self.alternativ_resname
        parm_interpolated = []
        force_name = "PeriodicTorsionForce"
        new_parms_offset = self._get_offset(new_name, force_name=force_name)
        old_parms_offset = self._get_offset(old_name, force_name=force_name)

        torsions = []  # list for atom indices in a dihedral

        # fill list with all dihedrals
        for old_idx, old_parm in enumerate(
            self.parameters[self.current_name]["PeriodicTorsionForce"]
        ):
            idx1, idx2, idx3, idx4 = (
                old_parm[0] - old_parms_offset,
                old_parm[1] - old_parms_offset,
                old_parm[2] - old_parms_offset,
                old_parm[3] - old_parms_offset,
            )
            ids = [idx1, idx2, idx3, idx4]
            torsions.append(ids)

        # match parameters
        parms_old = []
        parms_new = []

        for old_idx, old_parm in enumerate(self.parameters[old_name][force_name]):
            idx1, idx2, idx3, idx4, idx5 = (
                old_parm[0] - old_parms_offset,
                old_parm[1] - old_parms_offset,
                old_parm[2] - old_parms_offset,
                old_parm[3] - old_parms_offset,
                old_parm[4],
            )
            # first 4 parms: atoms in dihedral, 5.: multiplicity -> need all 5, different multiplicities for same dihedral possible
            ids = [idx1, idx2, idx3, idx4]  # get atoms in the current dihedral

            if (
                torsions.count(ids) == 1
            ):  # only 1 dihedral with 1 multiplicity, multiplicity can be different between old and new -> ignore multiplicity
                for new_idx, new_parm in enumerate(
                    self.parameters[new_name][force_name]
                ):
                    if {
                        new_parm[0] - new_parms_offset,
                        new_parm[1] - new_parms_offset,
                        new_parm[2] - new_parms_offset,
                        new_parm[3] - new_parms_offset,
                    } == {idx1, idx2, idx3, idx4}:
                        if old_idx != new_idx:
                            raise RuntimeError(
                                "Odering of dihedral parameters is different between the two topologies."
                            )

                        parms_old.append(old_parm)
                        parms_new.append(new_parm)
                        break
                else:
                    raise RuntimeError()

            else:  # count > 1: multiple dihedrals with different multiplicities for same set of atoms -> also match multiplicity
                for new_idx, new_parm in enumerate(
                    self.parameters[new_name][force_name]
                ):
                    if {
                        new_parm[0] - new_parms_offset,
                        new_parm[1] - new_parms_offset,
                        new_parm[2] - new_parms_offset,
                        new_parm[3] - new_parms_offset,
                        new_parm[4],
                    } == {idx1, idx2, idx3, idx4, idx5}:
                        if old_idx != new_idx:
                            raise RuntimeError(
                                "Odering of dihedral parameters is different between the two topologies."
                            )

                        parms_old.append(old_parm)
                        parms_new.append(new_parm)
                        break
                else:
                    raise RuntimeError()

        # interpolate parameters
        # omm dihedral: [atom1, atom2, atom3, atom4, periodicity, Quantity(value=delta/phase, unit=radian), Quantity(value=Kchi, unit=kilojoule/mole)]
        for parm_old_i, parm_new_i in zip(parms_old, parms_new):
            per_old, phase_old, k_old = parm_old_i[-3:]
            per_new, phase_new, k_new = parm_new_i[-3:]
            k_interpolated = (1 - lamb) * k_old + lamb * k_new

            if lamb <= 0.5:  # use per, phase from original residue
                parm_interpolated.append([per_old, phase_old, k_interpolated])

            if lamb > 0.5:  # use per, phase from final residue
                parm_interpolated.append([per_new, phase_new, k_interpolated])

        return parm_interpolated

    def _get_CustomTorsionForce_parameters_at_lambda(self, lamb):  # noqa: N802
        # returns CustomTorsionForce Forces (=impropers) ordered.
        assert lamb >= 0 and lamb <= 1
        # get the names of new and current state
        old_name = self.current_name
        new_name = self.alternativ_resname
        parm_interpolated = []
        force_name = "CustomTorsionForce"
        new_parms_offset = self._get_offset(new_name)
        old_parms_offset = self._get_offset(old_name)

        # match parameters
        parms_old = []
        parms_new = []

        for old_idx, old_parm in enumerate(self.parameters[old_name][force_name]):
            idx1, idx2, idx3, idx4 = old_parm[0], old_parm[1], old_parm[2], old_parm[3]
            for new_idx, new_parm in enumerate(self.parameters[new_name][force_name]):
                if {
                    new_parm[0] - new_parms_offset,
                    new_parm[1] - new_parms_offset,
                    new_parm[2] - new_parms_offset,
                    new_parm[3] - new_parms_offset,
                } == {
                    idx1 - old_parms_offset,
                    idx2 - old_parms_offset,
                    idx3 - old_parms_offset,
                    idx4 - old_parms_offset,
                }:
                    if old_idx != new_idx:
                        raise RuntimeError(
                            "Odering of improper parameters is different between the two topologies."
                        )
                    parms_old.append(old_parm)
                    parms_new.append(new_parm)
                    break
            else:
                raise RuntimeError()

        # interpolate parameters
        # omm improper: [atom1, atom2, atom3, atom4, k, psi0]
        for parm_old_i, parm_new_i in zip(parms_old, parms_new):
            k_old, psi0_old = parm_old_i[-1]
            k_new, psi0_new = parm_new_i[-1]
            k_interpolated = (1 - lamb) * k_old + lamb * k_new

            if lamb <= 0.5:  # use per, phase from original residue
                parm_interpolated.append([k_interpolated, psi0_old])

            if lamb > 0.5:  # use per, phase from final residue
                parm_interpolated.append([k_interpolated, psi0_new])

        return parm_interpolated

    def _get_DrudeForce_parameters_at_lambda(self, lamb):  # noqa: N802
        # Split in two parts, one for charge and polarizability one for thole
        # returns a list with the two, different than the other get methods!
        # returns Drude Forces ordered.
        assert lamb >= 0 and lamb <= 1
        # get the names of new and current state
        old_name = self.current_name
        new_name = self.alternativ_resname
        parm_interpolated = []
        force_name = "DrudeForce"
        new_parms_offset = self._get_offset(new_name)
        old_parms_offset = self._get_offset(old_name)

        # match parameters
        parms_old = []
        parms_new = []
        for old_idx, old_parm in enumerate(self.parameters[old_name][force_name]):
            idx1, idx2 = old_parm[0], old_parm[1]
            for new_idx, new_parm in enumerate(self.parameters[new_name][force_name]):
                if {new_parm[0] - new_parms_offset, new_parm[1] - new_parms_offset} == {
                    idx1 - old_parms_offset,
                    idx2 - old_parms_offset,
                }:
                    if old_idx != new_idx:
                        raise RuntimeError(
                            "Odering of bond parameters is different between the two topologies."
                        )
                    parms_old.append(old_parm)
                    parms_new.append(new_parm)
                    break
            else:
                raise RuntimeError()

        # interpolate parameters
        for parm_old_i, parm_new_i in zip(parms_old, parms_new):
            charge_old, pol_old, aniso12_old, aniso14_old = parm_old_i[-4:]
            charge_new, pol_new, aniso12_new, aniso14_new = parm_new_i[-4:]
            charge_interpolated = (1 - lamb) * charge_old + lamb * charge_new
            pol_interpolated = (1 - lamb) * pol_old + lamb * pol_new
            aniso12_interpolated = (1 - lamb) * aniso12_old + lamb * aniso12_new
            aniso14_interpolated = (1 - lamb) * aniso14_old + lamb * aniso14_new

            parm_interpolated.append(
                [
                    charge_interpolated,
                    pol_interpolated,
                    aniso12_interpolated,
                    aniso14_interpolated,
                ]
            )

        # Thole
        parm_interpolated_thole = []
        force_name = "DrudeForceThole"
        if force_name in self.parameters[old_name]:
            new_parms_offset = self._get_offset_thole(new_name)
            old_parms_offset = self._get_offset_thole(old_name)
            # print(f"{new_parms_offset=}, {old_parms_offset=}")
            # print(f"{self.parameters[old_name][force_name]=}")
            # print(f"{self.parameters[new_name][force_name]=}")

            # match parameters
            parms_old = []
            parms_new = []
            for old_idx, old_parm in enumerate(self.parameters[old_name][force_name]):
                idx1, idx2 = old_parm[0], old_parm[1]
                for new_idx, new_parm in enumerate(self.parameters[new_name][force_name]):
                    if {new_parm[0] - new_parms_offset, new_parm[1] - new_parms_offset} == {
                        idx1 - old_parms_offset,
                        idx2 - old_parms_offset,
                    }:
                        if old_idx != new_idx:
                            raise RuntimeError(
                                "Odering of bond parameters is different between the two topologies."
                            )
                        parms_old.append(old_parm)
                        parms_new.append(new_parm)
                        break
                else:
                    raise RuntimeError()

            # interpolate parameters
            for parm_old_i, parm_new_i in zip(parms_old, parms_new):
                thole_old = parm_old_i[-1]
                thole_new = parm_new_i[-1]
                thole_interpolated = (1 - lamb) * thole_old + lamb * thole_new

                parm_interpolated_thole.append(thole_interpolated)

        return [parm_interpolated, parm_interpolated_thole]

    # NOTE: this is a bug!
    def get_idx_for_atom_name(self, query_atom_name: str) -> int:
        for idx, atom_name in zip(self.atom_idxs, self.atom_names):
            if query_atom_name == atom_name:
                return idx
        raise RuntimeError(
            f"Atom name '{query_atom_name}' not in atom names of residue '{self.current_name}'."
        )

    # not used?
    @property
    def endstate_charge(self) -> int:
        """Charge of the residue at the endstate (will be int)."""
        charge = int(
            np.round(
                sum(
                    [
                        parm[0]._value
                        for parm in self.parameters[self.current_name]["NonbondedForce"]
                    ]
                ),
                4,
            )
        )
        return charge

    @property
    def current_charge(self) -> int:
        """Current charge of the residue."""
        charge = 0
        for force in self.system.getForces():
            if type(force).__name__ == "NonbondedForce":
                for idx in self.atom_idxs:
                    charge_idx, _, _ = force.getParticleParameters(idx)
                    charge += charge_idx._value

        return np.round(charge, 3)
