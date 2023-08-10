import itertools
from collections import deque

import numpy as np


class Residue:
    """Residue extends the OpenMM Residue Class by important features needed for the proton transfer.

    Parameters
    ----------
    residue: openmm.app.topology.Residue
        The residue from  an OpenMM Topology
    alternativ_name: str
        The name of the corresponding protonated/deprotonated form (eg. OAC for HOAC)
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
        Dictionary containnig the parameters for ``original_name`` and ``alternativ_name``
    record_charge_state: list
        deprecated 1.1?
        Records the charge state of that residue
    system: openmm.openmm.System
        The system generated with openMM, where all residues are in
    equivalent_atoms: dict[str, bool]
        if orignal_name and alternative name have equivalent atoms
    force_idxs:
    """

    def __init__(
        self,
        residue,
        alternativ_name,
        system,
        inital_parameters,
        alternativ_parameters,
        has_equivalent_atoms,
        force_idxs=dict(),
    ) -> None:
        self.residue = residue
        self.original_name = residue.name
        self.current_name = self.original_name
        self.system = system
        self.atom_idxs = [atom.index for atom in residue.atoms()]
        self.atom_names = [atom.name for atom in residue.atoms()]
        self.parameters = {
            self.original_name: inital_parameters,
            alternativ_name: alternativ_parameters,
        }
        self.record_charge_state = []
        self.record_charge_state.append(self.endstate_charge)  # Not used anywhere?
        if has_equivalent_atoms is not None:
            self.equivalent_atoms: dict[str, bool] = {
                self.original_name: has_equivalent_atoms[0],
                self.alternativ_name: has_equivalent_atoms[1],
            }
        self.equivalent_atom_pos_in_list: int = None
        self.used_equivalent_atom: bool = False
        self.force_idxs = force_idxs

    def __str__(self) -> str:
        return f"Residue {self.current_name}, {self.residue}"

    def __eq__(self, other) -> bool:
        return self.current_name == other.current_name and self.residue.index == other.residue.index

    def __hash__(self):
        return hash((self.current_name, self.residue.index))

    @property
    def has_equivalent_atom(self) -> bool:
        """Determines if the current residue has an equivalent atom defined.

        It depends i.e if the residue is currently OAC (-> two equivalent O's) or HOAC (no equivlent O's).

        Returns
        -------
        bool
            True if this residue currently has an equivalent atom, else otherwise
        """
        return self.equivalent_atoms[self.current_name]

    @property
    def alternativ_name(self) -> str:
        """Alternative name for the residue, e.g. the corresponding name for the protonated/deprotonated form.

        Returns
        -------
        str
            The alternative name
        """
        for name in self.parameters.keys():
            if name != self.current_name:
                return name

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
                "Force name {force_name=} is not covered, no updates will happen on this one!"
            )

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
        parms_nonb = deque(parms[0])
        parms_exclusions = deque(parms[1])
        for force in self.system.getForces():
            fgroup = force.getForceGroup()
            if type(force).__name__ == "CustomNonbondedForce":
                for parms_nonbonded, idx in zip(parms_nonb, self.atom_idxs):
                    force.setParticleParameters(idx, parms_nonbonded)
                try:  # use the fast way
                    lst = self.force_idxs[fgroup]["CustomNonbondedForceExclusions"]
                    for exc_idx, idx1, idx2 in lst:
                        excl_idx1, excl_idx2 = parms_exclusions.popleft()
                        force.setExclusionParticles(
                            exc_idx, excl_idx1, excl_idx2
                        )
                except KeyError:  # use the old slow way
                    for exc_idx in range(force.getNumExclusions()):
                        f = force.getExclusionParticles(exc_idx)
                        idx1 = f[0]
                        idx2 = f[1]
                        if idx1 in self.atom_idxs and idx2 in self.atom_idxs:
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

    def _set_DrudeForce_parameters(self, parms) -> None:  # noqa: N802
        parms_pol = deque(parms[0])
        parms_thole = deque(parms[1])
        particle_map = {}
        for force in self.system.getForces():
            if type(force).__name__ == "DrudeForce":
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
                try:
                    lst = self.force_idxs[fgroup]["DrudeForceThole"]
                    for thole_idx, idx1, idx2 in lst:
                        thole = parms_thole.popleft()
                        force.setScreenedPairParameters(thole_idx, idx1, idx2, thole)
                except KeyError:
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

    def _get_NonbondedForce_parameters_at_lambda(  # noqa: N802
        self, lamb: float
    ) -> list[list[int]]:
        # returns interpolated sorted nonbonded Forces.
        assert lamb >= 0 and lamb <= 1
        current_name = self.current_name
        new_name = self.alternativ_name

        nonbonded_parm_old = [
            parm for parm in self.parameters[current_name]["NonbondedForce"]
        ]
        nonbonded_parm_new = [
            parm for parm in self.parameters[new_name]["NonbondedForce"]
        ]
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

    def _get_CustomNonbondedForce_parameters_at_lambda(  # noqa: N802
        self, lamb: float
    ) -> list[list[int]]:
        # we cover the Customnonbonded force wihich depends on atom type, this is not interpolateable, hence it is just switched after lamb > 0.5
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
        else: #lamb >= 0.5
            #new
            new_name = self.alternativ_name
            cnb_parm = [
                parm for parm in self.parameters[new_name]["CustomNonbondedForce"]
            ]
            cnb_exclusions = [
                parm for parm in self.parameters[new_name]["CustomNonbondedForceExclusions"]
            ]

        return [cnb_parm, cnb_exclusions]

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
        new_name = self.alternativ_name
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
        new_name = self.alternativ_name
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
        new_name = self.alternativ_name
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
        new_name = self.alternativ_name
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
        new_name = self.alternativ_name
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
