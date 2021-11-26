import itertools
import numpy as np
import logging
from collections import ChainMap, defaultdict, deque
from simtk.openmm import openmm

logger = logging.getLogger(__name__)


class IonicLiquidTemplates:
    def __init__(self, states: list, allowed_updates: dict[frozenset]) -> None:

        self.pairs = [list(i.keys()) for i in states]
        self.states = dict(ChainMap(*states))
        self.names = list(itertools.chain(*self.pairs))
        self.allowed_updates = allowed_updates
        self.overall_max_distance = max(
            [value["r_max"] for value in self.allowed_updates.values()]
        )

    def get_canonical_name(self, name: str) -> str:
        assert name in self.names
        for state in self.states:
            if name in state:
                return self.states[name]["canonical_name"]

    def get_residue_name_for_coupled_state(self, name: str):
        """
        get_residue_name_of_paired_ion returns the paired residue name given a reisue name

        Parameters
        ----------
        name : str
            residue name

        Returns
        -------
        str

        Raises
        ------
        RuntimeError
            is raised if no paired residue name can be found
        """
        assert name in self.names

        for pair in self.pairs:
            if name in pair:
                state_1, state_2 = pair
                if state_1 == name:
                    return state_2
                else:
                    return state_1
        else:
            raise RuntimeError("something went wrong")

    def get_charge_template_for(self, name: str):
        """
        get_template_for returns the charge template for a residue

        Parameters
        ----------
        name : str
            Name of the residue

        Returns
        -------
        list
            charge state of residue

        Raises
        ------
        RuntimeError
            [description]
        """
        assert name in self.names

        return self.states[name]["charge"]


class Residue:
    def __init__(
        self,
        residue,
        alternativ_name,
        system,
        inital_parameters,
        alternativ_parameters,
        canonical_name: str,
        pair_12_13_exclusion_list,
    ) -> None:

        self.residue = residue
        self.original_name = residue.name
        self.current_name = self.original_name
        self.atom_idxs = [atom.index for atom in residue.atoms()]
        self.atom_names = [atom.name for atom in residue.atoms()]
        self.parameters = {
            self.original_name: inital_parameters,
            alternativ_name: alternativ_parameters,
        }
        self.record_charge_state = []
        self.canonical_name = canonical_name
        self.system = system
        self.record_charge_state.append(self.endstate_charge)
        self.pair_12_13_list = pair_12_13_exclusion_list

    @property
    def alternativ_name(self):
        for name in self.parameters.keys():
            if name != self.current_name:
                return name

    def update(
        self, force_name: str, lamb: float
    ):  # we don't need to call update in context since we are doing this in NaiveMCUpdate
        if force_name == "NonbondedForce":
            parms = self._get_NonbondedForce_parameters_at_lambda(lamb)
            self._set_NonbondedForce_parameters(parms)
        elif force_name == "HarmonicBondedForce":
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

    def _set_NonbondedForce_parameters(self, parms):
        for force in self.system.getForces():
            if type(force).__name__ == "NonbondedForce":
                for parms, idx in zip(parms, self.atom_idxs):
                    charge, sigma, epsiolon = parms
                    force.setParticleParameters(idx, charge, sigma, epsiolon)

    def _set_HarmonicBondForce_parameters(self, parms):

        parms = deque(parms)
        for force in self.system.getForces():
            if type(force).__name__ == "HarmonicBondForce":
                for bond_idx in range(force.getNumBonds()):
                    f = force.getBondParameters(bond_idx)
                    idx1 = f[0]
                    idx2 = f[1]
                    if idx1 in self.atom_idxs and idx2 in self.atom_idxs:
                        r, k = parms.popleft()
                        force.setBondParameters(bond_idx, idx1, idx2, r, k)

    def _set_HarmonicAngleForce_parameters(self, parms):
        parms = deque(parms)

        for force in self.system.getForces():
            if type(force).__name__ == "HarmonicAngleForce":
                for angle_idx in range(force.getNumAngles()):
                    f = force.getAngleParameters(angle_idx)
                    idx1 = f[0]
                    idx2 = f[1]
                    idx3 = f[2]
                    if (
                        idx1 in self.atom_idxs
                        and idx2 in self.atom_idxs
                        and idx3 in self.atom_idxs
                    ):
                        thetha, k = parms.popleft()
                        force.setAngleParameters(angle_idx, idx1, idx2, idx3, thetha, k)

    def _set_PeriodicTorsionForce_parameters(self, parms):
        parms = deque(parms)

        for force in self.system.getForces():
            if type(force).__name__ == "PeriodicTorsionForce":
                for torsion_idx in range(force.getNumTorsions()):
                    f = force.getTorsionParameters(torsion_idx)
                    idx1 = f[0]
                    idx2 = f[1]
                    idx3 = f[2]
                    idx4 = f[3]
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

    def _set_CustomTorsionForce_parameters(self, parms):
        parms = deque(parms)

        for force in self.system.getForces():
            if type(force).__name__ == "CustomTorsionForce":
                for torsion_idx in range(force.getNumTorsions()):
                    f = force.getTorsionParameters(torsion_idx)
                    idx1 = f[0]
                    idx2 = f[1]
                    idx3 = f[2]
                    idx4 = f[3]
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

    def _set_DrudeForce_parameters(self, parms):

        parms_pol = deque(parms[0])
        parms_thole = deque(parms[1])
        for force in self.system.getForces():
            if type(force).__name__ == "DrudeForce":
                for drude_idx in range(force.getNumParticles()):
                    f = force.getParticleParameters(drude_idx)
                    idx1 = f[0]
                    idx2 = f[1]
                    idx3 = f[2]
                    idx4 = f[3]
                    idx5 = f[4]
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
                for drude_idx in range(force.getNumScreenedPairs()):
                    f = force.getScreenedPairParameters(drude_idx)
                    idx1 = f[0]
                    idx2 = f[1]
                    parent1, parent2 = self.pair_12_13_list[drude_idx]
                    drude1, drude2 = parent1 + 1, parent2 + 1
                    if drude1 in self.atom_idxs and drude2 in self.atom_idxs:
                        thole = parms_thole.popleft()
                        force.setScreenedPairParameters(drude_idx, idx1, idx2, thole)

    def _get_NonbondedForce_parameters_at_lambda(self, lamb: float):
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
            print(f'{charge_old=}, {sigma_old=}, {epsilon_old=}')
            print(f'{charge_new=}, {sigma_new=}, {epsilon_new=}')
            charge_interpolated = (1 - lamb) * charge_old + lamb * charge_new
            sigma_interpolated = (1 - lamb) * sigma_old + lamb * sigma_new
            epsilon_interpolated = (1 - lamb) * epsilon_old + lamb * epsilon_new

            parm_interpolated.append(
                [charge_interpolated, sigma_interpolated, epsilon_interpolated]
            )

        return parm_interpolated

    def _get_offset(self, name):
        # get offset for atom idx
        force_name = "HarmonicBondForce"
        return min(
            itertools.chain(
                *[query_parm[0:2] for query_parm in self.parameters[name][force_name]]
            )
        )

    def _get_offset_thole(self, name):
        # get offset for atom idx for thole parameters
        force_name = "DrudeForceThole"
        return min(
            itertools.chain(
                *[query_parm[0:2] for query_parm in self.parameters[name][force_name]]
            )
        )

    def _get_HarmonicBondForce_parameters_at_lambda(self, lamb):
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
                if set(
                    [new_parm[0] - new_parms_offset, new_parm[1] - new_parms_offset]
                ) == set([idx1 - old_parms_offset, idx2 - old_parms_offset]):
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
            r_old, k_old = parm_old_i[-2:]
            r_new, k_new = parm_new_i[-2:]
            r_interpolated = (1 - lamb) * r_old + lamb * r_new
            k_interpolated = (1 - lamb) * k_old + lamb * k_new

            parm_interpolated.append([r_interpolated, k_interpolated])

        return parm_interpolated

    def _get_HarmonicAngleForce_parameters_at_lambda(self, lamb):
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
                if set(
                    [
                        new_parm[0] - new_parms_offset,
                        new_parm[1] - new_parms_offset,
                        new_parm[2] - new_parms_offset,
                    ]
                ) == set(
                    [
                        idx1 - old_parms_offset,
                        idx2 - old_parms_offset,
                        idx3 - old_parms_offset,
                    ]
                ):
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

    def _get_PeriodicTorsionForce_parameters_at_lambda(self, lamb):
        # returns PeriodicTorsionForce Forces ordered.
        assert lamb >= 0 and lamb <= 1
        # get the names of new and current state
        old_name = self.current_name
        new_name = self.alternativ_name
        parm_interpolated = []
        force_name = "PeriodicTorsionForce"
        new_parms_offset = self._get_offset(new_name)
        old_parms_offset = self._get_offset(old_name)

        # match parameters
        parms_old = []
        parms_new = []

        for old_idx, old_parm in enumerate(self.parameters[old_name][force_name]):
            idx1, idx2, idx3, idx4 = old_parm[0], old_parm[1], old_parm[2], old_parm[3]
            for new_idx, new_parm in enumerate(self.parameters[new_name][force_name]):
                if set(
                    [
                        new_parm[0] - new_parms_offset,
                        new_parm[1] - new_parms_offset,
                        new_parm[2] - new_parms_offset,
                        new_parm[3] - new_parms_offset,
                    ]
                ) == set(
                    [
                        idx1 - old_parms_offset,
                        idx2 - old_parms_offset,
                        idx3 - old_parms_offset,
                        idx4 - old_parms_offset,
                    ]
                ):
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
        # omm dihedral: [atom1, atom2, atom3, atom4, periodicity, Quantity(value=delta/phase, unit=radian), Quantity(value=Kchi, unit=kilojoule/mole)]
        for parm_old_i, parm_new_i in zip(parms_old, parms_new):
            # print(f"{parm_old_i=}")
            per_old, phase_old, k_old = parm_old_i[-3:]
            per_new, phase_new, k_new = parm_new_i[-3:]
            k_interpolated = (1 - lamb) * k_old + lamb * k_new

            if lamb <= 0.5:  # use per, phase from original residue
                parm_interpolated.append([per_old, phase_old, k_interpolated])

            if lamb > 0.5:  # use per, phase from final residue
                parm_interpolated.append([per_new, phase_new, k_interpolated])

        return parm_interpolated

    def _get_CustomTorsionForce_parameters_at_lambda(self, lamb):
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
                if set(
                    [
                        new_parm[0] - new_parms_offset,
                        new_parm[1] - new_parms_offset,
                        new_parm[2] - new_parms_offset,
                        new_parm[3] - new_parms_offset,
                    ]
                ) == set(
                    [
                        idx1 - old_parms_offset,
                        idx2 - old_parms_offset,
                        idx3 - old_parms_offset,
                        idx4 - old_parms_offset,
                    ]
                ):
                    if old_idx != new_idx:
                        raise RuntimeError(
                            "Odering of improper parameters is different between the two topologies."
                        )
                    # print("Here with index:")
                    # print(f"{old_idx=}, {new_idx=}")
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

    def _get_DrudeForce_parameters_at_lambda(self, lamb):
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

        # print(f"{old_name=}, {new_name=}")
        # print(f"{new_parms_offset=}, {old_parms_offset=}")

        # match parameters
        parms_old = []
        parms_new = []
        for old_idx, old_parm in enumerate(self.parameters[old_name][force_name]):
            idx1, idx2 = old_parm[0], old_parm[1]
            for new_idx, new_parm in enumerate(self.parameters[new_name][force_name]):
                if set(
                    [new_parm[0] - new_parms_offset, new_parm[1] - new_parms_offset]
                ) == set([idx1 - old_parms_offset, idx2 - old_parms_offset]):
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
                if set(
                    [new_parm[0] - new_parms_offset, new_parm[1] - new_parms_offset]
                ) == set([idx1 - old_parms_offset, idx2 - old_parms_offset]):
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
    def get_idx_for_atom_name(self, query_atom_name: str):
        for idx, atom_name in zip(self.atom_idxs, self.atom_names):
            if query_atom_name == atom_name:
                return idx
        else:
            raise RuntimeError()

    @property
    def endstate_charge(self) -> int:
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

        charge = 0
        for force in self.system.getForces():
            if type(force).__name__ == "NonbondedForce":
                for idx in self.atom_idxs:
                    charge_idx, _, _ = force.getParticleParameters(idx)
                    charge += charge_idx._value

        return np.round(charge, 3)


class IonicLiquidSystem:
    """
    This class defines the full system, performs the MD steps and offers an
    interface for protonation state updates.
    """

    def __init__(self, simulation, templates: IonicLiquidTemplates) -> None:
        self.system = simulation.system
        self.topology = simulation.topology
        self.simulation = simulation
        self.templates = templates
        self.residues = self._set_initial_states()

    def update_context(self, name: str):
        for force in self.system.getForces():
            print(type(force).__name__)
        print("#####")
        for force in self.system.getForces():
            if type(force).__name__ == name:
                force.updateParametersInContext(self.simulation.context)
                break

    def _build_exclusion_list(self):
        pair_12_set = set()
        pair_13_set = set()
        for bond in self.topology.bonds():
            a1, a2 = bond.atom1, bond.atom2
            if "H" not in a1.name and "H" not in a2.name:
                pair = (
                    min(a1.index, a2.index),
                    max(a1.index, a2.index),
                )
                pair_12_set.add(pair)
        for a in pair_12_set:
            for b in pair_12_set:
                shared = set(a).intersection(set(b))
                if len(shared) == 1:
                    pair = tuple(set(list(a) + list(b)) - shared)
                    pair_13_set.add(pair)

        self.pair_12_list = list(sorted(pair_12_set))
        self.pair_13_list = list(sorted(pair_13_set - pair_12_set))
        self.pair_12_13_list = self.pair_12_list + self.pair_13_list

    def _extract_templates(self, query_name: str) -> defaultdict:
        # returns the forces for the residue name
        forces_dict = defaultdict(list)

        for residue in self.topology.residues():
            if query_name == residue.name:
                atom_idxs = [atom.index for atom in residue.atoms()]
                atom_names = [atom.name for atom in residue.atoms()]
                logger.debug(atom_idxs)
                logger.debug(atom_names)

                for force in self.system.getForces():
                    # print(type(force).__name__)
                    if type(force).__name__ == "NonbondedForce":
                        forces_dict[type(force).__name__] = [
                            force.getParticleParameters(idx) for idx in atom_idxs
                        ]

                    if type(force).__name__ == "HarmonicBondForce":
                        for bond_id in range(force.getNumBonds()):
                            f = force.getBondParameters(bond_id)
                            idx1 = f[0]
                            idx2 = f[1]
                            if idx1 in atom_idxs and idx2 in atom_idxs:
                                forces_dict[type(force).__name__].append(f)

                    if type(force).__name__ == "HarmonicAngleForce":
                        for angle_id in range(force.getNumAngles()):
                            f = force.getAngleParameters(angle_id)
                            if (
                                f[0] in atom_idxs
                                and f[1] in atom_idxs
                                and f[2] in atom_idxs
                            ):
                                forces_dict[type(force).__name__].append(f)

                    if type(force).__name__ == "PeriodicTorsionForce":
                        for torsion_id in range(force.getNumTorsions()):
                            f = force.getTorsionParameters(torsion_id)
                            if (
                                f[0] in atom_idxs
                                and f[1] in atom_idxs
                                and f[2] in atom_idxs
                                and f[3] in atom_idxs
                            ):
                                forces_dict[type(force).__name__].append(f)

                    if type(force).__name__ == "CustomTorsionForce":
                        for torsion_id in range(force.getNumTorsions()):
                            f = force.getTorsionParameters(torsion_id)

                            if (
                                f[0] in atom_idxs
                                and f[1] in atom_idxs
                                and f[2] in atom_idxs
                                and f[3] in atom_idxs
                            ):
                                forces_dict[type(force).__name__].append(f)

                    if type(force).__name__ == "CMAPTorsionForce":
                        pass

                    # DrudeForce stores charge and polarizability in ParticleParameters and Thole values in ScreenedPairParameters
                    # Number of these two is not the same -> i did two loops, and called the thole parameters DrudeForceThole.
                    # Not ideal but i could not think of anything better, pay attention to the set and get methods for drudes.
                    if type(force).__name__ == "DrudeForce":
                        for drude_id in range(force.getNumParticles()):
                            f = force.getParticleParameters(drude_id)
                            idx1 = f[0]  # drude
                            idx2 = f[1]  # parentatom
                            if idx1 in atom_idxs and idx2 in atom_idxs:
                                forces_dict[type(force).__name__].append(f)
                        assert len(self.pair_12_13_list) == force.getNumScreenedPairs()
                        for drude_id in range(force.getNumScreenedPairs()):
                            f = force.getScreenedPairParameters(drude_id)
                            # idx1 = f[0]
                            # idx2 = f[1]
                            parent1, parent2 = self.pair_12_13_list[drude_id]
                            drude1, drude2 = parent1 + 1, parent2 + 1
                            # print(f"thole {idx1=}, {idx2=}")
                            # print(f"{drude_id=}, {f=}")
                            if drude1 in atom_idxs and drude2 in atom_idxs:
                                # print(f"Thole {query_name=}")
                                # print(f"{drude1=}, {drude2=}")
                                forces_dict[type(force).__name__ + "Thole"].append(f)
                break  # do this only for the relevant amino acid once
        return forces_dict

    @staticmethod
    def _check_nr_of_forces(forces_state1, forces_state2, name, name_of_paired_ion):
        # check if two forces lists have the same number of forces
        assert len(forces_state1) == len(forces_state2)  # check the number of forces
        for force_name in forces_state1:
            if len(forces_state1[force_name]) != len(
                forces_state2[force_name]  # check the number of entries in the forces
            ):
                logger.critical(force_name)
                logger.critical(name)
                logger.critical(len(forces_state1[force_name]))
                logger.critical(len(forces_state2[force_name]))

                for b1, b2, in zip(
                    forces_state1[force_name],
                    forces_state2[force_name],
                ):
                    logger.critical(f"{name}:{b1}")
                    logger.critical(f"{name_of_paired_ion}:{b2}")

                logger.critical(f"{name}:{forces_state1[force_name][-1]}")
                logger.critical(f"{name_of_paired_ion}:{forces_state2[force_name][-1]}")

                raise AssertionError("ohoh")

    def _set_initial_states(self) -> list:
        """
        set_initial_states For each ionic liquid residue in the system the protonation state
        is interfered from the provided openMM system object and the protonation site is defined.
        """

        self._build_exclusion_list()

        residues = []
        templates = dict()

        # for each residue type get forces
        for r in self.topology.residues():
            name = r.name
            name_of_paired_ion = self.templates.get_residue_name_for_coupled_state(name)

            if name in templates or name_of_paired_ion in templates:
                continue

            templates[name] = self._extract_templates(name)
            templates[name_of_paired_ion] = self._extract_templates(name_of_paired_ion)

        for r in self.topology.residues():
            name = r.name
            if name in self.templates.names:
                name_of_paired_ion = self.templates.get_residue_name_for_coupled_state(
                    name
                )

                parameters_state1 = templates[name]
                parameters_state2 = templates[name_of_paired_ion]
                # check that we have the same number of parameters
                self._check_nr_of_forces(
                    parameters_state1, parameters_state2, name, name_of_paired_ion
                )

                residues.append(
                    Residue(
                        r,
                        name_of_paired_ion,
                        self.system,
                        parameters_state1,
                        parameters_state2,
                        self.templates.get_canonical_name(name),
                        self.pair_12_13_list,
                    )
                )
                residues[-1].current_name = name

            else:
                raise RuntimeError("Found resiude not present in Templates: {r.name}")
        return residues

    def report_states(self) -> None:
        """
        report_states prints out a summary of the current protonation state of the ionic liquid
        """
        pass

    def report_charge_changes(self, filename: str, step=0, n_steps=None):
        """
        call for each round, before first update to have step 0 as initial step saved
        report_charge_changes reports the current charges after each update step in a dictionary format:
        {"step": [residue_charges]}
        additional header data is the dcd save frequency needed for later reconstruction of the charges at different steps
        """
        if step == 0:
            mode = "w"
        else:
            mode = "a"

        with open(filename, mode) as f:
            if mode == "w":
                f.write('{"dcd_save_freq": 100,\n')  # TODO: fetch automatically!
                f.write('"charges_at_step": {\n')
            if step != n_steps - 1:
                f.write(
                    f'"{step}": {[residue.endstate_charge for residue in self.residues]},\n'
                )
            elif step == n_steps - 1:
                f.write(
                    f'"{step}": {[residue.endstate_charge for residue in self.residues]}}}}}\n'
                )
