import logging
import os

from collections import defaultdict, deque
import numpy as np
import pytest
from scipy.spatial import distance_matrix

import protex
from ..system import IonicLiquidSystem, IonicLiquidTemplates
from ..testsystems import (
    IM1H_IM1,
    OAC_HOAC,
    generate_im1h_oac_system,
    generate_single_im1h_oac_system,
)
from ..update import NaiveMCUpdate, StateUpdate


def test_distance_calculation():
    simulation = generate_im1h_oac_system()
    # get ionic liquid templates
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "delta_e": 2.33}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "delta_e": -2.33}

    templates = IonicLiquidTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)

    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)
    distance_dict, res_dict = state_update._get_positions_for_mutation_sites()

    canonical_names = list(
        set([residue.canonical_name for residue in state_update.ionic_liquid.residues])
    )

    # calculate distance matrix between the two molecules
    distance = distance_matrix(
        distance_dict[canonical_names[0]], distance_dict[canonical_names[1]]
    )
    # get a list of indices for elements in the distance matrix sorted by increasing distance
    # NOTE: This always accepts a move!
    shape = distance.shape
    idx = np.dstack(np.unravel_index(np.argsort(distance.ravel()), shape))[0]
    distances = []
    for candidate_idx1, candidate_idx2 in idx:
        residue1 = res_dict[canonical_names[0]][candidate_idx1]
        residue2 = res_dict[canonical_names[1]][candidate_idx2]
        # is this combination allowed?
        if (
            frozenset([residue1.current_name, residue2.current_name])
            in state_update.ionic_liquid.templates.allowed_updates
        ):
            charge_candidate_idx1 = residue1.endstate_charge
            charge_candidate_idx2 = residue2.endstate_charge

            print(
                f"{residue1.original_name}:{residue1.current_name}:{residue1.residue.id}:{charge_candidate_idx1}-{residue2.original_name}:{residue2.current_name}:{residue2.residue.id}:{charge_candidate_idx2} pair suggested ..."
            )
            print(f"Distance between pairs: {distance[candidate_idx1,candidate_idx2]}")
            distances.append(distance[candidate_idx1, candidate_idx2])

    assert np.min(distances) == distances[0]
    assert np.max(distances) == distances[-1]


def test_get_and_interpolate_forces():

    simulation = generate_im1h_oac_system()
    # get ionic liquid templates
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "delta_e": 2.33}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "delta_e": -2.33}

    templates = IonicLiquidTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)

    # test _get*_parameters
    int_force_0 = ionic_liquid.residues[0]._get_NonbondedForce_parameters_at_lambda(0.0)
    int_force_1 = ionic_liquid.residues[0]._get_NonbondedForce_parameters_at_lambda(1.0)
    int_force_01 = ionic_liquid.residues[0]._get_NonbondedForce_parameters_at_lambda(
        0.1
    )
    assert (
        int_force_0[0][0][0]._value * 0.9 + int_force_1[0][0][0]._value * 0.1
        == int_force_01[0][0][0]._value
    )
    assert (
        int_force_0[0][0][1]._value * 0.9 + int_force_1[0][0][1]._value * 0.1
        == int_force_01[0][0][1]._value
    )
    # exceptions
    assert (
        int_force_0[1][0][0]._value * 0.9 + int_force_1[1][0][0]._value * 0.1
        == int_force_01[1][0][0]._value
    )
    assert (
        int_force_0[1][0][1]._value * 0.9 + int_force_1[1][0][1]._value * 0.1
        == int_force_01[1][0][1]._value
    )

    int_force_0 = ionic_liquid.residues[0]._get_HarmonicBondForce_parameters_at_lambda(
        0.0
    )
    int_force_1 = ionic_liquid.residues[0]._get_HarmonicBondForce_parameters_at_lambda(
        1.0
    )
    int_force_01 = ionic_liquid.residues[0]._get_HarmonicBondForce_parameters_at_lambda(
        0.1
    )
    assert (
        int_force_0[0][0]._value * 0.9 + int_force_1[0][0]._value * 0.1
        == int_force_01[0][0]._value
    )
    assert (
        int_force_0[0][1]._value * 0.9 + int_force_1[0][1]._value * 0.1
        == int_force_01[0][1]._value
    )

    int_force_0 = ionic_liquid.residues[0]._get_HarmonicAngleForce_parameters_at_lambda(
        0.0
    )
    int_force_1 = ionic_liquid.residues[0]._get_HarmonicAngleForce_parameters_at_lambda(
        1.0
    )
    int_force_01 = ionic_liquid.residues[
        0
    ]._get_HarmonicAngleForce_parameters_at_lambda(0.1)
    assert (
        int_force_0[0][0]._value * 0.9 + int_force_1[0][0]._value * 0.1
        == int_force_01[0][0]._value
    )
    assert (
        int_force_0[0][1]._value * 0.9 + int_force_1[0][1]._value * 0.1
        == int_force_01[0][1]._value
    )
    # Drude
    # charges, pol
    int_force_0 = ionic_liquid.residues[0]._get_DrudeForce_parameters_at_lambda(0.0)[0]
    int_force_1 = ionic_liquid.residues[0]._get_DrudeForce_parameters_at_lambda(1.0)[0]
    int_force_01 = ionic_liquid.residues[0]._get_DrudeForce_parameters_at_lambda(0.1)[0]
    assert (
        int_force_0[0][0]._value * 0.9 + int_force_1[0][0]._value * 0.1
        == int_force_01[0][0]._value
    )
    assert (
        int_force_0[0][1]._value * 0.9 + int_force_1[0][1]._value * 0.1
        == int_force_01[0][1]._value
    )
    # thole
    int_force_0 = ionic_liquid.residues[0]._get_DrudeForce_parameters_at_lambda(0.0)[1]
    int_force_1 = ionic_liquid.residues[0]._get_DrudeForce_parameters_at_lambda(1.0)[1]
    int_force_01 = ionic_liquid.residues[0]._get_DrudeForce_parameters_at_lambda(0.1)[1]
    assert int_force_0[0] * 0.9 + int_force_1[0] * 0.1 == int_force_01[0]


def test_setting_forces():

    simulation = generate_im1h_oac_system()
    allowed_updates = {}
    # allowed updates according to simple protonation scheme
    allowed_updates[frozenset(["IM1H", "OAC"])] = {
        "r_max": 0.16,
        "delta_e": 2.33,
    }  # r_max in nanometer, delta_e in kcal/mol
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "delta_e": -2.33}
    allowed_updates[frozenset(["IM1H", "IM1"])] = {"r_max": 0.16, "delta_e": 1.78}
    allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.16, "delta_e": 0.68}
    # get ionic liquid templates
    templates = IonicLiquidTemplates(
        [OAC_HOAC, IM1H_IM1],
        (allowed_updates),
    )
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)

    ##################################################
    ##################################################
    # test set*parameters HarmonicBondForce
    print("Testing HarmonicBondForce")
    print("Lambda: 0.0")
    parm_lambda_00 = []
    for force in ionic_liquid.system.getForces():
        if type(force).__name__ == "HarmonicBondForce":
            for bond_idx in range(force.getNumBonds()):
                f = force.getBondParameters(bond_idx)
                idx1 = f[0]
                idx2 = f[1]
                if (
                    idx1 in ionic_liquid.residues[0].atom_idxs
                    and idx2 in ionic_liquid.residues[0].atom_idxs
                ):
                    parm_lambda_00.append(f)

    # update HarmonicBondForce
    int_force_0a = ionic_liquid.residues[0]._get_HarmonicBondForce_parameters_at_lambda(
        0.5
    )
    ionic_liquid.residues[0]._set_HarmonicBondForce_parameters(int_force_0a)
    print("Lambda: 0.5")
    parm_lambda_05 = []
    for force in ionic_liquid.system.getForces():
        if type(force).__name__ == "HarmonicBondForce":
            for bond_idx in range(force.getNumBonds()):
                f = force.getBondParameters(bond_idx)
                idx1 = f[0]
                idx2 = f[1]
                if (
                    idx1 in ionic_liquid.residues[0].atom_idxs
                    and idx2 in ionic_liquid.residues[0].atom_idxs
                ):
                    parm_lambda_05.append(f)
    # update HarmonicBondForce
    int_force_0a = ionic_liquid.residues[0]._get_HarmonicBondForce_parameters_at_lambda(
        1.0
    )
    ionic_liquid.residues[0]._set_HarmonicBondForce_parameters(
        int_force_0a,
    )
    print("Lambda: 1.0")
    parm_lambda_10 = []
    for force in ionic_liquid.system.getForces():
        if type(force).__name__ == "HarmonicBondForce":
            for bond_idx in range(force.getNumBonds()):
                f = force.getBondParameters(bond_idx)
                idx1 = f[0]
                idx2 = f[1]
                if (
                    idx1 in ionic_liquid.residues[0].atom_idxs
                    and idx2 in ionic_liquid.residues[0].atom_idxs
                ):
                    parm_lambda_10.append(f)

    assert parm_lambda_00 != parm_lambda_05
    assert parm_lambda_00[5][3] != parm_lambda_05[5][3]
    for i, j, k in zip(parm_lambda_00, parm_lambda_05, parm_lambda_10):
        assert i[2]._value * 0.5 + k[2]._value * 0.5 == j[2]._value
        assert i[3]._value * 0.5 + k[3]._value * 0.5 == j[3]._value

    ##################################################
    ##################################################
    # test set*parameters HarmonicAngleForce
    print("Testing HarmonicAngleForce")
    print("Lambda: 0.0")
    parm_lambda_00 = []
    for force in ionic_liquid.system.getForces():
        if type(force).__name__ == "HarmonicAngleForce":
            for bond_idx in range(force.getNumAngles()):
                f = force.getAngleParameters(bond_idx)
                idx1 = f[0]
                idx2 = f[1]
                idx3 = f[2]
                if (
                    idx1 in ionic_liquid.residues[0].atom_idxs
                    and idx2 in ionic_liquid.residues[0].atom_idxs
                    and idx3 in ionic_liquid.residues[0].atom_idxs
                ):
                    parm_lambda_00.append(f)

    # update HarmonicAngleForce
    int_force_0a = ionic_liquid.residues[
        0
    ]._get_HarmonicAngleForce_parameters_at_lambda(0.5)
    ionic_liquid.residues[0]._set_HarmonicAngleForce_parameters(int_force_0a)
    print("Lambda: 0.5")
    parm_lambda_05 = []
    for force in ionic_liquid.system.getForces():
        if type(force).__name__ == "HarmonicAngleForce":
            for bond_idx in range(force.getNumAngles()):
                f = force.getAngleParameters(bond_idx)
                idx1 = f[0]
                idx2 = f[1]
                idx3 = f[2]
                if (
                    idx1 in ionic_liquid.residues[0].atom_idxs
                    and idx2 in ionic_liquid.residues[0].atom_idxs
                    and idx3 in ionic_liquid.residues[0].atom_idxs
                ):
                    parm_lambda_05.append(f)
    # update HarmonicAngleForce
    int_force_0a = ionic_liquid.residues[
        0
    ]._get_HarmonicAngleForce_parameters_at_lambda(1.0)
    ionic_liquid.residues[0]._set_HarmonicAngleForce_parameters(int_force_0a)
    print("Lambda: 1.0")
    parm_lambda_10 = []
    for force in ionic_liquid.system.getForces():
        if type(force).__name__ == "HarmonicAngleForce":
            for bond_idx in range(force.getNumAngles()):
                f = force.getAngleParameters(bond_idx)
                idx1 = f[0]
                idx2 = f[1]
                idx3 = f[2]
                if (
                    idx1 in ionic_liquid.residues[0].atom_idxs
                    and idx2 in ionic_liquid.residues[0].atom_idxs
                    and idx3 in ionic_liquid.residues[0].atom_idxs
                ):
                    parm_lambda_10.append(f)

    assert parm_lambda_00 != parm_lambda_05
    assert parm_lambda_00[5][3] != parm_lambda_05[5][3]
    for i, j, k in zip(parm_lambda_00, parm_lambda_05, parm_lambda_10):
        assert i[3]._value * 0.5 + k[3]._value * 0.5 == j[3]._value
        assert i[4]._value * 0.5 + k[4]._value * 0.5 == j[4]._value

    ##################################################
    ##################################################
    # test set*parameters PeriodicTorsionForce
    print("Testing PeriodicTorsionForce")
    print("Lambda: 0.0")
    parm_lambda_00 = []
    for force in ionic_liquid.system.getForces():
        if type(force).__name__ == "PeriodicTorsionForce":
            for bond_idx in range(force.getNumTorsions()):
                f = force.getTorsionParameters(bond_idx)
                idx1 = f[0]
                idx2 = f[1]
                idx3 = f[2]
                idx4 = f[3]
                if (
                    idx1 in ionic_liquid.residues[0].atom_idxs
                    and idx2 in ionic_liquid.residues[0].atom_idxs
                    and idx3 in ionic_liquid.residues[0].atom_idxs
                    and idx4 in ionic_liquid.residues[0].atom_idxs
                ):
                    parm_lambda_00.append(f)

    # update PeriodicTorsionForce
    int_force_0a = ionic_liquid.residues[
        0
    ]._get_PeriodicTorsionForce_parameters_at_lambda(0.5)
    ionic_liquid.residues[0]._set_PeriodicTorsionForce_parameters(int_force_0a)
    print("Lambda: 0.5")
    parm_lambda_05 = []
    for force in ionic_liquid.system.getForces():
        if type(force).__name__ == "PeriodicTorsionForce":
            for bond_idx in range(force.getNumTorsions()):
                f = force.getTorsionParameters(bond_idx)
                idx1 = f[0]
                idx2 = f[1]
                idx3 = f[2]
                idx4 = f[3]
                if (
                    idx1 in ionic_liquid.residues[0].atom_idxs
                    and idx2 in ionic_liquid.residues[0].atom_idxs
                    and idx3 in ionic_liquid.residues[0].atom_idxs
                    and idx4 in ionic_liquid.residues[0].atom_idxs
                ):
                    parm_lambda_05.append(f)

    # update PeriodicTorsionForce
    int_force_0a = ionic_liquid.residues[
        0
    ]._get_PeriodicTorsionForce_parameters_at_lambda(1.0)
    ionic_liquid.residues[0]._set_PeriodicTorsionForce_parameters(int_force_0a)
    print("Lambda: 1.0")
    parm_lambda_10 = []
    for force in ionic_liquid.system.getForces():
        if type(force).__name__ == "PeriodicTorsionForce":
            for bond_idx in range(force.getNumTorsions()):
                f = force.getTorsionParameters(bond_idx)
                idx1 = f[0]
                idx2 = f[1]
                idx3 = f[2]
                idx4 = f[3]
                if (
                    idx1 in ionic_liquid.residues[0].atom_idxs
                    and idx2 in ionic_liquid.residues[0].atom_idxs
                    and idx3 in ionic_liquid.residues[0].atom_idxs
                    and idx4 in ionic_liquid.residues[0].atom_idxs
                ):
                    parm_lambda_10.append(f)

    assert parm_lambda_00 != parm_lambda_05
    assert parm_lambda_00[5][6] != parm_lambda_05[5][6]
    for i, j, k in zip(parm_lambda_00, parm_lambda_05, parm_lambda_10):
        assert i[6]._value * 0.5 + k[6]._value * 0.5 == j[6]._value
        # assert i[4]._value * 0.5 + k[4]._value * 0.5 == j[4]._value

    ##################################################
    ##################################################
    # test set*parameters CustomTorsionForce
    print("Testing CustomTorsionForce")
    print("Lambda: 0.0")
    parm_lambda_00 = []
    for force in ionic_liquid.system.getForces():
        if type(force).__name__ == "CustomTorsionForce":
            for torsion_idx in range(force.getNumTorsions()):
                f = force.getTorsionParameters(torsion_idx)
                idx1 = f[0]
                idx2 = f[1]
                idx3 = f[2]
                idx4 = f[3]
                if (
                    idx1 in ionic_liquid.residues[200].atom_idxs
                    and idx2 in ionic_liquid.residues[200].atom_idxs
                    and idx3 in ionic_liquid.residues[200].atom_idxs
                    and idx4 in ionic_liquid.residues[200].atom_idxs
                ):
                    parm_lambda_00.append(f)

    # update CustomTorsionForce
    int_force_0a = ionic_liquid.residues[
        200
    ]._get_CustomTorsionForce_parameters_at_lambda(0.5)
    ionic_liquid.residues[200]._set_CustomTorsionForce_parameters(
        int_force_0a,
    )
    print("Lambda: 0.5")
    parm_lambda_05 = []
    for force in ionic_liquid.system.getForces():
        if type(force).__name__ == "CustomTorsionForce":
            for bond_idx in range(force.getNumTorsions()):
                f = force.getTorsionParameters(bond_idx)
                idx1 = f[0]
                idx2 = f[1]
                idx3 = f[2]
                idx4 = f[3]
                if (
                    idx1 in ionic_liquid.residues[200].atom_idxs
                    and idx2 in ionic_liquid.residues[200].atom_idxs
                    and idx3 in ionic_liquid.residues[200].atom_idxs
                    and idx4 in ionic_liquid.residues[200].atom_idxs
                ):
                    parm_lambda_05.append(f)

    # update CustomTorsionForce
    int_force_0a = ionic_liquid.residues[
        200
    ]._get_CustomTorsionForce_parameters_at_lambda(1.0)
    ionic_liquid.residues[200]._set_CustomTorsionForce_parameters(
        int_force_0a,
    )
    print("Lambda: 1.0")
    parm_lambda_10 = []
    for force in ionic_liquid.system.getForces():
        if type(force).__name__ == "CustomTorsionForce":
            for bond_idx in range(force.getNumTorsions()):
                f = force.getTorsionParameters(bond_idx)
                idx1 = f[0]
                idx2 = f[1]
                idx3 = f[2]
                idx4 = f[3]
                if (
                    idx1 in ionic_liquid.residues[200].atom_idxs
                    and idx2 in ionic_liquid.residues[200].atom_idxs
                    and idx3 in ionic_liquid.residues[200].atom_idxs
                    and idx4 in ionic_liquid.residues[200].atom_idxs
                ):
                    parm_lambda_10.append(f)
    assert parm_lambda_00 != parm_lambda_10
    # assert parm_lambda_00[0][6] != parm_lambda_05[5][6]
    for i, j, k in zip(parm_lambda_00, parm_lambda_05, parm_lambda_10):
        assert i[0] * 0.5 + k[0] * 0.5 == j[0]

    ###################################################################
    ###################################################################
    # test set*parameters DrudeForce
    print("Testing DrudeForce")
    print("Lambda: 0.0")
    parm_lambda_00_charges_pol = []
    parm_lambda_00_thole = []
    for force in ionic_liquid.system.getForces():
        if type(force).__name__ == "DrudeForce":
            for drude_idx in range(force.getNumParticles()):
                f = force.getParticleParameters(drude_idx)
                idx1 = f[0]
                idx2 = f[1]
                # idx3 = f[2]
                # idx4 = f[3]
                # idx5 = f[4]
                if (
                    idx1 in ionic_liquid.residues[0].atom_idxs
                    and idx2 in ionic_liquid.residues[0].atom_idxs
                ):
                    parm_lambda_00_charges_pol.append(f)
            for drude_idx in range(force.getNumScreenedPairs()):
                f = force.getScreenedPairParameters(drude_idx)
                idx1 = f[0]
                idx2 = f[1]
                if (
                    idx1 in ionic_liquid.residues[0].atom_idxs
                    and idx2 in ionic_liquid.residues[0].atom_idxs
                ):
                    parm_lambda_00_thole.append(f)
    parm_lambda_00 = [parm_lambda_00_charges_pol, parm_lambda_00_thole]

    # update DrudeForce
    int_force_0a = ionic_liquid.residues[0]._get_DrudeForce_parameters_at_lambda(0.5)
    ionic_liquid.residues[0]._set_DrudeForce_parameters(
        int_force_0a,
    )
    print("Lambda: 0.5")
    parm_lambda_05_charges_pol = []
    parm_lambda_05_thole = []
    for force in ionic_liquid.system.getForces():
        if type(force).__name__ == "DrudeForce":
            for drude_idx in range(force.getNumParticles()):
                f = force.getParticleParameters(drude_idx)
                idx1 = f[0]
                idx2 = f[1]
                # idx3 = f[2]
                # idx4 = f[3]
                # idx5 = f[4]
                if (
                    idx1 in ionic_liquid.residues[0].atom_idxs
                    and idx2 in ionic_liquid.residues[0].atom_idxs
                ):
                    parm_lambda_05_charges_pol.append(f)
            for drude_idx in range(force.getNumScreenedPairs()):
                f = force.getScreenedPairParameters(drude_idx)
                idx1 = f[0]
                idx2 = f[1]
                if (
                    idx1 in ionic_liquid.residues[0].atom_idxs
                    and idx2 in ionic_liquid.residues[0].atom_idxs
                ):
                    parm_lambda_05_thole.append(f)
    parm_lambda_05 = [parm_lambda_05_charges_pol, parm_lambda_05_thole]

    # update DrudeForce
    int_force_0a = ionic_liquid.residues[0]._get_DrudeForce_parameters_at_lambda(1.0)
    ionic_liquid.residues[0]._set_DrudeForce_parameters(
        int_force_0a,
    )
    print("Lambda: 1.0")
    parm_lambda_10_charges_pol = []
    parm_lambda_10_thole = []
    for force in ionic_liquid.system.getForces():
        if type(force).__name__ == "DrudeForce":
            for drude_idx in range(force.getNumParticles()):
                f = force.getParticleParameters(drude_idx)
                idx1 = f[0]
                idx2 = f[1]
                # idx3 = f[2]
                # idx4 = f[3]
                # idx5 = f[4]
                if (
                    idx1 in ionic_liquid.residues[0].atom_idxs
                    and idx2 in ionic_liquid.residues[0].atom_idxs
                ):
                    parm_lambda_10_charges_pol.append(f)
            for drude_idx in range(force.getNumScreenedPairs()):
                f = force.getScreenedPairParameters(drude_idx)
                idx1 = f[0]
                idx2 = f[1]
                if (
                    idx1 in ionic_liquid.residues[0].atom_idxs
                    and idx2 in ionic_liquid.residues[0].atom_idxs
                ):
                    parm_lambda_05_thole.append(f)
    parm_lambda_10 = [parm_lambda_10_charges_pol, parm_lambda_10_thole]

    assert parm_lambda_00 != parm_lambda_05
    # charges, pol
    for i, j, k in zip(parm_lambda_00[0], parm_lambda_05[0], parm_lambda_10[0]):
        assert i[5]._value * 0.5 + k[5]._value * 0.5 == j[5]._value
        assert i[6]._value * 0.5 + k[6]._value * 0.5 == j[6]._value
    # thole
    for i, j, k in zip(parm_lambda_00[1], parm_lambda_05[1], parm_lambda_10[1]):
        assert i[3] * 0.5 + k[3] * 0.5 == j[3]


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Will fail sporadicaly.",
)
def test_single_update():

    simulation = generate_im1h_oac_system()
    # get ionic liquid templates
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "delta_e": 2.33}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "delta_e": -2.33}

    templates = IonicLiquidTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)
    ionic_liquid.simulation.minimizeEnergy(maxIterations=500)
    ionic_liquid.simulation.step(500)

    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)
    pos = state_update.ionic_liquid.simulation.context.getState(
        getPositions=True
    ).getPositions(asNumpy=True)
    # check that we have all atoms
    assert len(pos) == 18000
    # check properties of residue that will be tested
    idx1 = 0
    idx2 = 200
    assert state_update.ionic_liquid.residues[idx1].current_name == "IM1H"
    assert state_update.ionic_liquid.residues[idx1].original_name == "IM1H"
    assert state_update.ionic_liquid.residues[idx1].current_charge == 1.00
    assert state_update.ionic_liquid.residues[idx1].endstate_charge == 1

    assert state_update.ionic_liquid.residues[idx2].current_name == "OAC"
    assert state_update.ionic_liquid.residues[idx2].original_name == "OAC"
    assert state_update.ionic_liquid.residues[idx2].current_charge == -1.00
    assert state_update.ionic_liquid.residues[idx2].endstate_charge == -1

    candidate_pairs = [
        (
            state_update.ionic_liquid.residues[idx1],
            state_update.ionic_liquid.residues[idx2],
        )
    ]
    ###### update
    state_update.updateMethod._update(candidate_pairs, 2)

    assert state_update.ionic_liquid.residues[idx1].current_name == "IM1"
    assert state_update.ionic_liquid.residues[idx1].original_name == "IM1H"
    assert state_update.ionic_liquid.residues[idx1].endstate_charge == 0
    assert state_update.ionic_liquid.residues[idx2].current_charge == 0.00

    assert state_update.ionic_liquid.residues[idx2].current_name == "HOAC"
    assert state_update.ionic_liquid.residues[idx2].original_name == "OAC"
    assert state_update.ionic_liquid.residues[idx2].endstate_charge == 0
    assert state_update.ionic_liquid.residues[idx2].current_charge == 0.00

    ###### update
    state_update.updateMethod._update(candidate_pairs, 2)

    assert state_update.ionic_liquid.residues[idx1].current_name == "IM1H"
    assert state_update.ionic_liquid.residues[idx1].original_name == "IM1H"
    assert state_update.ionic_liquid.residues[idx1].endstate_charge == 1
    assert state_update.ionic_liquid.residues[idx1].current_charge == 1

    assert state_update.ionic_liquid.residues[idx2].current_name == "OAC"
    assert state_update.ionic_liquid.residues[idx2].original_name == "OAC"
    assert state_update.ionic_liquid.residues[idx2].endstate_charge == -1
    assert state_update.ionic_liquid.residues[idx2].current_charge == -1


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Will fail sporadicaly.",
)
def test_check_updated_charges(caplog):
    caplog.set_level(logging.DEBUG)

    simulation = generate_im1h_oac_system()
    # get ionic liquid templates
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "delta_e": 2.33}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "delta_e": -2.33}

    templates = IonicLiquidTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)

    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)
    # define mutation
    idx1, idx2 = 0, 200

    candidate_pairs = [
        set(
            [
                state_update.ionic_liquid.residues[idx1],
                state_update.ionic_liquid.residues[idx2],
            ],
        )
    ]

    state_update.write_charges("output_initial.txt")
    par_initial = state_update.get_charges()
    state_update.updateMethod._update(candidate_pairs, 21)
    par_after_first_update = state_update.get_charges()
    state_update.updateMethod._update(candidate_pairs, 21)
    par_after_second_update = state_update.get_charges()

    print("####################################")
    print("Comparing intial charges with first update")
    print("####################################")
    for (idx1, atom1, charge1), (idx2, atom2, charge2) in zip(
        par_initial, par_after_first_update
    ):
        if charge1._value != charge2._value:
            print(
                f"{atom1.residue.name}:{atom1.residue.id}:{atom1.name}:{charge1._value}, {atom2.residue.name}:{atom2.residue.id}:{atom2.name}:{charge2._value}"
            )
    assert par_initial != par_after_first_update

    print("####################################")
    print("Comparing intial charges with second update")
    print("####################################")
    for (idx1, atom1, charge1), (idx2, atom2, charge2) in zip(
        par_initial, par_after_second_update
    ):

        if charge1._value != charge2._value:
            print(
                f"{atom1.residue.name}:{atom1.residue.id}:{atom1.name}:{charge1._value}, {atom2.residue.name}:{atom2.residue.id}:{atom2.name}:{charge2._value}"
            )
    assert par_after_first_update != par_after_second_update
    assert par_initial == par_after_second_update


def test_transfer_with_distance_matrix():

    import numpy as np

    simulation = generate_im1h_oac_system()
    # get ionic liquid templates
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "delta_e": 2.33}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "delta_e": -2.33}

    templates = IonicLiquidTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)

    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)
    state_update.write_charges("output_initial.txt")
    par_initial = state_update.get_charges()
    res_dict = state_update.get_num_residues()
    print(res_dict)
    for residue in ionic_liquid.residues:
        current_charge = 0
        for idx in residue.atom_idxs:
            current_charge += par_initial[idx][2]._value
        if not np.round(current_charge) == residue.endstate_charge:
            raise RuntimeError(
                f"{residue.residue.id=},{residue.current_name=},{residue.original_name=},{current_charge=},{residue.current_charge=}"
            )

    ##############################
    ###### FIRST UPDATE
    ##############################

    candidate_pairs1 = state_update.update()
    par_after_first_update = state_update.get_charges()
    res_dict = state_update.get_num_residues()

    for residue in ionic_liquid.residues:
        current_charge = 0
        atoms = []
        print(f"{residue.current_name=}")
        for idx in residue.atom_idxs:
            current_charge += par_after_first_update[idx][2]._value
            atoms.append(par_after_first_update[idx][1])
        if not np.round(current_charge) == residue.current_charge:
            print(atoms)
            raise RuntimeError(
                f"{residue.residue.id=},{residue.current_name=},{residue.original_name=},{current_charge=},{residue.current_charge=}"
            )

    ##############################
    ###### SECOND UPDATE
    #############################
    candidate_pairs2 = state_update.update()
    par_after_second_update = state_update.get_charges()
    res_dict = state_update.get_num_residues()

    for residue in ionic_liquid.residues:
        current_charge = 0
        for idx in residue.atom_idxs:
            current_charge += par_after_second_update[idx][2]._value
        if not np.round(current_charge) == residue.current_charge:
            raise RuntimeError(
                f"{residue.residue.id=},{residue.current_name=},{residue.original_name=},{current_charge=},{residue.current_charge=}"
            )

    # Number of atoms is constant
    assert (
        len(par_initial)
        == len(par_after_first_update)
        == len(par_after_second_update)
        == 18000
    )
    # check that total charge remains constant
    total_charge_init, total_charge_first, total_charge_second = 0.0, 0.0, 0.0
    print(candidate_pairs1)
    r1, r2 = candidate_pairs1[0]
    print(r1.current_name)
    print(r2.current_name)

    for (idx1, atom1, charge1), (idx2, atom2, charge2), (idx3, atom3, charge3) in zip(
        par_initial, par_after_first_update, par_after_second_update
    ):
        # if charge1._value != charge2._value or charge2._value != charge3._value:
        # print(f"{charge1._value=},{charge2._value=}, {charge3._value=}")
        total_charge_init += charge1._value
        total_charge_first += charge2._value
        total_charge_second += charge3._value
    print(f"{total_charge_init=}, {total_charge_first=}, {total_charge_second=}")

    # Total charge should be 0
    assert np.isclose(total_charge_init, 0.0)
    assert np.isclose(total_charge_first, 0.0)
    assert np.isclose(total_charge_second, 0.0)

    for _ in range(10):
        state_update.update(2)


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Will fail sporadicaly.",
)
def test_updates(caplog):
    caplog.set_level(logging.DEBUG)

    simulation = generate_im1h_oac_system()
    allowed_updates = {}
    # allowed updates according to simple protonation scheme
    allowed_updates[frozenset(["IM1H", "OAC"])] = {
        "r_max": 0.16,
        "delta_e": 2.33,
    }  # r_max in nanometer, delta_e in kcal/mol
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "delta_e": -2.33}
    allowed_updates[frozenset(["IM1H", "IM1"])] = {"r_max": 0.16, "delta_e": 1.78}
    allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.16, "delta_e": 0.68}
    print(allowed_updates.keys())
    # get ionic liquid templates
    templates = IonicLiquidTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)
    pars = []
    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)
    # ionic_liquid.simulation.minimizeEnergy(maxIterations=200)
    ionic_liquid.simulation.step(50)

    for _ in range(5):
        ionic_liquid.simulation.step(100)
        pars.append(state_update.get_charges())
        candidate_pairs = state_update.update(2)
        print(candidate_pairs)


def test_dry_updates(caplog):
    caplog.set_level(logging.DEBUG)

    simulation = generate_im1h_oac_system()
    # get ionic liquid templates
    allowed_updates = {}
    # allowed updates according to simple protonation scheme
    allowed_updates[frozenset(["IM1H", "OAC"])] = {
        "r_max": 0.16,
        "delta_e": 2.33,
    }  # r_max in nanometer, delta_e in kcal/mol
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "delta_e": -2.33}
    allowed_updates[frozenset(["IM1H", "IM1"])] = {"r_max": 0.16, "delta_e": 1.78}
    allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.16, "delta_e": 0.68}
    print(allowed_updates.keys())
    templates = IonicLiquidTemplates(
        # [OAC_HOAC_chelpg, IM1H_IM1_chelpg], (set(["IM1H", "OAC"]), set(["IM1", "HOAC"]))
        [OAC_HOAC, IM1H_IM1],
        (allowed_updates),
    )
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)
    pars = []
    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)
    ionic_liquid.simulation.minimizeEnergy(maxIterations=200)
    ionic_liquid.simulation.step(200)

    for _ in range(1):
        ionic_liquid.simulation.step(200)
        distance_dict, res_dict = state_update._get_positions_for_mutation_sites_new()
        # propose the update candidates based on distances
        state_update._print_start()
        candidate_pairs = state_update._propose_candidate_pair_new(
            distance_dict, res_dict
        )
        print(f"{candidate_pairs=}, {len(candidate_pairs)=}")
        state_update._print_stop()
        pars.append(state_update.get_charges())


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Will fail sporadicaly.",
)
def test_parameters_after_update():
    from simtk.openmm.app import StateDataReporter, DCDReporter

    simulation = generate_im1h_oac_system()
    # get ionic liquid templates
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "delta_e": 2.33}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "delta_e": -2.33}

    templates = IonicLiquidTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)
    ionic_liquid.simulation.reporters.append(DCDReporter(f"test_transfer.dcd", 1))
    ionic_liquid.simulation.reporters.append(
        StateDataReporter(
            f"test_transfer.out",
            1,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
            density=True,
        )
    )
    # ionic_liquid.simulation.minimizeEnergy(maxIterations=500)
    ionic_liquid.simulation.step(50)

    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)
    pos = state_update.ionic_liquid.simulation.context.getState(
        getPositions=True
    ).getPositions(asNumpy=True)
    # check that we have all atoms
    assert len(pos) == 18000
    # check properties of residue that will be tested
    idx1 = (
        150 + 150 + 89 - 1
    )  # soll IM1 mit residue.id=89 sein, da in residue list 0 based -> -1
    idx2 = 150 + 150 + 350 + 139 - 1  # HOAC 139
    assert state_update.ionic_liquid.residues[idx1].current_name == "IM1"
    assert state_update.ionic_liquid.residues[idx1].original_name == "IM1"
    assert state_update.ionic_liquid.residues[idx1].current_charge == 0.00
    assert state_update.ionic_liquid.residues[idx1].endstate_charge == 0

    assert state_update.ionic_liquid.residues[idx2].current_name == "HOAC"
    assert state_update.ionic_liquid.residues[idx2].original_name == "HOAC"
    assert state_update.ionic_liquid.residues[idx2].current_charge == 0.00
    assert state_update.ionic_liquid.residues[idx2].endstate_charge == 0

    # candidate_pairs = [
    #     (
    #         state_update.ionic_liquid.residues[idx1],
    #         state_update.ionic_liquid.residues[idx2],
    #     )
    # ]
    ###### update
    # state_update.updateMethod._update(candidate_pairs, 2)

    current_name = state_update.ionic_liquid.residues[idx1].current_name
    print("BEFORE START")
    print(f"{current_name=}")
    imp_before_list = []
    drude_before_list = []
    thole_before_list = []
    for force in ionic_liquid.system.getForces():
        if type(force).__name__ == "NonbondedForce":
            before_list = [
                force.getParticleParameters(idx)
                for idx in ionic_liquid.residues[idx1].atom_idxs
            ]

        #### Drude before ####
        if type(force).__name__ == "DrudeForce":
            for drude_id in range(force.getNumParticles()):
                f = force.getParticleParameters(drude_id)
                idx11 = f[0]  # drude
                idx22 = f[1]  # parentatom
                if (
                    idx11 in ionic_liquid.residues[idx1].atom_idxs
                    and idx22 in ionic_liquid.residues[idx1].atom_idxs
                ):
                    drude_before_list.append(f)
            assert (
                len(ionic_liquid.residues[idx1].pair_12_13_list)
                == force.getNumScreenedPairs()
            )
            for drude_id in range(force.getNumScreenedPairs()):
                f = force.getScreenedPairParameters(drude_id)
                # idx1 = f[0]
                # idx2 = f[1]
                parent1, parent2 = ionic_liquid.residues[idx1].pair_12_13_list[drude_id]
                drude1, drude2 = parent1 + 1, parent2 + 1
                if (
                    drude1 in ionic_liquid.residues[idx1].atom_idxs
                    and drude2 in ionic_liquid.residues[idx1].atom_idxs
                ):
                    thole_before_list.append(f)

        ## IMPROPERS Before##
        if type(force).__name__ == "CustomTorsionForce":
            for torsion_id in range(force.getNumTorsions()):
                f = force.getTorsionParameters(torsion_id)

                if (
                    f[0] in ionic_liquid.residues[idx1].atom_idxs
                    and f[1] in ionic_liquid.residues[idx1].atom_idxs
                    and f[2] in ionic_liquid.residues[idx1].atom_idxs
                    and f[3] in ionic_liquid.residues[idx1].atom_idxs
                ):
                    imp_before_list.append(f)
    print("BEFORE END")

    state_update.update(2)
    ionic_liquid.simulation.step(100)

    print("AFTER START")
    print(f"{current_name=}")
    current_name = state_update.ionic_liquid.residues[idx1].current_name

    imp_after_list = []
    drude_after_list = []
    thole_after_list = []
    for force in ionic_liquid.system.getForces():
        if type(force).__name__ == "NonbondedForce":
            after_list = [
                force.getParticleParameters(idx)
                for idx in ionic_liquid.residues[idx1].atom_idxs
            ]

        #### Drude after ####
        if type(force).__name__ == "DrudeForce":
            for drude_id in range(force.getNumParticles()):
                f = force.getParticleParameters(drude_id)
                idx11 = f[0]  # drude
                idx22 = f[1]  # parentatom
                if (
                    idx11 in ionic_liquid.residues[idx1].atom_idxs
                    and idx22 in ionic_liquid.residues[idx1].atom_idxs
                ):
                    drude_after_list.append(f)
            assert (
                len(ionic_liquid.residues[idx1].pair_12_13_list)
                == force.getNumScreenedPairs()
            )
            for drude_id in range(force.getNumScreenedPairs()):
                f = force.getScreenedPairParameters(drude_id)
                # idx1 = f[0]
                # idx2 = f[1]
                parent1, parent2 = ionic_liquid.residues[idx1].pair_12_13_list[drude_id]
                drude1, drude2 = parent1 + 1, parent2 + 1
                if (
                    drude1 in ionic_liquid.residues[idx1].atom_idxs
                    and drude2 in ionic_liquid.residues[idx1].atom_idxs
                ):
                    thole_after_list.append(f)

        ## IMPROPERS AFTER ##
        if type(force).__name__ == "CustomTorsionForce":
            for torsion_id in range(force.getNumTorsions()):
                f = force.getTorsionParameters(torsion_id)

                if (
                    f[0] in ionic_liquid.residues[idx1].atom_idxs
                    and f[1] in ionic_liquid.residues[idx1].atom_idxs
                    and f[2] in ionic_liquid.residues[idx1].atom_idxs
                    and f[3] in ionic_liquid.residues[idx1].atom_idxs
                ):
                    imp_after_list.append(f)
    print("AFTER END")
    # for residue in ionic_liquid.topology.residues():
    #     # if residue.id == 14:
    #     print(residue, residue.id)
    for (
        pos1,
        ((b_charge, b_epsilon, b_sigma), (a_charge, a_epsilon, a_sigma)),
    ) in enumerate(zip(before_list, after_list)):
        if b_charge != a_charge:
            print(f"Charge changed: {pos1=}, {b_charge=}, {a_charge=}")
        # if b_epsilon != a_epsilon:
        #     print(f"Epsilon changed: {pos1=}, {b_epsilon=}, {a_epsilon=}")
        # if b_sigma != a_sigma:
        #     print(f"Sigma changed: {pos1=}, {b_sigma=}, {a_sigma=}")

    print(f"{imp_before_list=}, {imp_after_list=}")

    for pos, (
        (b_drude, b_parent, _, _, _, b_charge, b_pol, _, _),
        (a_drude, a_parent, _, _, _, a_charge, a_pol, _, _),
    ) in enumerate(zip(drude_before_list, drude_after_list)):
        if b_charge != a_charge:
            print(f"Charge changed: {pos=}, {b_charge=}, {a_charge=}")
        else:
            print(f"Charge NOT changed: {pos=}, {b_charge=}, {a_charge=}")
        # if b_pol != a_pol:
        #     print(f"Polarizability changed: {pos=}, {b_pol=}, {a_pol=}")
    for pos, ((b_1, b_2, b_thole), (a_1, a_2, a_thole)) in enumerate(
        zip(thole_before_list, thole_after_list)
    ):
        if b_thole != a_thole:
            print(f"Thole changed: {pos=}, {b_thole=}, {a_thole=}")
    for pos, (
        (b1, b2, b3, b4, (b_k, b_psi0)),
        (a1, a2, a3, a4, (a_k, a_psi0)),
    ) in enumerate(zip(imp_before_list, imp_after_list)):
        if b_k != a_k:
            print(f"k changed: {pos=}, {b_k=}, {a_k=}")
        if b_psi0 != a_psi0:
            print(f"psi0 changed: {pos=}, {b_psi0=}, {a_psi0=}")

    assert state_update.ionic_liquid.residues[idx1].current_name == "IM1H"
    assert state_update.ionic_liquid.residues[idx1].original_name == "IM1"
    assert state_update.ionic_liquid.residues[idx1].endstate_charge == 1
    assert state_update.ionic_liquid.residues[idx1].current_charge == 1.00

    assert state_update.ionic_liquid.residues[idx2].current_name == "OAC"
    assert state_update.ionic_liquid.residues[idx2].original_name == "HOAC"
    assert state_update.ionic_liquid.residues[idx2].endstate_charge == -1
    assert state_update.ionic_liquid.residues[idx2].current_charge == -1.00

    ###### update
    # state_update.updateMethod._update(candidate_pairs, 2)
    # state_update.update(2)

    # assert state_update.ionic_liquid.residues[idx1].current_name == "IM1H"
    # assert state_update.ionic_liquid.residues[idx1].original_name == "IM1H"
    # assert state_update.ionic_liquid.residues[idx1].endstate_charge == 1
    # assert state_update.ionic_liquid.residues[idx1].current_charge == 1

    # assert state_update.ionic_liquid.residues[idx2].current_name == "OAC"
    # assert state_update.ionic_liquid.residues[idx2].original_name == "OAC"
    # assert state_update.ionic_liquid.residues[idx2].endstate_charge == -1
    # assert state_update.ionic_liquid.residues[idx2].current_charge == -1


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Will fail sporadicaly.",
)
def test_single_im1h_oac():
    from simtk.openmm.app import StateDataReporter, DCDReporter

    base = f"{protex.__path__[0]}/single_pairs"

    simulation = generate_single_im1h_oac_system()
    # get ionic liquid templates
    allowed_updates = {}
    # set distance criterion in order to have both possible transfers
    # -> then in the end it is checked that the new residue has the same exception parameters than the old one
    allowed_updates[frozenset(["IM1H", "OAC"])] = {
        "r_max": 1.3,
        "delta_e": 2.33,
    }  # distance is about 1.23
    allowed_updates[frozenset(["IM1", "HOAC"])] = {
        "r_max": 0.2,
        "delta_e": -2.33,
    }  # distance is about 0.18 -> verformt sich total

    templates = IonicLiquidTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)
    ionic_liquid.simulation.reporters.append(
        DCDReporter(f"{base}/test_single_im1h_oac.dcd", 10)
    )
    ionic_liquid.simulation.reporters.append(
        StateDataReporter(
            f"{base}/test_single_im1h_oac.out",
            10,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
            density=True,
        )
    )

    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)
    pos = state_update.ionic_liquid.simulation.context.getState(
        getPositions=True
    ).getPositions(asNumpy=True)
    # check that we have all atoms
    assert len(pos) == 72

    exceptions_before = {}
    for force in ionic_liquid.system.getForces():
        if type(force).__name__ == "NonbondedForce":
            print(f"{force.getNumParticles()=}")
            print(f"{force.getNumExceptions()=}")
            for exc_idx in range(force.getNumExceptions()):
                # print(force.getExceptionParameters(exc_idx))
                (
                    index1,
                    index2,
                    chargeprod,
                    sigma,
                    epsilon,
                ) = force.getExceptionParameters(exc_idx)
                for i in range(4):  # 0: IM1H, 1: OAC, 2: IM1, 3: HOAC
                    if (
                        index1 in ionic_liquid.residues[i].atom_idxs
                        and index2 in ionic_liquid.residues[i].atom_idxs
                    ):
                        if (
                            not ionic_liquid.residues[i].original_name
                            in exceptions_before.keys()
                        ):
                            exceptions_before[
                                ionic_liquid.residues[i].original_name
                            ] = [
                                index1,
                                index2,
                                chargeprod,
                                sigma,
                                epsilon,
                            ]
                if ["IM1H", "OAC", "IM1", "HOAC"] == list(exceptions_before.keys()):
                    break

    ionic_liquid.simulation.step(100)

    state_update.update(2)

    ionic_liquid.simulation.step(1000)

    exceptions_after = {}
    for force in ionic_liquid.system.getForces():
        if type(force).__name__ == "NonbondedForce":
            print(f"{force.getNumParticles()=}")
            print(f"{force.getNumExceptions()=}")
            for exc_idx in range(force.getNumExceptions()):
                (
                    index1,
                    index2,
                    chargeprod,
                    sigma,
                    epsilon,
                ) = force.getExceptionParameters(exc_idx)
                for i in range(4):  # 0: IM1H, 1: OAC, 2: IM1, 3: HOAC
                    if (
                        index1 in ionic_liquid.residues[i].atom_idxs
                        and index2 in ionic_liquid.residues[i].atom_idxs
                    ):
                        if (
                            not ionic_liquid.residues[i].current_name
                            in exceptions_after.keys()
                        ):
                            exceptions_after[ionic_liquid.residues[i].current_name] = [
                                index1,
                                index2,
                                chargeprod,
                                sigma,
                                epsilon,
                            ]
                if ["IM1H", "OAC", "IM1", "HOAC"] == list(exceptions_after.keys()):
                    break
    assert sorted(exceptions_before) == sorted(exceptions_after)
