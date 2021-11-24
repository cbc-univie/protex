from ..testsystems import (
    generate_im1h_oac_system_chelpg,
    OAC_HOAC_chelpg,
    IM1H_IM1_chelpg,
)
from ..system import IonicLiquidSystem, IonicLiquidTemplates
from ..update import NaiveMCUpdate, StateUpdate
from scipy.spatial import distance_matrix
import numpy as np
import logging


def test_distance_calculation():
    simulation = generate_im1h_oac_system_chelpg()
    # get ionic liquid templates
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "delta_e": 2.33}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "delta_e": -2.33}

    templates = IonicLiquidTemplates(
        [OAC_HOAC_chelpg, IM1H_IM1_chelpg], (allowed_updates)
    )
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
            charge_candidate_idx1 = residue1.current_charge
            charge_candidate_idx2 = residue2.current_charge

            print(
                f"{residue1.original_name}:{residue1.current_name}:{residue1.residue.id}:{charge_candidate_idx1}-{residue2.original_name}:{residue2.current_name}:{residue2.residue.id}:{charge_candidate_idx2} pair suggested ..."
            )
            print(f"Distance between pairs: {distance[candidate_idx1,candidate_idx2]}")
            distances.append(distance[candidate_idx1, candidate_idx2])

    assert np.min(distances) == distances[0]
    assert np.max(distances) == distances[-1]


def test_get_and_interpolate_forces():

    simulation = generate_im1h_oac_system_chelpg()
    # get ionic liquid templates
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "delta_e": 2.33}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "delta_e": -2.33}

    templates = IonicLiquidTemplates(
        [OAC_HOAC_chelpg, IM1H_IM1_chelpg], (allowed_updates)
    )
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)

    # test _get*_parameters
    int_force_0 = ionic_liquid.residues[0]._get_NonbondedForce_parameters_at_lambda(0.0)
    int_force_1 = ionic_liquid.residues[0]._get_NonbondedForce_parameters_at_lambda(1.0)
    int_force_01 = ionic_liquid.residues[0]._get_NonbondedForce_parameters_at_lambda(
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

    simulation = generate_im1h_oac_system_chelpg()
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
        [OAC_HOAC_chelpg, IM1H_IM1_chelpg],
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
    ionic_liquid.residues[0]._set_HarmonicBondForce_parameters(
        int_force_0a, ionic_liquid.simulation.context
    )
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
        int_force_0a, ionic_liquid.simulation.context
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
    ionic_liquid.residues[0]._set_HarmonicAngleForce_parameters(
        int_force_0a, ionic_liquid.simulation.context
    )
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
    ionic_liquid.residues[0]._set_HarmonicAngleForce_parameters(
        int_force_0a, ionic_liquid.simulation.context
    )
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
    ionic_liquid.residues[0]._set_PeriodicTorsionForce_parameters(
        int_force_0a, ionic_liquid.simulation.context
    )
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
    ionic_liquid.residues[0]._set_PeriodicTorsionForce_parameters(
        int_force_0a, ionic_liquid.simulation.context
    )
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
        int_force_0a, ionic_liquid.simulation.context
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
        int_force_0a, ionic_liquid.simulation.context
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
        int_force_0a, ionic_liquid.simulation.context
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
        int_force_0a, ionic_liquid.simulation.context
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


def test_single_update():

    simulation = generate_im1h_oac_system_chelpg()
    # get ionic liquid templates
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "delta_e": 2.33}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "delta_e": -2.33}

    templates = IonicLiquidTemplates(
        [OAC_HOAC_chelpg, IM1H_IM1_chelpg], (allowed_updates)
    )
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)

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
    assert state_update.ionic_liquid.residues[idx1].current_charge == 1
    assert state_update.ionic_liquid.residues[idx1].current_charge == 1

    assert state_update.ionic_liquid.residues[idx2].current_name == "OAC"
    assert state_update.ionic_liquid.residues[idx2].original_name == "OAC"
    assert state_update.ionic_liquid.residues[idx2].current_charge == -1
    assert state_update.ionic_liquid.residues[idx2].current_charge == -1

    candidate_pairs = [
        (
            state_update.ionic_liquid.residues[idx1],
            state_update.ionic_liquid.residues[idx2],
        )
    ]
    ###### update
    state_update.updateMethod._update(candidate_pairs, 11)

    assert state_update.ionic_liquid.residues[idx1].current_name == "IM1"
    assert state_update.ionic_liquid.residues[idx1].original_name == "IM1H"
    assert state_update.ionic_liquid.residues[idx1].current_charge == 0
    assert state_update.ionic_liquid.residues[idx1].current_charge == 0

    assert state_update.ionic_liquid.residues[idx2].current_name == "HOAC"
    assert state_update.ionic_liquid.residues[idx2].original_name == "OAC"
    assert state_update.ionic_liquid.residues[idx2].current_charge == 0
    assert state_update.ionic_liquid.residues[idx2].current_charge == 0

    ###### update
    state_update.updateMethod._update(candidate_pairs, 11)

    assert state_update.ionic_liquid.residues[idx1].current_name == "IM1H"
    assert state_update.ionic_liquid.residues[idx1].original_name == "IM1H"
    assert state_update.ionic_liquid.residues[idx1].current_charge == 1
    assert state_update.ionic_liquid.residues[idx1].current_charge == 1

    assert state_update.ionic_liquid.residues[idx2].current_name == "OAC"
    assert state_update.ionic_liquid.residues[idx2].original_name == "OAC"
    assert state_update.ionic_liquid.residues[idx2].current_charge == -1
    assert state_update.ionic_liquid.residues[idx2].current_charge == -1


def test_check_updated_charges(caplog):
    caplog.set_level(logging.DEBUG)

    simulation = generate_im1h_oac_system_chelpg()
    # get ionic liquid templates
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "delta_e": 2.33}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "delta_e": -2.33}

    templates = IonicLiquidTemplates(
        [OAC_HOAC_chelpg, IM1H_IM1_chelpg], (allowed_updates)
    )
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)

    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)
    # define mutation
    idx1, idx2 = 0, 200

    candidate_pairs = [
        (
            state_update.ionic_liquid.residues[idx1],
            state_update.ionic_liquid.residues[idx2],
        )
    ]

    state_update.write_charges("output_initial.txt")
    par_initial = state_update.get_charges()
    print(par_initial)
    state_update.updateMethod._update(candidate_pairs, 11)
    par_after_first_update = state_update.get_charges()
    state_update.updateMethod._update(candidate_pairs, 11)
    par_after_second_update = state_update.get_charges()

    assert par_initial == par_after_second_update
    assert par_initial != par_after_first_update

    print("####################################")
    print("Comparing intial charges with first update")
    print("####################################")
    for (idx1, atom1, charge1), (idx2, atom2, charge2) in zip(
        par_initial, par_after_first_update
    ):
        if charge1._value != charge2._value:
            print(
                f"{atom1.residue.name}:{atom1.residue.id}:{atom1.name}:{charge1._value}, {atom2.residue.name}:{atom2.residue.id}:{atom2.name}:{charge1._value}"
            )

    print("####################################")
    print("Comparing intial charges with second update")
    print("####################################")
    for (idx1, atom1, charge1), (idx2, atom2, charge2) in zip(
        par_initial, par_after_second_update
    ):

        if charge1._value != charge2._value:
            assert False  # should not happen!


def test_transfer_with_distance_matrix():

    import numpy as np

    simulation = generate_im1h_oac_system_chelpg()
    # get ionic liquid templates
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "delta_e": 2.33}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "delta_e": -2.33}

    templates = IonicLiquidTemplates(
        [OAC_HOAC_chelpg, IM1H_IM1_chelpg], (allowed_updates)
    )
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
        if not np.round(current_charge) == residue.current_charge:
            raise RuntimeError(
                f"{residue.residue.id=},{residue.current_name=},{residue.original_name=},{current_charge=},{residue.current_charge=}"
            )
            
    candidate_pairs1 = state_update.update(2)
    par_after_first_update = state_update.get_charges()
    res_dict = state_update.get_num_residues()
    print(res_dict)
    
    for residue in ionic_liquid.residues:
        current_charge = 0
        for idx in residue.atom_idxs:
            current_charge += par_after_first_update[idx][2]._value
        if not np.round(current_charge) == residue.current_charge:
            print(
                f"{residue.residue.id=},{residue.current_name=},{residue.original_name=},{current_charge=},{residue.current_charge=}"
            )  # -> why?
    candidate_pairs2 = state_update.update(2)
    par_after_second_update = state_update.get_charges()
    res_dict = state_update.get_num_residues()
    print(res_dict)
    for residue in ionic_liquid.residues:
        current_charge = 0
        for idx in residue.atom_idxs:
            current_charge += par_after_second_update[idx][2]._value
        if not np.round(current_charge) == residue.current_charge:
            print(
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

    for _ in range(100):
        state_update.update(11)


def test_updates(caplog):
    caplog.set_level(logging.DEBUG)

    simulation = generate_im1h_oac_system_chelpg()
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
    templates = IonicLiquidTemplates(
        [OAC_HOAC_chelpg, IM1H_IM1_chelpg], (allowed_updates)
    )
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)
    pars = []
    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)
    # ionic_liquid.simulation.minimizeEnergy(maxIterations=200)
    ionic_liquid.simulation.step(500)

    for _ in range(1):
        ionic_liquid.simulation.step(2000)
        pars.append(state_update.get_charges())
        candidate_pairs = state_update.update(1)


def test_dry_updates(caplog):
    caplog.set_level(logging.DEBUG)

    simulation = generate_im1h_oac_system_chelpg()
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
        [OAC_HOAC_chelpg, IM1H_IM1_chelpg],
        (allowed_updates),
    )
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)
    pars = []
    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)
    ionic_liquid.simulation.minimizeEnergy(maxIterations=200)
    ionic_liquid.simulation.step(500)

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
