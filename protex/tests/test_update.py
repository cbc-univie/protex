import logging
import os

import numpy as np
import pytest
from scipy.spatial import distance_matrix

try:
    from openmm import DrudeNoseHooverIntegrator, Platform, XmlSerializer
    from openmm.app import DCDReporter, StateDataReporter
    from openmm.unit import angstroms, kelvin, nanometers, picoseconds
except ImportError:
    from simtk.openmm import DrudeNoseHooverIntegrator, Platform, XmlSerializer
    from simtk.openmm.app import DCDReporter, StateDataReporter
    from simtk.unit import angstroms, kelvin, nanometers, picoseconds

import protex

from ..system import ProtexSystem, ProtexTemplates
from ..testsystems import (
    IM1H_IM1,
    OAC_HOAC,
    generate_im1h_oac_dummy_system,
    generate_im1h_oac_system,
    generate_single_im1h_oac_system,
    generate_small_box,
)
from ..update import KeepHUpdate, NaiveMCUpdate, StateUpdate, Update


#############
# small box
############
def test_create_update():
    # simulation = generate_im1h_oac_system()
    simulation = generate_small_box(use_plugin=False)
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 1}
    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    ionic_liquid = ProtexSystem(simulation, templates)
    try:
        update = Update(ionic_liquid)
    except TypeError:
        print("All fine, should raise TypeError, because abstract class")
        pass

    to_adapt = [("OAC", 15, frozenset(["IM1H", "OAC"]))]
    try:
        update = NaiveMCUpdate(
            ionic_liquid,
            all_forces=True,
            to_adapt=to_adapt,
            include_equivalent_atom=True,
            # reorient=True,
        )
    except NotImplementedError:
        pass

    update = NaiveMCUpdate(
        ionic_liquid,
        all_forces=True,
        to_adapt=to_adapt,
        include_equivalent_atom=True,
        # reorient=False,
    )

    state_update = StateUpdate(update)
    state_update.update(2)

    update = KeepHUpdate(
        ionic_liquid,
        all_forces=True,
        to_adapt=to_adapt,
        include_equivalent_atom=True,
        reorient=False,
    )

    state_update = StateUpdate(update)
    state_update.update(2)


def test_distance_calculation():
    # simulation = generate_im1h_oac_system()
    simulation = generate_small_box(use_plugin=False)
    # get ionic liquid templates
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 1}

    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    # wrap system in IonicLiquidSystem
    ionic_liquid = ProtexSystem(simulation, templates)

    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)
    distance_list, res_list = state_update._get_positions_for_mutation_sites()

    # calculate distance matrix between the two molecules
    distance = distance_matrix(distance_list, distance_list)
    # get a list of indices for elements in the distance matrix sorted by increasing distance
    # NOTE: This always accepts a move!
    shape = distance.shape
    idx = np.dstack(np.unravel_index(np.argsort(distance.ravel()), shape))[0]
    distances = []
    for candidate_idx1, candidate_idx2 in idx:
        residue1 = res_list[candidate_idx1]
        residue2 = res_list[candidate_idx2]
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
    # simulation = generate_im1h_oac_system()
    simulation = generate_small_box(use_plugin=False)
    # get ionic liquid templates
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 1}

    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    # wrap system in IonicLiquidSystem
    ionic_liquid = ProtexSystem(simulation, templates)

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
    #simulation = generate_im1h_oac_system()
    simulation = generate_small_box(use_plugin=False)
    allowed_updates = {}
    # allowed updates according to simple protonation scheme
    allowed_updates[frozenset(["IM1H", "OAC"])] = {
        "r_max": 0.16,
        "prob": 1,
    }  # r_max in nanometer, prob between 0 and 1
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["IM1H", "IM1"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.16, "prob": 1}
    # get ionic liquid templates
    templates = ProtexTemplates(
        [OAC_HOAC, IM1H_IM1],
        (allowed_updates),
    )
    # wrap system in IonicLiquidSystem
    ionic_liquid = ProtexSystem(simulation, templates)

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
                    idx1 in ionic_liquid.residues[20].atom_idxs
                    and idx2 in ionic_liquid.residues[20].atom_idxs
                    and idx3 in ionic_liquid.residues[20].atom_idxs
                    and idx4 in ionic_liquid.residues[20].atom_idxs
                ):
                    parm_lambda_00.append(f)

    # update CustomTorsionForce
    int_force_0a = ionic_liquid.residues[
        20
    ]._get_CustomTorsionForce_parameters_at_lambda(0.5)
    ionic_liquid.residues[20]._set_CustomTorsionForce_parameters(
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
                    idx1 in ionic_liquid.residues[20].atom_idxs
                    and idx2 in ionic_liquid.residues[20].atom_idxs
                    and idx3 in ionic_liquid.residues[20].atom_idxs
                    and idx4 in ionic_liquid.residues[20].atom_idxs
                ):
                    parm_lambda_05.append(f)

    # update CustomTorsionForce
    int_force_0a = ionic_liquid.residues[
        20
    ]._get_CustomTorsionForce_parameters_at_lambda(1.0)
    ionic_liquid.residues[20]._set_CustomTorsionForce_parameters(
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
                    idx1 in ionic_liquid.residues[20].atom_idxs
                    and idx2 in ionic_liquid.residues[20].atom_idxs
                    and idx3 in ionic_liquid.residues[20].atom_idxs
                    and idx4 in ionic_liquid.residues[20].atom_idxs
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
def test_single_update(caplog):
    # caplog.set_level(logging.DEBUG)

    #simulation = generate_im1h_oac_system()
    simulation = generate_small_box(use_plugin=False)
    # get ionic liquid templates
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 1}

    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    # wrap system in IonicLiquidSystem
    ionic_liquid = ProtexSystem(simulation, templates)
    ionic_liquid.simulation.minimizeEnergy(maxIterations=500)
    ionic_liquid.simulation.step(500)

    update = NaiveMCUpdate(ionic_liquid, all_forces=True)
    # initialize state update class
    state_update = StateUpdate(update)
    pos = state_update.ionic_liquid.simulation.context.getState(
        getPositions=True
    ).getPositions(asNumpy=True)
    # check that we have all atoms
    assert len(pos) == 1800
    # check properties of residue that will be tested
    idx1 = 0
    idx2 = 20
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
def test_check_updated_charges(caplog, tmp_path):
    caplog.set_level(logging.DEBUG)

    #simulation = generate_im1h_oac_system()
    simulation = generate_small_box(use_plugin=False)
    # get ionic liquid templates
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 1}

    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    # wrap system in IonicLiquidSystem
    ionic_liquid = ProtexSystem(simulation, templates)

    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)
    # define mutation
    idx1, idx2 = 0, 20

    candidate_pairs = [
        {
            state_update.ionic_liquid.residues[idx1],
            state_update.ionic_liquid.residues[idx2],
        }
    ]

    state_update.write_charges(f"{tmp_path}/output_initial1.txt")
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


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Takes too long for github actions",
)
def test_transfer_with_distance_matrix(tmp_path):
    simulation = generate_im1h_oac_system()
    # get ionic liquid templates
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 1}

    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    # wrap system in IonicLiquidSystem
    ionic_liquid = ProtexSystem(simulation, templates)

    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)
    state_update.write_charges(f"{tmp_path}/output_initial.txt")
    par_initial = state_update.get_charges()
    state_update.get_num_residues()
    # print(res_dict)
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
    state_update.get_num_residues()

    for residue in ionic_liquid.residues:
        current_charge = 0
        atoms = []
        # print(f"{residue.current_name=}")
        for idx in residue.atom_idxs:
            current_charge += par_after_first_update[idx][2]._value
            atoms.append(par_after_first_update[idx][1])
        if not np.round(current_charge) == residue.current_charge:
            # print(atoms)
            raise RuntimeError(
                f"{residue.residue.id=},{residue.current_name=},{residue.original_name=},{current_charge=},{residue.current_charge=}"
            )

    ##############################
    ###### SECOND UPDATE
    #############################
    state_update.update()
    par_after_second_update = state_update.get_charges()
    state_update.get_num_residues()

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
    # print(candidate_pairs1)
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

    for _ in range(5):
        state_update.update(2)


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Will fail sporadicaly.",
)
def test_updates(caplog):
    caplog.set_level(logging.DEBUG)

    #simulation = generate_im1h_oac_system()
    simulation = generate_small_box(use_plugin=True)
    allowed_updates = {}
    # allowed updates according to simple protonation scheme
    allowed_updates[frozenset(["IM1H", "OAC"])] = {
        "r_max": 0.16,
        "prob": 1,
    }  # r_max in nanometer, prob between 0 and 1
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["IM1H", "IM1"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.16, "prob": 1}
    print(allowed_updates.keys())
    # get ionic liquid templates
    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    # wrap system in IonicLiquidSystem
    ionic_liquid = ProtexSystem(simulation, templates)
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


def test_adapt_probabilities(caplog):
    #simulation = generate_im1h_oac_system()
    simulation = generate_small_box(use_plugin=False)
    allowed_updates = {}
    # allowed updates according to simple protonation scheme
    allowed_updates[frozenset(["IM1H", "OAC"])] = {
        "r_max": 0.16,
        "prob": 1,
    }  # r_max in nanometer, prob between 0 and 1
    # allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 1}
    # get ionic liquid templates
    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    # wrap system in IonicLiquidSystem
    ionic_liquid = ProtexSystem(simulation, templates)
    caplog.set_level(logging.DEBUG)
    # check that residue and frozeset match
    try:
        to_adapt = [("OAC", 15, frozenset(["IM1H", "HOAC"]))]
        update = NaiveMCUpdate(ionic_liquid, to_adapt)
        update._adapt_probabilities(to_adapt)
    except AssertionError as e:
        print("Check 1")
        print(e)
    # check that not duplicate sets
    try:
        to_adapt = [
            ("OAC", 15, frozenset(["IM1H", "OAC"])),
            ("OAC", 14, frozenset(["IM1H", "OAC"])),
        ]
        update = NaiveMCUpdate(ionic_liquid, to_adapt)
        update._adapt_probabilities(to_adapt)
    except AssertionError as e:
        print("Check 2")
        print(e)
    # check that set is an allowed update set
    try:
        to_adapt = [("HOAC", 35, frozenset(["IM1", "HOAC"]))]
        update = NaiveMCUpdate(ionic_liquid, to_adapt)
        update._adapt_probabilities(to_adapt)
    except RuntimeError as e:
        print("Check 3")
        print(e)


# @pytest.mark.skipif(
#     os.getenv("CI") == "true",
#     reason="Skipping tests that cannot pass in github actions",
# )
def test_dry_updates(caplog):
    caplog.set_level(logging.DEBUG)

    #simulation = generate_im1h_oac_system()
    simulation = generate_small_box(use_plugin=False)
    # get ionic liquid templates
    allowed_updates = {}
    # allowed updates according to simple protonation scheme
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["IM1H", "IM1"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.16, "prob": 1}
    print(allowed_updates.keys())
    templates = ProtexTemplates(
        # [OAC_HOAC_chelpg, IM1H_IM1_chelpg], (set(["IM1H", "OAC"]), set(["IM1", "HOAC"]))
        [OAC_HOAC, IM1H_IM1],
        (allowed_updates),
    )
    # wrap system in IonicLiquidSystem
    ionic_liquid = ProtexSystem(simulation, templates)
    pars = []
    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)
    ionic_liquid.simulation.minimizeEnergy(maxIterations=200)
    ionic_liquid.simulation.step(200)

    for _ in range(1):
        ionic_liquid.simulation.step(200)
        distance_dict, res_dict = state_update._get_positions_for_mutation_sites()
        # propose the update candidates based on distances
        state_update._print_start()
        candidate_pairs = state_update._propose_candidate_pair(distance_dict, res_dict)
        print(f"{candidate_pairs=}, {len(candidate_pairs)=}")
        state_update._print_stop()
        pars.append(state_update.get_charges())


def test_equivalence_pair1213_thole(tmp_path):
    def build_pair_12_13_list(topology):
        pair_12_set = set()
        pair_13_set = set()
        for bond in topology.bonds():
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
                    pair = tuple(sorted(set(list(a) + list(b)) - shared))
                    pair_13_set.add(pair)
                    # there were duplicates in pair_13_set, e.g. (1,3) and (3,1), needs to be sorted

        # self.pair_12_list = list(sorted(pair_12_set))
        # self.pair_13_list = list(sorted(pair_13_set - pair_12_set))
        # self.pair_12_13_list = self.pair_12_list + self.pair_13_list
        # change to return the list and set the parameters in the init method?
        pair_12_list = list(sorted(pair_12_set))
        pair_13_list = list(sorted(pair_13_set - pair_12_set))
        pair_12_13_list = pair_12_list + pair_13_list
        return pair_12_13_list
    simulation = generate_small_box(use_plugin=False)
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 1}
    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    ionic_liquid = ProtexSystem(simulation, templates)
    pair_12_13_list = build_pair_12_13_list(ionic_liquid.topology)

    idx1 = 1
    thole_before_list = []
    thole_before_list_pm = []
    for force in ionic_liquid.system.getForces():
        if type(force).__name__ == "DrudeForce":
            particle_map = {}
            for drude_id in range(force.getNumParticles()):
                f = force.getParticleParameters(drude_id)
                idx11 = f[0]  # drude
                #idx22 = f[1]  # parentatom
                particle_map[drude_id] = idx11
            assert (
                len(pair_12_13_list)
                == force.getNumScreenedPairs()
            )
            for drude_id in range(force.getNumScreenedPairs()):
                f = force.getScreenedPairParameters(drude_id)
                idx11 = f[0]
                idx22 = f[1]
                drude1pm = particle_map[idx11]
                drude2pm = particle_map[idx22]
                parent1, parent2 = pair_12_13_list[drude_id]
                drude1, drude2 = parent1 + 1, parent2 + 1
                #d = (drude1, drude2)
                #dpm = (drude1pm,drude2pm)
                #print(f"{d=}, {dpm=}")
                assert drude1 == drude1pm
                assert drude2 == drude2pm
                if (
                    drude1 in ionic_liquid.residues[idx1].atom_idxs
                    and drude2 in ionic_liquid.residues[idx1].atom_idxs
                ):
                    thole_before_list.append(f)
                if (
                    drude1pm in ionic_liquid.residues[idx1].atom_idxs
                    and drude2pm in ionic_liquid.residues[idx1].atom_idxs
                ):
                    thole_before_list_pm.append(f)
    for thole, tholepm in zip(thole_before_list, thole_before_list_pm):
        assert thole == tholepm
        #print(f"{thole=}, {tholepm=}")

@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Will fail sporadicaly.",
)
def test_parameters_after_update(tmp_path):
    simulation = generate_im1h_oac_system()
    #simulation = generate_small_box()
    # get ionic liquid templates
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 1}

    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    # wrap system in IonicLiquidSystem
    ionic_liquid = ProtexSystem(simulation, templates)
    ionic_liquid.simulation.reporters.append(
        DCDReporter(f"{tmp_path}/test_transfer.dcd", 1)
    )
    ionic_liquid.simulation.reporters.append(
        StateDataReporter(
            f"{tmp_path}/test_transfer.out",
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
            particle_map = {}
            for drude_id in range(force.getNumParticles()):
                f = force.getParticleParameters(drude_id)
                idx11 = f[0]  # drude
                idx22 = f[1]  # parentatom
                particle_map[drude_id] = idx11
                if (
                    idx11 in ionic_liquid.residues[idx1].atom_idxs
                    and idx22 in ionic_liquid.residues[idx1].atom_idxs
                ):
                    drude_before_list.append(f)
            for drude_id in range(force.getNumScreenedPairs()):
                f = force.getScreenedPairParameters(drude_id)
                idx11 = f[0]
                idx22 = f[1]
                drude1 = particle_map[idx11]
                drude2 = particle_map[idx22]
                #parent1, parent2 = ionic_liquid.residues[idx1].pair_12_13_list[drude_id]
                #drude1, drude2 = parent1 + 1, parent2 + 1
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
            particle_map = {}
            for drude_id in range(force.getNumParticles()):
                f = force.getParticleParameters(drude_id)
                idx11 = f[0]  # drude
                idx22 = f[1]  # parentatom
                particle_map[drude_id] = idx11
                if (
                    idx11 in ionic_liquid.residues[idx1].atom_idxs
                    and idx22 in ionic_liquid.residues[idx1].atom_idxs
                ):
                    drude_after_list.append(f)
            for drude_id in range(force.getNumScreenedPairs()):
                f = force.getScreenedPairParameters(drude_id)
                idx11 = f[0]
                idx22 = f[1]
                drude1 = particle_map[idx11]
                drude2 = particle_map[idx22]
                #parent1, parent2 = ionic_liquid.residues[idx1].pair_12_13_list[drude_id]
                #drude1, drude2 = parent1 + 1, parent2 + 1
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

    for (
        pos,
        (
            (b_drude, b_parent, _, _, _, b_charge, b_pol, _, _),
            (a_drude, a_parent, _, _, _, a_charge, a_pol, _, _),
        ),
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
    for (
        pos,
        (
            (b1, b2, b3, b4, (b_k, b_psi0)),
            (a1, a2, a3, a4, (a_k, a_psi0)),
        ),
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


# @pytest.mark.skipif(
#     os.getenv("CI") == "true",
#     reason="Will fail sporadicaly.",
# )
def test_pbc():
    #simulation = generate_im1h_oac_system()
    simulation = generate_small_box(use_plugin=False)
    # get ionic liquid templates
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 1}
    # allowed_updates[frozenset(["IM1H", "IM1"])] = {"r_max": 0.16, "prob": 1}
    # allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.16, "prob": 1}

    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    # wrap system in IonicLiquidSystem
    ionic_liquid = ProtexSystem(simulation, templates)

    boxl = ionic_liquid.boxlength.value_in_unit(nanometers)
    print(f"{boxl=}")

    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)

    pos_list, res_list = state_update._get_positions_for_mutation_sites()

    # calculate distance matrix between the two molecules
    distance_matrix(pos_list, pos_list)
    # print(f"{distance[0]=}")

    from scipy.spatial.distance import cdist

    def _rPBC(coor1, coor2, boxl=ionic_liquid.boxlength.value_in_unit(nanometers)):
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

    distance_pbc = cdist(pos_list, pos_list, _rPBC)
    # print(f"{distance_pbc[0]=}")
    # print(f"{distance_pbc[distance_pbc>boxl]=}")
    assert (
        len(distance_pbc[distance_pbc > boxl]) == 0
    ), "After correcting for PBC no distance should be larger than the boxlength"


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Will fail sporadicaly.",
)
def test_single_im1h_oac():
    base = f"{protex.__path__[0]}/forcefield/single_pairs"

    simulation = generate_single_im1h_oac_system()
    # get ionic liquid templates
    allowed_updates = {}
    # set distance criterion in order to have both possible transfers
    # -> then in the end it is checked that the new residue has the same exception parameters than the old one
    allowed_updates[frozenset(["IM1H", "OAC"])] = {
        "r_max": 1.3,
        "prob": 1,
    }  # distance is about 1.23
    allowed_updates[frozenset(["IM1", "HOAC"])] = {
        "r_max": 0.2,
        "prob": 1,
    }  # distance is about 0.18 -> verformt sich total

    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    # wrap system in IonicLiquidSystem
    ionic_liquid = ProtexSystem(simulation, templates)
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
                            ionic_liquid.residues[i].original_name
                            not in exceptions_before.keys()
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
                            ionic_liquid.residues[i].current_name
                            not in exceptions_after.keys()
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


def test_force_selection():
    simulation = generate_single_im1h_oac_system(use_plugin=False)
    allowed_updates = {}
    # allowed updates according to simple protonation scheme
    allowed_updates[frozenset(["IM1H", "OAC"])] = {
        "r_max": 0.16,
        "delta_e": 2.33,
    }  # r_max in nanometer, delta_e in kcal/mol
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "delta_e": -2.33}
    # allowed_updates[frozenset(["IM1H", "IM1"])] = {"r_max": 0.16, "delta_e": 1.78}
    # allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.16, "delta_e": 0.68}
    # get ionic liquid templates
    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    # wrap system in IonicLiquidSystem
    ionic_liquid = ProtexSystem(simulation, templates)
    update = NaiveMCUpdate(ionic_liquid)
    assert update.allowed_forces == ["NonbondedForce", "DrudeForce"]
    update = NaiveMCUpdate(ionic_liquid, all_forces=False)
    assert update.allowed_forces == ["NonbondedForce", "DrudeForce"]
    update = NaiveMCUpdate(ionic_liquid, all_forces=True)
    assert update.allowed_forces == [
        "NonbondedForce",
        "DrudeForce",
        "HarmonicBondForce",
        "HarmonicAngleForce",
        "PeriodicTorsionForce",
        "CustomTorsionForce",
    ]


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Will fail sporadicaly.",
)
def test_update_all_forces(caplog):
    caplog.set_level(logging.DEBUG)

    #simulation = generate_im1h_oac_system()
    simulation = generate_small_box(use_plugin=False)
    allowed_updates = {}
    # allowed updates according to simple protonation scheme
    allowed_updates[frozenset(["IM1H", "OAC"])] = {
        "r_max": 0.16,
        "prob": 1,
    }  # r_max in nanometer, prob between 0 and 1
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["IM1H", "IM1"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.16, "prob": 1}
    print(allowed_updates.keys())
    # get ionic liquid templates
    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    # wrap system in IonicLiquidSystem
    ionic_liquid = ProtexSystem(simulation, templates)
    # pars = []
    update = NaiveMCUpdate(ionic_liquid, all_forces=True)
    # initialize state update class
    state_update = StateUpdate(update)
    # ionic_liquid.simulation.minimizeEnergy(maxIterations=200)
    ionic_liquid.simulation.step(50)

    for _ in range(2):
        ionic_liquid.simulation.step(100)
        # pars.append(state_update.get_charges())
        candidate_pairs = state_update.update(2)
        print(candidate_pairs)


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Will fail sporadicaly.",
)
def test_energy_before_after():
    def get_time_energy(simulation, print=False):
        time = simulation.context.getState().getTime()
        e_pot = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        if print:
            print(f"time: {time}, e_pot: {e_pot}")
        return time, e_pot

    def save_il(ionic_liquid, number):
        ionic_liquid.write_psf(
            "protex/forcefield/dummy/im1h_oac_150_im1_hoac_350.psf",
            f"test_{number}.psf",
        )
        ionic_liquid.saveCheckpoint(f"test_{number}.rst")

    def load_sim(psf, rst):
        sim = generate_im1h_oac_system(psf_file=psf)
        sim.loadCheckpoint(rst)
        return sim

    def load_il(psf, rst, templates):
        sim = generate_im1h_oac_system(psf_file=psf)
        il = ProtexSystem(sim, templates)
        il.loadCheckpoint(rst)
        return il

    def print_force_contrib(simulation):
        for i, f in enumerate(simulation.system.getForces()):
            group = f.getForceGroup()
            state = simulation.context.getState(getEnergy=True, groups={group})
            print(f.getName(), state.getPotentialEnergy())

    sim0 = generate_im1h_oac_system()
    allowed_updates = {}
    # allowed updates according to simple protonation scheme
    allowed_updates[frozenset(["IM1H", "OAC"])] = {
        "r_max": 0.17,
        "prob": 1,
    }
    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    ionic_liquid = ProtexSystem(sim0, templates)
    update = NaiveMCUpdate(ionic_liquid, all_forces=True)
    state_update = StateUpdate(update)

    t_tmp, e_tmp = get_time_energy(ionic_liquid.simulation, print=False)
    t0, e0 = get_time_energy(sim0)
    assert t_tmp == t0
    assert e_tmp == e0
    save_il(ionic_liquid, 0)
    sim0_1 = load_sim("test_0.psf", "test_0.rst")
    t0_1, e0_1 = get_time_energy(sim0_1, print=False)
    assert t0 == t0_1
    print(e0, e0_1)
    np.testing.assert_almost_equal(e0._value, e0_1._value, 0)

    ionic_liquid.simulation.step(5)
    t1, e1 = get_time_energy(ionic_liquid.simulation)
    save_il(ionic_liquid, 1)
    sim1_1 = load_sim("test_1.psf", "test_1.rst")
    il1_1 = load_il("test_1.psf", "test_1.rst", templates)
    t1_1, e1_1 = get_time_energy(sim1_1)
    t1_2, e1_2 = get_time_energy(il1_1.simulation)
    print("Orig")
    print_force_contrib(ionic_liquid.simulation)
    print("Loaded")
    print_force_contrib(sim1_1)
    print("######")
    assert t1 == t1_1
    assert t1_1 == t1_2
    np.testing.assert_almost_equal(
        e1_2._value, e1_1._value, 0, err_msg=f"il {e1_2=} should be equal to {e1_1=}"
    )
    np.testing.assert_almost_equal(
        e1._value, e1_1._value, 0, err_msg=f"{e1=} should be equal to {e1_1=}"
    )

    state_update.update(2)
    t2, e2 = get_time_energy(ionic_liquid.simulation)
    save_il(ionic_liquid, 2)
    sim2_1 = load_sim("test_2.psf", "test_2.rst")
    t2_1, e2_1 = get_time_energy(sim2_1)
    print("Orig")
    print_force_contrib(ionic_liquid.simulation)
    print("Loaded")
    print_force_contrib(sim2_1)
    print("######")
    assert t2 == t2_1
    print(f"{e2=} should be equal to {e2_1=}")
    # np.testing.assert_almost_equal(e2._value, e2_1._value, 0)
    # assert e2 == e2_1, f"{e2=} should be equal to {e2_1=}"


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Will fail sporadicaly.",
)
def test_single_energy_before_after(caplog):
    caplog.set_level(logging.DEBUG)

    def get_time_energy(simulation, print=False):
        time = simulation.context.getState().getTime()
        e_pot = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        if print:
            print(f"time: {time}, e_pot: {e_pot}")
        return time, e_pot

    def save_il(ionic_liquid, number):
        ionic_liquid.write_psf(
            "protex/forcefield/single_pairs/im1h_oac_im1_hoac_1_secondtry.psf",
            f"test_{number}.psf",
        )
        ionic_liquid.saveCheckpoint(f"test_{number}.rst")

    def load_sim(psf, rst):
        sim = generate_single_im1h_oac_system(psf_file=psf)
        sim.loadCheckpoint(rst)
        return sim

    def load_il(psf, rst, templates):
        sim = generate_single_im1h_oac_system(psf_file=psf)
        il = ProtexSystem(sim, templates)
        il.loadCheckpoint(rst)
        return il

    def save_system(system, number):
        with open(f"system_{number}.xml", "w") as output:
            output.write(XmlSerializer.serialize(system))

    def load_system(number):
        with open(f"system_{number}.xml") as input:
            system = XmlSerializer.deserialize(input.read())
            return system

    def print_force_contrib(simulation):
        for i, f in enumerate(simulation.system.getForces()):
            group = f.getForceGroup()
            state = simulation.context.getState(getEnergy=True, groups={group})
            print(f.getName(), state.getPotentialEnergy())

    # integrator = VVIntegrator(
    integrator = DrudeNoseHooverIntegrator(
        300 * kelvin,
        10 / picoseconds,
        1 * kelvin,
        100 / picoseconds,
        0.0005 * picoseconds,
    )
    integrator.setMaxDrudeDistance(0.25 * angstroms)
    Platform.getPlatformByName("CUDA")
    dict(CudaPrecision="single")  # default is single

    sim0 = generate_single_im1h_oac_system()
    allowed_updates = {}
    # allowed updates according to simple protonation scheme
    allowed_updates[frozenset(["IM1H", "OAC"])] = {
        "r_max": 0.17,
        "prob": 1,
    }
    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    ionic_liquid = ProtexSystem(sim0, templates)
    update = NaiveMCUpdate(ionic_liquid, all_forces=True)
    state_update = StateUpdate(update)

    t_tmp, e_tmp = get_time_energy(ionic_liquid.simulation, print=False)
    t0, e0 = get_time_energy(sim0)
    assert t_tmp == t0
    assert e_tmp == e0
    save_il(ionic_liquid, 0)
    sim0_1 = load_sim("test_0.psf", "test_0.rst")
    t0_1, e0_1 = get_time_energy(sim0_1, print=False)
    assert t0 == t0_1
    assert e0 == e0_1

    ionic_liquid.simulation.step(5)
    t1, e1 = get_time_energy(ionic_liquid.simulation)
    save_il(ionic_liquid, 1)
    sim1_1 = load_sim("test_1.psf", "test_1.rst")
    il1_1 = load_il("test_1.psf", "test_1.rst", templates)
    t1_1, e1_1 = get_time_energy(sim1_1)
    t1_2, e1_2 = get_time_energy(il1_1.simulation)
    assert t1 == t1_1
    assert t1_1 == t1_2
    assert e1_2 == e1_1, f"il {e1_2=} should be equal to {e1_1=}"
    assert e1 == e1_1, f"{e1=} should be equal to {e1_1=}"
    logging.debug(t1)
    logging.debug(e1)
    logging.debug(t1_1)
    logging.debug(e1_1)
    print("### Orig Il ###")
    print_force_contrib(ionic_liquid.simulation)
    print("loaded il")
    print_force_contrib(sim1_1)
    print("####")

    state_update.update(2)
    t2, e2 = get_time_energy(ionic_liquid.simulation)
    save_il(ionic_liquid, 2)
    # save_system(ionic_liquid.system, 2)
    # positions = ionic_liquid.simulation.context.getState(
    #    getPositions=True
    # ).getPositions()
    # PDBFile.writeFile(ionic_liquid.topology, positions, file=open("test_2.pdb", "w"))
    sim2_1 = load_sim("protex/forcefield/single_pairs/im1_hoac_2.psf", "test_2.rst")
    sim2_2 = load_sim("test_2.psf", "test_2.rst")
    # sys2_2 = load_system(2)
    # pdb = PDBFile("test_2.pdb")
    # sim2_2 = Simulation(
    #    pdb.topology, sys2_2, integrator, platform=platform, platformProperties=prop
    # )
    # sim2_2.context.setPositions(pdb.positions)
    print("### Orig Il ###")
    print_force_contrib(ionic_liquid.simulation)
    print("loaded il")
    print_force_contrib(sim2_1)
    print("loaded il wrong psf")
    print_force_contrib(sim2_2)
    print("####")
    t2_1, e2_1 = get_time_energy(sim2_1)
    t2_2, e2_2 = get_time_energy(sim2_2)

    logging.debug(t2)
    logging.debug(e2)
    logging.debug(t2_1)
    logging.debug(e2_1)
    logging.debug(t2_2)
    logging.debug(e2_2)
    # ionic_liquid.simulation.saveState("state_2.xml")
    # print(ionic_liquid.simulation.context.getState(getParameters=True).getParameters())
    # with open("integrator.xml", "w") as f:
    #    f.write(
    #        XmlSerializer.serialize(
    #            ionic_liquid.simulation.context.getState(getParameters=True)
    #        )
    #    )
    # sim2_3 = Simulation.loadState("state_2.xml")
    # t2_3, e2_3 = get_time_energy(sim2_3)
    # print(t2_3, e2_3)

    # assert t2 == t2_1
    # assert e2 == e2_1, f"{e2=} should be equal to {e2_1=}"


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Will fail sporadicaly.",
)
def test_dummy_energy_before_after(caplog):
    caplog.set_level(logging.DEBUG)

    def get_time_energy(simulation, print=False):
        time = simulation.context.getState().getTime()
        e_pot = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        if print:
            print(f"time: {time}, e_pot: {e_pot}")
        return time, e_pot

    def save_il(ionic_liquid, number):
        ionic_liquid.write_psf(
            "protex/forcefield/dummy/im1h_oac_im1_hoac_1.psf",
            f"test_{number}.psf",
        )
        ionic_liquid.saveCheckpoint(f"test_{number}.rst")

    def load_sim(psf, rst):
        sim = generate_im1h_oac_dummy_system(psf_file=psf)
        sim.loadCheckpoint(rst)
        return sim

    def load_il(psf, rst, templates):
        sim = generate_im1h_oac_dummy_system(psf_file=psf)
        il = ProtexSystem(sim, templates)
        il.loadCheckpoint(rst)
        return il

    def save_system(system, number):
        with open(f"system_{number}.xml", "w") as output:
            output.write(XmlSerializer.serialize(system))

    def load_system(number):
        with open(f"system_{number}.xml") as input:
            system = XmlSerializer.deserialize(input.read())
            return system

    def print_force_contrib(simulation):
        for i, f in enumerate(simulation.system.getForces()):
            group = f.getForceGroup()
            state = simulation.context.getState(getEnergy=True, groups={group})
            print(f.getName(), state.getPotentialEnergy())

    # integrator = VVIntegrator(
    integrator = DrudeNoseHooverIntegrator(
        300 * kelvin,
        10 / picoseconds,
        1 * kelvin,
        100 / picoseconds,
        0.0005 * picoseconds,
    )
    integrator.setMaxDrudeDistance(0.25 * angstroms)
    Platform.getPlatformByName("CUDA")
    dict(CudaPrecision="single")  # default is single

    sim0 = generate_im1h_oac_dummy_system()
    allowed_updates = {}
    # allowed updates according to simple protonation scheme
    allowed_updates[frozenset(["IM1H", "OAC"])] = {
        "r_max": 0.17,
        "prob": 1,
    }
    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    ionic_liquid = ProtexSystem(sim0, templates)
    update = NaiveMCUpdate(ionic_liquid, all_forces=True)
    state_update = StateUpdate(update)

    t_tmp, e_tmp = get_time_energy(ionic_liquid.simulation, print=False)
    t0, e0 = get_time_energy(sim0)
    assert t_tmp == t0
    assert e_tmp == e0
    save_il(ionic_liquid, 0)
    sim0_1 = load_sim("test_0.psf", "test_0.rst")
    t0_1, e0_1 = get_time_energy(sim0_1, print=False)
    assert t0 == t0_1
    assert e0 == e0_1

    ionic_liquid.simulation.step(5)
    t1, e1 = get_time_energy(ionic_liquid.simulation)
    save_il(ionic_liquid, 1)
    sim1_1 = load_sim("test_1.psf", "test_1.rst")
    il1_1 = load_il("test_1.psf", "test_1.rst", templates)
    t1_1, e1_1 = get_time_energy(sim1_1)
    t1_2, e1_2 = get_time_energy(il1_1.simulation)
    assert t1 == t1_1
    assert t1_1 == t1_2
    assert e1_2 == e1_1, f"il {e1_2=} should be equal to {e1_1=}"
    assert e1 == e1_1, f"{e1=} should be equal to {e1_1=}"
    logging.debug(t1)
    logging.debug(e1)
    logging.debug(t1_1)
    logging.debug(e1_1)
    print("### Orig Il ###")
    print_force_contrib(ionic_liquid.simulation)
    print("loaded il")
    print_force_contrib(sim1_1)
    print("####")

    state_update.update(2)
    t2, e2 = get_time_energy(ionic_liquid.simulation)
    save_il(ionic_liquid, 2)
    # save_system(ionic_liquid.system, 2)
    # positions = ionic_liquid.simulation.context.getState(
    #    getPositions=True
    # ).getPositions()
    # PDBFile.writeFile(ionic_liquid.topology, positions, file=open("test_2.pdb", "w"))
    # sim2_1 = load_sim("protex/forcefield/single_pairs/im1_hoac_2.psf", "test_2.rst")
    sim2_2 = load_sim("test_2.psf", "test_2.rst")
    # sys2_2 = load_system(2)
    # pdb = PDBFile("test_2.pdb")
    # sim2_2 = Simulation(
    #    pdb.topology, sys2_2, integrator, platform=platform, platformProperties=prop
    # )
    # sim2_2.context.setPositions(pdb.positions)
    print("### Orig Il ###")
    print_force_contrib(ionic_liquid.simulation)
    print("loaded il")
    # print_force_contrib(sim2_1)
    # print("loaded il wrong psf")
    print_force_contrib(sim2_2)
    print("####")
    # t2_1, e2_1 = get_time_energy(sim2_1)
    t2_2, e2_2 = get_time_energy(sim2_2)

    logging.debug(t2)
    logging.debug(e2)
    # logging.debug(t2_1)
    # logging.debug(e2_1)
    logging.debug(t2_2)
    logging.debug(e2_2)
    # ionic_liquid.simulation.saveState("state_2.xml")
    # print(ionic_liquid.simulation.context.getState(getParameters=True).getParameters())
    # with open("integrator.xml", "w") as f:
    #    f.write(
    #        XmlSerializer.serialize(
    #            ionic_liquid.simulation.context.getState(getParameters=True)
    #        )
    #    )
    # sim2_3 = Simulation.loadState("state_2.xml")
    # t2_3, e2_3 = get_time_energy(sim2_3)
    # print(t2_3, e2_3)

    assert t2 == t2_2
    assert e2 == e2_2, f"{e2=} should be equal to {e2_2=}"


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Will fail sporadicaly.",
)
def test_periodictorsionforce_energy(caplog, tmp_path):
    # caplog.set_level(logging.DEBUG)

    def get_time_energy(simulation, print=False):
        time = simulation.context.getState().getTime()
        e_pot = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        if print:
            print(f"time: {time}, e_pot: {e_pot}")
        return time, e_pot

    def save_il(ionic_liquid, number):
        ionic_liquid.write_psf(
            "protex/forcefield/single_pairs/im1h_oac_im1_hoac_1_secondtry.psf",
            f"{tmp_path}/test_{number}.psf",
        )
        ionic_liquid.saveCheckpoint(f"{tmp_path}/test_{number}.rst")

    def load_sim(psf, rst):
        sim = generate_single_im1h_oac_system(psf_file=psf)
        sim.loadCheckpoint(rst)
        return sim

    def load_il(psf, rst, templates):
        sim = generate_single_im1h_oac_system(psf_file=psf)
        il = ProtexSystem(sim, templates)
        il.loadCheckpoint(rst)
        return il

    def print_force_contrib(simulation):
        for i, f in enumerate(simulation.system.getForces()):
            group = f.getForceGroup()
            state = simulation.context.getState(getEnergy=True, groups={group})
            print(f.getName(), state.getPotentialEnergy())

    def print_periodictorsionforce(simulation):
        f = simulation.system.getForce(3)
        state = simulation.context.getState(getEnergy=True, groups={3})
        print(f.getName(), state.getPotentialEnergy())

    # integrator = VVIntegrator(
    integrator = DrudeNoseHooverIntegrator(
        300 * kelvin,
        10 / picoseconds,
        1 * kelvin,
        100 / picoseconds,
        0.0005 * picoseconds,
    )
    integrator.setMaxDrudeDistance(0.25 * angstroms)
    Platform.getPlatformByName("CUDA")
    dict(CudaPrecision="single")  # default is single

    sim0 = generate_single_im1h_oac_system()
    allowed_updates = {}
    # allowed updates according to simple protonation scheme
    allowed_updates[frozenset(["IM1H", "OAC"])] = {
        "r_max": 0.17,
        "prob": 1,
    }
    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    ionic_liquid = ProtexSystem(sim0, templates)
    update = NaiveMCUpdate(ionic_liquid, all_forces=True)
    state_update = StateUpdate(update)

    t_tmp, e_tmp = get_time_energy(ionic_liquid.simulation, print=False)
    t0, e0 = get_time_energy(sim0)
    assert t_tmp == t0
    assert e_tmp == e0
    save_il(ionic_liquid, 0)
    sim0_1 = load_sim(f"{tmp_path}/test_0.psf", f"{tmp_path}/test_0.rst")
    t0_1, e0_1 = get_time_energy(sim0_1, print=False)
    assert t0 == t0_1
    assert e0 == e0_1

    ionic_liquid.simulation.step(5)
    t1, e1 = get_time_energy(ionic_liquid.simulation)
    save_il(ionic_liquid, 1)
    sim1_1 = load_sim(f"{tmp_path}/test_1.psf", f"{tmp_path}/test_1.rst")
    il1_1 = load_il(f"{tmp_path}/test_1.psf", f"{tmp_path}/test_1.rst", templates)
    t1_1, e1_1 = get_time_energy(sim1_1)
    t1_2, e1_2 = get_time_energy(il1_1.simulation)
    assert t1 == t1_1
    assert t1_1 == t1_2
    assert e1_2 == e1_1, f"il {e1_2=} should be equal to {e1_1=}"
    assert e1 == e1_1, f"{e1=} should be equal to {e1_1=}"
    # print("### Orig Il ###")
    # print_periodictorsionforce(ionic_liquid.simulation)
    # print("loaded il")
    # print_periodictorsionforce(sim1_1)
    # print("####")
    # print("IL before update")
    # print_force_contrib(ionic_liquid.simulation)

    state_update.update(2)
    # t2, e2 = get_time_energy(ionic_liquid.simulation)
    save_il(ionic_liquid, 2)

    load_sim("protex/forcefield/single_pairs/im1_hoac_2.psf", f"{tmp_path}/test_2.rst")
    load_sim("protex/forcefield/single_pairs/im1_hoac_2.psf", f"{tmp_path}/test_1.rst")
    # sim2_2 = load_sim("test_2.psf", "test_2.rst")

    # print("### Orig Il ###")
    # print_periodictorsionforce(ionic_liquid.simulation)
    # print("loaded il")
    # print_periodictorsionforce(sim2_1)
    ## print("loaded il wrong psf")
    ## print_force_contrib(sim2_2)
    # print("####")
    # print("### Orig Il ###")
    # print_force_contrib(ionic_liquid.simulation)
    # print("loaded il")
    # print_force_contrib(sim2_1)
    # print("loaded il old coord")
    # print_force_contrib(sim_2_oldcoord)
    # t2_1, e2_1 = get_time_energy(sim2_1)
    # t2_2, e2_2 = get_time_energy(sim2_2)
    # logging.debug(t2)
    # logging.debug(e2)
    # logging.debug(t2_1)
    # logging.debug(e2_1)
    # logging.debug(t2_2)
    # logging.debug(e2_2)


def test_ubforce_update(caplog):
    # caplog.set_level(logging.DEBUG)

    def print_force_contrib(simulation):
        for i, f in enumerate(simulation.system.getForces()):
            group = f.getForceGroup()
            state = simulation.context.getState(getEnergy=True, groups={group})
            print(f.getName(), state.getPotentialEnergy())

    sim0 = generate_single_im1h_oac_system(use_plugin=False)
    allowed_updates = {}
    # allowed updates according to simple protonation scheme
    allowed_updates[frozenset(["IM1H", "OAC"])] = {
        "r_max": 0.17,
        "prob": 1,
    }
    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    ionic_liquid = ProtexSystem(sim0, templates)

    print_force_contrib(ionic_liquid.simulation)
    for force in ionic_liquid.system.getForces():
        print(force, force.getForceGroup())
        if (
            type(force).__name__ == "HarmonicBondForce"
        ):  # and force.getForceGroup() == 3:
            f0 = force.getBondParameters(0)
            print(f"Initial params, {f0=}")
            force.setBondParameters(0, *f0[0:2], 0.15, 500000)
            f0_new = force.getBondParameters(0)
            print(f"New params, {f0_new=}")
        if type(force).__name__ == "HarmonicAngleForce":
            f0 = force.getAngleParameters(0)
            # print(f"Initial params, {f0=}")
            # force.setAngleParameters(0, *f0[0:3], 20, 100)
            # f0_new = force.getAngleParameters(0)
            # print(f"New params, {f0_new=}")
        if type(force).__name__ == "PeriodicTorsionForce":
            f0 = force.getTorsionParameters(0)
            # print(f"Initial params, {f0=}")
            # force.setTorsionParameters(0, *f0[0:4], 2, np.pi, 30)
            # f0_new = force.getTorsionParameters(0)
            # print(f"New params, {f0_new=}")
    print_force_contrib(ionic_liquid.simulation)
    for force in ionic_liquid.system.getForces():
        # print(force)
        if (
            type(force).__name__ == "HarmonicBondForce"
        ):  # and force.getForceGroup() == 3:
            print(f"hier: {force}")
            force.updateParametersInContext(ionic_liquid.simulation.context)
        # if type(force).__name__ == "HarmonicAngleForce":
        #    force.updateParametersInContext(ionic_liquid.simulation.context)
        # if type(force).__name__ == "PeriodicTorsionForce":
        #    force.updateParametersInContext(ionic_liquid.simulation.context)

    print_force_contrib(ionic_liquid.simulation)


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Will fail sporadicaly.",
)
def test_single_energy_molecule(caplog):
    caplog.set_level(logging.DEBUG)

    def get_time_energy(simulation, print=False):
        time = simulation.context.getState().getTime()
        e_pot = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        if print:
            print(f"time: {time}, e_pot: {e_pot}")
        return time, e_pot

    def print_force_contrib(simulation):
        for i, f in enumerate(simulation.system.getForces()):
            group = f.getForceGroup()
            state = simulation.context.getState(getEnergy=True, groups={group})
            print(f.getName(), state.getPotentialEnergy())

    def get_ub_contrib_from_(ionic_liquid, name=None):
        force_index = 2  # assuming that urey bradly term is the second force...
        ub_force = ionic_liquid.system.getForce(force_index)
        force_group = ub_force.getForceGroup()
        # print(force_group)
        # print(ub_force)

        if name is None:
            state = ionic_liquid.simulation.context.getState(
                getEnergy=True, groups={force_group}
            )
            print("ALL", ub_force.getName(), state.getPotentialEnergy())
            return state.getPotentialEnergy()

        orig_values = []
        for bondid in range(ub_force.getNumBonds()):
            f = ub_force.getBondParameters(bondid)
            idx1, idx2 = f[0:2]
            for residue in ionic_liquid.residues:
                if residue.current_name != name:
                    if idx1 in residue.atom_idxs and idx2 in residue.atom_idxs:
                        orig_values.append([f[2], f[3]])
                        f[2] = 0
                        f[3] = 0
                        ub_force.setBondParameters(bondid, *f)
                        # break
        ub_force.updateParametersInContext(ionic_liquid.simulation.context)
        state = ionic_liquid.simulation.context.getState(
            getEnergy=True, groups={force_group}
        )
        print(name, ub_force.getName(), state.getPotentialEnergy())
        # reset
        orig_values = iter(orig_values)
        for bondid in range(ub_force.getNumBonds()):
            f = ub_force.getBondParameters(bondid)
            idx1, idx2 = f[0:2]
            for residue in ionic_liquid.residues:
                if residue.current_name != name:
                    if idx1 in residue.atom_idxs and idx2 in residue.atom_idxs:
                        orig_value = next(orig_values)
                        f[2] = orig_value[0]
                        f[3] = orig_value[1]
                        ub_force.setBondParameters(bondid, *f)
                        # break
        ub_force.updateParametersInContext(ionic_liquid.simulation.context)
        return (
            state.getPotentialEnergy()
        )  # E_pot from ub with only the contributions from "name"

    def get_angle_contrib_from_(ionic_liquid, name=None):
        force_index = 1  # assuming that angle term is the second force...
        force = ionic_liquid.system.getForce(force_index)
        force_group = force.getForceGroup()
        # print(force_group)
        # print(ub_force)

        if name is None:
            state = ionic_liquid.simulation.context.getState(
                getEnergy=True, groups={force_group}
            )
            print("ALL", force.getName(), state.getPotentialEnergy())
            return state.getPotentialEnergy()

        orig_values = []
        for bondid in range(force.getNumAngles()):
            f = force.getAngleParameters(bondid)
            idx1, idx2, idx3 = f[0:3]
            for residue in ionic_liquid.residues:
                if residue.current_name != name:
                    if (
                        idx1 in residue.atom_idxs
                        and idx2 in residue.atom_idxs
                        and idx3 in residue.atom_idxs
                    ):
                        orig_values.append([f[3], f[4]])
                        f[3] = 0
                        f[4] = 0
                        force.setAngleParameters(bondid, *f)
                        # break
        force.updateParametersInContext(ionic_liquid.simulation.context)
        state = ionic_liquid.simulation.context.getState(
            getEnergy=True, groups={force_group}
        )
        print(name, force.getName(), state.getPotentialEnergy())
        # reset
        orig_values = iter(orig_values)
        for bondid in range(force.getNumAngles()):
            f = force.getAngleParameters(bondid)
            idx1, idx2, idx3 = f[0:3]
            for residue in ionic_liquid.residues:
                if residue.current_name != name:
                    if (
                        idx1 in residue.atom_idxs
                        and idx2 in residue.atom_idxs
                        and idx3 in residue.atom_idxs
                    ):
                        orig_value = next(orig_values)
                        f[3] = orig_value[0]
                        f[4] = orig_value[1]
                        force.setAngleParameters(bondid, *f)
                        # break
        force.updateParametersInContext(ionic_liquid.simulation.context)
        return (
            state.getPotentialEnergy()
        )  # E_pot from angle with only the contributions from "name"

    # integrator = VVIntegrator(
    integrator = DrudeNoseHooverIntegrator(
        300 * kelvin,
        10 / picoseconds,
        1 * kelvin,
        100 / picoseconds,
        0.0005 * picoseconds,
    )
    integrator.setMaxDrudeDistance(0.25 * angstroms)
    Platform.getPlatformByName("CUDA")
    dict(CudaPrecision="single")  # default is single

    # sim0 = generate_single_im1h_oac_system()
    sim0 = generate_im1h_oac_system()
    allowed_updates = {}
    # allowed updates according to simple protonation scheme
    allowed_updates[frozenset(["IM1H", "OAC"])] = {
        "r_max": 0.17,
        "prob": 1,
    }
    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    ionic_liquid = ProtexSystem(sim0, templates)

    print_force_contrib(ionic_liquid.simulation)
    # zero forces
    # all = get_ub_contrib_from_(ionic_liquid)
    # im1h = get_ub_contrib_from_(ionic_liquid, name="IM1H")
    # im1 = get_ub_contrib_from_(ionic_liquid, name="IM1")
    # oac = get_ub_contrib_from_(ionic_liquid, name="OAC")
    # hoac = get_ub_contrib_from_(ionic_liquid, name="HOAC")

    get_angle_contrib_from_(ionic_liquid)
    get_angle_contrib_from_(ionic_liquid, name="IM1H")
    get_angle_contrib_from_(ionic_liquid, name="IM1")
    get_angle_contrib_from_(ionic_liquid, name="OAC")
    get_angle_contrib_from_(ionic_liquid, name="HOAC")

    # print_force_contrib(ionic_liquid.simulation)

    # update = NaiveMCUpdate(ionic_liquid, all_forces=True)
    # state_update = StateUpdate(update)


# @pytest.mark.skipif(
#     os.getenv("CI") == "true",
#     reason="Will fail sporadicaly.",
# )
def test_wrong_atom_name(caplog):
    caplog.set_level(logging.DEBUG)

    #simulation = generate_im1h_oac_system()
    simulation = generate_small_box(use_plugin=False)
    allowed_updates = {}
    # allowed updates according to simple protonation scheme
    allowed_updates[frozenset(["IM1H", "OAC"])] = {
        "r_max": 0.16,
        "prob": 1,
    }  # r_max in nanometer, prob between 0 and 1
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["IM1H", "IM1"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.16, "prob": 1}
    # get ionic liquid templates
    IM1H_IM1 = {"IM1H": {"atom_name": "H72313"}, "IM1": {"atom_name": "wrong_name"}}
    OAC_HOAC = {"OAC": {"atom_name": "O2"}, "HOAC": {"atom_name": "H"}}
    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))
    # wrap system in IonicLiquidSystem
    ionic_liquid = ProtexSystem(simulation, templates)
    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)
    # ionic_liquid.simulation.minimizeEnergy(maxIterations=200)
    try:
        state_update.update(2)
    except RuntimeError as e:
        print("This is fine. Atom name is not present")
        print(e)


def test_save_load_updates(caplog, tmp_path):
    caplog.set_level(logging.DEBUG)

    #simulation = generate_im1h_oac_system()
    simulation = generate_small_box(use_plugin=False)
    # get ionic liquid templates
    allowed_updates = {}
    # allowed updates according to simple protonation scheme
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["IM1H", "IM1"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.16, "prob": 1}
    print(allowed_updates.keys())
    templates = ProtexTemplates(
        # [OAC_HOAC_chelpg, IM1H_IM1_chelpg], (set(["IM1H", "OAC"]), set(["IM1", "HOAC"]))
        [OAC_HOAC, IM1H_IM1],
        (allowed_updates),
    )
    # wrap system in IonicLiquidSystem
    ionic_liquid = ProtexSystem(simulation, templates)
    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)
    state_update.update_trial = 100

    # idea:
    update.dump(f"{tmp_path}/naivemcupdate.pkl")
    state_update.dump(f"{tmp_path}/stateupdate.pkl")
    del update
    del state_update
    update = NaiveMCUpdate.load(f"{tmp_path}/naivemcupdate.pkl", ionic_liquid)
    state_update = StateUpdate.load(f"{tmp_path}/stateupdate.pkl", update)
    assert update.all_forces is False
    assert state_update.update_trial == 100
