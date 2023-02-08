import logging
import os
from collections import defaultdict
from sys import stdout

import pytest

import protex

try:  # Syntax changed in OpenMM 7.6
    import openmm as mm
    from openmm import (
        Context,
        DrudeNoseHooverIntegrator,
        OpenMMException,
        Platform,
        XmlSerializer,
    )
    from openmm.app import (
        PME,
        CharmmCrdFile,
        CharmmParameterSet,
        CharmmPsfFile,
        DCDReporter,
        HBonds,
        PDBReporter,
        Simulation,
        StateDataReporter,
    )
    from openmm.unit import angstroms, kelvin, md_kilocalories, nanometers, picoseconds
except ImportError:
    import simtk.openmm as mm
    from simtk.openmm import (
        OpenMMException,
        Platform,
        Context,
        DrudeNoseHooverIntegrator,
        XmlSerializer,
    )
    from simtk.openmm.app import DCDReporter, PDBReporter, StateDataReporter
    from simtk.openmm.app import CharmmCrdFile, CharmmParameterSet, CharmmPsfFile
    from simtk.openmm.app import PME, HBonds
    from simtk.openmm.app import Simulation
    from simtk.unit import angstroms, kelvin, picoseconds, nanometers

from ..reporter import ChargeReporter, EnergyReporter
from ..residue import Residue
from ..system import ProtexSystem, ProtexTemplates
from ..testsystems import (
    IM1H_IM1,
    OAC_HOAC,
    generate_im1h_oac_dummy_system,
    generate_im1h_oac_system,
    generate_im1h_oac_system_clap,
    generate_single_im1h_oac_system,
    generate_small_box,
)
from ..update import NaiveMCUpdate, StateUpdate

LOGGER = logging.getLogger(__name__)

###########################################
# Short tests with only single moleucle box
###########################################
def test_update_single():
    """Test the residue update function and therefore all get and set methods if they work
    check if the parameters before and after have changed
    """
    simulation = generate_single_im1h_oac_system()
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 1}

    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))

    ionic_liquid = ProtexSystem(simulation, templates)
    FORCES = [
        "HarmonicBondForce",
        "HarmonicAngleForce",
        "PeriodicTorsionForce",
        "CustomTorsionForce",
        "DrudeForce",
        "NonbondedForce",
    ]
    LOGGER.debug(f"{FORCES=}")
    name_function = {
        "HarmonicBondForce": [
            Residue._get_HarmonicBondForce_parameters_at_lambda,
            Residue._set_HarmonicBondForce_parameters,
        ],
        "HarmonicAngleForce": [
            Residue._get_HarmonicAngleForce_parameters_at_lambda,
            Residue._set_HarmonicAngleForce_parameters,
        ],
        "PeriodicTorsionForce": [
            Residue._get_PeriodicTorsionForce_parameters_at_lambda,
            Residue._set_PeriodicTorsionForce_parameters,
        ],
        "CustomTorsionForce": [
            Residue._get_CustomTorsionForce_parameters_at_lambda,
            Residue._set_CustomTorsionForce_parameters,
        ],
        "DrudeForce": [
            Residue._get_DrudeForce_parameters_at_lambda,
            Residue._set_DrudeForce_parameters,
        ],
        "NonbondedForce": [
            Residue._get_NonbondedForce_parameters_at_lambda,
            Residue._set_DrudeForce_parameters,
        ],
    }

    pair_12_13_list = ionic_liquid._build_exclusion_list(ionic_liquid.topology)

    def get_params(force, force_name, atom_idxs, forces_dict):
        params = []

        if force_name == "HarmonicBondForce":
            for bond_id in range(force.getNumBonds()):
                f = force.getBondParameters(bond_id)
                idx1 = f[0]
                idx2 = f[1]
                if (
                    idx1 in atom_idxs and idx2 in atom_idxs
                ):  # atom index of bond force needs to be in atom_idxs
                    params.append(f)
                    forces_dict[force_name].append(f)

        elif force_name == "HarmonicAngleForce":
            for bond_id in range(force.getNumAngles()):
                f = force.getAngleParameters(bond_id)
                idx1 = f[0]
                idx2 = f[1]
                idx3 = f[2]
                if (
                    idx1 in atom_idxs and idx2 in atom_idxs and idx3 in atom_idxs
                ):  # atom index of bond force needs to be in atom_idxs
                    params.append(f)
                    forces_dict[force_name].append(f)

        elif force_name == "PeriodicTorsionForce" or force_name == "CustomTorsionForce":
            for torsion_id in range(force.getNumTorsions()):
                f = force.getTorsionParameters(torsion_id)
                idx1 = f[0]
                idx2 = f[1]
                idx3 = f[2]
                idx4 = f[3]
                if (
                    idx1 in atom_idxs
                    and idx2 in atom_idxs
                    and idx3 in atom_idxs
                    and idx4 in atom_idxs
                ):  # atom index of bond force needs to be in atom_idxs
                    params.append(f)
                    forces_dict[force_name].append(f)

        elif force_name == "DrudeForce":
            for drude_id in range(force.getNumParticles()):
                f = force.getParticleParameters(drude_id)
                idx1, idx2 = f[0], f[1]
                if idx1 in atom_idxs and idx2 in atom_idxs:
                    params.append(f)
                    forces_dict[force_name].append(f)
            # thole
            for drude_id in range(force.getNumScreenedPairs()):
                f = force.getScreenedPairParameters(drude_id)
                parent1, parent2 = pair_12_13_list[drude_id]
                drude1, drude2 = parent1 + 1, parent2 + 1
                if drude1 in atom_idxs and drude2 in atom_idxs:
                    params.append(f)
                    forces_dict[force_name + "Thole"].append(f)

        elif force_name == "NonbondedForce":
            if type(force).__name__ == "NonbondedForce":
                params = [force.getParticleParameters(idx) for idx in atom_idxs]
                forces_dict[force_name] = [
                    force.getParticleParameters(idx) for idx in atom_idxs
                ]
                # Also add exceptions
                for exc_id in range(force.getNumExceptions()):
                    f = force.getExceptionParameters(exc_id)
                    idx1 = f[0]
                    idx2 = f[1]
                    if idx1 in atom_idxs and idx2 in atom_idxs:
                        params.append(f)
                        forces_dict[force_name + "Exceptions"].append(f)
        else:
            print("Force not found")

        return params

    def get_quantities(forces_dict, name):
        quantities = []
        for lst in forces_dict[name]:
            for value in lst:
                if isinstance(value, mm.unit.Quantity):
                    quantities.append(value)
        return quantities

    for residue in ionic_liquid.residues:
        LOGGER.debug(f"Processing {residue}")
        res_name = residue.current_name
        alternativ_name = residue.alternativ_name
        forces_orig = residue.parameters[res_name]
        forces_alternativ = residue.parameters[alternativ_name]
        # print(f"hallo2 {residue}") # uses __str__
        # print(f"hallo3 {residue=}") # uses __repr__
        forces_dict0 = defaultdict(list)
        forces_dict1 = defaultdict(list)

        for force in ionic_liquid.system.getForces():
            force_name = type(force).__name__
            if force_name in FORCES:
                LOGGER.debug(f"At force {force_name}")
                atom_idxs = residue.atom_idxs
                params0 = get_params(force, force_name, atom_idxs, forces_dict0)

        assert forces_dict0 == forces_orig

        for force in ionic_liquid.system.getForces():
            force_name = type(force).__name__
            if force_name in FORCES:
                residue.update(force_name, 1)
                ionic_liquid.update_context(force_name)
                params2 = get_params(force, force_name, atom_idxs, forces_dict1)
        # assert forces_dict1 == forces_alternativ # npt working because indices change
        for key in forces_dict1:
            # after update 1 -> changes
            q1 = get_quantities(forces_dict1, key)
            q2 = get_quantities(forces_alternativ, key)
            assert q1 == q2


def test_residues():
    simulation = generate_single_im1h_oac_system()
    topology = simulation.topology
    for idx, r in enumerate(topology.residues()):
        if r.name == "IM1H":  # and idx == 0:
            atom_idxs = [atom.index for atom in r.atoms()]
            atom_names = [atom.name for atom in r.atoms()]
            print(atom_idxs)
            print(atom_names)
            assert atom_idxs == [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
            ]
            assert atom_names == [
                "C1",
                "DC1",
                "H1",
                "H2",
                "H3",
                "N1",
                "DN1",
                "C2",
                "DC2",
                "H4",
                "C3",
                "DC3",
                "H5",
                "C4",
                "DC4",
                "H6",
                "N2",
                "DN2",
                "H7",
                "LPN21",
            ]
        if r.name == "HOAC":  # and idx == 650:
            atom_idxs = [atom.index for atom in r.atoms()]
            atom_names = [atom.name for atom in r.atoms()]
            print(atom_idxs)
            print(atom_names)
            print(idx)
            assert atom_idxs == [
                56,
                57,
                58,
                59,
                60,
                61,
                62,
                63,
                64,
                65,
                66,
                67,
                68,
                69,
                70,
                71,
            ]

            assert atom_names == [
                "C1",
                "DC1",
                "C2",
                "DC2",
                "H1",
                "H2",
                "H3",
                "O1",
                "DO1",
                "O2",
                "DO2",
                "H",
                "LPO11",
                "LPO12",
                "LPO21",
                "LPO22",
            ]

    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 1}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 1}

    templates = ProtexTemplates([OAC_HOAC, IM1H_IM1], (allowed_updates))

    ionic_liquid = ProtexSystem(simulation, templates)

    for residue in ionic_liquid.residues:
        if residue.residue.name == "IM1H":
            assert residue.current_name == "IM1H"
            assert residue.alternativ_name == "IM1"
        if residue.residue.name == "IM1":
            assert residue.current_name == "IM1"
            assert residue.alternativ_name == "IM1H"
        if residue.residue.name == "OAC":
            assert residue.current_name == "OAC"
            assert residue.alternativ_name == "HOAC"
        if residue.residue.name == "HOAC":
            assert residue.current_name == "HOAC"
            assert residue.alternativ_name == "OAC"

    assert len(ionic_liquid.residues) == 4

    residue = ionic_liquid.residues[0]
    charge = residue.endstate_charge

    assert charge == 1
    print(residue.atom_names)
    assert (residue.get_idx_for_atom_name("H7")) == 18

    residue = ionic_liquid.residues[1]

    assert (residue.get_idx_for_atom_name("H")) == 31


def test_single_harmonic_force(caplog):
    caplog.set_level(logging.DEBUG)

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
    StateUpdate(update)

    for force in ionic_liquid.system.getForces():
        if type(force).__name__ == "HarmonicBondForce" and force.getForceGroup() == 0:
            print(force.getNumBonds())
        if type(force).__name__ == "HarmonicBondForce" and force.getForceGroup() == 3:
            print(force.getNumBonds())

    for residue in ionic_liquid.residues:
        print(residue.current_name)
        params = residue.parameters[residue.current_name]["HarmonicBondForce"]
        ub_forces = []
        for force in ionic_liquid.system.getForces():
            if force.getForceGroup() == 3:  # UreyBradley
                for idx in range(force.getNumBonds()):
                    f = force.getBondParameters(idx)
                    for p in params:
                        if f[0:2] == p[0:2]:
                            ub_forces.append(f)
                            print(f[3] / 2 / md_kilocalories * angstroms**2)
        print(len(ub_forces))
