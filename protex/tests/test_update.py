from ..testsystems import generate_im1h_oac_system, OAC_HOAC, IM1H_IM1
from ..system import IonicLiquidSystem, IonicLiquidTemplates
from ..update import NaiveMCUpdate, StateUpdate
from scipy.spatial import distance_matrix
import numpy as np


def test_distance_calculation():
    simulation = generate_im1h_oac_system()
    # get ionic liquid templates
    templates = IonicLiquidTemplates(
        [OAC_HOAC, IM1H_IM1], (set(["IM1H", "OAC"]), set(["IM1", "HOAC"]))
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
            set([residue1.current_name, residue2.current_name])
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


def test_single_update():

    simulation = generate_im1h_oac_system()
    # get ionic liquid templates
    templates = IonicLiquidTemplates(
        [OAC_HOAC, IM1H_IM1], (set(["IM1H", "OAC"]), set(["IM1", "HOAC"]))
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
    assert state_update.ionic_liquid.residues[idx1]._current_charge == 1

    assert state_update.ionic_liquid.residues[idx2].current_name == "OAC"
    assert state_update.ionic_liquid.residues[idx2].original_name == "OAC"
    assert state_update.ionic_liquid.residues[idx2].current_charge == -1
    assert state_update.ionic_liquid.residues[idx2]._current_charge == -1

    candidate_pairs = (
        state_update.ionic_liquid.residues[idx1],
        state_update.ionic_liquid.residues[idx2],
    )
    state_update.updateMethod._update(candidate_pairs, 11)

    assert state_update.ionic_liquid.residues[idx1].current_name == "IM1"
    assert state_update.ionic_liquid.residues[idx1].original_name == "IM1H"
    assert state_update.ionic_liquid.residues[idx1].current_charge == 0
    assert state_update.ionic_liquid.residues[idx1]._current_charge == 0

    assert state_update.ionic_liquid.residues[idx2].current_name == "HOAC"
    assert state_update.ionic_liquid.residues[idx2].original_name == "OAC"
    assert state_update.ionic_liquid.residues[idx2].current_charge == 0
    assert state_update.ionic_liquid.residues[idx2]._current_charge == 0


def test_check_updated_charges(caplog):
    caplog.set_level(logging.DEBUG)

    simulation = generate_im1h_oac_system()
    # get ionic liquid templates
    templates = IonicLiquidTemplates(
        [OAC_HOAC, IM1H_IM1], (set(["IM1H", "OAC"]), set(["IM1", "HOAC"]))
    )
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)

    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)
    # define mutation
    idx1, idx2 = 0, 200

    candidate_pairs = (
        state_update.ionic_liquid.residues[idx1],
        state_update.ionic_liquid.residues[idx2],
    )

    state_update.write_parameters("output_initial.txt")
    par_initial = state_update.get_parameters()
    state_update.updateMethod._update(candidate_pairs, 11)
    par_after_first_update = state_update.get_parameters()
    state_update.updateMethod._update(candidate_pairs, 11)
    par_after_second_update = state_update.get_parameters()

    assert par_initial == par_after_second_update
    assert par_initial != par_after_first_update

    print("####################################")
    print("Comparing intial charges with first update")
    print("####################################")
    for (p1, a1), (p2, a2) in zip(par_initial, par_after_first_update):
        if p1[0]._value != p2[0]._value:
            print(
                f"{a1.residue.name}:{a1.residue.id}:{a1.name}:{p1[0]._value}, {a2.residue.name}:{a2.residue.id}:{a2.name}:{p2[0]._value}"
            )

    print("####################################")
    print("Comparing intial charges with second update")
    print("####################################")
    for (p1, a1), (p2, a2) in zip(par_initial, par_after_second_update):

        if p1[0]._value != p2[0]._value:
            assert False  # should not happen!


def test_transfer_with_distance_matrix():

    import numpy as np

    simulation = generate_im1h_oac_system()
    # get ionic liquid templates
    templates = IonicLiquidTemplates(
        [OAC_HOAC, IM1H_IM1], (set(["IM1H", "OAC"]), set(["IM1", "HOAC"]))
    )
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)

    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)
    state_update.write_parameters("output_initial.txt")
    par_initial = state_update.get_parameters()
    candidate_pairs1 = state_update.update(11)
    par_after_first_update = state_update.get_parameters()
    candidate_pairs2 = state_update.update(11)
    par_after_second_update = state_update.get_parameters()

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
    r1, r2 = candidate_pairs1
    print(r1.current_name)
    print(r2.current_name)

    for (p1, a1), (p2, a2), (p3, a3) in zip(
        par_initial, par_after_first_update, par_after_second_update
    ):
        total_charge_init += p1[0]._value
        total_charge_first += p2[0]._value
        total_charge_second += p3[0]._value

    # Total charge should be 0
    assert np.isclose(total_charge_init, 0.0)
    assert np.isclose(total_charge_first, 0.0)
    assert np.isclose(total_charge_second, 0.0)


def test_updates(caplog):
    caplog.set_level(logging.DEBUG)

    simulation = generate_im1h_oac_system()
    # get ionic liquid templates
    templates = IonicLiquidTemplates(
        [OAC_HOAC, IM1H_IM1], (set(["IM1H", "OAC"]), set(["IM1", "HOAC"]))
    )
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)
    pars = []
    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)
    ionic_liquid.simulation.minimizeEnergy(maxIterations=200)
    ionic_liquid.simulation.step(500)

    for _ in range(15):
        ionic_liquid.simulation.step(200)
        pars.append(state_update.get_parameters())
        candidate_pairs = state_update.update(1001)
    assert False


def test_dry_updates(caplog):
    caplog.set_level(logging.DEBUG)

    simulation = generate_im1h_oac_system()
    # get ionic liquid templates
    templates = IonicLiquidTemplates(
        [OAC_HOAC, IM1H_IM1], (set(["IM1H", "OAC"]), set(["IM1", "HOAC"]))
    )
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)
    pars = []
    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)
    ionic_liquid.simulation.minimizeEnergy(maxIterations=200)
    ionic_liquid.simulation.step(500)

    for _ in range(15):
        ionic_liquid.simulation.step(200)
        distance_dict, res_dict = state_update._get_positions_for_mutation_sites()
        # propose the update candidates based on distances
        state_update._print_start()
        candidate_pairs = state_update._propose_candidate_pair(distance_dict, res_dict)
        state_update._print_stop()
        pars.append(state_update.get_parameters())
    assert False
