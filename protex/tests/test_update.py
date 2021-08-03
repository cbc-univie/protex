from ..testsystems import generate_im1h_oac_system, OAC_HOAC, IM1H_IM1
from ..system import IonicLiquidSystem, IonicLiqudTemplates
from ..update import NaiveMCUpdate, StateUpdate
import logging


def test_perform_charge_muatation(caplog):
    caplog.set_level(logging.DEBUG)

    simulation = generate_im1h_oac_system()
    # get ionic liquid templates
    templates = IonicLiqudTemplates(
        [OAC_HOAC, IM1H_IM1], (set(["IM1H", "OAC"]), set(["IM1", "HOAC"]))
    )
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)

    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)
    state_update.write_parameters("output_initial.txt")
    par_initial = state_update.get_parameters()
    state_update.update()
    par_after_first_update = state_update.get_parameters()
    state_update.update()
    par_after_second_update = state_update.get_parameters()
    state_update.update()
    par_after_third_update = state_update.get_parameters()
    state_update.update()
    par_after_fourth_update = state_update.get_parameters()

    print("####################################")
    print("Comparing intial charges with first update")
    print("####################################")
    for (p1, a1), (p2, a2) in zip(par_initial, par_after_first_update):
        if p1[0]._value != p2[0]._value:
            print(a1, a2)
            print(p1[0]._value, p2[0]._value)
            print(a1.residue)
            print(a2.residue)

            # assert int(a1.residue.id) in [350, 2]  # [649, 651]
            # assert int(a2.residue.id) in [350, 2]  # [649, 651]

            assert a1.residue.name in ["IM1H", "OAC"]
            assert a2.residue.name in ["IM1H", "OAC"]

    print("####################################")
    print("Comparing intial charges with second update")
    print("####################################")
    for (p1, a1), (p2, a2) in zip(par_initial, par_after_second_update):

        if p1[0]._value != p2[0]._value:
            print(a1, a2)
            print(p1[0]._value, p2[0]._value)
            print(a1.residue)
            print(a2.residue)


def test_transfer_with_distance_matrix():

    import numpy as np

    simulation = generate_im1h_oac_system()
    # get ionic liquid templates
    templates = IonicLiqudTemplates(
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
        == 17500
    )
    total_charge_init, total_charge_first, total_charge_second = 0.0, 0.0, 0.0
    print(candidate_pairs1)
    r1, r2 = candidate_pairs1
    print(r1.name)
    print(r2.name)

    assert False
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
    templates = IonicLiqudTemplates(
        [OAC_HOAC, IM1H_IM1], (set(["IM1H", "OAC"]), set(["IM1", "HOAC"]))
    )
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)
    pars = []
    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)
    for _ in range(15):
        pars.append(state_update.get_parameters())
        candidate_pairs1 = state_update.update(101)
