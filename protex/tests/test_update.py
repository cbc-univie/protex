from ..testsystems import generate_im1h_oac_system, OAC_HOAC, IM1H_IM1
from ..system import IonicLiquidSystem, IonicLiqudTemplates
from ..update import NaiveMCUpdate, StateUpdate
import logging


def test_perform_charge_muatation(caplog):
    caplog.set_level(logging.DEBUG)

    simulation = generate_im1h_oac_system()
    # get ionic liquid templates
    templates = IonicLiqudTemplates([OAC_HOAC, IM1H_IM1])
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)

    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)
    state_update.write_parameters("output_initial.txt")
    par_initial = state_update.get_parameters()
    state_update.update()
    par_after_first_update = state_update.get_parameters()
    state_update.write_parameters("output_final.txt")
    state_update.update()
    par_after_second_update = state_update.get_parameters()

    print("####################################")
    print("Comparing intial charges with first update")
    print("####################################")
    for (p1, a1), (p2, a2) in zip(par_initial, par_after_first_update):
        if p1[0]._value != p2[0]._value:
            # print(a1, a2)
            # print(a1.residue)
            # print(a2.residue)

            assert int(a1.residue.id) in [350, 2]  # [649, 651]
            assert int(a2.residue.id) in [350, 2]  # [649, 651]

            assert a1.residue.name in ["IM1", "HOAC"]
            assert a2.residue.name in ["IM1", "HOAC"]

    print("####################################")
    print("Comparing intial charges with second update")
    print("####################################")
    for (p1, a1), (p2, a2) in zip(par_initial, par_after_second_update):

        if p1[0]._value != p2[0]._value:
            print(a1, a2)
            print(p1[0]._value, p2[0]._value)
            print(a1.residue)
            print(a2.residue)
            raise RuntimeError()

def test_transfer_with_distance_matrix():
    simulation = generate_im1h_oac_system()
    # get ionic liquid templates
    templates = IonicLiqudTemplates([OAC_HOAC, IM1H_IM1])
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)

    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)
    state_update.write_parameters("output_initial.txt")
    par_initial = state_update.get_parameters()
    state_update.update()
    par_after_first_update = state_update.get_parameters()
    state_update.write_parameters("output_final.txt")
    state_update.update()
    par_after_second_update = state_update.get_parameters()

    # Number of atoms is constant
    assert len(par_initial) == len(par_after_first_update) == len(par_after_second_update) == 17500
    total_charge_init, total_charge_first, total_charge_second = 0., 0., 0.
    charge_changes_first, charge_changes_second = 0, 0
    for (p1, a1), (p2,a2), (p3,a3) in zip(par_initial, par_after_first_update, par_after_second_update):
        total_charge_init += p1[0]._value
        total_charge_first += p2[0]._value
        total_charge_second += p3[0]._value
        #first update
        if p1[0]._value != p2[0]._value:
            charge_changes_first += 1
            for template in [IM1H_IM1, OAC_HOAC]:
                for resname, atom_charges in template.items():
                    if atom_charges[0] == p1[0]._value:
                        init_name = resname
                    if atom_charges[0] == p2[0]._value:
                        first_name = resname
            alternative_init_residue = ionic_liquid.templates.get_residue_name_for_other_charged_state(init_name)
            assert alternative_init_residue == first_name
        #second update
        if p2[0]._value != p3[0]._value:
            charge_changes_second += 1
            for template in [IM1H_IM1, OAC_HOAC]:
                for resname, atom_charges in template.items():
                    if atom_charges[0] == p2[0]._value:
                        first_name = resname
                    if atom_charges[0] == p3[0]._value:
                        second_name = resname
            alternative_first_residue = ionic_liquid.templates.get_residue_name_for_other_charged_state(first_name)
            assert alternative_first_residue == second_name

    # Total charge should be 0
    assert total_charge_init < 10e-10 
    assert total_charge_first < 10e-10 
    assert total_charge_second < 10e-10 

    # 1 Transfer per update -> maximum of 19 IM1H/OAC + 16 OAC/HOAC Atoms updated => 35 (effectively 34 bc IM1H/IM1 drude on C1 has same charge)
    print(charge_changes_first, charge_changes_second)
    assert charge_changes_first == 34
    assert charge_changes_second == 34