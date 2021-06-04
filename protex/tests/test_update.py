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
