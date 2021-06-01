from sys import stdout
from ..testsystems import generate_im1h_oac_system, OAC_HOAC, IM1H_IM1
from ..system import IonicLiquidSystem, IonicLiqudTemplates
from ..update import NaiveMCUpdate, StateUpdate
from simtk import unit
import numpy as np


def test_perform_charge_muatation():
    simulation = generate_im1h_oac_system()
    # get ionic liquid templates
    templates = IonicLiqudTemplates([OAC_HOAC, IM1H_IM1])
    # wrap system in IonicLiquidSystem
    ionic_liquid = IonicLiquidSystem(simulation, templates)

    update = NaiveMCUpdate(ionic_liquid)
    # initialize state update class
    state_update = StateUpdate(update)

    state_update.update()
