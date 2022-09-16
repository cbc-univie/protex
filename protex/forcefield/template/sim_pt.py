import os
import sys
import warnings
from pathlib import Path

from simtk.openmm import (Context, DrudeLangevinIntegrator,
                          DrudeNoseHooverIntegrator, OpenMMException, Platform,
                          XmlSerializer)
from simtk.openmm.app import (PME, CharmmCrdFile, CharmmParameterSet,
                              CharmmPsfFile, DCDReporter, HBonds, PDBReporter,
                              Simulation, StateDataReporter)
from simtk.unit import angstroms, kelvin, picoseconds

import protex
from protex.reporter import ChargeReporter
from protex.scripts.ommhelper import DrudeTemperatureReporter
from protex.system import IonicLiquidSystem, IonicLiquidTemplates
from protex.testsystems import IM1H_IM1, OAC_HOAC, generate_im1h_oac_system
from protex.update import NaiveMCUpdate, StateUpdate

allowed_updates = {}
# allowed updates according to simple protonation scheme
allowed_updates[frozenset(["IM1H", "OAC"])] = { # 2+3
    "r_max": 0.155,
    "prob": 0.994,
}  # r_max in nanometer, prob zwischen 0 und 1
allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.155, "prob": 0.098} #1+4
allowed_updates[frozenset(["IM1H", "IM1"])] = {"r_max": 0.155, "prob": 0.201} # 1+2
allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.155, "prob": 0.684} # 3+4

file_nr: int = int(sys.argv[1])
last_run: int = int(sys.argv[2])


Path("./psf").mkdir(parents=True, exist_ok=True)
psf_file = "im1h_oac_150_im1_hoac_350.psf"
if file_nr > 1:
    psf_file = f"psf/nvt_{file_nr-1}.psf"


npt_restart_file = "traj/im1h_oac_150_im1_hoac_350_npt_7.rst"
# obtain simulation object
simulation = generate_im1h_oac_system(psf_file=psf_file,restart_file=npt_restart_file)

# get ionic liquid templates
templates = IonicLiquidTemplates(
    [OAC_HOAC, IM1H_IM1], (allowed_updates)
)
# wrap system in IonicLiquidSystem
ionic_liquid = IonicLiquidSystem(simulation, templates)
# load names to get the correct protonation states
if file_nr > 1:
    ionic_liquid.load_updates(f"out/updates_{file_nr-1}.txt")
    ionic_liquid.loadCheckpoint(f"traj/nvt_checkpoint_{file_nr-1}.rst")
# initialize update method
to_adapt = [("IM1H", 150, frozenset(["IM1H", "OAC"])), ("IM1", 350, frozenset(["IM1", "HOAC"]))]
update = NaiveMCUpdate(ionic_liquid, all_forces=True, to_adapt = to_adapt)
# initialize state update class
state_update = StateUpdate(update)

# adding reporter
dcd_save_freq = 200
state_save_freq = 200
ionic_liquid.simulation.reporters.append(DCDReporter(f"traj/nvt_{file_nr}.dcd", int(dcd_save_freq)))
ionic_liquid.simulation.reporters.append(
    StateDataReporter(
        sys.stdout,
        state_save_freq,
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
        volume=True,
        density=False,
    )
)
ionic_liquid.simulation.reporters.append(DrudeTemperatureReporter(f"out/drude_temp_{file_nr}.out", state_save_freq))

dt = 0.0005 #ps
time_between_updates = 10 #ps
sim_time = 500  # ps
update_steps: int = 2 #only even numbers! is it working for steps > 2??? 
steps_from_update_before_reporter: int = update_steps // 2
steps_from_update_after_reporter: int = update_steps // 2
if update_steps % 2 != 0:
    warnings.warn("Probably working, not entirely tested. To be on the safe side use an even number for 'update_steps'", UserWarning)
    steps_from_update_after_reporter += 1

steps_between_updates = int(time_between_updates/dt) #int division not working, dt too small
if ((1/dt)*time_between_updates)%(1/dt)!=0: #BUG not working if time_between_updates <1
    time_between_updates = round(steps_between_updates*dt,8)
    msg = f"Not possible to have an integer step value with {time_between_updates}/{dt}! \
            It is rounded down. Hence the effective time between updates is now {steps_between_updates*dt}"
    warnings.warn(msg, UserWarning)

#init_steps_between_updates = steps_between_updates-update_steps/2
init_steps_between_updates = steps_between_updates-steps_from_update_before_reporter
real_steps_between_updates = steps_between_updates-update_steps #remove the steps taken during the updates process
final_steps_between_updates = steps_from_update_after_reporter

transfer_cycles = sim_time/time_between_updates
total_steps = sim_time/dt
if file_nr == 1:
    actual_transfer_cycles = transfer_cycles -1 #-1 da die ersten steps außerhalb der loop stattfinden
    assert total_steps == init_steps_between_updates+(real_steps_between_updates+update_steps)*(actual_transfer_cycles)+final_steps_between_updates
    simulation.context.setTime(0.)
else:
    actual_transfer_cycles = transfer_cycles
    assert total_steps == (real_steps_between_updates+update_steps)*(actual_transfer_cycles)
    simulation.currentStep = int(total_steps*(file_nr-1)-1)


infos = {"dcd_save_freq": dcd_save_freq, "sim_time(ps)": sim_time, "update_steps": update_steps, "steps_between_updates": steps_between_updates, "dt(ps)": dt}
charge_reporter = ChargeReporter(f"out/charge_changes_{file_nr}.out", dcd_save_freq, ionic_liquid, header_data=infos)
ionic_liquid.simulation.reporters.append(charge_reporter)

print(f"{init_steps_between_updates=}")
print(f"{update_steps=}")
print(f"{real_steps_between_updates=}")
print(f"{final_steps_between_updates=}")
if file_nr == 1:
    print("First run")
    ionic_liquid.simulation.step(int(init_steps_between_updates))
for step in range(1,int(actual_transfer_cycles+1)):
    print(step)
    state_update.update(int(update_steps))
    ionic_liquid.simulation.step(int(real_steps_between_updates))
if file_nr == last_run:
    print(f"Last run (Nr. {last_run})")
    ionic_liquid.simulation.step(int(final_steps_between_updates))

# restart files
state = ionic_liquid.simulation.context.getState( getPositions=True, getVelocities=True )
with open(f"traj/nvt_{file_nr}.rst", 'w') as f:
    f.write(XmlSerializer.serialize(state))

ionic_liquid.save_updates(f"out/updates_{file_nr}.txt")
ionic_liquid.write_psf("im1h_oac_150_im1_hoac_350.psf", f"psf/nvt_{file_nr}.psf")
ionic_liquid.saveCheckpoint(f"traj/nvt_checkpoint_{file_nr}.rst")
