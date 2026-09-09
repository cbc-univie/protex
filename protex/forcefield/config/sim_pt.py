# exemplary run script for OpenMM, using a config file for setting up protex

import sys 
import warnings
import os
from pathlib import Path

from simtk.openmm.app import StateDataReporter, PDBReporter, DCDReporter
from simtk.openmm import XmlSerializer
from simtk.openmm.app import CharmmCrdFile, CharmmParameterSet, CharmmPsfFile
from simtk.openmm.app import PME, HBonds
from simtk.openmm.app import Simulation
from simtk.unit import angstroms, kelvin, picoseconds
from simtk.openmm import (
        OpenMMException,
        Platform,
        Context,
        DrudeLangevinIntegrator,
        DrudeNoseHooverIntegrator,
        XmlSerializer,
    )

import protex
from protex.testsystems import generate_system_from_config
from protex.system import ProtexSystem, ProtexTemplates
from protex.reporter import ChargeReporter, DrudeTemperatureReporter, EnergyReporter
from protex.update import KeepHUpdate, StateUpdate

# usage: python sim_pt.py [current segment nr] [last segment nr]
file_nr: int = int(sys.argv[1])
last_run: int = int(sys.argv[2])

Path("./out").mkdir(parents=True, exist_ok=True)
Path("./traj").mkdir(parents=True, exist_ok=True)

simulation = generate_system_from_config("config_test.toml")
simulation_for_parameters = generate_system_from_config("config_test.toml")

templates = ProtexTemplates.from_config("config_test.toml")

if file_nr == 1:
    ionic_liquid = ProtexSystem(simulation, templates, simulation_for_parameters)
else:
    ionic_liquid = ProtexSystem.load(f"out/system_{file_nr-1}.pkl", simulation, simulation_for_parameters)
# load names to get the correct protonation states
if file_nr > 1:
    ionic_liquid.load_updates(f"out/updates_{file_nr-1}.txt")
    ionic_liquid.loadState(f"traj/nvt_state_{file_nr-1}.rst")

update = KeepHUpdate.from_config(ionic_liquid, "config_test.toml")
prob_function = "linear"
if file_nr == 1:
    state_update = StateUpdate(update, prob_function=prob_function)
else:
    state_update = StateUpdate.load(f"out/state_update_{file_nr-1}.pkl", update)

#  adding reporter
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
        density=True,
    )
)
ionic_liquid.simulation.reporters.append(DrudeTemperatureReporter(f"out/drude_temp_{file_nr}.out", state_save_freq))
ionic_liquid.simulation.reporters.append(EnergyReporter(f"out/energy_{file_nr}.out", state_save_freq*10))

# time step and update frequency
dt = 0.0005 #ps
time_between_updates = 7 #ps
sim_time = 500  # ps
update_steps: int = 2 # "lambda states", only even numbers!
steps_from_update_before_reporter: int = update_steps // 2
steps_from_update_after_reporter: int = update_steps // 2
if update_steps % 2 != 0:
    warnings.warn("Probably working, not entirely tested. To be on the safe side use an even number for 'update_steps'", UserWarning)
    steps_from_update_after_reporter += 1

steps_between_updates = int(time_between_updates/dt) 
if ((1/dt)*time_between_updates)%(1/dt)!=0:
    time_between_updates = round(steps_between_updates*dt,8)
    msg = f"Not possible to have an integer step value with {time_between_updates}/{dt}! \
            It is rounded down. Hence the effective time between updates is now {steps_between_updates*dt}"
    warnings.warn(msg, UserWarning)

init_steps_between_updates = steps_between_updates-steps_from_update_before_reporter
real_steps_between_updates = steps_between_updates-update_steps #remove the steps taken during the updates process
final_steps_between_updates = steps_from_update_after_reporter

transfer_cycles = sim_time/time_between_updates
total_steps = sim_time/dt
if file_nr == 1:
    actual_transfer_cycles = transfer_cycles -1 
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
ionic_liquid.save_updates(f"out/updates_{file_nr}.txt")
ionic_liquid.saveState(f"traj/nvt_state_{file_nr}.rst") 
ionic_liquid.dump(f"out/system_{file_nr}.pkl")
state_update.dump(f"out/state_update_{file_nr}.pkl")