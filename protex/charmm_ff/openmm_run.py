#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Minimization of PACKMOL structure
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#general Imports
import sys
import argparse
import os

#OpenMM Imports
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

from omm_readinputs import *
from omm_readparams import *

parser = argparse.ArgumentParser()
parser.add_argument('-cnt', dest='counter', help='Counter variable fpr consecutive runs', required=True)
parser.add_argument('-i', dest='inpfile', help='Input parameter file', required=True)
parser.add_argument('-p', dest='psffile', help='Input CHARMM PSF file', required=True)
parser.add_argument('-c', dest='crdfile', help='Input CHARMM CRD file (EXT verison)', required=True)
parser.add_argument('-t', dest='toppar', help='Input CHARMM-GUI toppar stream file', required=True)
parser.add_argument('-n', dest='name', help='Input base name of dcd and restart files', required=True)

parser.add_argument('-b', dest='sysinfo', help='Input CHARMM-GUI sysinfo stream file (optional)', default=None)
parser.add_argument('-icrst', metavar='RSTFILE', dest='icrst', help='Input CHARMM RST file (optional)', default=None)
parser.add_argument('-irst', metavar='RSTFILE', dest='irst', help='Input restart file (optional)', default=None)
parser.add_argument('-ichk', metavar='CHKFILE', dest='ichk', help='Input checkpoint file (optional)', default=None)
parser.add_argument('-opdb', metavar='PDBFILE', dest='opdb', help='Output PDB file (optional)', default=None)
parser.add_argument('-minpdb', metavar='MINIMIZEDPDBFILE', dest='minpdb', help='Output PDB file after minimization (optional)', default=None)
parser.add_argument('-orst', metavar='RSTFILE', dest='orst', help='Output restart file (optional)', default=None)
parser.add_argument('-ochk', metavar='CHKFILE', dest='ochk', help='Output checkpoint file (optional)', default=None)
parser.add_argument('-odcd', metavar='DCDFILE', dest='odcd', help='Output trajectory file (optional)', default=None)
args = parser.parse_args()


# Load parameters
print("Loading parameters")
inputs = read_inputs(args.inpfile)
if inputs.ensemble == "NVT" and int(args.counter) == 1:
    args.irst = "traj/" + args.name + "_npt_11.rst"

f_name = args.name + '_' + str(inputs.ensemble).lower()
if not any([args.orst, args.ochk]):
    args.orst = "traj/" + f_name + '_' + args.counter + '.rst'
if not args.odcd:
    args.odcd = "traj/" + f_name + '_' + args.counter + '.dcd'
if not args.opdb:
    args.opdb = f_name + '.pdb'

pcnt = int(args.counter) - 1
if int(args.counter) > 1 and not any([args.icrst, args.irst, args.ichk]):
    args.irst = "traj/" + f_name + '_' + str(pcnt) + '.rst'
#=======================================================================
# Force field
#=======================================================================
#Loading CHARMM files
print("Loading CHARMM files...")
params = read_params(args.toppar)
#psf = CharmmPsfFile("im1h_oac_200_im1_hoac_300_xplor.psf")
psf = CharmmPsfFile(args.psffile)
#cooridnates can be provieded by CharmmCrdFile, CharmmRstFile or PDBFile classes
#crd = CharmmCrdFile("im1h_oac_200_im1_hoac_300_charmmext.crd")
crd = CharmmCrdFile(args.crdfile)
#pdb = PDBFile('input.pdb')

# Create the OpenMM system
print('Creating OpenMM System...')
#boxlength
xtl = inputs.boxl*angstroms
psf.setBox(xtl,xtl,xtl)
system = psf.createSystem(params, nonbondedMethod=inputs.coulomb, 
                                  nonbondedCutoff=inputs.r_off*angstroms, 
                                  switchDistance=inputs.r_on*angstroms,
                                  ewaldErrorTolerance=inputs.ewald_Tol,
                                  constraints=inputs.cons)
#for force in system.getForces():
#    if isinstance(force, NonbondedForce): force.setUseDispersionCorrection(True)
#    if isinstance(force, CustomNonbondedForce) and force.getNumTabulatedFunctions() == 2:
#        force.setUseLongRangeCorrection(True)

#isotropoic barostat type (ist sowohl bei equil als auch production aktiviert)
if inputs.pcouple == "yes":
    barostat = MonteCarloBarostat(inputs.p_ref*atmosphere, inputs.temp*kelvin) #400 in timestep units, bei charmm war es 0.2ps 
    system.addForce(barostat)

#for stabilitiy reasons make first run (20ps) with timestep of 0.1fs
if int(args.counter) == 1 and inputs.ensemble == "NPT":
	inputs.dt = inputs.pre_dt
	inputs.nstep = inputs.pre_nstep
	print(f"First run with time step of {inputs.dt} ps")
# set up integrator and dual Langevin thermostat
#integrator = DrudeLangevinIntegrator(inputs.temp*kelvin, inputs.fric_coeff/picosecond, inputs.drude_temp*kelvin, inputs.drude_fric_coeff/picosecond, inputs.dt*picoseconds)
integrator = DrudeNoseHooverIntegrator(inputs.temp*kelvin, inputs.coll_freq/picosecond, inputs.drude_temp*kelvin, inputs.drude_coll_freq/picosecond, inputs.dt*picoseconds)

#Drude hard wall
integrator.setMaxDrudeDistance(inputs.drude_hardwall*angstroms)
if integrator.getMaxDrudeDistance() == 0:
    print("No Drude Hard Wall Contraint in use")
else:
    print("Drude Hard Wall set to {}".format(integrator.getMaxDrudeDistance()))

# Set platform
platform = Platform.getPlatformByName('CUDA')
prop = dict(CudaPrecision='mixed')
print(platform.getName())

#simulation = Simulation(psf.topology, system, integrator)
simulation = Simulation(psf.topology, system, integrator, platform, prop)
simulation.context.setPositions(crd.positions)
if args.icrst:
    charmm_rst = read_charmm_rst(args.icrst)
    simulation.context.setPositions(charmm_rst.positions)
    simulation.context.setVelocities(charmm_rst.velocities)
    simulation.context.setPeriodicBoxVectors(charmm_rst.box[0], charmm_rst.box[1], charmm_rst.box[2])
if args.irst:
    with open(args.irst, 'r') as f:
        simulation.context.setState(XmlSerializer.deserialize(f.read()))
if args.ichk:
    with open(args.ichk, 'rb') as f:
        simulation.context.loadCheckpoint(f.read())

#Drude VirtualSites
simulation.context.computeVirtualSites()
# print out energy of initial configuration
state=simulation.context.getState(getPositions=True, getEnergy=True)
print("\nInitial system energy")
#print(state.getPotentialEnergy().value_in_unit(kilocalories_per_mole))
print(state.getPotentialEnergy())

if int(args.counter) == 1 and not any([args.irst, args.ichk, args.icrst]):
    print("\nEnergy minimization: %s steps" % inputs.mini_nstep)
    simulation.minimizeEnergy(tolerance=inputs.mini_Tol*kilojoule/mole, maxIterations=inputs.mini_nstep)
    print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
    print("Saving minimized pdb...")
    if not args.minpdb: args.minpdb = args.name + "_min.pdb"
    positions = simulation.context.getState(getPositions=True).getPositions()
    PDBFile.writeFile(simulation.topology, positions, open(args.minpdb,'w'))

# Generate initial velocities
if inputs.gen_vel == 'yes':
    print("\nGenerate initial velocities")
    if inputs.gen_seed:
        simulation.context.setVelocitiesToTemperature(inputs.gen_temp, inputs.gen_seed)
    else:
        simulation.context.setVelocitiesToTemperature(inputs.gen_temp)

# save every step for IR spectrum calculation
if int(args.counter) == 101 and inputs.ensemble == "NVT":
    inputs.nstdcd = 1
#Production
if inputs.nstep > 0:
    print("\nMD run: %s steps" % inputs.nstep)
    if inputs.nstdcd > 0:
        if not args.odcd: args.odcd = 'output.dcd'
        simulation.reporters.append(DCDReporter(args.odcd, inputs.nstdcd))
    simulation.reporters.append(StateDataReporter(sys.stdout, inputs.nstout, step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, volume=True))
    simulation.step(inputs.nstep)

# Write restart file
if not (args.orst or args.ochk): args.orst = 'output.rst'
if args.orst:
    state = simulation.context.getState( getPositions=True, getVelocities=True )
    with open(args.orst, 'w') as f:
        f.write(XmlSerializer.serialize(state))
if args.ochk:
    with open(args.ochk, 'wb') as f:
        f.write(simulation.context.createCheckpoint())
if args.opdb:
    crd = simulation.context.getState(getPositions=True).getPositions()
    PDBFile.writeFile(psf.topology, crd, open(args.opdb, 'w'))
