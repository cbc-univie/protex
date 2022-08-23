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

#OpenMM Plugin Imports
# https://github.com/z-gong/openmm-velocityVerlet
from velocityverletplugin import VVIntegrator

#extra stuff from VVplugin repo
sys.path.append("/home/florian/software/openmm-velocityVerlet/examples/")
from ommhelper import DrudeTemperatureReporter, ViscosityReporter

# user specified
sys.path.append("/home/florian/pythonfiles")
from pystat import read_data 

parser = argparse.ArgumentParser()
parser.add_argument('-cnt', dest='counter', help='Counter variable fpr consecutive runs', required=True)
parser.add_argument('-i', dest='inpfile', help='Input parameter file', required=True)
parser.add_argument('-p', dest='psffile', help='Input CHARMM PSF file', required=True)
parser.add_argument('-c', dest='crdfile', help='Input CHARMM CRD file (EXT verison)', required=True)
parser.add_argument('-t', dest='toppar', help='Input CHARMM-GUI toppar stream file', required=True)
parser.add_argument('-n', dest='name', help='Input base name of dcd and restart files', required=True)
parser.add_argument('-npt', dest='n_npt', help='Total number of npt runs', required=False)
parser.add_argument('-nvt', dest='n_nvt', help='Total number of nvt runs', required=False)

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
#input restart file
#if inputs.ensemble == "NVT" and int(args.counter) == int(args.n_npt)+1:
#    args.irst = "traj/" + args.name + "_npt_" + args.n_npt + ".rst"
#    if not os.path.isfile(args.irst):
#        print("Restart file not found, ", args.irst)
        #args.irst = "traj/" + args.name + "_npt_6.rst"
#    print("Restart file", args.irst)

#if inputs.ensemble == "NVT" and int(args.counter) > int(args.n_npt):
#    args.counter = int(args.counter) - int(args.n_npt)
#print("Counter is: ", args.counter)

# output_files
f_name = args.name + '_' + str(inputs.ensemble).lower()
if not any([args.orst, args.ochk]):
    args.orst = "traj/" + f_name + '_' + str(args.counter) + '.rst'
if not args.odcd:
    args.odcd = "traj/" + f_name + '_' + str(args.counter) + '.dcd'
if not args.opdb:
    args.opdb = f_name + '.pdb'
drude_temp_file = f"out/drude_temp_{str(inputs.ensemble).lower()}_{str(args.counter)}.out"

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

if 1 < int(args.counter) <= 3 and inputs.ensemble == "NPT":
    inputs.temp = 500

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
#integrator = DrudeNoseHooverIntegrator(inputs.temp*kelvin, inputs.coll_freq/picosecond, inputs.drude_temp*kelvin, inputs.drude_coll_freq/picosecond, inputs.dt*picoseconds)
integrator = VVIntegrator(inputs.temp*kelvin, inputs.coll_freq/picosecond, inputs.drude_temp*kelvin, inputs.drude_coll_freq/picosecond, inputs.dt*picoseconds)

#Drude hard wall
#if int(args.counter) == 1 and inputs.ensemble == "NPT":
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
        #state = XmlSerializer.deserialize(f.read())
        #pos = state.getState(getPositions=True).getPositions()
        #vel = state.getState(getVelocities=True).getVelocities()
        #print(state)
        simulation.context.setState(XmlSerializer.deserialize(f.read()))
        #if inputs.ensemble == "NVT":
        #    simulation.context.setPeriodicBoxVectors(psf.boxVectors[0],psf.boxVectors[1],psf.boxVectors[2])
        #    simulation.context.setPositions(pos)
        #    simulation.context.setVelocities(vel)
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
#if int(args.counter) == 101 and inputs.ensemble == "NVT":
#    inputs.nstdcd = 1
if int(args.counter) == 7 and inputs.ensemble == "NPT":
    #make file with npt box lengths
    os.system('python analysis/get_boxl.py')
    #extract mean from step 18000 to end
    xtl,_,_,_,_ = read_data("analysis/boxl.dat", ['-x', '18000'])
    #xtl = inputs.avg_boxl/10
    xtl_low = (xtl - xtl*0.0005)
    xtl_high = (xtl + xtl*0.0005)
    curr_boxl = 0.0
    print("xtl_high, xtl_low", xtl_high, xtl_low)
    #while xtl_low <= curr_boxl <= xtl_high:
    for i in range(500):
        simulation.reporters.append(StateDataReporter("find_mw.out", 10, step=True, time=True, totalEnergy=True, volume=True))
        simulation.step(100)
        curr_boxl = (float(os.popen('tail -n1 find_mw.out | cut -d "," -f4').read())**(1/3))*10
        print(i)
        print("curr_boxl",curr_boxl)
        if xtl_low <= curr_boxl <= xtl_high:
            print("Final boxl for nvt:", curr_boxl)
            break
    state = simulation.context.getState( getPositions=True, getVelocities=True )
    with open(args.orst, 'w') as f:
        f.write(XmlSerializer.serialize(state))
    quit()
#Production
if inputs.nstep > 0:
    print("\nMD run: %s steps" % inputs.nstep)
    if inputs.nstdcd > 0:
        if not args.odcd: args.odcd = 'output.dcd'
        simulation.reporters.append(DCDReporter(args.odcd, inputs.nstdcd))
    simulation.reporters.append(StateDataReporter(sys.stdout, inputs.nstout, step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, volume=True))
    simulation.reporters.append(DrudeTemperatureReporter(drude_temp_file, inputs.nstout))
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
