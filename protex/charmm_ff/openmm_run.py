#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Minimization of PACKMOL structure
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#general Imports
import sys

#OpenMM Imports
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

#Variables
if len(sys.argv) > 1:
    cnt = int(sys.argv[1])
else:
    cnt = -1

omin = "output_mini.pdb"
odcd = "output_" + str(cnt) + ".dcd"
orst = "output_" + str(cnt) + ".rst"
ochk = "output_" + str(cnt) + ".chk"
opdb = "output_" + str(cnt) + ".pdb"


#=======================================================================
# Force field
#=======================================================================

#Loading CHARMM files
print("Loading CHARMM files...")
params = CharmmParameterSet("toppar/toppar_drude_master_protein_2013f_lj04.str","toppar/im1h_d_fm_lj.str","toppar/oac_d_lj.str","toppar/im1_d_fm_lj.str","toppar/hoac_d.str")

psf = CharmmPsfFile("im1h_oac_200_im1_hoac_300_xplor.psf")
#cooridnates can be provieded by CharmmCrdFile, CharmmRstFile or PDBFile classes
crd = CharmmCrdFile("im1h_oac_200_im1_hoac_300_ext.crd")
#pdb = PDBFile('input.pdb')

# Create the OpenMM system
print('Creating OpenMM System...')
#boxlength
xtl = 48.0*angstroms
psf.setBox(xtl,xtl,xtl)
system = psf.createSystem(params, nonbondedMethod=PME, 
                                  nonbondedCutoff=11.0*angstroms, 
                                  switchDistance=10.0*angstroms,
                                  ewaldErrorTolerance=0.0001,
                                  constraints=HBonds)
for force in system.getForces():
    if isinstance(force, NonbondedForce): force.setUseDispersionCorrection(True)
    if isinstance(force, CustomNonbondedForce) and force.getNumTabulatedFunctions() == 2:
        force.setUseLongRangeCorrection(True)

#barostat
barostat = MonteCarloBarostat(1.0*bar, 300*kelvin, 400) #400 in timestep units, bei charmm war es 0.2ps
system.addForce(barostat)

# set up integrator and dual Langevin thermostat
integrator = DrudeLangevinIntegrator(300*kelvin, 5/picosecond, 1*kelvin, 20/picosecond, 0.0005*picoseconds)
#Drude hard wall
integrator.setMaxDrudeDistance(0.2*angstroms)

# Set platform
#platform = Platform.getPlatformByName('CUDA')
#prop = dict(CudaPrecision='mixed')

simulation = Simulation(psf.topology, system, integrator)
#simulation = Simulation(psf.topology, system, integrator, platform, prop)
simulation.context.setPositions(crd.positions)
if cnt > 1:
    pcnt = cnt -1
    ichk = "output_" + str(pcnt) + ".chk"
    with open(ichk, 'rb') as f:
        simulation.context.loadCheckpoint(f.read())

#Drude VirtualSites
simulation.context.computeVirtualSites()
# print out energy of initial configuration
state=simulation.context.getState(getPositions=True, getForces=True, getEnergy=True, getVelocities=True)
print("\nInitial system energy")
print(state.getPotentialEnergy().value_in_unit(kilocalories_per_mole))
print(state.getKineticEnergy().value_in_unit(kilocalories_per_mole))
#print(state.getVelocities())

if cnt == 1:
    mini_nstep = 5000
    print(f"Minimizing {mini_nstep} steps...")
    simulation.minimizeEnergy(maxIterations=mini_nstep)
    print("Saving minimized pdb...")
    positions = simulation.context.getState(getPositions=True).getPositions()
    PDBFile.writeFile(simulation.topology, positions, open(omin,'w'))
    state1=simulation.context.getState(getPositions=True, getForces=True, getEnergy=True, getVelocities=True)
    print("\nenergy after mini (pot, kin)")
    print(state1.getPotentialEnergy().value_in_unit(kilocalories_per_mole))
    print(state1.getKineticEnergy().value_in_unit(kilocalories_per_mole))
    #print(state.getVelocities())

#Production
nstep = 200 #200000 #nstep*timestep = simulation time
simulation.reporters.append(DCDReporter(odcd, 200))
simulation.reporters.append(StateDataReporter(sys.stdout, 100, step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, volume=True))
print("Running dynmamics...")
print(f"MD run: {nstep} steps")
simulation.step(nstep)

# Write restart file
state = simulation.context.getState( getPositions=True, getVelocities=True )
with open(orst, 'w') as f:
    f.write(XmlSerializer.serialize(state))
with open(ochk, 'wb') as f:
    f.write(simulation.context.createCheckpoint())
crd = simulation.context.getState(getPositions=True).getPositions()
PDBFile.writeFile(simulation.topology, crd, open(opdb, 'w'))
