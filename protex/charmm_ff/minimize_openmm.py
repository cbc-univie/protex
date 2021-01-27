#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Minimization of PACKMOL structure
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#general Imports
from sys import stdout

#OpenMM Imports
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

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

# add Drudes, LP, ... to system
#modeller = Modeller(psf.topology, crd.positions)
#modeller.addExtraParticles(params)

# Create the OpenMM system
print('Creating OpenMM System')
#boxlength
xtl = 48.0*angstroms
psf.setBox(xtl,xtl,xtl)
system = psf.createSystem(params, nonbondedMethod=PME, 
                                  nonbondedCutoff=11.0*angstroms, 
                                  switchDistance=10*angstroms,
                                  constraints=HBonds,
)

integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.005*picoseconds)
simulation = Simulation(psf.topology, system, integrator)
simulation.context.setPositions(crd.positions)

print("Minimizing...")
simulation.minimizeEnergy(maxIterations=100)
print("Saving...")
positions = simulation.context.getState(getPositions=True).getPositions()
PDBFile.writeFile(simulation.topology, positions, open('mini_out.pdb','w'))

#Specify stuff to save
simulation.reporters.append(PDBReporter('output.pdb', 10))
simulation.reporters.append(DCDReporter('output.dcd', 10))
simulation.reporters.append(StateDataReporter(stdout, 1, step=True, potentialEnergy=True, temperature=True, time=True,volume=True, density=True ))
print("Running dynmamics...")
simulation.step(10)
