import argparse
import sys

from simtk.openmm import DrudeNoseHooverIntegrator, MonteCarloBarostat
from simtk.openmm.app import (PME, CharmmCrdFile, CharmmParameterSet,
                              CharmmPsfFile, HBonds, Simulation)
from simtk.unit import angstroms, atmosphere, kelvin, picoseconds

# user specified
sys.path.append("/home/florian/pythonfiles")
from pystat import read_data

parser = argparse.ArgumentParser()
parser.add_argument('-cnt', dest='counter', help='Counter variable fpr consecutive runs', required=True)
#parser.add_argument('-i', dest='inpfile', help='Input parameter file', required=True)
#parser.add_argument('-p', dest='psffile', help='Input CHARMM PSF file', required=True)
#parser.add_argument('-c', dest='crdfile', help='Input CHARMM CRD file (EXT verison)', required=True)
#parser.add_argument('-t', dest='toppar', help='Input CHARMM-GUI toppar stream file', required=True)
parser.add_argument('-n', dest='name', help='Input base name of dcd and restart files', required=True)

#parser.add_argument('-b', dest='sysinfo', help='Input CHARMM-GUI sysinfo stream file (optional)', default=None)
#parser.add_argument('-icrst', metavar='RSTFILE', dest='icrst', help='Input CHARMM RST file (optional)', default=None)
parser.add_argument('-irst', metavar='RSTFILE', dest='irst', help='Input restart file (optional)', default=None)
#parser.add_argument('-ichk', metavar='CHKFILE', dest='ichk', help='Input checkpoint file (optional)', default=None)
parser.add_argument('-opdb', metavar='PDBFILE', dest='opdb', help='Output PDB file (optional)', default=None)
#parser.add_argument('-minpdb', metavar='MINIMIZEDPDBFILE', dest='minpdb', help='Output PDB file after minimization (optional)', default=None)
parser.add_argument('-orst', metavar='RSTFILE', dest='orst', help='Output restart file (optional)', default=None)
#parser.add_argument('-ochk', metavar='CHKFILE', dest='ochk', help='Output checkpoint file (optional)', default=None)
parser.add_argument('-odcd', metavar='DCDFILE', dest='odcd', help='Output trajectory file (optional)', default=None)
args = parser.parse_args()
f_name = args.name #+ '_npt'
if not any([args.orst, args.ochk]):
    args.orst = "traj/" + f_name + '_' + args.counter + '.rst'
if not args.odcd:
    args.odcd = "traj/" + f_name + '_' + args.counter + '.dcd'
if not args.opdb:
    args.opdb = f_name + '.pdb'

pcnt = int(args.counter) - 1
if int(args.counter) > 1 and not any([args.icrst, args.irst, args.ichk]):
    args.irst = "traj/" + f_name + '_' + str(pcnt) + '.rst'


print("Loading CHARMM files...")
PARA_FILES = [
    "toppar_drude_master_protein_2013f_lj02.str",
    "hoac_d.str",
    "im1h_d_fm_lj_lp.str",
    "im1_d_fm_lj_dummy_lp.str",
    "oac_d_lj.str",
]
base = "../charmm_ff" 
params = CharmmParameterSet(
    *[f"{base}/toppar/{para_files}" for para_files in PARA_FILES]
)

psf = CharmmPsfFile(f"{base}/im1h_oac_150_im1_hoac_350_lp.psf")
# cooridnates can be provieded by CharmmCrdFile, CharmmRstFile or PDBFile classes
crd = CharmmCrdFile(f"{base}/im1h_oac_150_im1_hoac_350_lp.crd")
xtl = 48.0 * angstroms
psf.setBox(xtl, xtl, xtl)
system = psf.createSystem(
    params,
    nonbondedMethod=PME,
    nonbondedCutoff=11.0 * angstroms,
    switchDistance=10 * angstroms,
    constraints=HBonds,
)

barostat = MonteCarloBarostat(1.0*atmosphere, 300*kelvin)
system.addForce(barostat)

integrator = DrudeNoseHooverIntegrator(
    300 * kelvin,
    5 / picoseconds,
    1 * kelvin,
    100 / picoseconds,
    0.0005 * picoseconds,
)
integrator.setMaxDrudeDistance(0.25 * angstroms)
simulation = Simulation(psf.topology, system, integrator)
simulation.context.setPositions(crd.positions)
#restart
with open(args.irst, 'r') as f:
    simulation.context.setState(XmlSerializer.deserialize(f.read()))
#    if inputs.ensemble == "NVT":
#        simulation.context.setPeriodicBoxVectors(psf.boxVectors[0],psf.boxVectors[1],psf.boxVectors[2])
simulation.context.computeVirtualSites()
simulation.context.setVelocitiesToTemperature(300 * kelvin)

#Minimize
if int(args.counter) == 1 and not any([args.irst, args.ichk, args.icrst]):
    simulation.minimizeEnergy(maxIterations=200)

# get average of boxl after 11 runs
if int(args.counter) == 12 and inputs.ensemble == "NPT":
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


simulation.reporters.append(DCDReporter(args.odcd, 100))
simulation.reporters.append(StateDataReporter(sys.stdout, 100, step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, volume=True))

simulation.step(200000)

if args.orst:
    state = simulation.context.getState( getPositions=True, getVelocities=True )
    with open(args.orst, 'w') as f:
        f.write(XmlSerializer.serialize(state))
if args.opdb:
    crd = simulation.context.getState(getPositions=True).getPositions()
    PDBFile.writeFile(psf.topology, crd, open(args.opdb, 'w'))
