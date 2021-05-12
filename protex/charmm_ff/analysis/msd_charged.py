from __future__ import print_function
import MDAnalysis
import numpy as np
from newanalysis.diffusion import msdCOM, unfoldTraj, msdMJ
#from MDAnalysis.newanalysis.diffusion import msdCOM, msdMJ, unfoldTraj
import os
import time
from pathlib import Path
import re
import argparse

#for use with conda env "mda_py3"
#python msd_charged.py [--insidebash]

############
# argparse
###########
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--insidebash", help="use if script is run inside another script", action="store_true")
parser.add_argument("-s", "--short", help="non verbose option, no printing of current frame calculation", action="store_true")
args = parser.parse_args()

#################################################################################################################################
# Trajectory
#################################################################################################################################
firstfile=1
lastfile=100
skip=10

if args.insidebash:
    # automatically tries to find correct psf and dcd files.
    # needed structure: dir1/dir2, dir1/traj/polarizable_nvt
    # file must be executed from within dir2
    # psf needs to be in dir1

    #path = pathlib.Path(__file__).parent.absolute #path for dir of script being run
    path = Path().absolute() #current working dir
    base = path.parent
    try:
        psf_generator = base.glob("*.psf")
        psf = list(psf_generator)[0]
        print("PSF file:", psf)
    except:
        print("Error, no psf found")
    try:
        dcd_base = base / "traj/"
        f = [f for f in os.listdir(dcd_base) if re.match("[a-zA-Z0-9]+_[a-zA-Z0-9]+_[0-9]+_[a-zA-Z0-9]+_[a-zA-Z0-9]+_[0-9]+_nvt_1.dcd", f)]
        file_base = f[0][:-5]
        print("DCD base:", dcd_base/file_base)
        dcd = ["%s%d.dcd" % (dcd_base/file_base,i) for i in range(firstfile,lastfile+1)]
    except:
        print("Error, no dcd found")

else:    
    # check manually for right paths to psf and dcd files!

    #base="/site/raid11a/student3/Florian_Joerg/python_ph_1/"
    base="/site/raid2/florian/pt/tests/npt_nvt_sd200/"
    
    psf=base+"im1h_oac_200_im1_hoac_300.psf"
    dcd_base = "traj/im1h_oac_200_im1_hoac_300_nvt_"
    dcd = ["%s%d.dcd" % (base+dcd_base,i) for i in range(firstfile,lastfile+1)]
    print("PSF file:", psf)
    print("DCD base:", base+dcd_base)

print("Using dcd files from",firstfile, "to", lastfile)
print("Skip is", skip)

u=MDAnalysis.Universe(psf,dcd)
boxl = np.float64(round(u.coord.dimensions[0],4))
dt=round(u.trajectory.dt,4)

#n = u.trajectory.numframes//skip  # used integer division (//) instead of (/) -> so dtype is int, otherwise problems with line 42 com_cat=np.zeros(...)
n = int(u.trajectory.n_frames/skip)
if u.trajectory.n_frames%skip != 0:
    n+=1

print("boxl", boxl)
#################################################################################################################################
# molecular species
#################################################################################################################################
sel_cat = u.select_atoms("resname IM1H")
sel_an  = u.select_atoms("resname OAC")

ncat = sel_cat.n_residues
nan  = sel_an.n_residues

mass_cat=sel_cat.masses
mass_an =sel_an.masses

print("Number of cations",ncat)
print("Number of anions ",nan )


com_cat = np.zeros((n,ncat,3),dtype=np.float64)
com_an  = np.zeros((n,nan, 3),dtype=np.float64)

#################################################################################################################################
# Running through trajectory
#################################################################################################################################
ctr=0

start=time.time()
print("")

for ts in u.trajectory[::skip]:
    if not args.short:
        print("\033[1AFrame %d of %d" % (ts.frame,len(u.trajectory)), "\tElapsed time: %.2f hours" % ((time.time()-start)/3600))
    
    # efficiently calculate center-of-mass coordinates
    #coor_an  = sel_an.positions #not needed any more?
    #coor_cat = sel_cat.positions
    com_an[ctr] = sel_an.center_of_mass(compound='residues')
    com_cat[ctr]= sel_cat.center_of_mass(compound='residues')

    ctr+=1

#################################################################################################################################
# Post-processing
#################################################################################################################################
    
print("unfolding coordinates ..")
unfoldTraj(com_cat,boxl)
unfoldTraj(com_an, boxl)

print("calculating msd ..")
msd_cat = msdCOM(com_cat)
msd_an  = msdCOM(com_an )
filename1 = "im1h_msd_{}-{}_skip{}.dat".format(firstfile, lastfile, skip)
filename2 = "oac_msd_{}-{}_skip{}.dat".format(firstfile, lastfile, skip)
f1=open(filename1,'w')
f2=open(filename2,'w')

for i in range(len(msd_cat)):
    f1.write("%5.5f\t%5.5f\n" % (i*skip*dt, msd_cat[i]))
    f2.write("%5.5f\t%5.5f\n" % (i*skip*dt, msd_an[i]))
f1.close()
f2.close()

print("calculating msdMJ ..")
msdmj = msdMJ(com_cat,com_an)
filename1 = "msdMJ_{}-{}_skip{}.dat".format(firstfile, lastfile, skip)
f1=open(filename1,'w')
for i in range(len(msdmj)):
    f1.write("%5.5f\t%5.5f\n" % (i*skip*dt, msdmj[i]))
f1.close()
