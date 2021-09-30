from __future__ import print_function
import MDAnalysis
import numpy as np
from MDAnalysis.newanalysis.diffusion import msdCOM, msdMJ, unfoldTraj
import os, sys
import time
import argparse
import re
from pathlib import Path

############
# argparse
###########
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--insidebash", help="use if script is run inside another script", action="store_true")
parser.add_argument("-s", "--short", help="non verbose option, no printing of current frame calculation", action="store_true")
parser.add_argument("-n", "--nfile", help="specify first and last file")
args = parser.parse_args()


################################################################################################################################
# trajectory
################################################################################################################################
firstfile=10
lastfile=10
#firstfile=int(args.nfile)
#lastfile=int(args.nfile)
skip = 10
#if args.insidebash:
#    # automatically tries to find correct psf and dcd files.
#    # needed structure: dir1/dir2, dir1/traj/polarizable_nvt
#    # file must be executed from within dir2
#    # psf needs to be in dir1
#
#    #path = pathlib.Path(__file__).parent.absolute #path for dir of script being run
#    path = Path().absolute() #current working dir
#    base = path.parent
#    try:
#        psf_generator = base.glob("*.psf")
#        psf = list(psf_generator)[0]
#        print("PSF file:", psf)
#    except:
#        print("Error, no psf found")
#    try:
#        dcd_base = base / "traj/"
#        f = [f for f in os.listdir(dcd_base) if re.match("[a-zA-Z0-9]+_[a-zA-Z0-9]+_[0-9]+_[a-zA-Z0-9]+_[a-zA-Z0-9]+_[0-9]+_nvt_1.dcd", f)]
#        file_base = f[0][:-5]
#        print("DCD base:", dcd_base/file_base)
#        dcd = ["%s%d.dcd" % (dcd_base/file_base,i) for i in range(firstfile,lastfile+1)]
#    except:
#        print("Error, no dcd found")
#
#else:    
#    #origin: base="/site/raid1/esther/Charmm_Benchmark/properties/emim_dca_1000_04_1000/"
#    base="/site/raid2/florian/conductivity/mix_30_70/lj02/rep1/"
#    psf = base + "im1h_oac_150_im1_hoac_350.psf"
#    dcd_base = "/traj/im1h_oac_150_im1_hoac_350_nvt_"
#    dcd = ["%s%d.dcd" % (base+dcd_base,i) for i in range(firstfile,lastfile+1)]
#    print("PSF file:", psf)
#    print("DCD base:", base+dcd_base)
#    #u=MDAnalysis.Universe(base+"im1h_oac_200_im1_hoac_300_nvt.psf",["%s%d.dcd" % (base+"/traj/polarizable_npt/im1h_oac_200_im1_hoac_300_",i) for i in range(firstfile,lastfile+1)])
#print("Using dcd files from",firstfile, "to", lastfile)
#print("Skip is", skip)

psf = "../charmm_ff/im1h_oac_150_im1_hoac_350_lp.psf"
dcd = "output.dcd"
u=MDAnalysis.Universe(str(psf),dcd)
boxl=round(u.coord.dimensions[0],4)
dt=round(u.trajectory.dt,4)

n = int(u.trajectory.numframes/skip)
if u.trajectory.numframes%skip != 0:
    n+=1

print("boxl", boxl)
################################################################################################################################
# Molecules
################################################################################################################################
sel_cat = u.selectAtoms("resname IM1H")
sel_an  = u.selectAtoms("resname OAC")
sel_cat2 = u.selectAtoms("resname IM1")
sel_an2  = u.selectAtoms("resname HOAC")

ncat = sel_cat.numberOfResidues()
nan  = sel_an.numberOfResidues()
ncat2 = sel_cat2.numberOfResidues()
nan2  = sel_an2.numberOfResidues()

charges_cat = sel_cat.charges()
charges_an  = sel_an.charges()
charges_cat2 = sel_cat2.charges()
charges_an2  = sel_an2.charges()

apr_cat=sel_cat.atomsPerResidue()[0]
apr_an=sel_an.atomsPerResidue()[0]
apr_cat2=sel_cat2.atomsPerResidue()[0]
apr_an2=sel_an2.atomsPerResidue()[0]

drude_cat=np.zeros(apr_cat)
drude_cat_name=[]
for j in range(apr_cat):
    if len(u.selectAtoms("resname IM1H and resid 1").selectAtoms("bynum "+str(j+1)+" and name D*"))==1:
        drude_cat[j]=1
        drude_cat_name.append(u.selectAtoms("resname IM1H and resid 1").residues.atoms[j].name)
drude_an=np.zeros(apr_an)
drude_an_name=[]
for j in range(apr_an):
    if len(u.selectAtoms("resname OAC and resid 1").selectAtoms("bynum "+str(j+1)+" and name D*"))==1:
        drude_an[j]=1
        drude_an_name.append(u.selectAtoms("resname OAC and resid 1").residues.atoms[j].name)

drude_cat2=np.zeros(apr_cat2)
drude_cat_name2=[]
for j in range(apr_cat2):
    if len(u.selectAtoms("resname IM1 and resid 1").selectAtoms("bynum "+str(j+1)+" and name D*"))==1:
        drude_cat2[j]=1
        drude_cat_name2.append(u.selectAtoms("resname IM1 and resid 1").residues.atoms[j].name)
drude_an2=np.zeros(apr_an2)
drude_an_name2=[]
for j in range(apr_an2):
    if len(u.selectAtoms("resname HOAC and resid 1").selectAtoms("bynum "+str(j+1)+" and name D*"))==1:
        drude_an2[j]=1
        drude_an_name2.append(u.selectAtoms("resname OAC and resid 1").residues.atoms[j].name)
#fs = 0
#fe = 10
#n = int(fe -fs)
#mu_ind_cat=np.zeros((n,ncat,int(drude_cat.sum())))
#mu_ind_an =np.zeros((n,nan,int(drude_an.sum())))
d_dis_cat=np.zeros((n,ncat,int(drude_cat.sum())))
d_dis_an =np.zeros((n,nan,int(drude_an.sum())))
d_dis_cat2=np.zeros((n,ncat2,int(drude_cat2.sum())))
d_dis_an2 =np.zeros((n,nan2,int(drude_an2.sum())))
################################################################################################################################
# Analysis
################################################################################################################################
ctr=0
start=time.time()
print("")

#for ts in u.trajectory[fs:fe:skip]:
for ts in u.trajectory[::skip]:
    if not args.short:
        print("\033[1AFrame %d of %d" % (ts.frame,u.trajectory.numframes), "\tElapsed time: %.2f hours" % ((time.time()-start)/3600))
    coor_an  = sel_an.get_positions()
    coor_cat = sel_cat.get_positions()
    coor_an2  = sel_an2.get_positions()
    coor_cat2 = sel_cat2.get_positions()

    for i in range(ncat):
        ctr2=0
        for j in range(apr_cat-1):
            idx=i*apr_cat+j
            if drude_cat[j+1]==1:
                #x=charges_cat[idx+1]*coor_cat[idx+1,0]-charges_cat[idx+1]*coor_cat[idx,0]
                #y=charges_cat[idx+1]*coor_cat[idx+1,1]-charges_cat[idx+1]*coor_cat[idx,1]
                #z=charges_cat[idx+1]*coor_cat[idx+1,2]-charges_cat[idx+1]*coor_cat[idx,2]
                #mu_ind_cat[ctr,i,ctr2]=np.sqrt(x*x+y*y+z*z)
                x=coor_cat[idx+1,0]-coor_cat[idx,0]
                y=coor_cat[idx+1,1]-coor_cat[idx,1]
                z=coor_cat[idx+1,2]-coor_cat[idx,2]
                d_dis_cat[ctr,i,ctr2]=np.sqrt(x*x+y*y+z*z)
                ctr2+=1
    for i in range(nan):
        ctr2=0
        for j in range(apr_an-1):
            idx=i*apr_an+j
            if drude_an[j+1]==1:
                #x=charges_an[idx+1]*coor_an[idx+1,0]-charges_an[idx+1]*coor_an[idx,0]
                #y=charges_an[idx+1]*coor_an[idx+1,1]-charges_an[idx+1]*coor_an[idx,1]
                #z=charges_an[idx+1]*coor_an[idx+1,2]-charges_an[idx+1]*coor_an[idx,2]
                #mu_ind_an[ctr,i,ctr2]=np.sqrt(x*x+y*y+z*z)
                x=coor_an[idx+1,0]-coor_an[idx,0]
                y=coor_an[idx+1,1]-coor_an[idx,1]
                z=coor_an[idx+1,2]-coor_an[idx,2]
                d_dis_an[ctr,i,ctr2]=np.sqrt(x*x+y*y+z*z)
                ctr2+=1
    for i in range(ncat2):
        ctr2=0
        for j in range(apr_cat2-1):
            idx=i*apr_cat2+j
            if drude_cat2[j+1]==1:
                x=coor_cat2[idx+1,0]-coor_cat2[idx,0]
                y=coor_cat2[idx+1,1]-coor_cat2[idx,1]
                z=coor_cat2[idx+1,2]-coor_cat2[idx,2]
                d_dis_cat2[ctr,i,ctr2]=np.sqrt(x*x+y*y+z*z)
                ctr2+=1
    for i in range(nan2):
        ctr2=0
        for j in range(apr_an2-1):
            idx=i*apr_an2+j
            if drude_an2[j+1]==1:
                x=coor_an2[idx+1,0]-coor_an2[idx,0]
                y=coor_an2[idx+1,1]-coor_an2[idx,1]
                z=coor_an2[idx+1,2]-coor_an2[idx,2]
                d_dis_an2[ctr,i,ctr2]=np.sqrt(x*x+y*y+z*z)
                ctr2+=1
    ctr+=1
        
#with open("d_dis_{}-{}_{}_{}.dat".format(firstfile, lastfile, fs, fe), 'w') as f:
with open("d_dis_{}-{}.dat".format(firstfile, lastfile), 'w') as f:
    for i in range(int(drude_cat.sum())):
        tmp=np.ascontiguousarray(d_dis_cat[:,:,i])
        print("IM1H-"+drude_cat_name[i],np.mean(tmp),np.std(tmp,ddof=1),file=f)

    for i in range(int(drude_an.sum())):
        tmp=np.ascontiguousarray(d_dis_an[:,:,i])
        print("OAC-"+drude_an_name[i],np.mean(tmp),np.std(tmp,ddof=1),file=f)

    for i in range(int(drude_cat2.sum())):
        tmp=np.ascontiguousarray(d_dis_cat2[:,:,i])
        print("IM1-"+drude_cat_name2[i],np.mean(tmp),np.std(tmp,ddof=1),file=f)

    for i in range(int(drude_an2.sum())):
        tmp=np.ascontiguousarray(d_dis_an2[:,:,i])
        print("HOAC-"+drude_an_name2[i],np.mean(tmp),np.std(tmp,ddof=1),file=f)
