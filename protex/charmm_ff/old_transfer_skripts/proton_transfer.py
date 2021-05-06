from __future__ import print_function
import MDAnalysis
import numpy as np
import random
import sys

#########################################################################################
# Trajectory setup
#########################################################################################
base="/site/raid11a/student3/Florian_Joerg/python_ph_1/"
try:
    dcd = base+'traj/'+sys.argv[1]
    psf = base+'psf/'+sys.argv[2]

    print("DCD = %s"%dcd)
    print("PSF = %s"%psf)
    
    u=MDAnalysis.Universe(psf,dcd)
    boxl = np.float64(round(u.coord.dimensions[0],4))
except:
    print("Files do not exist!") 
    exit()
    
charge_imidazolium = [0.3573,-0.1650,-0.0059,0.2024,0.2117,-0.1316,0.2015,-0.0556,0.2012,-0.3083,0.1641,0.1641,0.1641]
charge_imidazole = [0.0000,-0.5374,0.1205,0.0920,0.2521,-0.3853,0.1765,0.1897,0.0645,-0.3497,0.1257,0.1257,0.1257]

charge_acetate = [-0.2824,0.0095,0.0095,0.0095,0.9948,-0.8718,-0.8691,0.0000]
charge_acetic_acid = [-0.3410,0.1111,0.1111,0.1111,0.7680,-0.5735,-0.5911,0.4043]

# charged molecules
sel_HNA1 = u.selectAtoms("name HNA1")
sel_O1   = u.selectAtoms("name O1")

nHNA1 = sel_HNA1.numberOfAtoms()
nO1   = sel_O1.numberOfAtoms()

# distance between uncharged molecules
sel_NA1  = u.selectAtoms("name NA1")
sel_H3   = u.selectAtoms("name H3")

nNA1 = sel_NA1.numberOfAtoms()
nH3  = sel_H3.numberOfAtoms()


#########################################################################################
# distance including periodic boundary conditions
#########################################################################################
def rPBC(coor1,coor2,boxl):
    dx = coor1[0] - coor2[0]
    dy = coor1[1] - coor2[1]
    dz = coor1[2] - coor2[2]

    dx -= round(dx/boxl)
    dy -= round(dy/boxl)
    dz -= round(dz/boxl)
    return np.sqrt(dx*dx+dy*dy+dz*dz)

#########################################################################################
# deprotonation
#########################################################################################
def deprotonate(current_molecule,charges):
    index = 0
    for k in current_molecule:
        tmp = k.charge
        k.charge = charges[index]
        index += 1
        print("%4s: %7.4f --> %7.4f"%(k.name,tmp,k.charge))
    print("\n")
        
#########################################################################################
# Trajectory analysis
#########################################################################################
random.seed()
u.trajectory[u.trajectory.numframes-1]
coor_HNA1 = sel_HNA1.get_positions()
coor_O1   = sel_O1.get_positions()
coor_NA1  = sel_NA1.get_positions()
coor_H3   = sel_H3.get_positions()

# Deprotonation
for i in sel_HNA1:
    # looking only for protonated imidazoliums
        
    
    res_i = i.resid

    for j in sel_O1:
        # looking only for acetates
        if j.charge == charge_acetic_acid[j.id]:
            continue
        
        res_j = j.resid
        r = rPBC(coor_HNA1[res_i-1],coor_O1[res_j-1],boxl)

        if r <1.5 and np.exp(-0.5*r)>random.random():
            if i.charge == charge_imidazole[i.id]:
                continue

            print("Deprotonation %5s --> %5s"%(res_i,res_j))
            current_i = u.selectAtoms("resname IM1H and resid "+str(res_i))
            deprotonate(current_i,charge_imidazole)
            current_j = u.selectAtoms("resname OAC and resid "+str(res_j))
            deprotonate(current_j,charge_acetic_acid)

# Protonation            
for i in sel_NA1:
    # looking only for deprotonated imidazoles
            
    
    res_i = i.resid
    
    for j in sel_H3:
        # looking only for protonated acetic acids
        if j.charge == charge_acetate[j.id]:
            continue
        
        res_j = j.resid
        r = rPBC(coor_NA1[res_i-1],coor_H3[res_j-1],boxl)
        
        if r <1.7 and np.exp(-0.1*r)>random.random():
            if i.charge == charge_imidazolium[i.id]:
                continue
    
            print("Protonation %5s --> %5s"%(res_i,res_j))
            current_i = u.selectAtoms("resname IM1H and resid "+str(res_i))
            deprotonate(current_i,charge_imidazolium)
            current_j = u.selectAtoms("resname OAC and resid "+str(res_j))
            deprotonate(current_j,charge_acetate)

#########################################################################################
# Stream file for CHARMM
#########################################################################################
f = open('current_charges.str','w')
f.write('*****************************************************************\n')
f.write('* current charges \n')
f.write('*****************************************************************\n')
f.write('* \n\n')
        
for k in u.selectAtoms("all"):
    f.write('scalar charge set %7.4f sele atom * %s %s end\n'%(k.charge,k.resid,k.name))
f.close()
