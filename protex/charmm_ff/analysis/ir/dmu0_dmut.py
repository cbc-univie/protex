from __future__ import print_function
import MDAnalysis
from newanalysis.correl import correlateParallel
from newanalysis.functions import atomsPerResidue, residueFirstAtom # "apr", "rfa", needed for faster calculation of centerOfMassByResidue, dipoleMomentByResidue
from newanalysis.functions import centerOfMassByResidue, dipoleMomentByResidue
#from MDAnalysis.newanalysis.correl import correlateParallel
import numpy as np
import time
import sys
import json

print("~"*120)
print("< dmu/dt(0) * dmu/dt(t) >")
print("~"*120)

#################################################################################
class InputClass:
#################################################################################
    def __init__(self):
        self.directory = ''
        self.psf = ''
        self.dcd = ''
        self.vel = ''
        self.firstfile = None
        self.lastfile  = None
        self.skip = 1
        self.molecules = {}
        self.coor = None
        
    def info(self):
        print('< Trajectory:')
        print('\tPSF = %s\n'%(self.directory+self.psf))

        if self.coor:
            for i in range(self.firstfile,self.lastfile+1):
                print('\tDCD = %s'%(self.directory+self.dcd+str(i)+'.dcd'))
            print('\n\t\tSkip = %s'%self.skip)
        else:
            for i in range(self.firstfile,self.lastfile+1):
                print('\tVEL = %s'%(self.directory+self.dcd+str(i)+'.vel'))
            print('\n\t\tSkip = %s'%self.skip)

        print('\n<Molecules:')
        for i in sorted(self.molecules.keys()):
            print('\tType = %8s: Resname = %8s'%(i,self.molecules[i]))
            
    def fromJson(self,data):
        if "directory" in data:
            self.directory = data["directory"]
        if "psf" in data:
            self.psf = data["psf"]
        if "trajectory" in data:
            if "dcd" in data["trajectory"]:
                self.coor = True
                try:
                    self.dcd       = data["trajectory"]["dcd"]
                    self.firstfile = int(data["trajectory"]["first"])
                    self.lastfile  = int(data["trajectory"]["last"])
                    self.skip      = data["trajectory"]["skip"]
                except KeyError:
                    print('!\ Error!')
                    print('\tTrajectory information incomplete!')
                    sys.exit()
            elif "vel" in data["trajectory"]:
                self.coor = False
                try:
                    self.vel       = data["trajectory"]["vel"]
                    self.firstfile = int(data["trajectory"]["first"])
                    self.lastfile  = int(data["trajectory"]["last"])
                    self.skip      = data["trajectory"]["skip"]
                except KeyError:
                    print('!\ Error!')
                    print('\tTrajectory information incomplete!')
                    sys.exit()
            else:
                print('!\ Error!')
                print('\tTrajectory information incomplete!')
                sys.exit()

        if "molecules" in data:
            for i in data["molecules"]:
                self.molecules[i["key"]] = i["resname"]
              
    def toJson(self):
        f = open('dmu_dmu.json','w')
        f.write('{\n')
        f.write('\t"directory" : "%s",\n'%self.directory)
        f.write('\t"trajectory": { ')
        f.write('"dcd": "%s", '%self.dcd)
        f.write('"first": %s, '%self.firstfile)
        f.write('"last": %s, '%self.lastfile)
        f.write('"skip": %s },\n\n'%self.skip)

        f.write('\t"molecules":\n')
        f.write('\t[\n')

        maxkey = len(self.molecules.keys())-1
        for ctr,i in enumerate(sorted(self.molecules.keys())):
            f.write('\t\t{ "key": "%s", "resname": "%s" }'%(i,self.molecules[i][0]))
            if ctr == maxkey:
                f.write('\n')
            else:
                f.write(',\n')
        f.write('\t]\n')
        f.write('}\n')
        f.close()
        
    def ExampleInput(self):
        self.directory = "/site/raid1/veronika/STUDENTS/Elen_Neureiter/data_raid9/ionpair_gasphase/"
        self.psf = "prol_bdoa.psf"
        self.dcd = "prol_bdoa_"
        self.firstfile = 1
        self.lastfile  = 10
        self.skip = 1

        self.molecules["cation"]  = ["PROL",None]
        self.molecules["anion"]   = ["BDOA",None]
        self.molecules["solvent"] = ["BUOH",None]
        self.toJson()

#################################################################################
# MAIN PROGRAM
#################################################################################
input = InputClass()        
try:
    JsonFile = sys.argv[1]
    with open(JsonFile) as infile:
        data = json.load(infile)
except IndexError:
    print('\n! Error!')
    print('!\t Json input file is missing: python3 mdmd_fit.py ___.json. ')
    print('!\t Writing example input dmu_dmu.json')
    input.ExampleInput()
    sys.exit()
except FileNotFoundError:
    print('\n! Error!')
    print('!\t Json input file %s not found!'%JsonFile)
    sys.exit()

input.fromJson(data)
input.info()
    
############################################################################################################################################################
# Trajectory
############################################################################################################################################################
print('\n> Open trajectories and PSF file ...')

if input.coor:
    try:
        u=MDAnalysis.Universe(input.directory+input.psf,["%s%d.dcd" % (input.directory+input.dcd,i) for i in range(input.firstfile,input.lastfile+1)])
    except FileNotFoundError:
        print('\n! Error')
        print('\tDCD or PSF file not found!')
        sys.exit()
else:
    try:
        u=MDAnalysis.Universe(input.directory+input.psf,["%s%d.vel" % (input.directory+input.vel,i) for i in range(input.firstfile,input.lastfile+1)])
    except FileNotFoundError:
        print('\n! Error')
        print('\tVEL or PSF file not found!')
        sys.exit()


boxl = np.float64(round(u.coord.dimensions[0],4))
dt=round(u.trajectory.dt,4)

n = int(u.trajectory.n_frames/input.skip)
if u.trajectory.n_frames%input.skip != 0:
    n+=1

############################################################################################################################################################
# molecular species
############################################################################################################################################################
print('\n> Defining selections ...')
selection = {}
number_of_residues = {}
mass = {}
charge = {}
mu = {}
apr = {}
rfa = {}

for i in sorted(input.molecules.keys()):
    resname = input.molecules[i]
    tmp = u.select_atoms('resname '+resname)
    selection[resname] = tmp
    
    tmp_number = tmp.n_residues
    number_of_residues[resname] = tmp_number
    mass[resname] = tmp.masses
    charge[resname] = tmp.charges
    apr[resname] = atomsPerResidue(tmp) # needed as kwargs for centerOfMassByresidue, dipoleMomentByResidue
    rfa[resname] = residueFirstAtom(tmp) # needed as kwargs for centerOfMassByresidue, dipoleMomentByResidue

    if tmp_number >0:
        mu[resname] = np.zeros((n,tmp_number,3))
        print('\t %8s: %6s molecules'%(input.molecules[i],tmp_number))
    else:
        del selection[resname]

############################################################################################################################################################
# Reading trajectory
############################################################################################################################################################
ctr=0
start=time.time()
print('\n')
for ts in u.trajectory[::input.skip]:
    print("\033[1A> Frame %d of %d" % (ts.frame,u.trajectory.n_frames), "\tElapsed time: %.2f hours" % ((time.time()-start)/3600))
    for i in selection.keys():
        tmp = selection[i]

        if input.coor:
            coor = np.ascontiguousarray(tmp.positions, dtype='double') #array shape needed for C
            com = centerOfMassByResidue(tmp, coor=coor, masses=mass[i], apr=apr[i], rfa=rfa[i])
            #com = tmp.center_of_mass(compound='residues')
        else:
            coor = np.contiguousarray(tmp.velocities, dtype='double')
            com = centerOfMassByResidue(tmp, coor=coor, masses=mass[i], apr=apr[i], rfa=rfa[i])
            #com = tmp.center_of_mass(compound='residues') #centerOfMassByResidue(coor=coor,masses=mass[i])
        mu[i][ctr] = dipoleMomentByResidue(tmp, coor=coor,charges=charge[i],masses=mass[i],com=com, apr=apr[i], rfa=rfa[i], axis=0)
        #mu[i][ctr] = tmp.dipoleMomentByResidue(coor=coor,charges=charge[i],masses=mass[i],com=com,axis=0)
    ctr+=1

############################################################################################################################################################
# Post processing
############################################################################################################################################################
print("\nPost-processing ...")

def mu_from_dcd(mu,nmol):
    corfun   = np.zeros(n)
    corfun_i = np.zeros(n)
    mudot = np.zeros((n,3))
    for i in range(nmol):
        mudot[:,0] = np.gradient(mu[:,i,0],input.skip*dt)
        mudot[:,1] = np.gradient(mu[:,i,1],input.skip*dt)
        mudot[:,2] = np.gradient(mu[:,i,2],input.skip*dt)
        correlateParallel(mudot.T,mudot.T,corfun_i,ltc=1)
        corfun[:] += corfun_i[:]/nmol
    return corfun
  
def mu_from_vel(mu,nmol):
    corfun   = np.zeros(n)
    corfun_i = np.zeros(n)
    mudot = np.zeros((n,3))
    for i in range(nmol):
        mudot = mu[:,i,:]
        correlateParallel(mudot.T,mudot.T,corfun_i,ltc=2)
        corfun[:] += corfun_i[:]/nmol
    return corfun
        
def apodization(alpha,time,corfun):
    # J. Non.-Cryst. Solids (1992), 140, 350
    # Phys. Rev. Lett. (1996), 77, 4023
    corfun_shortened = []
    
    for i in range(len(time)):
        # apodization
        corfun_shortened.append(corfun[i] * np.exp(-alpha*time[i]*time[i]))

        # 100 ps simulations corresponds to 0.2 cm^-1 resolution on IR spectrum
        if time[i]>100.0:
            break
    xlen = i
    corfun_shortened.pop()
    
    time_shortened   = time[0:xlen]
    return time_shortened, corfun_shortened

def harmonic_quantum_correction(frequencies,fourier):
    # beta = 1. / (1.3806 10^-23 J/K * 300 K )
    # hbar = 1.0545718*10^-34 J/s
    # factor = beta * hbar ( 10^12 ps/s)
    factor = 0.0254617
    for i in range(len(frequencies)):
        if frequencies[i]==0.:
            continue
        q = factor * frequencies[i] / (1.0 - np.exp(-factor*frequencies[i]))
        fourier[i] *= q
    return fourier

def laplacetransform(time,corfun):
    xlen = len(time)
    x = np.zeros(2*xlen-1)
    y = np.zeros(2*xlen-1)
    x[0:xlen]  = -time[::-1]
    x[xlen-1:] = time
    y[xlen-1:] = corfun

    #   simple Fourier transform    
    dt = x[1]-x[0]
    freq = 1./dt
    fourier = np.fft.fft(y)/freq
    N = int(len(fourier)/2)+1
    frequencies = np.linspace(0,freq/2,N,endpoint=True)
    return frequencies, fourier

############################################################################################################################################################
for i in sorted(selection.keys()):
    time = np.linspace(0.0,n*input.skip*dt,n)
    print('\n>\tCorrelation for %s'%i)
    nmol = number_of_residues[i]

    if input.coor:
        corfun = mu_from_dcd(mu[i],nmol)
    else:
        corfun = mu_from_vel(mu[i],nmol)
        
    f1 = open(i+'.mudot_mudot',"w")
    for j in range(n):
        f1.write("%10.5f %10.5f\n" % (j*input.skip*dt, corfun[j]))
    f1.close()
    
    alpha = 0.002
    print('\t\tApodization of correlation function with alpha = %10.3f'%alpha)
    time, corfun = apodization(alpha,time,corfun)
    
    print('>\tLaplace transform for %s'%i)
    frequencies, fourier = laplacetransform(time,corfun)

    #fourier = harmonic_quantum_correction(frequencies,fourier)
    
    THz2cm = 33.356  
    f1 = open(i+'.IR','w')
    flen = int(len(fourier)/2)+1
    for j in range(flen):
        f1.write("%10.4f %10.4f\n" %(frequencies[j]*THz2cm,np.abs(fourier[j])))
    f1.close()

