import argparse
import os
import re
import time
from pathlib import Path

import MDAnalysis
import numpy as np

try:
    from cbchelpers.cond_helpers import Charges
    from cbchelpers.helpers import msd_com
except ModuleNotFoundError as e:
    print("Did you install the cbchelpers in your protex environment?")
    print("git clone git@github.com:florianjoerg/cbchelpers.git")
    print("cd cbchelpers")
    print("pip install .")
    raise e
try:
    from newanalysis.diffusion import unfoldTraj
except ModuleNotFoundError as e:
    print("Did you install the newanalysis module in your protex environment?")
    print("git clone git@github.com:cbc-univie/mdy-newanalysis-package.git")
    print("cd mdy-newanalysis-packagei/newanalysis_source")
    print("python setup.py install")
    raise e


# python msd_charged.py [--insidebash]

############
# argparse
###########
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--insidebash",
    help="use if script is run inside another script",
    action="store_true",
)
parser.add_argument(
    "-s",
    "--short",
    help="non verbose option, no printing of current frame calculation",
    action="store_true",
)
args = parser.parse_args()

#################################################################################################################################
# Trajectory
#################################################################################################################################
firstfile = 1
lastfile = 100
skip = 1

identifier = "marta"
if identifier:
    identifier = "_" + identifier

if args.insidebash:
    # automatically tries to find correct psf and dcd files.
    # needed structure: dir1/dir2, dir1/traj/polarizable_nvt
    # file must be executed from within dir2
    # psf needs to be in dir1

    # path = pathlib.Path(__file__).parent.absolute #path for dir of script being run
    path = Path().absolute()  # current working dir
    base = path.parent
    try:
        psf_generator = base.glob("*.psf")
        psf = list(psf_generator)[0]
        print("PSF file:", psf)
    except FileNotFoundError:
        print("Error, no psf found")
    try:
        dcd_base = base / "traj/"
        # f = [f for f in os.listdir(dcd_base) if re.match("[a-zA-Z0-9]+_[a-zA-Z0-9]+_[0-9]+_[a-zA-Z0-9]+_[a-zA-Z0-9]+_[0-9]+_nvt_1.dcd", f)]
        f = [f for f in os.listdir(dcd_base) if re.match("nvt_1.dcd", f)]
        file_base = f[0][:-5]
        print("DCD base:", dcd_base / file_base)
        dcd = [
            "%s%d.dcd" % (dcd_base / file_base, i)
            for i in range(firstfile, lastfile + 1)
        ]
    except FileNotFoundError:
        print("Error, no dcd found")
    try:
        charges_base = base / "out/"
        file_base = "charge_changes"
        print("Charge base:", charges_base / file_base)
        charge_files = [
            f"{charges_base}/{file_base}_{i}.out"
            for i in range(firstfile, lastfile + 1)
        ]
    except FileNotFoundError as err:
        print("Error, charges file not found")
        print(err)

else:
    # check manually for right paths to psf and dcd files!
    base = "/site/raid4/florian/pt/test_pt/test_complete/"
    psf = base + "im1h_oac_150_im1_hoac_350.psf"
    dcd_base = "traj/nvt_"
    dcd = ["%s%d.dcd" % (base + dcd_base, i) for i in range(firstfile, lastfile + 1)]
    charges_base = base + "/out/"
    file_base = "charge_changes"
    charge_files = [
        f"{charges_base}/{file_base}_{i}.out" for i in range(firstfile, lastfile + 1)
    ]
    print("PSF file:", psf)
    print("DCD base:", base + dcd_base)
    print("charge base:", base + charges_base + file_base)

print("Using dcd files from", firstfile, "to", lastfile)
print("Skip is", skip)


u = MDAnalysis.Universe(psf, dcd)
boxl = np.float64(round(u.coord.dimensions[0], 4))
dt = round(u.trajectory.dt, 4)

# n = u.trajectory.numframes//skip  # used integer division (//) instead of (/) -> so dtype is int, otherwise problems with line 42 com_cat=np.zeros(...)
n = int(u.trajectory.n_frames / skip)
if u.trajectory.n_frames % skip != 0:
    n += 1
print("Frames to process:", n)

print("boxl", boxl)


charges = Charges()
charges.from_file(charge_files)
charges.info()
charge_steps = charges.data.keys()
assert len(charge_steps) == len(
    u.trajectory
), f"{len(charge_steps)=}, {len(u.trajectory)=}"
# print(charges.data)
# assert charges.charges_for_step(0) == charges.charges_for_step(1)
# assert charges.charges_for_step(100000) == charges.charges_for_step(99999)
# assert charges.charges_for_step(0) != charges.charges_for_step(100000)

points = int(charges.time_between_updates / (dt * skip))
print(f"{points=}, {charges.steps_between_updates=}")
#################################################################################################################################
# molecular species
#################################################################################################################################
residues = ["IM1H", "OAC", "IM1", "HOAC"]

coms = {}
sel = {}
n_res = {}
n_at = {}
apr = {}
idx_firstlast = {}
idx = 0
for residue in residues:
    sel[residue] = u.select_atoms(f"resname {residue}")
    n_res[residue] = sel[residue].n_residues
    n_at[residue] = sel[residue].n_atoms
    apr[residue] = sel[residue].n_atoms // n_res[residue]
    idx_firstlast[residue] = [idx]
    idx += n_res[residue]
    idx_firstlast[residue].append(idx)
    # masses[residue] = sel[residue].masses
    # charges[residue] = sel[residue].charges
    # apr[residue] = atomsPerResidue(sel[residue])
    # rfa[residue] = residueFirstAtom(sel[residue])
    # coms[residue] = np.zeros((n,n_res[residue],3), dtype=np.float64)
    # md[residue] = np.zeros((n,3),dtype=np.float64)
    print(
        f"Number of {residue}: {n_res[residue]} ranging from {idx_firstlast[residue]}"
    )

assert n_res["IM1H"] == n_res["OAC"]
assert apr["IM1H"] == apr["IM1"]
assert apr["OAC"] == apr["HOAC"]

sel_im = u.select_atoms(
    "(resname IM1H and not name H7) or (resname IM1 and not name H7)"
)
sel_im2 = u.select_atoms("resname IM1H or resname IM1")
sel_ac = u.select_atoms("(resname OAC and not name H) or (resname HOAC and not name H)")
print(sel_im.n_atoms)
print(sel_im2.n_atoms)
n_im = sel_im.n_residues
n_ac = sel_ac.n_residues

com_im = np.zeros((n, n_im, 3), dtype=np.float64)
com_im2 = np.zeros((n, n_im, 3), dtype=np.float64)
com_im1h = np.zeros((n, n_im, 3), dtype=np.float64)
com_im1 = np.zeros((n, n_im, 3), dtype=np.float64)
com_ac = np.zeros((n, n_ac, 3), dtype=np.float64)
com_oac = np.zeros((n, n_ac, 3), dtype=np.float64)
com_hoac = np.zeros((n, n_ac, 3), dtype=np.float64)

charges_im = np.zeros((n, n_im), dtype=np.float64)
charges_ac = np.zeros((n, n_ac), dtype=np.float64)


#################################################################################################################################
# Running through trajectory
#################################################################################################################################
ctr = 0

start = time.time()
print()
for ts, step in zip(u.trajectory[::skip], list(charge_steps)[::skip]):
    if not args.short:
        print(
            "\033[1AFrame %d of %d" % (ts.frame, len(u.trajectory)),
            "\tElapsed time: %.2f hours" % ((time.time() - start) / 3600),
        )

    # efficiently calculate center-of-mass coordinates
    # coor_an  = sel_an.positions #not needed any more?
    # coor_cat = sel_cat.positions
    com_im[ctr] = sel_im.center_of_mass(compound="residues")
    # com_im2[ctr] = sel_im2.center_of_mass(compound="residues")
    # print(np.testing.assert_allclose(com_im[ctr], com_im2[ctr]))
    # quit()
    com_ac[ctr] = sel_ac.center_of_mass(compound="residues")
    # ts.frame is 0-based, therefore i.e. 10000 is already frame 10001, hence use ts.frame+1
    # charges_tmp = np.asarray(charges.charges_for_step(ts.frame + 1))
    charges_tmp = np.asarray(charges.data[step])
    charges_im[ctr] = charges_tmp[
        np.r_[
            idx_firstlast["IM1H"][0] : idx_firstlast["IM1H"][1],
            idx_firstlast["IM1"][0] : idx_firstlast["IM1"][1],
        ]
    ]
    charges_ac[ctr] = charges_tmp[
        np.r_[
            idx_firstlast["OAC"][0] : idx_firstlast["OAC"][1],
            idx_firstlast["HOAC"][0] : idx_firstlast["HOAC"][1],
        ]
    ]
    assert sum(charges_im[ctr]) + sum(charges_ac[ctr]) == 0

    com_im1h[ctr] = np.copy(com_im[ctr])
    # set everything zero where im1 and not im1h is
    com_im1h[ctr][charges_im[ctr] == 0] = 0

    com_im1[ctr] = np.copy(com_im[ctr])
    # set everything zero where im1h and not im1 is
    com_im1[ctr][charges_im[ctr] == 1] = 0

    com_oac[ctr] = np.copy(com_ac[ctr])
    com_oac[ctr][charges_ac[ctr] == 0] = 0

    com_hoac[ctr] = np.copy(com_ac[ctr])
    com_hoac[ctr][charges_ac[ctr] == -1] = 0

    # quit()
    ctr += 1


#################################################################################################################################
# Post-processing
#################################################################################################################################
print("unfolding coordinates ..")
unfoldTraj(com_im, boxl)
unfoldTraj(com_ac, boxl)

time_axis = np.arange(0, n * skip * dt, skip * dt)

msdfolder = "./msd"
Path(msdfolder).mkdir(parents=True, exist_ok=True)

print("calculating msd ..")
msd_im = msd_com(com_im)
np.savetxt(
    f"{msdfolder}/msd_im_{firstfile}_{lastfile}_{skip}{identifier}.dat",
    np.c_[time_axis, msd_im],
)
msd_ac = msd_com(com_ac)
np.savetxt(
    f"{msdfolder}/msd_ac_{firstfile}_{lastfile}_{skip}{identifier}.dat",
    np.c_[time_axis, msd_ac],
)

msds_im1h = []
msds_oac = []
msds_im1 = []
msds_hoac = []
for residue in range(len(com_im1h[1])):
    ts_res_im1h = com_im1h[:, residue, :]
    # print(np.split(ts_res, np.where(ts_res==0)[0]))
    # print([x[x!=0] for x in np.split(ts_res, np.where(ts_res==0)[0]) if x[x!=0].size != 0])
    # print([x[x!=0].reshape(-1,3)[:,np.newaxis] for x in np.split(ts_res, np.where(ts_res==0)[0]) if x[x!=0].size != 0])

    # split the array where there is zeros (-> not im1h) and get separate arrays, then unfold them if there are elements inside
    arrs_im1h = [
        unfoldTraj(x[x != 0].reshape(-1, 3)[:, np.newaxis], boxl)
        for x in np.split(ts_res_im1h, np.where(ts_res_im1h == 0)[0])
        if x[x != 0].size != 0
    ]
    if arrs_im1h:
        for arr in arrs_im1h:
            # for all arrays calculate the msd
            msd = msd_com(arr)
            msds_im1h.append(msd)

    ts_res_oac = com_oac[:, residue, :]
    arrs_oac = [
        unfoldTraj(x[x != 0].reshape(-1, 3)[:, np.newaxis], boxl)
        for x in np.split(ts_res_oac, np.where(ts_res_oac == 0)[0])
        if x[x != 0].size != 0
    ]
    if arrs_oac:
        for arr in arrs_oac:
            # for all arrays calculate the msd
            msd = msd_com(arr)
            msds_oac.append(msd)

    ts_res_im1 = com_im1[:, residue, :]
    arrs_im1 = [
        unfoldTraj(x[x != 0].reshape(-1, 3)[:, np.newaxis], boxl)
        #for x in np.split(ts_res_im1, np.where(ts_res_im1 == 1)[0])
        for x in np.split(ts_res_im1, np.where(ts_res_im1 == 0)[0])
        if x[x != 0].size != 0
    ]
    if arrs_im1:
        for arr in arrs_im1:
            # for all arrays calculate the msd
            msd = msd_com(arr)
            msds_im1.append(msd)

    ts_res_hoac = com_hoac[:, residue, :]
    arrs_hoac = [
        unfoldTraj(x[x != 0].reshape(-1, 3)[:, np.newaxis], boxl)
        #for x in np.split(ts_res_hoac, np.where(ts_res_hoac == -1)[0])
        for x in np.split(ts_res_hoac, np.where(ts_res_hoac == 0)[0])
        if x[x != 0].size != 0
    ]
    if arrs_hoac:
        for arr in arrs_hoac:
            # for all arrays calculate the msd
            msd = msd_com(arr)
            msds_hoac.append(msd)

# caculate the average of all msd for each residue (pieces)
def calc_avg(msds: list[list]):
    max_len = len(max(msds, key=len))
    minimum_length = int(max_len * 0.5)
    print(f"{max_len=}, {minimum_length=}")
    try:
        averages = [
            np.mean([x[i] for x in msds if (len(x) > i and len(x) > minimum_length)])
            for i in range(max_len)
        ]
    except RuntimeWarning:
        minimum_length = int(max_len * 0.2)
        averages = [
            np.mean([x[i] for x in msds if (len(x) > i and len(x) > minimum_length)])
            for i in range(max_len)
        ]
    print(f"{len(averages)=}, {len(time_axis)=}")
    return averages


averages = calc_avg(msds_im1h)
np.savetxt(
    f"{msdfolder}/msd_im1h_{firstfile}_{lastfile}_{skip}{identifier}.dat",
    np.c_[time_axis[0 : len(averages)], averages],
)

averages = calc_avg(msds_oac)
np.savetxt(
    f"{msdfolder}/msd_oac_{firstfile}_{lastfile}_{skip}{identifier}.dat",
    np.c_[time_axis[0 : len(averages)], averages],
)
averages = calc_avg(msds_im1)
np.savetxt(
    f"{msdfolder}/msd_im1_{firstfile}_{lastfile}_{skip}{identifier}.dat",
    np.c_[time_axis[0 : len(averages)], averages],
)

averages = calc_avg(msds_hoac)
np.savetxt(
    f"{msdfolder}/msd_hoac_{firstfile}_{lastfile}_{skip}{identifier}.dat",
    np.c_[time_axis[0 : len(averages)], averages],
)

quit()
