import argparse
import os
import re
import time
import warnings
from pathlib import Path

import numpy as np

#import MDAnalysis

#import tidynamics

try:
    from cbchelpers.cond_helpers import Charges, UpdatePair, get_pairs
    from cbchelpers.derivative import derivative
    from cbchelpers.helpers import rPBC
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

########################################################################################
# Trajectory
########################################################################################
firstfile = 1
lastfile = 100
skip = 1
if skip != 1:
    warnings.warn("It is probably not working. Use a skip of 1!!!", UserWarning)


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
    try:
        pair_base = base / "out/"
        file_base = "nvt"
        print("Pair base", pair_base / file_base)
        pair_files = [
            f"{pair_base}/{file_base}_{i}.out" for i in range(firstfile, lastfile + 1)
        ]
    except FileNotFoundError as err:
        print("Error, pairs file not found")
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
# charges.from_file("test.out")
charges.info()

charge_steps = charges.data.keys()
assert len(charge_steps) == len(
    u.trajectory
), f"{len(charge_steps)=}, {len(u.trajectory)=}"

points = int(
    charges.time_between_updates / (dt * skip)
)  # = steps_between_updates/dcd_save_freq
print(f"{points=}, {charges.steps_between_updates=}")

########################################################################################
# molecular species
########################################################################################
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

sel_im = u.select_atoms("(resname IM1H or resname IM1) and not name H7")
sel_ac = u.select_atoms("(resname OAC or resname HOAC) and not name H")
n_im = sel_im.n_residues
n_ac = sel_ac.n_residues
print(sel_im.n_atoms)

sel_all = u.select_atoms("all and not name H and not name H7")
n_all = sel_all.n_residues
com_all = np.zeros((n, n_all, 3), dtype=np.float64)
charges_all = np.zeros((n, n_all), dtype=np.float64)

com_im = np.zeros((n, n_im, 3), dtype=np.float64)
com_ac = np.zeros((n, n_ac, 3), dtype=np.float64)
charges_im = np.zeros((n, n_im), dtype=np.float64)
charges_ac = np.zeros((n, n_ac), dtype=np.float64)
deltaM_Js = np.zeros((n, 3), dtype=np.float64)
M_J_diff_tot = np.zeros((n, 3), dtype=np.float64)

UpdatePair.skip_value = skip
pairs = get_pairs(pair_files)
pairs_idx = []
for i, pair_at_t in enumerate(pairs):
    pairs_idx.append([])
    pairs_list = []
    for pair in pair_at_t:
        pairs_idx[i].append(pair.idx1)
        pairs_idx[i].append(pair.idx2)
        pairs_list.append(pair.idx1)
        pairs_list.append(pair.idx2)

print(len(pairs))
print(len(u.trajectory) / (points * skip))
assert (len(pairs) <= (len(u.trajectory) / (points * skip))) & (
    len(pairs) >= (len(u.trajectory) / (points * skip) - 1)
)  # -1 weil am ende der ersten durchgang das erste update ist, wenn nicht alle wh eingelesen werden dann ist es <1 weniger
pairs = iter(pairs)


########################################################################################
# Running through trajectory
########################################################################################
ctr = 0

start = time.time()
print()
for ts, step in zip(u.trajectory[::skip], list(charge_steps)[::skip]):
    if not args.short:
        print(
            "\033[1AFrame %d of %d" % (ts.frame, len(u.trajectory)),
            "\tElapsed time: %.2f hours" % ((time.time() - start) / 3600),
        )

    com_all[ctr] = sel_all.center_of_mass(compound="residues")
    com_im[ctr] = sel_im.center_of_mass(compound="residues")
    com_ac[ctr] = sel_ac.center_of_mass(compound="residues")
    # charges_tmp = np.asarray(charges.charges_for_step(ts.frame+1)) # ts.frame is 0-based, therefore i.e. 10000 is already frame 10001, hence use ts.frame+1
    charges_tmp = np.asarray(charges.data[step])
    charges_all[ctr] = charges_tmp
    # charges_im[ctr] = charges_tmp[np.r_[0:150,300:650]]
    # charges_ac[ctr] = charges_tmp[np.r_[150:300,650:1000]]
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
    assert sum(charges_all[ctr]) == 0

    if (
        ctr % points == 0 and ctr != 0
    ):  # after_transfer #zb 100 der erste frame nach dem transfer
        # if no transfer dM_J = 0 -> dann wird bei removal nix abgezogen
        dM_J = []
        next_pairs = next(pairs)
        pair_list = []
        for update_pair in next_pairs:
            idx1, idx2 = update_pair.idx1, update_pair.idx2
            idx1_corr, idx2_corr = update_pair.correct_pos()
            pair_list.append(idx1)
            pair_list.append(idx2)
            r_ij = rPBC(
                com_all[ctr + idx2_corr][idx2], com_all[ctr + idx1_corr][idx1], boxl
            )  # xyz
            r_ij = r_ij * update_pair.multiply_factor()
            dM_J.append(r_ij)
        dM_J_tot = np.sum(dM_J, axis=0)
        deltaM_Js[ctr] = np.sum(dM_J, axis=0)  # wert wo transfer, sonst null
        M_J_diff = []
        for idx in range(n_all):
            if idx not in pair_list:
                com_now = com_all[ctr][idx]
                com_before = com_all[ctr - 1][idx]
                charge_now = charges_all[ctr][idx]
                charge_now3 = [charge_now, charge_now, charge_now]
                r_ij = rPBC(com_now, com_before, boxl)
                M_J_diff.append(r_ij * charge_now3)
        M_J_diff_tot[ctr] = np.sum(M_J_diff, axis=0)

    ctr += 1

########################################################################################
# Post-processing
########################################################################################
time_axis = np.arange(0, n * skip * dt, skip * dt)
# if firstfile == 1 and lastfile != 100:
#    time_axis = time_axis[:-1]
charges_im3 = np.stack(
    3 * (charges_im,), axis=2
)  # Expand the 2d charges array to 3d to fit the xyz of the coms
charges_ac3 = np.stack(
    3 * (charges_ac,), axis=2
)  # Expand the 2d charges array to 3d to fit the xyz of the coms
charges_all3 = np.stack(3 * (charges_all,), axis=2)

msdmjfolder = "./msdmj_jumps"
Path(msdmjfolder).mkdir(parents=True, exist_ok=True)
# com_im_folded = np.copy(com_im)
# com_ac_folded = np.copy(com_ac)

print("unfolding coordinates ..")
unfoldTraj(com_im, boxl)
unfoldTraj(com_ac, boxl)
unfoldTraj(com_all, boxl)
print(f"{com_im.shape=}")
print(f"{com_ac.shape=}")
print(f"{com_all.shape=}")

print("calculating msdMJ ..")
# collective translational dipolemoment M_J(t)
M_J_i_cat = charges_im3 * com_im  # M_J(t) per molecule
M_J_i_an = charges_ac3 * com_ac
M_J_cat = np.sum(M_J_i_cat, axis=1)  # total M_J(t) per timestep
M_J_an = np.sum(M_J_i_an, axis=1)  # total M_J(t) per timestep
np.savetxt(
    f"{msdmjfolder}/M_J_cat_{firstfile}_{lastfile}_{skip}{identifier}.dat",
    np.c_[time_axis, M_J_cat],
)
np.savetxt(
    f"{msdmjfolder}/M_J_an_{firstfile}_{lastfile}_{skip}{identifier}.dat",
    np.c_[time_axis, M_J_an],
)

M_J = M_J_cat + M_J_an
M_J_i_all = charges_all3 * com_all
M_J_all = np.sum(M_J_i_all, axis=1)
assert np.allclose(M_J, M_J_all)
np.savetxt(
    f"{msdmjfolder}/M_J_{firstfile}_{lastfile}_{skip}{identifier}.dat",
    np.c_[time_axis, M_J],
)

msdmj = tidynamics.msd(M_J)
np.savetxt(
    f"{msdmjfolder}/msdmj_{firstfile}_{lastfile}_{skip}{identifier}.dat",
    np.c_[time_axis, msdmj],
)

d_msdmj = derivative(time_axis, msdmj)
np.savetxt(
    f"{msdmjfolder}/d_msdmj_{firstfile}_{lastfile}_{skip}{identifier}.dat",
    np.c_[time_axis[:-1], d_msdmj],
)


# correct the data
# Which is better?
def simple_reduction(data, points_between_updates):
    for i in range(int(len(data) // points_between_updates) - 1):
        jump = (
            data[i * points_between_updates + points_between_updates]
            - data[i * points_between_updates + points_between_updates - 1]
        )
        data[(i + 1) * points_between_updates :] -= jump
    return data


def mw_reduction(data, points_between_updates):
    for i in range(int(len(data) // points_between_updates) - 1):
        jump = data[(i + 1) * points_between_updates] - np.mean(
            data[i * points_between_updates : (i + 1) * points_between_updates], axis=0
        )
        data[(i + 1) * points_between_updates :] -= jump
    return data


M_J = simple_reduction(M_J, points)
# M_J = mw_reduction(M_J, points)

np.savetxt(
    f"{msdmjfolder}/M_J_corrected_{firstfile}_{lastfile}_{skip}{identifier}.dat",
    np.c_[time_axis, M_J],
)

msdmj = tidynamics.msd(M_J)

np.savetxt(
    f"{msdmjfolder}/msdmj_corrected_{firstfile}_{lastfile}_{skip}{identifier}.dat",
    np.c_[time_axis, msdmj],
)

d_msdmj = derivative(time_axis, msdmj)

np.savetxt(
    f"{msdmjfolder}/d_msdmj_corrected_{firstfile}_{lastfile}_{skip}{identifier}.dat",
    np.c_[time_axis[:-1], d_msdmj],
)

np.savetxt(
    f"{msdmjfolder}/deltaM_Js_{firstfile}_{lastfile}_{skip}{identifier}.dat",
    np.c_[time_axis, deltaM_Js],
)
deltaM_Js = np.cumsum(deltaM_Js, axis=0)
np.savetxt(
    f"{msdmjfolder}/deltaM_Js_cumsum{firstfile}_{lastfile}_{skip}{identifier}.dat",
    np.c_[time_axis, deltaM_Js],
)

M_J += deltaM_Js
np.savetxt(
    f"{msdmjfolder}/M_J_corrected_jumps_{firstfile}_{lastfile}_{skip}{identifier}.dat",
    np.c_[time_axis, M_J],
)
msdmj = tidynamics.msd(M_J)

np.savetxt(
    f"{msdmjfolder}/msdmj_corrected_jumps_{firstfile}_{lastfile}_{skip}{identifier}.dat",
    np.c_[time_axis, msdmj],
)

d_msdmj = derivative(time_axis, msdmj)

np.savetxt(
    f"{msdmjfolder}/d_msdmj_corrected_jumps_{firstfile}_{lastfile}_{skip}{identifier}.dat",
    np.c_[time_axis[:-1], d_msdmj],
)

np.savetxt(
    f"{msdmjfolder}/M_J_diff_tot_{firstfile}_{lastfile}_{skip}{identifier}.dat",
    np.c_[time_axis, M_J_diff_tot],
)
M_J_diff_tot = np.cumsum(M_J_diff_tot, axis=0)
np.savetxt(
    f"{msdmjfolder}/M_J_diff_tot_cumsum_{firstfile}_{lastfile}_{skip}{identifier}.dat",
    np.c_[time_axis, M_J_diff_tot],
)
M_J += M_J_diff_tot
np.savetxt(
    f"{msdmjfolder}/M_J_corrected_jumps_diff_tot_{firstfile}_{lastfile}_{skip}{identifier}.dat",
    np.c_[time_axis, M_J],
)

msdmj = tidynamics.msd(M_J)

np.savetxt(
    f"{msdmjfolder}/msdmj_corrected_jumps_diff_{firstfile}_{lastfile}_{skip}{identifier}.dat",
    np.c_[time_axis, msdmj],
)

d_msdmj = derivative(time_axis, msdmj)

np.savetxt(
    f"{msdmjfolder}/d_msdmj_corrected_jumps_diff_{firstfile}_{lastfile}_{skip}{identifier}.dat",
    np.c_[time_axis[:-1], d_msdmj],
)

quit()
