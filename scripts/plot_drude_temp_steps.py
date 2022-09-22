import glob

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

plt.rcParams.update({'font.size': 14})

#colors of uni vienna
colordict = {
    1  : '#0063a6'  , # blue
    11 : '#0063a655', # blue 66% noch da(55)
    12 : '#0063a6AA', # blue 33% noch da (AA -> 66% von 255(fully transparent) in hex)
    2  : '#dd4814'  , # orange
    21 : '#dd481455', # orange 66%
    22 : '#dd4814AA', # orange 33% 
    3  : '#a71c49'  , # dark red/bordeaux
    31 : '#a71c49AA', # dark red 33%
    4  : '#94c154'  , # green
    41 : '#94c15455', # green 66%
    42 : '#94c154AA', # green 33%
    5  : '#666666'  , # gray
    6  : '#f6a800'  , # yellow
    7  : '#11897a'  , # mint
    8  : '#000000'    # black
        }

#data
# drude temp reporter data
steps = ["~/software/protex_data/scripts/data/drude_temp_1_d40.out","~/software/protex_data/scripts/data/drude_temp_11.out","~/software/protex_data/scripts/data/drude_temp_31.out","~/software/protex_data/scripts/data/drude_temp_51.out","~/software/protex_data/scripts/data/drude_temp_101.out"]
print(steps)


dt = 0.0005 #ps
dt = dt/1000 #ns

fig1 = plt.figure(0)
ax1 = plt.gca()
fig2 = plt.figure(1)
ax2 = plt.gca()
fig3 = plt.figure(2)
ax3 = plt.gca()

data = {}
for pos, fname in enumerate(steps):
    if 0 <= pos < 5: #and (pos==0 or pos==4):
        #data["sth"] = [step, T_COM, T_Atom, T_Drude, KE_COM, KE_Atom, KE_Drude]
        data[fname] = np.loadtxt(fname)
        step = int(fname.split("/")[-1].split(".")[0].split("_")[2])
        print(step)
        #other temp
        ax1.plot(data[fname][::,0]*dt, data[fname][::,1], ls="", marker="x", label=f"{step=} T= {round(data[fname][:,1].mean(),1)}$\pm${round(data[fname][:,1].std(),1)} K")
        ax2.plot(data[fname][::,0]*dt, data[fname][::,2], ls="", marker="x", label=f"{step=} T= {round(data[fname][:,2].mean(),1)}$\pm${round(data[fname][:,2].std(),1)} K")
        #drude temp
        ax3.plot(data[fname][::,0]*dt, data[fname][::,3], label=f"{step=} T={round(data[fname][:,3].mean(),1)}$\pm${round(data[fname][:,3].std(),1)} K", ls="--")#, marker="x")
        #print(data[fname][:,1].mean(), data[fname][:,1].std())
        #print(data[fname][:,2].mean(), data[fname][:,2].std())
        dtemp = data[fname][:,3]
        peaks, _ = find_peaks(dtemp, distance=1000, height=20)
        print(peaks)
        ax3.plot(data[fname][peaks,0]*dt, dtemp[peaks], "x", label=f"peaks T={round(dtemp[peaks].mean(),1)}$\pm${round(dtemp[peaks].std(),1)} K")
        idx = np.where(dtemp<10)[0]
        #ax3.plot(data[fname][idx,0]*dt, dtemp[dtemp<10], label=f"dtemp<10 T={round(dtemp[dtemp<10].mean(),1)}$\pm${round(dtemp[dtemp<10].std(),1)} K")

ax1.set_xlabel("time (ns)")
ax1.set_ylabel("COM Temp (K)")
ax1.legend(loc="upper center", bbox_to_anchor=(0.5,-0.25), ncol=2)
ax2.set_xlabel("time (ns)")
ax2.set_ylabel("Atom Temp (K)")
ax2.legend(loc="upper center", bbox_to_anchor=(0.5,-0.25), ncol=2)
ax3.set_xlabel("time (ns)")
ax3.set_ylabel("Drude Temp (K)")
ax3.legend(loc="upper center", bbox_to_anchor=(0.5,-0.25), ncol=2)
#fig1.tight_layout()
#f = fig1.gcf()
fig1.set_size_inches(9,6)
fig1.savefig("fig_com_temp_steps.png", bbox_inches="tight")
fig2.set_size_inches(9,6)
fig2.savefig("fig_atom_temp_steps.png", bbox_inches="tight")
fig3.set_size_inches(9,6)
fig3.savefig("fig_drude_temp_steps.png", bbox_inches="tight")
