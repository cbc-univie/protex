import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import glob
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
freqs = ["data/drude_temp_1_d20.out","data/drude_temp_1_d40.out","data/drude_temp_1_d80.out","data/drude_temp_1_d100.out", "data/drude_temp_1_d120.out", "data/drude_temp_1_d140.out","data/drude_temp_1_d160.out"]
print(freqs)

dt = 0.0005 #ps
dt = dt/1000 #ns

fig1 = plt.figure(0)
ax1 = plt.gca()
fig2 = plt.figure(1)
ax2 = plt.gca()
fig3 = plt.figure(2)
ax3 = plt.gca()

data = {}
colors = ["r", "b", "g", "y", "c", "lightblue", "black"]
for pos, fname in enumerate(freqs): 
    #data["sth"] = [step, T_COM, T_Atom, T_Drude, KE_COM, KE_Atom, KE_Drude]
    #if 0 <= pos < 6 and pos != 1  and pos != 4: 
        data[fname] = np.loadtxt(fname)
        d_coll = int(fname.split("/")[-1].split("_")[-1].split(".")[-2:-1][0].strip("d"))
        print(d_coll)
        #other temp
        ax1.plot(data[fname][:,0]*dt, data[fname][:,1], ls="", marker="x", label=f"{d_coll=} T={round(data[fname][:,1].mean(),1)}$\pm${round(data[fname][:,1].std(),1)} K", color=colors[pos])
        ax2.plot(data[fname][:,0]*dt, data[fname][:,2], ls="", marker="x", label=f"{d_coll=} T={round(data[fname][:,2].mean(),1)}$\pm${round(data[fname][:,2].std(),1)} K", color=colors[pos])
        #print(data[fname][:,1].mean(), data[fname][:,1].std())
        #print(data[fname][:,2].mean(), data[fname][:,2].std())
        #drude temp
        ax3.plot(data[fname][:,0]*dt, data[fname][:,3], label=f"{d_coll=} T={round(data[fname][:,3].mean(),1)}$\pm${round(data[fname][:,3].std(),1)} K", ls="--", color=colors[pos])#, marker="x")
        dtemp = data[fname][:,3]
        peaks, _ = find_peaks(dtemp, distance=10000)
        print(peaks)
        ax3.plot(data[fname][peaks,0]*dt, dtemp[peaks], "x", label=f"peaks T={round(dtemp[peaks].mean(),1)}$\pm${round(dtemp[peaks].std(),1)} K", color=colors[pos])
        idx = np.where(dtemp<10)[0]
        #ax3.plot(data_80[fname][idx,0]*dt, dtemp[dtemp<10], label=f"dtemp<10 T={round(dtemp[dtemp<10].mean(),1)}$\pm${round(dtemp[dtemp<10].std(),1)} K")

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
fig1.savefig("fig_com_temp.png", bbox_inches="tight")
fig2.set_size_inches(9,6)
fig2.savefig("fig_atom_temp.png", bbox_inches="tight")
fig3.set_size_inches(9,6)
fig3.savefig("fig_drude_temp.png", bbox_inches="tight")
