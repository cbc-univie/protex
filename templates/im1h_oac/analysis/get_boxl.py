import pathlib


def read_outfile(file):
    with open(file) as f:
        lines = f.readlines()
    return lines



def get_volume(lines):
    volume = []
    boxl = []
    found_volume = False
    for line in lines:
        if "#" in line:
            n_entries = len(line.split(","))
            found_volume = True
            continue
        if found_volume:
            try:
                volume_value = float(line.split(",")[n_entries-1].strip())
            except:
                continue
            volume.append(volume_value)
            boxl_value = volume_value**(1/float(3))
            boxl.append(boxl_value)
    return volume, boxl

######## MAIN #######

#base = "../out/npt_"
base = str(pathlib.Path(__file__).parent.absolute().parent) + "/out/npt_"
total_volume = []
total_boxl = []
for i in range(1,7):
    f = base + str(i) + ".out"
    lines = read_outfile(f)
    volume, boxl = get_volume(lines)
    total_volume.extend(volume)
    total_boxl.extend(boxl)

print("Entries:", len(total_volume))

with open(str(pathlib.Path(__file__).parent.absolute())+"/volume.dat", "w") as f:
#with open("volume.dat", "w") as f:
    for pos,val in enumerate(total_volume,1):
       f.write(str(pos) + "\t" + str(val)+"\n")

with open(str(pathlib.Path(__file__).parent.absolute())+"/boxl.dat", "w") as f:
#with open("boxl.dat", "w") as f:
    for pos,val in enumerate(total_boxl,1):
       f.write(str(pos) + "\t" + str(val*10)+"\n")
