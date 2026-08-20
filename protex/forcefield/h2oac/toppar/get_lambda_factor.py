import sys
import collections
##############
# Scale LJ parameters accoridng to
# epsilon(LJ, beta) = epsilon(LJ,orig,beta)*(delta(alpha)+lambda*max(alpha(beta)))/(max(alpha(beta))+lambda*delta(alpha))
# alpha = polarizability, lambda = scaling factor, delta(alpha) = max(alpha)-alpha

#Usage: python3 scale_lj.py sclaing_factor
###############
def read_file(filename):
    try:
        f = open(filename, "r")
    except:
        print("File not found")
    
    return f.readlines()

    
def read_atomtypes(lines, *residuenames):
    '''
    read atomtypes with polarizabilities from a stream files molecule section
    '''
    #TODO: dict with residues und atomtypes, pol -> problem: only 1 atomtype alpha in total, ev check bevor nächste rresidue kommt!
    nlines = len(lines)
    ilines = 0
    atomtypes = collections.defaultdict(list)
    #atomtypes[alpha] = [] #Polarizability
    residues = []
    for residue in residuenames:
       residues.append(residue) 
    while ilines<nlines:
        if any(x in lines[ilines] for x in residues):
            #print(lines[ilines])
            ilines+=1
            #print(lines[ilines])
            while "!" in lines[ilines].split()[0]:
                ilines+=1
            if "GROUP" in lines[ilines]:
                ilines+=1
                while "ATOM" in lines[ilines]:
                    if "ALPHA" in lines[ilines]:
                        atomtypes[lines[ilines].split()[2]].append(float(lines[ilines].split()[5]))
                    else:
                        atomtypes[lines[ilines].split()[2]].append(0)
                    ilines += 1
        ilines += 1
        #print(lines[ilines])
    return atomtypes


def read_toppar(lines, atomtypes):
    nlines = len(lines)
    ilines = 0
    header = []
    toppar_types = {}
    while ilines<nlines:
        if lines[ilines].startswith("NONB"): # in lines[ilines]:
            header.append(lines[ilines])
            header.append(lines[ilines+1])
            while not lines[ilines].startswith("NBFIX"):
                try:
                    a_type = lines[ilines].split()[0]
                except IndexError:
                    a_type = None
                #if any(a_type in lines[ilines].split()[:-4] for a_type in atomtypes):
                if a_type in atomtypes:
                    toppar_types[lines[ilines].split()[0]] = [float(lines[ilines].split()[2]), float(lines[ilines].split()[3])]
                ilines+=1
        ilines+=1
    
    return toppar_types
        
def scale_lj(alpha, max_alpha, orig_lj, scale_factor):
    # Phys. Chem.Chem.Phys. 2018, 20, 15106 
    delta_alpha = max_alpha-alpha
    new_lj = round(orig_lj * (delta_alpha + scale_factor*max_alpha) / (max_alpha + scale_factor * delta_alpha), 4)
    return new_lj   

def get_lambda_for_atomtype(epsilon_new, epsilon_old, alpha, max_alpha):
    delta_alpha = max_alpha-alpha
    print(f"{alpha=}, {max_alpha=}, {delta_alpha=}, {epsilon_new=}, {epsilon_old=}")
    try:
        new_lambda = round((epsilon_new * max_alpha - epsilon_old * delta_alpha)/(epsilon_old * max_alpha - delta_alpha*epsilon_new), 4)
    except:
        if alpha == 0:
            print("No scaling -> no polarizable atom type")
            new_lambda = -1
        else:
            raise ZeroDivisionError
    print(new_lambda)
    return new_lambda


def write_new_file(new_lj, old_toppar_lines, new_toppar_file_name):
    nlines = len(old_toppar_lines)
    ilines = 0
    #header = []
    with open(new_toppar_file_name, 'w') as f:
        while ilines<nlines:
            if old_toppar_lines[ilines].startswith("NONB"): # in lines[ilines]:
                #f.write(f"{old_toppar_lines[ilines]}\n")
                #f.write(f"{old_toppar_lines[ilines+1]}\n")
                #header.append(lines[ilines])
                #header.append(lines[ilines+1])
                while not old_toppar_lines[ilines].startswith("NBFIX"):
                    try:
                        a_type = old_toppar_lines[ilines].split()[0]
                    except IndexError:
                        a_type = None
                    if a_type in new_lj.keys():
                    #if any(a_type in old_toppar_lines[ilines].split() for a_type in new_lj.keys()):
                        #a_type = old_toppar_lines[ilines].split()[0]
                        new_line = '  '.join(value if pos != 2 else str(new_lj[a_type][0]) for pos, value in enumerate(old_toppar_lines[ilines].split()))
                        f.write(f"{new_line}\n")
                    else:
                        f.write(old_toppar_lines[ilines])
                    ilines+=1
                f.write(old_toppar_lines[ilines])
            else:
                f.write(old_toppar_lines[ilines])
            ilines+=1
    print(f"Worte new toppar file {new_toppar_file_name} with scaled values, only for needed atomtypes, everyrthing else is untouched.\n")

def test_write(lines):
    nlines = len(lines)
    print("nlines", nlines)
    ilines = 0
    with open("test_toppar.txt", "w") as f:
        while ilines<nlines:
            f.write(lines[ilines])
            ilines+=1


stream_file1 = read_file("./im1h_d.str")
stream_file2 = read_file("./oac_d.str")
stream_file3 = read_file("./im1_d.str")
stream_file4 = read_file("./hoac_d.str")
stream_file5 = read_file("./tfa_d.str")
stream_file6 = read_file("./htfa_d.str")
#stream_file5 = read_file("tfa_d.str")
#stream_file6 = read_file("htfa_d.str")
big_file = stream_file1
big_file.extend(stream_file2)
big_file.extend(stream_file3)
big_file.extend(stream_file4)
big_file.extend(stream_file5)
big_file.extend(stream_file6)
#atomtypes = read_atomtypes(big_file, "IM1H", "OAC", "IM1", "HOAC")
atomtypes = read_atomtypes(big_file, "IM1H", "OAC", "IM1", "HOAC", "TFA", "HTFA")
#print(atomtypes)
# make average and find max, same atomtypes need same LJ
unique_atomtypes = {}
for key, value in atomtypes.items():
   if len(set(value)) > 1 and sum(value)/len(value) != 0:
       print("ATTENTION!!!!! NEED TO AVERAGE POLARIZABILITITES FOR ATOMTYPE", key)
       print("THERE ARE", len(value), "ENTRIES")
       print("IMPORTANT!!! SCALING IS DONE WITH AVERAGE")
   unique_atomtypes[key] = round(sum(value)/len(value), 4)
#print(unique_atomtypes) 
toppar_file_unscaled = read_file("./toppar_drude_master_protein_2013f_modified.str")
toppar_types_unscaled = read_toppar(toppar_file_unscaled, atomtypes)
toppar_file_scaled = read_file("./toppar_drude_master_protein_2013f_lj025.str")
toppar_types_scaled = read_toppar(toppar_file_scaled, atomtypes)
#print(toppar_types)
#scale_factor = float(sys.argv[1])
#print("scale_factor", scale_factor)

#global maximum of all atoms of all residues used in simulation
max_alpha = min(unique_atomtypes.values())
print("max", max_alpha)
new_lj = {}
new_scaling_factor = {}
for key, alpha in unique_atomtypes.items():
    print(f"{key=}")
    #orig_lj = toppar_types.get(key)[0]
    epsilon_old = toppar_types_unscaled.get(key)[0]
    epsilon_new = toppar_types_scaled.get(key)[0]
    sigma = toppar_types_scaled.get(key)[1]
    assert sigma == toppar_types_unscaled.get(key)[1] #sigma should not change
#    print(key, orig_lj)
    #new_lj[key] = [scale_lj(alpha, max_alpha, orig_lj, scale_factor), sigma]
    new_scaling_factor[key] = get_lambda_for_atomtype(epsilon_new, epsilon_old, alpha, max_alpha)
print()
print("Lambda Values (Theory: 0.25): ", sorted(new_scaling_factor.items()))
#print("Old LJ", sorted(toppar_types.items()))
#print()
#print("New LJ", sorted(new_lj.items()))
#print()
print("polarizabilities", sorted(unique_atomtypes.items()))
print()
with open("scaling_per_atomtype.txt", "w") as of:
    for atom_type, scaling in new_scaling_factor.items():
        print(f"{atom_type} {scaling}", file=of)

#test_write(toppar_file)
#write_new_file(new_lj, toppar_file, "toppar_drude_master_protein_2013f_lj"+str(scale_factor)+".str")

print("Manually adjust polarizabiliies in topologies if needed or not already done!")
    
