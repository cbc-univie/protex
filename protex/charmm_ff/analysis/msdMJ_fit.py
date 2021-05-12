from lmfit import Minimizer, Parameters
from lmfit.printfuncs import report_fit
import numpy as np
import sys
import json
import copy
from decimal import Decimal

print('~'*120)
print('msdMJ_fit.py'),
print('~'*120)

#################################################################################
class InputClass:
#################################################################################
    def __init__(self):
        self.infile  = ''
        self.outfile = ''
        self.charmmfile = ''
        self.boxl = ''
        self.maxtime = 0

        self.fit_model = 'leastsq'
        #self.fit_model = 'ampgo'
        self.initialexp = []
        self.exp = []
        self.number_of_exp = 0

        self.initialline = LineClass()
        self.line = LineClass()
        
    def info(self):
        print('< Json Input:')
        print('\tdata         = ',self.infile)
        print('\tresiduals    = ',self.outfile)
        print('\tmaximal time = ',self.maxtime)
        print('\tcharmmfile   = ',self.charmmfile)
        print('\tboxlength   = ',self.boxl)
        print('\n\texp:')
        for i in self.exp:
            i.info()
        print('\n\tline:')
        self.line.info()
            
        print('\n\tFit model = ',self.fit_model)
    
    def fromJson(self,data):
        if "data" in data:
            self.infile = data["data"]
        if "maxtime" in data:
            self.maxtime = data["maxtime"]
        if "residuals" in data:
            self.outfile = data["residuals"]
            self.outfile = self.outfile[:-4] + "_" + str(self.maxtime) + ".dat"
        if "exp" in data:
            for i in data["exp"]:
                tmp = ExpClass()
                tmp.fromJson(i)
                self.exp.append(tmp)
            self.number_of_exp = len(self.exp)
            self.initialexp = copy.deepcopy(self.exp)
        if "fit_model" in data:
            self.fit_model = data["fit_model"]
        if "line" in data:
            # taking only the last slope and overwrite former declarations
            for i in data["line"]:
                self.line = LineClass()
                self.line.fromJson(i)
            self.initialline = copy.deepcopy(self.line) 
        
        ##### optional ######
        if "charmmfile" in data:
            self.charmmfile = data["charmmfile"]
        if "boxl" in data: #not needed if self.charmmfile
            self.boxl = data["boxl"]
            
    def toJson(self,JsonFile):
        print('>\t Writing Json File ',JsonFile)
        f = open(JsonFile,'w')
        f.write('{\n')
        f.write('\t"data":      "%s", \n'%self.infile)
        f.write('\t"residuals": "%s",\n'%self.outfile)
        f.write('\t"charmmfile": "%s",\n'%self.charmmfile)
        f.write('\t"boxl": "%s",\n'%self.boxl)
        f.write('\t"maxtime":   %s,\n\n'%self.maxtime)

        if len(self.initialexp) >0:
            f.write('\t"initialexp":\n')
            f.write('\t[\n')
            for ctr,i in enumerate(self.initialexp):
                if i.a_vary:
                    f.write('\t\t{ "a": {"value": %10.5f, "vary": %s}, '%(i.a,'true'))
                else:
                    f.write('\t\t{ "a": {"value": %10.5f, "vary": %s}, '%(i.a,'false'))
                if i.tau_vary:
                    f.write('"tau": {"value": %10.5f, "vary": %s} }' %(i.tau,'true'))
                else:
                    f.write('"tau": {"value": %10.5f, "vary": %s} }' %(i.tau,'false'))
                if not ctr == self.number_of_exp-1:
                    f.write(',')
                f.write('\n')
            f.write('\t],\n\n')
        f.write('\t"exp":\n')
        f.write('\t[\n')

        for ctr,i in enumerate(self.exp):
            if i.a_vary:
                f.write('\t\t{ "a": {"value": %10.5f, "vary": %s}, '%(i.a,'true'))
            else:
                f.write('\t\t{ "a": {"value": %10.5f, "vary": %s}, '%(i.a,'false'))
            if i.tau_vary:
                f.write('"tau": {"value": %10.5f, "vary": %s} }' %(i.tau,'true'))
            else:
                f.write('"tau": {"value": %10.5f, "vary": %s} }' %(i.tau,'false'))
            if not ctr == self.number_of_exp-1:
                f.write(',')
            f.write('\n')
        f.write('\t],\n')

        if self.initialline.slope>0.0:
            f.write('\n\t"initialline":\n')
            f.write('\t[\n')
            if self.initialline.slope_vary:
                f.write('\t\t{ "slope": {"value": %.8f, "vary": %s} }\n'%(self.initialline.slope,'true'))
            else:
                f.write('\t\t{ "slope": {"value": %.8f, "vary": %s} }\n'%(self.initialline.slope,'false'))
            f.write('\t],\n')
        f.write('\n\t"line":\n')
        f.write('\t[\n')
        if self.line.slope_vary:
            f.write('\t\t{ "slope": {"value": %.8f, "vary": %s} }\n'%(self.line.slope,'true'))
        else:
            f.write('\t\t{ "slope": {"value": %.8f, "vary": %s} }\n'%(self.line.slope,'false'))
        f.write('\t]\n')
        f.write('}\n')
        f.close()

    def ExampleInput(self):
        JsonFile = "msdMJ.json"
        self.infile  = "msdMJ.dat"
        self.outfile = "msdMJ_fit.dat"
        self.maxtime = 500

        # example tri exponential fit
        tmp = ExpClass()
        tmp.a         = 20.0
        tmp.a_vary    = True
        tmp.tau       = 0.2
        tmp_tau_vary = True
        self.exp.append(tmp)
        
        tmp = ExpClass()
        tmp.a         = 50.0
        tmp.a_vary    = True
        tmp.tau       = 50.0
        tmp_tau_vary = True
        self.exp.append(tmp)

        tmp = ExpClass()
        tmp.a         = 200.0
        tmp.a_vary    = True
        tmp.tau       = 200.0
        tmp_tau_vary = True
        self.exp.append(tmp)
        self.number_of_exp = len(self.exp)

        self.line.slope      = 6.0
        self.line.slope_vary = True
        
        self.toJson(JsonFile)
            
#################################################################################
class ExpClass:
#################################################################################
    def __init__(self):
        self.a      = 0.0
        self.a_vary = True
        
        self.tau      = 1.0
        self.tau_vary = True

    def info(self):
        if self.a_vary:
            print('\t\ta = %10.5f        '%self.a, end='')
        else:
            print('\t\ta = %10.5f (fixed)'%self.a, end='')

        if self.tau_vary:
            print('\ttau = %10.5f        '%self.tau)
        else:
            print('\ttau = %10.5f (fixed)'%self.tau)

    def fromJson(self,data):
        self.a        = data["a"]["value"]
        self.a_vary   = data["a"]["vary"]
        self.tau      = data["tau"]["value"]
        self.tau_vary = data["tau"]["vary"]

    def fromFit(self,i,result):
        self.a   = abs(result.params['a'+str(i+1)].value)
        self.tau = abs(result.params['tau'+str(i+1)].value)

#################################################################################
class LineClass:
#################################################################################
    def __init__(self):
        self.slope = 0.0
        self.slope_vary = True

    def info(self):
        if self.slope_vary:
            print('\t\tslope = %.8f        '%self.slope)
        else:
            print('\t\tslope = %.8f (fixed)'%self.slope)

    def fromJson(self,data):
        self.slope      = data["slope"]["value"]
        self.slope_vary = data["slope"]["vary"]

    def fromFit(self,result):
        self.slope = abs(result.params['slope'].value)
    
#################################################################################
class DataClass:
#################################################################################
    def __init__(self):
        self.time = []
        self.f    = []
        self.derivative = []

    def fromFile(self,filename,maxtime):
        try:
            infile = open(filename,'r')
        except FileNotFoundError:
            print('\n! Error!')
            print('\t data file %s not found!'%filename)
            sys.exit()

        tmp_time = []
        tmp_f = []
        for line in infile:
            line_element = line.split()
            current_time = float(line_element[0])
            if current_time > maxtime:
                break
            if len(line)>1:
                tmp_time.append(current_time)
                tmp_f.append(float(line_element[1]))
        self.time = np.asarray(tmp_time)
        self.f    = np.asarray(tmp_f)

    def compute_derivative(self):
        ###################################################################
        # "Numerische Methoden" J. Douglas Faiers / R.L. Burden, p. 168
        ###################################################################
        print('\n> Computing derivative of the time series ...')
        h = self.time[1] - self.time[0]
        tmp_derivative = []
        
        # end point calculation of the first two values of d/dt self.f
        tmp = -2.0833333*self.f[0]+4.0*self.f[1]-3.0*self.f[2]+1.3333333*self.f[3]-0.25*self.f[4]
        tmp_derivative.append(tmp/h)

        tmp = -0.25*self.f[0]-0.8333333*self.f[1]+1.5*self.f[2]-0.5*self.f[3]+0.08333333*self.f[4]
        tmp_derivative.append(tmp/h)

        # mid point calculation of d/dt self.f
        lenf = len(self.f)
        for i in range(lenf-5):
            tmp = 0.0833333*self.f[i]-0.666666*self.f[i+1]+0.666666*self.f[i+3]-0.0833333*self.f[i+4]
            tmp_derivative.append(tmp/h)

        # end point calculation of the last values of d/dt self.f
        tmp = -0.0833333*self.f[i]+0.5*self.f[i+1]-1.5*self.f[i+2]+0.8333333*self.f[i+3]+0.25*self.f[i+4]
        tmp_derivative.append(tmp/h)

        tmp = 0.25*self.f[i]-1.333333*self.f[i+1]+3.0*self.f[i+2]-4.0*self.f[i+3]+2.08333333*self.f[i+4]
        tmp_derivative.append(tmp/h)
        tmp_derivative.append(0.0)
        self.derivative = np.asarray(tmp_derivative)
        
#################################################################################
# Fit model
#################################################################################
def residual(parfit, t, fdata=None, ddata = None):
    model1 = 0.0
    model2 = 0.0
    for i in range(input.number_of_exp):
        tmp_a   = parfit['a'+str(i+1)].value
        tmp_tau = parfit['tau'+str(i+1)].value
        
        # correlation function
        model1 += abs(tmp_a) * (1.0 - np.exp(-t/abs(tmp_tau)))
        
        # its derivative
        model2 += abs(tmp_a) / abs(tmp_tau) * np.exp(-t/abs(tmp_tau))
    
    tmp_line = parfit['slope'].value
    model1 += abs(tmp_line) * t
    model2 += abs(tmp_line)
    
    if fdata is None:
        resid1 = model1
    else:
        resid1 = model1 - fdata

    if ddata is None:
        resid2 = model2
    else:
        resid2 = model2 - ddata
    return np.concatenate(( resid1, resid2 ))

#################################################################################
# Main program
#################################################################################

# Generating input object
input = InputClass()
try:
    JsonFile = sys.argv[1]
    with open(JsonFile) as infile:
        data = json.load(infile)
    input.fromJson(data)
    input.info()
except IndexError:
    print('\n! Error!')
    print('!\t Json input file is missing: python3 mdmd_fit.py ___.json. ')
    print('!\t Writing example input msdMJ.json')
    input.ExampleInput()
    sys.exit()
except FileNotFoundError:
    print('\n! Error!')
    print('!\t Json input file %s not found!'%JsonFile)
    sys.exit()

# Generating data object
data = DataClass()
data.fromFile(input.infile,input.maxtime)
data.compute_derivative()

# Generating fit model
print('\n> Fitting ...')

print('\t Setting up fit model ...')
parfit = Parameters()
parfit.add('slope', value = input.line.slope,   vary = input.line.slope_vary)
a0 = min(data.f[-1]-input.line.slope*data.time[-1],0.01*data.f[-1])
for i,exp in enumerate(input.exp):
    parfit.add('a'+str(i+1),   value = exp.a, vary = exp.a_vary, min = a0, max = 0.9*data.f[-1])

    if i>0:
        min_tau = parfit['tau'+str(i)]*2.0
    else:
        min_tau = 0.1

    max_tau = 0.5*float((i+1)/input.number_of_exp)*data.time[-1]
    parfit.add('tau'+str(i+1), value = exp.tau, vary = exp.tau_vary, min = min_tau, max = max_tau)
        
    
print('\t Fitting data, minimizing residuals ...\n')
myfit  = Minimizer(residual, parfit,fcn_args=(data.time,), fcn_kws={'fdata': data.f, 'ddata':data.derivative}, scale_covar=True)
result = myfit.minimize(input.fit_model)
fit    = residual(result.params, data.time)
resids = residual(result.params, data.time, data.f, data.derivative)
report_fit(result)
print('\n')

# update exp section in Json
for i,exp in enumerate(input.exp):
    exp.fromFit(i,result)
input.line.fromFit(result)
input.toJson(JsonFile)

print('\n>\t Writing residual file = ',input.outfile)
f = open(input.outfile,'w')
fitlen = int(len(fit)/2)
for i in range(fitlen):
    f.write("%10.5f "%data.time[i])
    f.write("%10.5f "%data.f[i])
    f.write("%10.5f "%fit[i])
    f.write("%10.5f "%resids[i])
    f.write("%10.5f "%data.derivative[i])
    f.write("%10.5f "%fit[i+fitlen])
    f.write("%10.5f "%resids[i+fitlen])
    f.write("\n")
    
slope = input.line.slope
try:
    with open(input.charmmfile, "r") as f:
        lines = f.readlines()
        for line in lines:
            l = line.split() 
            if len(l) >= 3 and l[0] == "set" and l[1] == "XTL":  
                boxl = float(l[3])
            if len(l) >= 7 and l[1] == "@?TEMP":
                temp = float(l[7])
    print(f"\n>\tExtracted temperature (= {temp} K) and box length (= {boxl} Angstrom) from file {input.charmmfile}.")
except:
    temp = 300
    boxl = float(input.boxl)
    print(f"\n>\tTemperature is set to {temp} K and box length (= {boxl} Angstrom) is taken from input file.")
conductivity = ( 1.602176634 * slope ) / (6 * 8.617332478 * 10**-8 * temp * boxl**3 )
print(f"\n>\tConductivity is: {Decimal(conductivity):.10e} [S/m]")
#    conductivity = ( 1.602176634 * slope ) / (6 * 8.617332478 * 10**-8 * temp * boxl**3 )
#    print(f"\n>\tConductivity is: {Decimal(conductivity):.10e} [S/m]")
#except FileNotFoundError:
#    print("No Charmm input file supplied or not found.")
#    print("No Conductivity was calculated.")
#    print("It can be calcualted with:")
#    print("conductivity = ( 1.602176634 * slope ) / (6 * 8.617332478 * 10**-8 * temp * boxl**3 )")
#    print("temp in K, boxl in Angstrom, conductivity is in S/m")
