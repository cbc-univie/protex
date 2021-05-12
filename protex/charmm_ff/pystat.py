#!/usr/bin/python3
import sys
import math

##########################################################################
# SUBROUTINES
##########################################################################
def mean(y):
    """
    Return the sample arithmetic mean of y.
    >>> mean([1, 2, 3, 4, 4])
    2.8
    """
    return sum(y)/len(y)

def median(y):
    """
    Return the median (middle value) of numeric y.

    When the number of y points is odd, return the middle y point.
    When the number of y points is even, the median is interpolated by
    taking the average of the two middle values:

    >>> median([1, 3, 5])
    3
    >>> median([1, 3, 5, 7])
    4.0
    """
    y = sorted(y)
    n = len(y)
    if n%2 == 1:
        return y[n//2]
    else:
        i = n//2
        return (y[i - 1] + y[i])/2

def variance(y):
    """
    Return the sample variance of y.
    >>> y = [2.75, 1.75, 1.25, 0.25, 0.5, 1.25, 3.5]
    >>> variance(y)
    1.3720238095238095
    """
    av = mean(y)
    ss = sum((x-av)**2 for x in y)
    return ss/(len(y)-1)

def standard_deviation(y):
    """
    Return the standard deviation of y.
    >>> y = [2.75, 1.75, 1.25, 0.25, 0.5, 1.25, 3.5]
    >>> standard_deviation(y)
    1.171
    """
    return math.sqrt(variance(y))
    
def linear_regression(x,y):
    """ 
    Return slope and axis intercept of a linear regression of the y.
    """
    n=len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x2 = sum(map(lambda a: a*a,x))
    sum_products = sum([x[i]*y[i] for i in range(n)])
    m = (sum_products - (sum_x*sum_y)/n) / (sum_x2-((sum_x**2)/n))
    b = (sum_y - m*sum_x)/n
    return m,b

##########################################################################
#                 M A I N    P R O G R A M
##########################################################################
def read_data(filename, args):
    #filename = sys.argv[1]
    f=open(filename,"r")
    
    column = 1
    start = None
    stop  = None
    xmin  = None
    xmax  = None
    #for i in range(len(sys.argv)):
    for i in range(len(args)):
        #if (sys.argv[i]=="-c"):
        if (args[i]=="-c"):
            column = int(args[i+1])-1
        #if (sys.argv[i]=="-l"):
        if (args[i]=="-l"):
            argument = (args[i+1]).split(":")        
            start    = int(argument[0])-1
            if (len(argument)>1):
                stop = int(argument[1])-1
        #if (sys.argv[i]=="-x"):
        if (args[i]=="-x"):
            #argument = (sys.argv[i+1]).split(":")  
            argument = (args[i+1]).split(":")  
            xmin     = float(argument[0])
            if (len(argument)>1):
                xmax = float(argument[1])
    
    print("Filename: ",filename)
    print("Column: ",column+1)
    if (start is not None):
        print("Start:  ",start+1)
    if (stop is not None):
        print("Stop:  ",stop+1)
    if (xmin is not None):
        print("Xmin: ",xmin)
    if (xmax is not None):
        print("Xmax: ",xmax)
        
    # Reading data
    x = []
    y = []
    nframe = 0
    low    = None
    high   = None
    for line in f:
        nframe += 1
    #   empty lines    
        if (len(line)<2):
            break
    #   only lines between start and stop are considered    
        if (start is not None):
            if (nframe<start):
                continue
        if (stop is not None):
            if (nframe>stop):
                break
        
        buffer = line.split()
        current_x = float(buffer[0])
        current_y = float(buffer[column])
    
    #   only lines with x-values between xmin and xmax are considered    
        if (xmin is not None):
            if (current_x<xmin):
                continue
            
        if (xmax is not None):
            if (current_x>xmax):
                break
        x.append(current_x)
        y.append(current_y)
        
    #   lowest and highest value
        if (low is None):
            low = current_y
        if (high is None):
            high = current_y
        if (current_y<low):
            low = current_y
        if (current_y>high):
            high = current_y
            
    # Output
    print("Mean value         = " ,mean(y))
    print("Standard deviation = ",standard_deviation(y))
    (m,b) = linear_regression(x,y)
    print("Lowest  y-value    = ",low)
    print("Highest y-value    = ",high)
    print("Linear regression:")
    print("\tm = ",m)
    print("\tb = ",b)

    return mean(y), standard_deviation(y), m, b, y

if __name__ == "__main__":
    filename = sys.argv[1]
    args = sys.argv
    _,_,_,_,_ = reading_data(filename, args)
