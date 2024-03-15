from params import *
import numpy as np

def deceleration(t, V0, Cd):
    A = np.pi*R**2
    Vol = (4/3)*np.pi*R**3
    return V0/((1+V0*(0.5*Cd*A*(1/Vol))*t))
    # return V0*np.exp(-0.5 * t)