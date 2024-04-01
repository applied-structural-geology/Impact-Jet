import numpy as np
from scipy.integrate import odeint

a = 0.5  # Tillotson parameter a
b = 1.63  # Tillotson parameter b
A = 0.75  # Bulk modulus
B = 0.65  # Tillotson parameter B
rho_0 = 2.7  # Initial density in g/cm^3
E_0 = 0.05  # Energy constant in the Tillotson EOS
E_iv = 5.0  # Energy at incipient vaporization
E_cv = 10.0  # Energy at complete vaporization
C_v = 0.1  # Specific heat at constant volume
C = 900

# The pressure, density, and specific internal energy (e) would be inputs to the function in a real scenario.
# Here we provide example values.
P = 0.83  # Pressure in GPa
rho = 1.0973*rho_0  # Current density in g/cm^3
e = 5.0  # Specific internal energy in MJ/kg, this is what we want to calculate.

# Equation (1) Tillotson EOS for compressed states
def tillotson_eos_compressed(P, rho, E):
    mu = rho / rho_0 
    return (a + b / (E/(E_0*(mu + 1) ** 2) + 1)) * rho * E + A * mu + B * mu**2



# Equation (8) defines dE_c/dP for the cold curve, we need to integrate this to find E_c
def dE_c_dP(P, E_c, rho):
    return P / rho**2

# Implementing the 4th order Runge-Kutta scheme to numerically integrate the cold curve from E_c
# We assume the initial temperature is 0 for the cold energy calculation, hence initial E_c is 0.
def runge_kutta(P, E_c_initial, rho):
    h = 0.1  # Step size
    n = int((P - 0) / h)  # Number of steps
    E_c_value = E_c_initial

    for i in range(1, n + 1):
        k1 = h * dE_c_dP(0 + (i-1) * h, E_c_value, rho)
        k2 = h * dE_c_dP(0 + (i-1) * h + h/2, E_c_value + k1/2, rho)
        k3 = h * dE_c_dP(0 + (i-1) * h + h/2, E_c_value + k2/2, rho)
        k4 = h * dE_c_dP(0 + (i-1) * h + h, E_c_value + k3, rho)
        E_c_value += (k1 + 2*k2 + 2*k3 + k4) / 6

    return E_c_value

# Initial cold energy
E_c_initial = 0

# Calculating the cold energy using Runge-Kutta
E_c_rk = runge_kutta(P, E_c_initial, rho)

# Now we need to solve for E using the Tillotson EOS for the compressed state, this is a numerical root-finding problem.
from scipy.optimize import fsolve

# Define the equation to find the roots of.
def equation(E, P, rho):
    return tillotson_eos_compressed(P, rho, E) - P

# Calculate the internal energy E
E_calculated = fsolve(equation, e, args=(P, rho))
temp = np.abs((E_calculated[0] - E_c_rk)*1e6/C)

print("Temperature: ", temp)