import numpy as np
import math
import matplotlib.pyplot as plt
import warnings 
from params import *
from tillotson import solve_tillotson
from plotter import plot
from utils import *
warnings.filterwarnings('ignore') 

# T is half penetration time, T is split into 1000 parts to plot parameters with time

T = R / (V0 * np.sin(np.deg2rad(theta)))
To = np.linspace(0,T, 1000)
T = To[1:1000]
t=T.T
Vimp = np.zeros_like(t)

for i in range(len(t)):
    Vimp[i] = deceleration(t[i], V0, Cd)

alpha=np.zeros((1,len(t)))
for i in range(len(t)):
   alpha[0:i] = np.round(np.degrees(np.arccos(1 - ((Vimp[i] * np.sin(np.radians(theta))) / R) * t)),decimals=1)
steps = 10 * alpha + 1
spare = 901 - steps

V1 = Vimp[i] * np.sin(np.deg2rad(theta)) / np.sin(np.deg2rad(alpha[:,:]))
V2 = ((Vimp[i] * np.sin(np.deg2rad(theta))) / np.tan(np.deg2rad(alpha))) + Vimp[i]  * np.cos(np.deg2rad(theta))
v1=np.reshape(V1,(1,-1))
v2=np.reshape(V2,(1,-1))
M1 = V1 / c1
M2 = V2 / c2

# Setting guess values for phi1 and phi2
# Simultaneously iterating over mu to get values for plotting Hodograph
U1 = np.zeros((len(t), len(t))) #Velocity of shocked projectile parallel to V1
U2 = np.zeros((len(t), len(t))) #Velocity of shocked target parallel to V2
W1=np.zeros((len(t), len(t))) #Velocity of shocked projectile perp to V1
W2=np.zeros((len(t), len(t))) #Velocity of shocked projectile perp to V2
# W2=np.nan_to_num(W2)

# mu11 = np.zeros((1,len(t)))
# mu22 = np.zeros((1,len(t)))
mu11 = 0
mu22 = 0
temp1=np.zeros((999,999))
# P1=np.zeros((1,999))
phi1=np.zeros([901,999])
phi2=np.zeros([901,999])
for i in range(999):
  mu11 += 0.01 #To solve the equation for u & v, we find the material pressure before shock by iterating mu
  mu22 += 0.01
  P1 = (d1 * (c1 ** 2) * (mu11 * (mu11 + 1))) / ((mu11 * (1 - s1) + 1) ** 2) #Non-Norm Shock Pressure
  P2 = (d2 * (c2 ** 2) * (mu22 * (mu22 + 1))) / ((mu22 * (1 - s2) + 1) ** 2) #Non-Norm Shock Pressure
  U1[i, :] = (d1 * (V1 ** 2) - P1) / (d1 * V1) #Velocity of shocked projectile parallel to V1
  U2[i, :] = (d2 * (V2 ** 2) - P2) / (d2 * V2) #Velocity of shocked target perpendicular to V2
  W1[i, :] = (np.sqrt(P1 * ((d1 * (mu11 * (V1 ** 2)) / (mu11 + 1)) - P1)) / (d1 * V1)) #Velocity of shocked projectile perp to V1
  W2[i, :] = (np.sqrt(P2 * ((d2 * (mu22 * (V2 ** 2)) / (mu22 + 1)) - P2)) / (d2 * V2)) #Velocity of shocked projectile perp to V2
  # W1[i, :] = np.sqrt(P1 * ((d1 * (mu11 * (V1 ** 2)) / (mu11 + 1)) - P1)) / (d1 * V1)
  # W2[i, :] = np.sqrt(P2 * ((d2 * (mu22 * (V2 ** 2)) / (mu22 + 1)) - P2)) / (d2 * V2)
  temp1=np.linspace(0,alpha[0,i],int(steps[0,i])) #temp1 stores values from 0 - alpha in steps
  for j in range(901):
      if(j<int(steps[0,i])):
          phi1[j][i]=round(temp1[j],4)
phi2=abs(alpha-phi1) #Storing guess values of phi upto alpha

# Extracting meaningful values for ploting hodograph.
matrices = [U1, U2, W1, W2]
for idx, matrix in enumerate(matrices, start=1):
    for col in range(matrix.shape[1]):
        min_idx = np.argmin(matrix[:, col])
        matrix[min_idx + 1:, col] = 0

# Setting coeff for use in Rankine-Hugoniot condition and relation of phi with mu & P.
q12 = ((np.round(((M1 * (1 - s1) * np.sin(np.radians(phi1)))**2) - 1 - (np.cos(np.radians(phi1))**2), 4)).T).flatten()
q11 = ((np.round(2 * (1 - s1) * ((M1 * np.sin(np.radians(phi1)))**2) - 1 - 2 * (np.cos(np.radians(phi1))**2), 4)).T).flatten()
q10 = ((np.round((M1 * np.sin(np.radians(phi1)))**2, 4)).T).flatten()
q22 = ((np.round(((M2 * (1 - s2) * np.sin(np.radians(phi2)))**2) - 1 - (np.cos(np.radians(phi2))**2), 4)).T).flatten()
q21 = ((np.round(2 * (1 - s2) * ((M2 * np.sin(np.radians(phi2)))**2) - 1 - 2 * (np.cos(np.radians(phi2))**2), 4)).T).flatten()
q20 = ((np.round((M2 * np.sin(np.radians(phi2)))**2, 4)).T).flatten()


# Calculating absolute pressure difference |P1-P2| for all guess values of phi1 and phi2.
# The guess combination for which the difference is minimum
# is the required combination for phi1&phi2
# Final variables 'phi1' & 'phi2' gives value of phi(s) as a function of
# time.

M11 = np.column_stack((q12, q11, q10))
M22 = np.column_stack((q22, q21, q20))

mu1 = np.zeros(len(M11))
mu2 = np.zeros(len(M22))
for i in range(len(M11)):
  mu1[i] = np.round((-1 * M11[i, 1] - np.sqrt((M11[i, 1] ** 2 - 4 * M11[i, 0] * M11[i, 2]))) / (2 * M11[i, 0]),4)
  mu2[i] = np.round((-1 * M22[i, 1] - np.sqrt((M22[i, 1] ** 2 - 4 * M22[i, 0] * M22[i, 2]))) / (2 * M22[i, 0]),4) 
m1o=(np.reshape(mu1,(999,901))).T
m2o=(np.reshape(mu2,(999,901))).T
p1 = ((m1o * (m1o + 1)) / ((M1 * (1 + (1 - s1) * m1o)) ** 2))* (d1 * (V1 ** 2))#Shock Pressure
p2 = (m2o * (m2o + 1)) / ((M2 * (1 + (1 - s2) * m2o)) ** 2)*(d2 * (V2 ** 2)) #Shock Pressure
a = np.abs(p2 - p1)

min_v=np.min(a,axis=0)
min_index=np.argmin(a,axis=0)

Phi1 =phi1
Phi2 =phi2
phi11 = Phi1[min_index]
phi22 = Phi2[min_index]
PHI1=phi11[:,998]
PHI2 = np.diagonal(phi22)
# Phi11 is the value of phi1 where the shock pressure difference is the minimum
# Calculating critical phi values as a function of time
cphi1 = (np.arctan(np.max(np.nan_to_num(np.real(W1) / U1), axis=0)))* 180 / np.pi 
cphi2 = (np.arctan(np.max(np.nan_to_num(np.real(W2) / U2), axis=0)))* 180 / np.pi 

# Pinpointing the onset of jetting and determining jetting side.
diff1 = cphi1 - PHI1
diff2 = cphi2 - PHI2

z_ind1 = np.where(diff1 <= 0)[0]
z_ind2 = np.where(diff2 <= 0)[0] 

if z_ind1.size == 0:
    z_ind1 = len(t)
else:
    z_ind1 = z_ind1[0]

if z_ind2.size == 0:
    z_ind2 = len(t)
else:
    z_ind2 = z_ind2[0]

z_ind = min(z_ind1, z_ind2)

if z_ind1 < z_ind2:
    V = V1[:,z_ind]
    J = 1
else:
    V = V2[:,z_ind]
    J = 2

#mu for projectile and target at jetting regime
zr_ind1 = min_index[z_ind1]
zr_ind2 = min_index[z_ind2]
m1 = m1o[zr_ind1][z_ind1]
m2 = m2o[zr_ind2][z_ind2]
Pf1 = p1[zr_ind1][z_ind1]
Pf2 = p2[zr_ind2][z_ind2]

# Tilloston EOS for Temperature
if(J == 1):
    temp = solve_tillotson(a1, b1, d1*1e-3, Pf1*1e-11, m1, A1, B1, Eo1, C1)
    
elif(J == 2):
    temp = solve_tillotson(a2, b2, d2*1e-3, Pf2*1e-11, m2, A2, B2, Eo2, C2)

# calculating V_jet and time of jetting
v_jet = (0.64 * V * np.cos(np.deg2rad(PHI2[z_ind])) + V2[0, z_ind])
jetting_angle = PHI2[z_ind]
T = t[z_ind]
T= round(T * 1e9,4)
Th = round(t[-1]*1e9,4)
v_jet=np.round(v_jet/10000,4)
del_t = Th - T

print(f"Velocity of jet {v_jet[0]}e+04 m/s")
print("Jet onset time: ", T, "ns")
print("Jetting from: ", J)
print("Temperature: ", np.round(temp, 0), "K")
print(f"Pressure: {repr(Pf1/10e9)} GPa")

plot(t, alpha, T, PHI1, PHI2, cphi1, cphi2)