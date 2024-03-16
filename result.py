import numpy as np
from result_jet import jet
from params import *
import matplotlib.pyplot as plt

theta_range = np.arange(20, 91, 2)

# Arrays to store results
result_var1 = []
result_var2 = []
result_var3 = []

# Loop through theta values and store results
for theta in theta_range:
    output_var1 = jet(theta)
    result_var1.append(output_var1)

temp = np.array(result_var1)

x = np.arange(len(temp))
mask = np.isnan(temp)
temp_interp = np.interp(x, x[~mask], temp[~mask])

plt.figure(figsize=(10, 10)) 
plt.plot(theta_range, temp_interp, linestyle='-', color='red')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Impact Angle (degree)', fontsize=20)
plt.ylabel('Jet Temperature (K)', fontsize=20) 
plt.title('Temperature vs Impacter Angle Fe-Cu (10.3 km/s)', fontsize=30)
plt.show()
 

