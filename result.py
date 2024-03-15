import numpy as np
from result_jet import jet
from params import *

theta_range = np.arange(4, 20, 1.2)

# Arrays to store results
result_var1 = []
result_var2 = []
result_var3 = []

# Loop through theta values and store results
for theta in theta_range:
    output_var1 = jet(theta*1000)
    result_var1.append(output_var1)

temp = np.array(result_var1)

print(temp)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10)) 
plt.plot(theta_range, temp, marker='o', linestyle='-', label='Jet Velocity', color='black')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Impact Velocity(km/s)', fontsize = 20)
plt.ylabel('Jet Temperature (K)', fontsize = 20) 
plt.title('Temperature vs Impacter Size Fe-Cu', fontsize = 30 )
plt.show()
 

