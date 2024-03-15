import matplotlib.pyplot as plt

# ploting alpha,phi,critical_phi as a function of time

def plot(t, alpha, T, PHI1, PHI2, cphi1, cphi2):
    t = t * 1e9
    plt.plot(t,alpha.T)
    plt.plot(t,PHI1)
    plt.plot(t,PHI2)
    plt.plot(t,cphi1)
    plt.plot(t,cphi2)
    plt.xlim(0, 200)
    plt.ylim(0)
    plt.axvline(x=T, color='black')
    plt.xlabel('Time in nanoseconds')
    plt.ylabel('Angle in degrees')
    plt.legend(['alpha', 'phi(p)', 'phi(t)', 'phi(cp)', 'phi(ct)'],loc='upper right')
    plt.show()