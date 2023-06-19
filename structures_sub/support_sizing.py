import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sp
thickness = np.array([0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.020]) * 100
deflection = np.array([0.664, 0.504, 0.392, 0.311, 0.251, 0.205, 0.17, 0.142, 0.121, 0.103, 0.0892])
max_stress = np.array([4.04*10**7,  4.01*10**7, 4*10**7])
xdata = np.arange(1, 2, 0.01)
def func(x, a, b, c):
    return a * np.exp(-b * x) + c
if __name__ == '__main__':
    args, pcov = sp.curve_fit(func,thickness, deflection)

    plt.plot(thickness, deflection/thickness * 10, 'o', label = 'Simulated points')
    plt.plot(xdata, func(xdata, *args) / xdata * 10, 'r-', label='Curve fit')
    plt.plot(xdata, np.ones(len(xdata))*2, label = '2% deformation requirement')
    plt.ylabel('Max $\delta$ spacecraft panel [%]')
    plt.xlabel('Thickness spacecraft panel [cm]')
    plt.legend()
    plt.grid()
    plt.show()
