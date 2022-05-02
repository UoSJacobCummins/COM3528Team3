import numpy as np
import matplotlib.pyplot as plt 

def plot_boundaries( fig, ax ):
    A = np.arange(0, 2, 0.01)
    B = np.arange(0, 2, 0.01)
    A, B = np.meshgrid(A, B)
    X = (A + B - 2)/2.0
    Y = (A - B)/2.0
    Z = -4*X**3 - 27*Y**2
    s = ax.pcolormesh(A,B,Z, cmap='RdBu', vmin=-0.2, vmax=0.1)
    #fig.colorbar(s, shrink=0.5, aspect=10)
