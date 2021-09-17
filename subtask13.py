import numpy as np


def subtask13(H, fc, da):
    LAMBDA = 3e8 / fc
    rho = np.arange(1, 11, da)
    phi = np.arange(-70, 71, da)
    theta = np.arange(-20, 81, da)

    return rho, phi, theta