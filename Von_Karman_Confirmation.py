import numpy as np
import matplotlib.pyplot as plt
from aotools.turbulence import phasescreen
from scipy.special import laguerre
from aotools import opticalpropagation
from aotools.turbulence.phasescreen import ft_phase_screen as ps
from aotools.turbulence.phasescreen import ft_sh_phase_screen as psh


"""
r0 (float) – r0 parameter of scrn in metres
N (int) – Size of phase scrn in pxls
delta (float) – size in Metres of each pxl
L0 (float) – Size of outer-scale in metres
l0 (float) – inner scale in metres
"""
r0 = 1
N = 512
delta = 1e-3
L0 = 1
l0 = 1e-5
seed = 1
ps_test = ps(r0, N, delta, L0, l0, FFT=None, seed=None)
plt.imshow(ps_test)
plt.show()