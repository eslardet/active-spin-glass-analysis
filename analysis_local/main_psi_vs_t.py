import numpy as np
import sys
sys.path.insert(1, '/Users/el2021/Code/2D_ActiveSpinGlass_EL/Active_Spin_Glass/analysis')
from analysis_functions import *
import os
import matplotlib.pyplot as plt
import time
import csv

mode = "G"
nPart = 1000
phi = 1.0
noise = "0.20"
K = "1.0_1.0"
Rp = 1.0
xTy = 1.0
seed = 1

plot_porder_time(mode, nPart, phi, noise, K, Rp, xTy, seed)