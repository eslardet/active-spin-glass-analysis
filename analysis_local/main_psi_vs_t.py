import numpy as np
from analysis.analysis_functions import *
import os
import matplotlib.pyplot as plt
import sys
import time
import csv

mode = "G"
nPart = 1000
phi = 1.0
noise = "0.20"
K = "1.0_0.0"
Rp = 1.0
xTy = 1.0
seed = 1

plot_porder_time(mode, nPart, phi, noise, K, Rp, xTy, seed)