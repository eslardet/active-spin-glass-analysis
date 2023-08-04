import numpy as np
import analysis_functions_vicsek_new as fun
import os
import matplotlib.pyplot as plt
import sys
import time
import csv

mode = "F"
nPart = 1000
phi = 1.0
noise = "0.20"
K = "0.0_8.0_Kn-5.0"
Rp = 1.0
xTy = 1.0
seed = 1

fun.plot_porder_time(mode, nPart, phi, noise, K, Rp, xTy, seed)