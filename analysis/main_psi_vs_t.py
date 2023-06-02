import numpy as np
import analysis_functions_vicsek_new as fun
import os
import matplotlib.pyplot as plt
import sys
import time
import csv

mode = "G"
nPart = 10000
phi = 1.0
noise = "0.20"
K = "0.0_8.0"
Rp = 5.0
xTy = 1.0
seed = 101

fun.plot_porder_time(mode, nPart, phi, noise, K, Rp, xTy, seed, max_T=200)