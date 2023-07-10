import numpy as np
import analysis_functions_vicsek_new as fun
import os
import matplotlib.pyplot as plt
import sys
import csv


mode = "G"
nPart = 20000
phi = 1.0
noise = "0.20"
K = "0.5_8.0"
Rp = 1.0
xTy = 1.0
seed = 2

fun.snapshot(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, pos_ex=False, timestep=5)

