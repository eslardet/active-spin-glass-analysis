import numpy as np
import analysis_functions_vicsek as fun
import os
import matplotlib.pyplot as plt
import sys
import time

mode = "C"
nPart = 10000
phi = 1.0
noise = 0.3
K = 1.0
seed = 1


# fun.snapshot(mode, nPart, phi, noise, K, seed, view_time=10.0)
fun.animate(mode, nPart, phi, noise, K, seed)
