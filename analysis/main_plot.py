import numpy as np
import analysis_functions_lattice as fun
import os
import matplotlib.pyplot as plt
import sys

mode = 'C'
nPart_range = [1024]
K_range = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
Rp = 2.0
seed_range = [1]


fun.plot_vorder_k(mode=mode, nPart_range=nPart_range, K_range=K_range, Rp=Rp, seed_range=seed_range)

