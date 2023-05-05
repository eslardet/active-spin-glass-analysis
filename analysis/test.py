import numpy as np
import analysis_functions_vicsek_new as fun
import matplotlib.pyplot as plt
import time
import scipy.stats as sps
import os
import freud

mode = "G"
nPart = 10
phi = 1.0
noise = "0.20"
Rp = 1.0
K = "0.0_1.0"
xTy = 1.0
seed = 1
min_grid_size=2


filename = "test_file"

save_file = open(filename + ".txt", "w")

some_data_x = [1.0, 2.0, 3.0, 4.0]
some_data_y = [0.0, 0.0, 0.0, 1.0]

for x in some_data_x:
    save_file.write(str(x) + "\t")
save_file.close()

read_file = open(filename + ".txt", "r")
for line in read_file:
    print(line)