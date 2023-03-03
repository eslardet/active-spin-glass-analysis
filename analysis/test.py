import numpy as np
import analysis_functions_vicsek_nd as fun
import matplotlib.pyplot as plt


noise_range = np.round(np.concatenate((np.arange(-0.5,0.0,0.1), np.arange(0.0,1.1,0.1))),1)

print(noise_range)