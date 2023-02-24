import numpy as np
import analysis_functions_vicsek_nd as fun
import matplotlib.pyplot as plt


noise_range = np.arange(0.1,1.1,0.05)

noise_range = [format(i, '.2f') for i in np.arange(0.1,0.65,0.05)]

noise_range += ['0.70', '0.85', '1.00']

print(np.round(np.arange(0.1,2.1,0.1),1))