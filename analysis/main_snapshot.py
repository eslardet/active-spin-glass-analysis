import numpy as np
import analysis_functions as fun
import os
import matplotlib.pyplot as plt

# f.plot_vorder_ksd(mode="G", nPart=500, phi=0.2, KAVG=1.0, KSTD_range=np.arange(0, 41, 1.0), seed=2, save=True, view=False)

# f.snapshot(mode="C", nPart=1000, phi=0.4, K=1.0, seed=2, view_time=10)

fun.snapshot(mode="C", nPart=5000, phi=0.2, K="1.0", seed=1, view_time=20)


