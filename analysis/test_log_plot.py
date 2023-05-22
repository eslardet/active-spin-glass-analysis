import numpy as np
import matplotlib.pyplot as plt

bin_ratio = 2

fig, ax = plt.subplots(figsize=(10,10/bin_ratio))

ax.set_xlabel(r"$K_{ij}$", fontsize=48)
ax.set_ylabel(r"$r_{ij}$", fontsize=48)
plt.show()