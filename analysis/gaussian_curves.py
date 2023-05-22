import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

fig, ax = plt.subplots(figsize=(10,5))

x_plot = np.arange(-5, 5, 0.001)
plt.plot(x_plot, norm.pdf(x_plot, 0, 1), label=r"$K_{AVG}=0.0; K_{STD}=1.0$")
plt.plot(x_plot, norm.pdf(x_plot, 1, 1), label=r"$K_{AVG}=1.0; K_{STD}=1.0$")
plt.plot(x_plot, norm.pdf(x_plot, 1, 2), label=r"$K_{AVG}=1.0;  K_{STD}=2.0$")

plt.ylabel("Probability Density", fontsize=16)
plt.xlabel(r"$K_{ij}$", fontsize=16)
plt.legend(loc="upper left", fontsize=16)
plt.show()