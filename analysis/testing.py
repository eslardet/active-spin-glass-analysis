import numpy as np
import analysis_functions_vicsek_nd as fun
import matplotlib.pyplot as plt
from matplotlib import cm, colors


x = np.zeros(10)
y = np.zeros(10)
theta = np.linspace(0,2*np.pi,10)

cols = np.mod(theta, 2*np.pi)
norm = colors.Normalize(vmin=0.0, vmax=2*np.pi, clip=True)



fig, ax = plt.subplots()
colormap = cm.hsv
plt.set_cmap('hsv')

arrows = ax.quiver(x, y, np.cos(theta), np.sin(theta), norm(cols))
# arrows.set_UVC(np.cos(theta), np.sin(theta), norm(cols))

mapper = cm.ScalarMappable(norm=norm, cmap=cm.hsv)
plt.colorbar(mappable=mapper, ax=ax)

plt.show()