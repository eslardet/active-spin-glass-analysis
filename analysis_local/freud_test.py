import numpy as np
sys.path.insert(1, '/Users/el2021/Code/2D_ActiveSpinGlass_EL/Active_Spin_Glass/analysis')
from analysis_functions import *
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import freud

mode = 'G'
nPart = 10000
phi = 1.0
noise = "0.60"
K = "2.0_1.0"
xTy = 5.0
seed = 1
timestep_range = range(1)


inparFile, posFile = get_files(mode, nPart, phi, noise, K, xTy, seed)
posExactFile = get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed, file_name="pos_exact")
L = np.sqrt(nPart / (phi*xTy))
Ly = L
Lx = L*xTy

r_max = Ly / 2.01
print(r_max)
print(Lx, Ly)
cf = freud.density.CorrelationFunction(bins=25, r_max=r_max)

for t in timestep_range:
    # x, y, theta = get_pos_snapshot(posFile=posFile, nPart=nPart, timestep=t)
    x, y, theta, view_time = get_pos_ex_snapshot(file=posExactFile)

    points = np.zeros((nPart, 3))
    points[:,0] = x
    points[:,1] = y
    box = freud.Box.from_box([Lx, Ly])
    points = box.wrap(points)

    theta = np.array(theta)
    values = np.array(np.exp(theta * 1j))

    cf.compute(system=(box, points), values=values, query_points=points, query_values=values, reset=False)

fig, ax = plt.subplots()

# cf.plot(ax=ax)
ax.plot(cf.bin_centers, cf.correlation, label="check")
# ax.set_label("check")
ax.legend()

ax.hlines(y=0, xmin=0, xmax=r_max, color="grey", linestyle="dashed")
plt.show()
