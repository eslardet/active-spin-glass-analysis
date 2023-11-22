import sys
sys.path.insert(1, './analysis/analysis_functions')
from bands_analysis import *
from matplotlib import colors


mode = "G"
nPart = 50000
phi = 1.0
noise = "0.70"
K_avg_range = [1.0]
K_std_range = [1.0, 4.0, 8.0]
Rp = 1.0
xTy = 5.0
seed = 1
min_grid_size = 5

colors = plt.cm.BuPu(np.linspace(0.2, 1, 9))

fig, ax = plt.subplots()
for K_avg in K_avg_range:
    for K_std in K_std_range:
        K = str(K_avg) + "_" + str(K_std)
        posFileExact = get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, file_name='pos_exact')
        x, y, theta, view_time = get_pos_ex_snapshot(file=posFileExact)

        L = np.sqrt(nPart / (phi*xTy))
        Ly = L
        Lx = L*xTy

        x = pbc_wrap(x,Lx)

        ngrid_x = int(Lx // min_grid_size)
        grid_size_x = Lx / ngrid_x

        grid_area = grid_size_x*Ly

        grid_counts = np.zeros(ngrid_x)

        for i in range(nPart):
            gridx = int(x[i]//grid_size_x)
            grid_counts[gridx] += 1
        n_density = grid_counts / grid_area

        x_vals = np.arange(0, Lx, grid_size_x)
        if K_std == 4.0:
            n_density = np.roll(n_density, shift=78)
        if K_std == 5.0:
            n_density = np.roll(n_density, shift=18)
        ax.plot(x_vals, n_density, label = r"$\overline{K}=$" + str(K_avg) + r"; $\sigma_K=$" + str(K_std), color=colors[int(K_std)])

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"Local density")
ax.legend(loc=(0.45,0.7), frameon=False)


folder = os.path.abspath('../plots/density_profile/')
filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed) + '.png'
if not os.path.exists(folder):
    os.makedirs(folder)
plt.savefig(os.path.join(folder, filename))
plt.close()