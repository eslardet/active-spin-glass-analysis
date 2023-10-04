import numpy as np
import sys
sys.path.insert(1, '/Users/el2021/Code/2D_ActiveSpinGlass_EL/Active_Spin_Glass/analysis')
from analysis_functions import *
from scipy.signal import fftconvolve, correlate2d


# def get_distance_matrix(ngridx, ngridy):
#     """
#     Output matrix is distance shift matrix in terms of x, y distance wrapped by number of grid points
#     """
#     x = pbc_wrap_calc(np.tile(np.arange(0,ngridy), (ngridx,1)),ngridy)
#     y = pbc_wrap_calc(np.tile(np.arange(0,ngridx), (ngridy,1)),ngridx).T
#     dist = np.sqrt(x**2+y**2)
#     return dist

# def get_r_corr(x,y,inparFile, min_grid_size=1):

#     params = get_params(inparFile)
#     nPart = params['nPart']
#     phi = params['phi']
#     xTy = params['xTy']

#     L = np.sqrt(nPart / (phi*xTy))
#     Ly = L
#     Lx = L*xTy

#     ngridx = int(Lx // min_grid_size)
#     ngridy = int(Ly // min_grid_size)

#     grid_size_x = Lx / ngridx
#     grid_size_y = Ly / ngridy

#     count_arr = np.zeros((ngridx, ngridy))
#     for i in range(nPart):
#         ix = int(pbc_wrap(x[i],Lx) // grid_size_x)
#         iy = int(pbc_wrap(y[i],Ly) // grid_size_y)
#         count_arr[ix, iy] += 1

#     density_arr = count_arr / (grid_size_x * grid_size_y)
#     density_fluc_arr = density_arr - np.mean(density_arr) # for fluctuations

#     a=np.fft.fft2(density_fluc_arr)
#     b=np.fft.fft2(density_fluc_arr[::-1,::-1])
#     corr=np.round(np.real(np.fft.ifft2(a*b))[::-1,::-1],0)
#     corr = np.abs(corr/corr[0,0]) # normalization
#     dist = get_distance_matrix(ngridx, ngridy)
#     return dist.flatten(), corr.flatten()

# def get_r_corr_all(mode, nPart, phi, noise, K, Rp, xTy, seed_range, pos_ex=False, timestep_range=[0], min_grid_size=1):
#     r_all = []
#     corr_all = []
#     for seed in seed_range:
#         for timestep in timestep_range:
#             inparFile, posFile = get_files(mode, nPart, phi, noise, K, Rp, xTy, seed)
#             if pos_ex:
#                 posFileExact = get_file_path(mode, nPart, phi, noise, K, Rp, xTy, seed, file_name="pos_exact")
#                 x, y, theta, view_time = get_pos_ex_snapshot(file=posFileExact)
#             else:
#                 x, y, theta = get_pos_snapshot(posFile, nPart, timestep)
#             dist, corr = get_r_corr(x,y,inparFile, min_grid_size)
#             r_all += list(dist)
#             corr_all += list(corr)
#     return np.array(r_all), np.array(corr_all)

# def write_corr_all(mode, nPart, phi, noise, K, Rp, xTy, seed_range, pos_ex=True, timestep_range=[0], min_grid_size=1):
#     folder = os.path.abspath('../plot_data/correlation_density_grid/')
#     if not os.path.exists(folder):
#         os.makedirs(folder)
#     filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed_range[-1]) + '_g' + str(min_grid_size) + ".txt"
    
#     corrFile = open(os.path.join(folder, filename), "w")

#     corrFile.write(str(min_grid_size) + "\n")
#     corrFile.write(str(timestep_range[0]) + "\t" + str(timestep_range[-1]) + "\n")

#     r_all, corr_all = get_r_corr_all(mode, nPart, phi, noise, K, Rp, xTy, seed_range, pos_ex, timestep_range, min_grid_size)

#     for i in len(r_all):
#         corrFile.write(str(r_all[i]) + "\t" + str(corr_all[i]) + "\n")
#     corrFile.close()


# def read_corr_density(mode, nPart, phi, noise, K, Rp, xTy, seed_range, min_grid_size=1):
#     r_all = []
#     corr_all = []

#     folder = os.path.abspath('../plot_data/correlation_density_grid/')
#     filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed_range[-1]) + '_g' + str(min_grid_size) + ".txt"
#     corrFile = os.path.join(folder, filename)
#     with open(corrFile) as f:
#         line_count = 1
#         for line in f:
#             if line_count > 2:
#                 r_all.append(float(line.split('\t')[0]))
#                 corr_all.append(float(line.split('\t')[1]))
#             line_count += 1

#     return r_all, corr_all

# def get_corr_binned(dist, corr, bin_size=1):
#     r_plot = np.linspace(0, np.max(dist), num=int(np.max(dist)/bin_size))
#     corr_plot = []
#     r_plot_2 = []
#     for i in range(len(r_plot)):
#         lower = r_plot[i]
#         try:
#             upper = r_plot[i+1]
#         except:
#             upper = np.max(dist)
#         idx = np.where((dist>=lower) & (dist<upper))[0]
#         if len(idx)>0:
#             c = np.mean(corr[idx])
#             corr_plot.append(c)
#             r_plot_2.append(r_plot[i]+bin_size/2)
#     # r_plot += bin_size/2
#     return r_plot_2, corr_plot

# def get_corr_binned_unique(dist, corr, min_r=0, max_r=10):
#     r_plot = np.unique(dist)
#     r_plot2 = r_plot[np.where((r_plot>=min_r) & (r_plot<=max_r))[0]]
#     corr = np.array(corr)
#     corr_plot = []
#     for i in range(len(r_plot2)):
#         idx = np.where(dist == r_plot[i])[0]
#         if len(idx)>0:
#             c = np.mean(corr[idx])
#             corr_plot.append(c)
#     return r_plot2, corr_plot

mode = 'G'
nPart = 10000
phi = 1.0
noise = "0.20"
K = "0.0_8.0"
# K = "1.0_0.0"
Rp = 1.0
xTy = 1.0
seed = 1
seed_range = np.arange(1,21,1)
min_grid_size=1

# plot_corr_density_file(mode, nPart, phi, noise, K, Rp, xTy, seed_range, log_y=True, min_grid_size=1)
dist, corr = read_corr_density(mode, nPart, phi, noise, K, Rp, xTy, seed_range, min_grid_size)
r_plot, corr_plot = get_corr_binned(dist, corr)
# print(r_plot)
# dist, corr = read_corr_density(mode, nPart, phi, noise, K, Rp, xTy, seed_range, min_grid_size=0.5)
# dist, corr = get_r_corr_all(mode, nPart, phi, noise, K, Rp, xTy, seed_range, pos_ex=False, timestep_range=[0], min_grid_size=1)
# r_plot2, corr_plot2 = get_corr_binned(dist, corr)
print(r_plot)

# plot_corr_density_file(mode, nPart, phi, noise, K, Rp, xTy, seed_range, log_y=True, min_grid_size=1)

# inparFile = get_file_path(mode, nPart, phi, noise, K, Rp, xTy, seed, file_name="inpar")
# posFileExact = get_file_path(mode, nPart, phi, noise, K, Rp, xTy, seed, file_name="pos_exact")

# # snapshot(mode, nPart, phi, noise, K, Rp, xTy, seed, pos_ex=True, show_plot=True, save_plot=False)
# # print(get_distance_matrix(4,4))
# # get_r_corr(posFileExact)

# # plot_corr_scatter(posFileExact)

fig, ax = plt.subplots()
# # # dist, corr = get_r_corr(inparFile, posFileExact)
# # # dist, corr = get_r_corr_all(mode, nPart, phi, noise, K, Rp, xTy, seed_range, pos_ex=True, timestep_range=[0])
# # ax.scatter(dist, corr)

# # # r_plot, corr_plot = get_corr_binned(dist, corr, bin_size=1)
ax.plot(r_plot, corr_plot)
# ax.plot(r_plot2, corr_plot2)

# # ax.set_xscale('log')
ax.set_yscale('log')
# # # ax.set_xlabel("Distance")
# # # ax.set_ylabel("Correlation")
# # ax.set_xlim(0,10)
# # # ax.set_ylim(1e-3,1)


# filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed_range[-1]) + '_g' + str(min_grid_size) + ".png"
# folder = os.path.abspath('../plots/correlation_density_grid/')
# if not os.path.exists(folder):
#     os.makedirs(folder)
# plt.savefig(os.path.join(folder, filename), bbox_inches="tight")
plt.show()
