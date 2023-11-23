import sys
sys.path.insert(1, './analysis/analysis_functions')
from bands_analysis import *


mode = "G"
nPart = 100000
phi = 1.0
noise = "0.20"
#K_avg = 0.0
K_avg_range = [1.0]
#K_avg_range = np.round(np.arange(0.0,0.1,0.5),1)
#K_avg_range = np.concatenate((np.round(np.arange(-0.5,0.0,0.1),1), np.round(np.arange(0.0,1.1,0.1),1)))
# K_std = 0.0
#K_std_range = np.round(np.arange(1,8.1,1),1)
K_std_range = [0.0]
K = "-0.1_8.0"
Rp = 1.0
xTy = 5.0
seed_range = np.arange(1,2,1)
seed = 1
timestep_range = range(1)
#r_max = 10
#r_bin_num = 100
# timestep_range = range(10)
min_grid_size = 5
min_T = 0
max_T = 1000

# plot_density_profile(mode, nPart, phi, noise, K, Rp, xTy, seed, min_grid_size=20)

##plot_band_profiles(mode, nPart, phi, noise, K, Rp, xTy, seed)
# plot_average_band_profile(mode, nPart, phi, noise, K_avg_range, K_std_range, Rp, xTy, seed_range, pos_ex=True, timestep_range=timestep_range, min_grid_size=10)

# plot_density_profile_superimpose(mode, nPart, phi, noise, K_avg_range, K_std_range, Rp, xTy, seed, min_grid_size=5)

# animate_density_profile(mode, nPart, phi, noise, K, Rp, xTy, seed, min_grid_size, min_T, max_T)

del_pos(mode, nPart, phi, noise, K, Rp, xTy, seed)