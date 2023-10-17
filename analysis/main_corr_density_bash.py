import sys
sys.path.insert(1, './analysis/analysis_functions')
from correlation import *

import time


mode = str(sys.argv[1])
nPart = int(sys.argv[2])
phi = float(sys.argv[3])
noise = sys.argv[4]
K = str(sys.argv[5]) + "_" + str(sys.argv[6])
Rp = sys.argv[7]
xTy = float(sys.argv[8])
seed = int(sys.argv[9])
seed_range = np.arange(1,seed+1,1)
max_time = int(sys.argv[10])
timestep_range = np.arange(0,max_time+1,1)
corr_r_max = float(sys.argv[11])
r_bin_num = int(sys.argv[12])
r_scale = str(sys.argv[13])
corr_r_min = float(sys.argv[14])
rho_r_max = float(sys.argv[15])


log_y = True
bin_ratio = 1

#linlin = bool(sys.argv[10])
#loglin = bool(sys.argv[11])
#loglog = bool(sys.argv[12])

t0 = time.time()
#plot_corr_density(mode, nPart, phi, noise, K, Rp, xTy, seed_range, linlin=linlin, loglin=loglin, loglog=loglog)
write_corr_density_grid(mode, nPart, phi, noise, K, Rp, xTy, seed_range, pos_ex=False, timestep_range=np.arange(0,6,1), min_grid_size=1)

print("Time taken: " + str(time.time() - t0))

# plot_corr_density_file(mode, nPart, phi, noise, K, Rp, xTy, seed_range, r_scale=r_scale, log_y=log_y, bin_ratio=bin_ratio)

#for seed in seed_range:
#    del_pos(mode, nPart, phi, noise, K, Rp, xTy, seed)

#print("Time taken: " + str(time.time() - t0))

