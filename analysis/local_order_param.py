import numpy as np
import sys
from analysis_functions import *
import time
import numba
import seaborn as sns

@numba.jit(nopython=True)
def pbc_wrap_calc(x, L):
    """
    Wrap points into periodic box with length L (from -L/2 to L/2) for calculations
    """
    return x - L*np.round(x/L)

# Calculate distances between all particles
@numba.jit(nopython=True)
def get_particle_distances(nPart, phi, xTy, x, y):
    nPart = len(x)
    L = np.sqrt(nPart / (phi*xTy))
    Ly = L
    Lx = L*xTy
    D = np.zeros((nPart, nPart)) # Could do as vector instead
    for i in range(nPart):
        for j in range(i+1, nPart):
            xij = pbc_wrap_calc(x[i]-x[j], Lx)
            yij = pbc_wrap_calc(y[i]-y[j], Ly)
            rij = np.sqrt(xij**2 + yij**2)
            D[i,j] = rij
            D[j,i] = rij
    return D

@numba.jit(nopython=True)
def local_order_param(D, theta, r_max):
    nPart = len(theta)
    order_param = np.zeros(nPart)
    for i in range(nPart):
        idx = np.where(D[i,:] <= r_max)[0]
        theta_i = theta[idx]
        order_param[i] = np.sqrt(np.sum(np.cos(theta_i))**2 + np.sum(np.sin(theta_i))**2)/len(theta_i)

    return order_param

def global_order_param(theta):
    return np.sqrt(np.sum(np.cos(theta))**2 + np.sum(np.sin(theta))**2)/len(theta)

def local_order_param_mean(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max_range):
    posExFile = get_file_path(mode, nPart, phi, noise, K, Rp, xTy, seed, file_name='pos_exact')
    x, y, theta, viewtime = get_pos_ex_snapshot(posExFile)
    x = np.array(x)
    y = np.array(y)
    theta = np.array(theta)

    D = get_particle_distances(nPart, phi, xTy, x, y)

    o_plot = []
    for r_max in r_max_range:
        o_plot.append(np.mean(local_order_param(D, theta, r_max)))

    g = global_order_param(theta)

    return o_plot, g

def local_order_param_all(mode, nPart, phi, noise, K, Rp, xTy, seed_range, r_max):
    o_list = []
    for seed in seed_range:
        posExFile = get_file_path(mode, nPart, phi, noise, K, Rp, xTy, seed, file_name='pos_exact')
        x, y, theta, viewtime = get_pos_ex_snapshot(posExFile)
        x = np.array(x)
        y = np.array(y)
        theta = np.array(theta)

        D = get_particle_distances(nPart, phi, xTy, x, y)

        
        o_list += local_order_param(D, theta, r_max).tolist()

    return o_list

def plot_local_order_vs_l(mode, nPart, phi, noise, K_avg_range, K_std_range, Rp, xTy, seed_range, r_max_range, show_g=False):
    fig, ax = plt.subplots()

    for K_avg in K_avg_range:
        for K_std in K_std_range:
            K = str(K_avg) + "_" + str(K_std)
            o_plot_all = []
            g_all = []
            for seed in seed_range:
                o_plot, g = local_order_param_mean(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max_range)
                o_plot_all.append(o_plot)
                g_all.append(g)
            o_plot_mean = np.mean(o_plot_all, axis=0)
            g_mean = np.mean(g_all)
            ax.plot(r_max_range, o_plot_mean, '-o', label = r"$\overline{K}=$"+ str(K_avg) + r", $\sigma_K=$" + str(K_std))
            if show_g == True:
                ax.hlines(g_mean, 0, r_max_range[-1], linestyle='dashed', color='gray')

    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\Psi(\ell)$')
    ax.legend()

    folder = os.path.abspath('../plots/p_order_local_vs_l/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '.png'
    # filename = "N10000_K0.0_8.0_loglog.png"
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))
    plt.show()

def plot_local_order_vs_l_decay(mode, nPart, phi, noise, K_avg_range, K_std_range, Rp, xTy, seed_range, r_max_range):
    fig, ax = plt.subplots()

    for K_avg in K_avg_range:
        for K_std in K_std_range:
            K = str(K_avg) + "_" + str(K_std)
            o_plot_all = []
            for seed in seed_range:
                o_plot, g = local_order_param_mean(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max_range)
                o_plot_all.append(o_plot-g)
            o_plot_mean = np.mean(o_plot_all, axis=0)
            ax.plot(r_max_range, o_plot_mean, '-o', label = r"$\overline{K}=$"+ str(K_avg) + r", $\sigma_K=$" + str(K_std))

    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\Psi(\ell)$')
    ax.legend()

    ax.set_xscale('log')
    ax.set_yscale('log')

    folder = os.path.abspath('../plots/p_order_local_vs_l_decay/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '.png'
    # filename = "N10000_K0.0_8.0_loglog.png"
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))
    plt.show()

def plot_local_order_hist(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed_range, r_max_range):
    fig, ax = plt.subplots()
    for r_max in r_max_range:
        K = str(K_avg) + "_" + str(K_std)
        o_list = local_order_param_all(mode, nPart, phi, noise, K, Rp, xTy, seed_range, r_max)
        # ax.hist(o_list, bins=100, range=(0,1), density=True, label = r"$\overline{K}=$" + str(K_avg) + r", $\sigma_K=$" + str(K_std) + r", $\ell=$" + str(r_max), alpha=0.5)
        sns.histplot(o_list, bins=100, binrange=(0,1), stat='probability', kde=False, label = r"$\overline{K}=$" + str(K_avg) + r", $\sigma_K=$" + str(K_std) + r", $\ell=$" + str(r_max), alpha=0.5)
    ax.legend()
    ax.set_xlabel(r'$\Psi(\ell)$')
    ax.set_ylabel(r'$P(\Psi(\ell))$')
    ax.set_ylim(0, 0.2)

    folder = os.path.abspath('../plots/p_order_local_hist/')
    filename = "N" + str(nPart) + "_K" + str(K) + ".png"
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

mode = "G"
nPart = 10000
phi = 1.0
noise = "0.20"
K = "0.0_8.0"
Rp = 1.0
xTy = 1.0
seed = 1

K_avg_range = [-1.0, 0.0, 1.0]
K_std_range = [0.0, 1.0, 4.0, 8.0]
seed_range = np.arange(1,21,1)
# r_max_range = np.arange(0,21,1)
r_max_range = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0]

# t0 = time.time()
# plot_local_order_vs_l(mode, nPart, phi, noise, K_avg_range, K_std_range, Rp, xTy, seed_range, r_max_range, show_g=True)
# plot_local_order_vs_l_decay(mode, nPart, phi, noise, K_avg_range, K_std_range, Rp, xTy, seed_range, r_max_range)
# print(time.time()-t0)

t0 = time.time()
for K_avg in K_avg_range:
    for K_std in K_std_range:
        plot_local_order_hist(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed_range, r_max_range)

print(time.time()-t0)
