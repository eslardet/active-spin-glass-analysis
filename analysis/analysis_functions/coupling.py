import sys
sys.path.insert(1, '././analysis_functions')
from import_files import *
from stats import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

def get_couplings(mode, nPart, phi, noise, K, Rp, xTy, seed):
    sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
    couplingFile = os.path.join(sim_dir, "coupling")
    with open(couplingFile) as f:
        K_list = []
        for Kij in f:
            K_list.append(float(Kij))
    return K_list

def get_coupling_rij(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max=None, pos_ex=True, init_pos=False, timestep_range=[0]):
    if pos_ex == True:
        posFileExact = get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, file_name='pos_exact')
        x, y = get_pos_ex_snapshot(file=posFileExact)[:2]
        x_all = [x]
        y_all = [y]
    elif init_pos == True:
        initFile = get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, file_name='initpos')
        x, y = get_pos_snapshot(initFile, nPart, timestep=0)[:2]
        x_all = [x]
        y_all = [y]     
    else:
        inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
        x_all = []
        y_all = []
        for t in timestep_range:
            x, y = get_pos_snapshot(posFile, nPart, timestep=t)[:2]
            x_all.append(x)
            y_all.append(y)

    sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
    couplingFile = os.path.join(sim_dir, "coupling")

    L = np.sqrt(nPart / (phi*xTy))
    Ly = L
    Lx = L*xTy

    if r_max == None:
        r_max = Lx
    
    num_t = len(timestep_range)

    with open(couplingFile) as f:
        line = 0
        i = 0
        j = i+1
        k = nPart-1
        start_row = 0
        K_list = []
        rij_list = []
        for Kij in f:
            for t in range(num_t):
                x = x_all[t]
                y = y_all[t]
                xij = x[i] - x[j]
                xij = xij - Lx*round(xij/Lx)
                yij = y[i] - y[j]
                yij = yij - Ly*round(yij/Ly)
                rij = np.sqrt(xij**2 + yij**2)
                if rij < r_max:
                    K_list.append(float(Kij))
                    rij_list.append(rij)

            line += 1

            if line == start_row + k:
                i += 1
                j = i+1
                k -= 1
                start_row = line
            else:
                j += 1
    return K_list, rij_list


def plot_dist_coupling_hist(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed_range, 
                            pos_ex=True, init_pos=False, timestep_range=[0], bin_size=100, bin_ratio=1, r_max=None, K_max=None, shift=False, save_data=False):

    K = str(K_avg) + "_" + str(K_std)
    folder = os.path.abspath('../plots/dist_coupling/')
    if init_pos == True:
        filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_hist_init'
    else:
        filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_hist'
    if not os.path.exists(folder):
        os.makedirs(folder)

    K_list = []
    rij_list = []
    for seed in seed_range:
        K_seed, rij_seed = get_coupling_rij(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, r_max=r_max, pos_ex=pos_ex, timestep_range=timestep_range, init_pos=init_pos)
        K_list.extend(K_seed)
        rij_list.extend(rij_seed)

    if save_data == True:
        save_file = open(os.path.join(folder, filename + '.txt'), "w")
        for i in range(len(K_list)):
            save_file.write(str(K_list[i]) + "\t" + str(rij_list[i]) + "\n")
        save_file.close()

    ## Shift to origin
    if shift == True:
        K_list = [k - K_avg for k in K_list]

    fig, ax = plt.subplots(figsize=(10,10/bin_ratio)) 
    # plt.tight_layout()
    
    if K_max != None:
        ax.hist2d(K_list, rij_list, bins=(bin_size, int(bin_size/bin_ratio)), range=[[-K_max,K_max], [0,r_max]], cmap=cm.jet)
    else:
        ax.hist2d(K_list, rij_list, bins=(bin_size, int(bin_size/bin_ratio)), cmap=cm.jet)
        
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$K_{ij}$")
    ax.set_ylabel(r"$r_{ij}$")

    plt.savefig(os.path.join(folder, filename + ".png"))

def plot_dist_coupling_hist_diff(mode, nPart, phi, noise, K_avg, K_avg_compare, K_std, Rp, xTy, seed, pos_ex=True, timestep_range=[0], bin_size=100, bin_ratio=1, r_max=None, K_max=None):
    K = str(K_avg_compare) + "_" + str(K_std)
    K_list_compare, rij_list_compare = get_coupling_rij(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, r_max=r_max, pos_ex=pos_ex, timestep_range=timestep_range)
    K = str(K_avg) + "_" + str(K_std)
    K_list, rij_list = get_coupling_rij(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, r_max=r_max, pos_ex=pos_ex, timestep_range=timestep_range)
    
    ## Shift to origin
    K_list = [k - K_avg for k in K_list]
    K_list_compare = [k - K_avg_compare for k in K_list_compare]

    # fig, ax = plt.subplots(3, figsize=(3,9))
    fig, ax = plt.subplots(figsize=(10,10/bin_ratio)) 

    # plt.tight_layout()
    if K_max != None:
        h1, xedges1, yedges1, image_1 = plt.hist2d(K_list, rij_list, bins=(bin_size, int(bin_size/bin_ratio)), range= [[-K_max,K_max], [0,r_max]], cmap=cm.jet)
    else: 
        h1, xedges1, yedges1, image_1 = plt.hist2d(K_list, rij_list, bins=(bin_size, int(bin_size/bin_ratio)), cmap=cm.jet)
    h0, xedges0, yedges0, image_0 = plt.hist2d(K_list_compare, rij_list_compare, bins=(xedges1, yedges1), cmap=cm.jet)
    ax.clear()
    ax.pcolormesh(xedges1, yedges1, (h1-h0).T)
    
    # for a in ax:
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$K_{ij}$")
    ax.set_ylabel(r"$r_{ij}$")
    # ax.set_ylabel(r"$\langle r_{ij}\rangle_t$") ## when time average uncomment

    # plt.show()
    folder = os.path.abspath('../plots/dist_coupling/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed) + '_histdiff.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))


def plot_dist_coupling_hist_diff_init(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed_range, pos_ex=True, timestep_range=[0], bin_size=100, bin_ratio=1, r_max=None, K_max=None, shift=False):
    K = str(K_avg) + "_" + str(K_std)
    K_list = []
    rij_list = []
    K_list_compare = []
    rij_list_compare = []
    for seed in seed_range:
        K_seed, rij_seed = get_coupling_rij(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, r_max=r_max, pos_ex=pos_ex, timestep_range=timestep_range)
        K_list.extend(K_seed)
        rij_list.extend(rij_seed)

        K_list_compare, rij_list_compare = get_coupling_rij(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, r_max=r_max, timestep_range=timestep_range, pos_ex=False, init_pos=True)
        K_list_compare.extend(K_seed)
        rij_list_compare.extend(rij_seed)
    
    if shift == True:
        K_list = [k - K_avg for k in K_list]
        K_list_compare = [k - K_avg for k in K_list_compare]

    # fig, ax = plt.subplots(3, figsize=(3,9))
    fig, ax = plt.subplots(figsize=(10,10/bin_ratio)) 

    # plt.tight_layout()
    if K_max != None:
        h1, xedges1, yedges1, image_1 = plt.hist2d(K_list, rij_list, bins=(bin_size, int(bin_size/bin_ratio)), range= [[-K_max,K_max], [0,r_max]], cmap=cm.jet)
    else: 
        h1, xedges1, yedges1, image_1 = plt.hist2d(K_list, rij_list, bins=(bin_size, int(bin_size/bin_ratio)), cmap=cm.jet)
    h0, xedges0, yedges0, image_0 = plt.hist2d(K_list_compare, rij_list_compare, bins=(xedges1, yedges1), cmap=cm.jet)
    ax.clear()
    ax.pcolormesh(xedges1, yedges1, (h1-h0).T)
    
    # for a in ax:
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$K_{ij}$")
    ax.set_ylabel(r"$r_{ij}$")
    # ax.set_ylabel(r"$\langle r_{ij}\rangle_t$") ## when time average uncomment

    # plt.show()
    folder = os.path.abspath('../plots/dist_coupling/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed) + '_histdiff.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))