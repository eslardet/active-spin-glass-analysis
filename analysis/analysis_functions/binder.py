import sys
sys.path.insert(1, '././analysis_functions')
from import_files import *
from stats import *

import numpy as np
import matplotlib.pyplot as plt


def get_binder(mode, nPart, phi, noise, K, Rp, xTy, seed_range):
    p_2 = []
    p_4 = []
    for seed in seed_range:
        stats_dir = read_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
        if np.isnan(stats_dir["p_2"]) or np.isnan(stats_dir["p_4"]):
            print("Nan, s=" + str(seed))
        else:
            p_2.append(stats_dir["p_2"])
            p_4.append(stats_dir["p_4"])
    
    p_2_av = np.mean(p_2)
    p_4_av = np.mean(p_4)

    binder = 1 - p_4_av/(3*(p_2_av**2))

    return binder

def plot_binder_Kavg(mode, nPart_range, phi, noise_range, K_avg_range, K_std_range, Rp_range, xTy, seed_range, save_data=False):
    """
    Plot steady state binder cumulant against Kavg, for each fixed K_std value and noise value
    Averaged over a number of realizations
    """
    folder = os.path.abspath('../plots/binder_vs_Kavg/')
    if not os.path.exists(folder):
        os.makedirs(folder)
    if save_data == True:
        save_file = open(os.path.join(folder, 'data.txt'), "w")

    fig, ax = plt.subplots()
    for nPart in nPart_range:
        for Rp in Rp_range:
            for noise in noise_range:
                for K_std in K_std_range:
                    binder = []
                    for K_avg in K_avg_range:
                        K = str(K_avg) + "_" + str(K_std)
                        p_2_sum = 0
                        p_4_sum = 0
                        error_count = 0
                        for seed in seed_range:
                            sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
                            if not os.path.exists(os.path.join(sim_dir, 'stats')):
                                print(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed)
                                error_count += 1
                                # write_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed)
                            else:
                                stats_dir = read_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
                                if np.isnan(stats_dir["p_2"]) or np.isnan(stats_dir["p_4"]):
                                    print("Nan, s=" + str(seed))
                                    error_count += 1
                                else:
                                    p_2_sum += stats_dir["p_2"]
                                    p_4_sum += stats_dir["p_4"]
                        p_2_av = p_2_sum/(len(seed_range)-error_count)
                        p_4_av = p_4_sum/(len(seed_range)-error_count)

                        binder.append(1 - p_4_av/(3*(p_2_av**2)))

                    ax.plot([float(k) for k in K_avg_range], binder, '-o', label=r"$N=$" + str(nPart) + r"; $\sigma_K=$" + str(K_std) + r"; $\eta=$" + str(noise) + r"; $R_p=$" + str(Rp))
                    if save_data == True:
                        save_file.write(str(nPart) + "\t" + str(Rp) + "\t" + str(phi) + "\t" + str(K_std) + "\n")
                        for k in K_avg_range:
                            save_file.write(str(k) + "\t")
                        save_file.write("\n")
                        for b in binder:
                            save_file.write(str(b) + "\t")
                        save_file.write("\n")

    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_Kstd' + str(K_std) + '_Rp' + str(Rp) + '_xTy' + str(xTy)

    if save_data == True:
        save_file.close()
        os.rename(os.path.join(folder, "data.txt"), os.path.join(folder, filename + '.txt'))
    ax.set_xlabel(r"$K_{AVG}$")
    ax.set_ylabel(r"Binder cumulant, $G$")
    # ax.set_ylim([0,1])
    # ax.set_xlim([-1,2])
    ax.legend()

    plt.savefig(os.path.join(folder, filename + '.png'))

def plot_binder_noise(mode, nPart_range, phi, noise_range, K_avg_range, K_std_range, Rp_range, xTy, seed_range):
    """
    Plot steady state binder cumulant against noise
    Averaged over a number of realizations
    """
    fig, ax = plt.subplots()
    for nPart in nPart_range:
        for Rp in Rp_range:
            for K_avg in K_avg_range:
                for K_std in K_std_range:
                    binder = []
                    for noise in noise_range:
                        K = str(K_avg) + "_" + str(K_std)
                        p_2_sum = 0
                        p_4_sum = 0
                        error_count = 0
                        for seed in seed_range:
                            sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
                            if not os.path.exists(os.path.join(sim_dir, 'stats')):
                                print(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed)
                                error_count += 1
                                # write_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed)
                            else:
                                stats_dir = read_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
                                if np.isnan(stats_dir["p_2"]) or np.isnan(stats_dir["p_4"]):
                                    print("Nan, s=" + str(seed))
                                    error_count += 1
                                else:
                                    p_2_sum += stats_dir["p_2"]
                                    p_4_sum += stats_dir["p_4"]
                        p_2_av = p_2_sum/(len(seed_range)-error_count)
                        p_4_av = p_4_sum/(len(seed_range)-error_count)

                        binder.append(1 - p_4_av/(3*(p_2_av**2)))

                    ax.plot([float(k) for k in noise_range], binder, '-o')
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"Binder cumulant, $G$")
    # ax.set_ylim([0,1])
    # ax.set_xlim([-1,2])
    # ax.legend()

    folder = os.path.abspath('../plots/binder_vs_noise/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_Kavg' + str(K_avg) + '_Kstd' + str(K_std) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))