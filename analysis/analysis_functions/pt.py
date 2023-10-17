import sys
sys.path.insert(1, '././analysis_functions')
from import_files import *
from stats import *

import numpy as np
import matplotlib.pyplot as plt

## Polar order ##
def plot_porder_Kavg(mode, nPart_range, phi_range, noise_range, K_avg_range, K_std_range, Rp_range, xTy, seed_range, save_data=False):
    """
    Plot steady state polar order parameter against Kavg, for each fixed K_std value and noise value
    Averaged over a number of realizations
    """
    folder = os.path.abspath('../plots/p_order_vs_Kavg/')
    if not os.path.exists(folder):
        os.makedirs(folder)
    if save_data == True:
        save_file = open(os.path.join(folder, "data.txt"), "w")

    fig, ax = plt.subplots()
    for nPart in nPart_range:
        for phi in phi_range:
            for Rp in Rp_range:
                for noise in noise_range:
                    for K_std in K_std_range:
                        p_ss = []
                        for K_avg in K_avg_range:
                            K = str(K_avg) + "_" + str(K_std)
                            p_ss_sum = 0
                            error_count = 0
                            for seed in seed_range:
                                sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
                                if not os.path.exists(os.path.join(sim_dir, 'stats')):
                                    print(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed)
                                    error_count += 1
                                    # write_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed)
                                else:
                                    p_mean = read_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)["p_mean"]
                                    if np.isnan(p_mean):
                                        print("Nan")
                                        print(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed)
                                        error_count += 1
                                    else:
                                        p_ss_sum += p_mean
                            p_ss.append(p_ss_sum/(len(seed_range)-error_count))

                        ax.plot([float(k) for k in K_avg_range], p_ss, '-o', label=r"$N=$" + str(nPart) + r"; $K_{STD}=$" + str(K_std) + r"; $\rho=$" + str(phi) + r"; $\eta=$" + str(noise) + r"; $R_p=$" + str(Rp))
                        if save_data == True:
                            save_file.write(str(nPart) + "\t" + str(Rp) + "\t" + str(phi) + "\t" + str(noise) + "\t" + str(K_std) + "\n")
                            for K_avg in K_avg_range:
                                save_file.write(str(K_avg) + "\t")
                            save_file.write("\n")
                            for p in p_ss:
                                save_file.write(str(p) + "\t")
                            save_file.write("\n")

    ax.set_xlabel(r"$K_{AVG}$")
    ax.set_ylabel(r"Polar order parameter, $\Psi$")
    ax.set_ylim([0,1])
    # ax.set_xlim([-1,2])
    ax.legend()

    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_Kstd' + str(K_std) + '_Rp' + str(Rp) + '_xTy' + str(xTy)
    if save_data == True:
        save_file.close()
        os.rename(os.path.join(folder, "data.txt"), os.path.join(folder, filename + '.txt'))
    plt.savefig(os.path.join(folder, filename + '.png'))

def plot_porder_noise(mode, nPart_range, phi, noise_range, K_avg_range, K_std_range, Rp_range, xTy, seed_range, save_data=False):
    """
    Plot steady state polar order parameter against noise
    Averaged over a number of realizations
    """
    folder = os.path.abspath('../plots/p_order_vs_noise/')
    if not os.path.exists(folder):
        os.makedirs(folder)
    if save_data == True:
        save_file = open(os.path.join(folder, "data.txt"), "w")

    fig, ax = plt.subplots()
    for nPart in nPart_range:
        for Rp in Rp_range:
            for K_avg in K_avg_range:
                for K_std in K_std_range:
                    p_ss = []
                    for noise in noise_range:
                        K = str(K_avg) + "_" + str(K_std)
                        p_ss_sum = 0
                        error_count = 0
                        for seed in seed_range:
                            sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
                            if not os.path.exists(os.path.join(sim_dir, 'stats')):
                                print(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed)
                                error_count += 1
                                # write_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed)
                            else:
                                p_mean = read_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)["p_mean"]
                                if np.isnan(p_mean):
                                    print("Nan")
                                    print(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed)
                                    error_count += 1
                                else:
                                    p_ss_sum += p_mean
                        p_ss.append(p_ss_sum/(len(seed_range)-error_count))

                    ax.plot([float(k) for k in noise_range], p_ss, '-o')
                    if save_data == True:
                        save_file.write(str(nPart) + "\t" + str(Rp) + "\t" + str(phi) + "\t" + str(K_avg) + "\t" + str(K_std) + "\n")
                        for noise in noise_range:
                            save_file.write(str(noise) + "\t")
                        save_file.write("\n")
                        for p in p_ss:
                            save_file.write(str(p) + "\t")
                        save_file.write("\n")

    # noise_range = [float(i) for i in noise_range]
    # ax.plot(noise_range, p_ss, '-o')
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"Polar order parameter, $\Psi$")
    ax.set_ylim([0,1])
    
    folder = os.path.abspath('../plots/p_order_vs_noise/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_Kavg' + str(K_avg)+ '_Kstd' + str(K_std) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

def plot_porder_phi(mode, nPart, phi_range, noise, K, Rp, xTy, seed_range):
    """
    Plot steady state polar order parameter against phi
    Averaged over a number of realizations
    """
    fig, ax = plt.subplots()
    p_ss = []
    for phi in phi_range:
        p_ss_sum = 0
        for seed in seed_range:
            sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
            if not os.path.exists(os.path.join(sim_dir, 'stats')):
                write_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
            p_ss_sum += read_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)["p_mean"]
        p_ss.append(p_ss_sum/len(seed_range))

    ax.plot(phi_range, p_ss, '-o')
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"Polar order parameter, $\Psi$")
    ax.set_ylim([0,1])
    
    folder = os.path.abspath('../plots/p_order_vs_phi/')
    filename = mode + '_N' + str(nPart) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

def plot_porder_K0(mode, nPart, phi, noise, K_range, Rp, xTy, seed_range):
    """
    Plot steady state polar order parameter against K0
    Averaged over a number of realizations
    """
    fig, ax = plt.subplots()
    p_ss = []
    for K in K_range:
        p_ss_sum = 0
        for seed in seed_range:
            sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
            if not os.path.exists(os.path.join(sim_dir, 'stats')):
                write_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
            p_ss_sum += read_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)["p_mean"]
        p_ss.append(p_ss_sum/len(seed_range))

    ax.plot(K_range, p_ss, '-o')
    ax.set_xlabel(r"$K_{0}$")
    ax.set_ylabel(r"Polar order parameter, $\Psi$")
    ax.set_ylim([0,1])
    ax.legend()

    folder = os.path.abspath('../plots/p_order_vs_Kavg/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_Kstd0.0' + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

def plot_porder_Kavg_t(mode, nPart_range, phi, noise_range, K_avg_range, K_std_range, Rp_range, xTy, seed_range, t, save_data=False):
    """
    Plot steady state polar order parameter against Kavg, for each fixed K_std value and noise value
    Averaged over a number of realizations
    """
    folder = os.path.abspath('../plots/p_order_vs_Kavg/')
    if not os.path.exists(folder):
        os.makedirs(folder)
    if save_data == True:
        save_file = open(os.path.join(folder, "data.txt"), "w")

    fig, ax = plt.subplots()
    for nPart in nPart_range:
        for Rp in Rp_range:
            for noise in noise_range:
                for K_std in K_std_range:
                    p_ss = []
                    for K_avg in K_avg_range:
                        K = str(K_avg) + "_" + str(K_std)
                        p_ss_sum = 0
                        error_count = 0
                        for seed in seed_range:
                            sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
                            posFileExact = get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, file_name='pos_exact')
                            x, y, theta, view_time = get_pos_ex_snapshot(file=posFileExact)
                            if not os.path.exists(os.path.join(sim_dir, 'stats')):
                                print(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed)
                                error_count += 1
                                # write_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed)
                            elif int(view_time) != t:
                                error_count += 1
                            else:
                                p_mean = read_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)["p_mean"]
                                if np.isnan(p_mean):
                                    print("Nan")
                                    print(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed)
                                    error_count += 1
                                else:
                                    p_ss_sum += p_mean
                        p_ss.append(p_ss_sum/(len(seed_range)-error_count))

                    ax.plot([float(k) for k in K_avg_range], p_ss, '-o', label=r"$N=$" + str(nPart) + r"; $K_{STD}=$" + str(K_std) + r"; $\eta=$" + str(noise) + r"; $R_p=$" + str(Rp))
                    if save_data == True:
                        save_file.write(str(nPart) + "\t" + str(Rp) + "\t" + str(phi) + "\t" + str(noise) + "\t" + str(K_std) + "\n")
                        for K_avg in K_avg_range:
                            save_file.write(str(K_avg) + "\t")
                        save_file.write("\n")
                        for p in p_ss:
                            save_file.write(str(p) + "\t")
                        save_file.write("\n")

    ax.set_xlabel(r"$K_{AVG}$")
    ax.set_ylabel(r"Polar order parameter, $\Psi$")
    ax.set_ylim([0,1])
    # ax.set_xlim([-1,2])
    ax.legend()

    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_Kstd' + str(K_std) + '_Rp' + str(Rp) + '_xTy' + str(xTy)
    if save_data == True:
        save_file.close()
        os.rename(os.path.join(folder, "data.txt"), os.path.join(folder, filename + '.txt'))
    plt.savefig(os.path.join(folder, filename + '.png'))

def plot_porder_Kavg_ax(mode, nPart, phi, noise_range, K_avg_range, K_std_range, Rp, xTy, seed_range, ax):
    """
    Plot steady state polar order parameter against Kavg, for each fixed K_std value and noise value
    Averaged over a number of realizations
    Return axis for further plotting
    """
    # fig, ax = plt.subplots()
    for noise in noise_range:
        for K_std in K_std_range:
            p_ss = []
            for K_avg in K_avg_range:
                K = str(K_avg) + "_" + str(K_std)
                p_ss_sum = 0
                error_count = 0
                for seed in seed_range:
                    sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
                    if not os.path.exists(os.path.join(sim_dir, 'stats')):
                        print(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed)
                        error_count += 1
                        # write_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed)
                    p_ss_sum += read_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)["p_mean"]
                p_ss.append(p_ss_sum/(len(seed_range)-error_count))

            ax.plot(K_avg_range, p_ss, '-o', label=r"$K_{STD}=$" + str(K_std) + r"; $\eta=$" + str(noise))
    ax.set_xlabel(r"$K_{AVG}$")
    ax.set_ylabel(r"Polar order parameter, $\Psi$")
    ax.set_ylim([0,1])
    ax.legend()

    return ax


def plot_porder_alpha_old(mode, nPart_range, phi, noise_range, K0_range, alpha_range, Rp_range, xTy, seed_range, save_data=False):
    """
    Plot steady state polar order parameter against alpha, for each fixed K0 & K1 value and noise value
    Averaged over a number of realizations
    """
    folder = os.path.abspath('../plots/p_order_vs_alpha/')
    if not os.path.exists(folder):
        os.makedirs(folder)
    if save_data == True:
        save_file = open(os.path.join(folder, "data.txt"), "w")

    fig, ax = plt.subplots()
    for nPart in nPart_range:
        for Rp in Rp_range:
            for noise in noise_range:
                for K0 in K0_range:
                    p_ss = []
                    for alpha in alpha_range:
                        K = str(K0) + "_" + str(alpha)
                        p_ss_sum = 0
                        error_count = 0
                        for seed in seed_range:
                            sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
                            if not os.path.exists(os.path.join(sim_dir, 'stats')):
                                print(mode, nPart, phi, noise, K0, alpha, Rp, xTy, seed)
                                error_count += 1
                                # write_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed)
                            else:
                                p_mean = read_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)["p_mean"]
                                if np.isnan(p_mean):
                                    print("Nan")
                                    print(mode, nPart, phi, noise, K0, alpha, Rp, xTy, seed)
                                    error_count += 1
                                else:
                                    p_ss_sum += p_mean
                        p_ss.append(p_ss_sum/(len(seed_range)-error_count))

                    # ax.plot([float(a) for a in alpha_range], p_ss, '-o', label=r"$N=$" + str(nPart) + r"; $K_{0}=$" + str(K0) + r"; $\eta=$" + str(noise) + r"; $R_p=$" + str(Rp))
                    ax.plot([float(a) for a in alpha_range], p_ss, '-o', label=r"$N=$" + str(nPart) + r"; $K_{0}=$" + str(K0))
                    if save_data == True:
                        save_file.write(str(nPart) + "\t" + str(Rp) + "\t" + str(phi) + "\t" + str(noise) + "\t" + str(K0) + "\n")
                        for alpha in alpha_range:
                            save_file.write(str(alpha) + "\t")
                        save_file.write("\n")
                        for p in p_ss:
                            save_file.write(str(p) + "\t")
                        save_file.write("\n")

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"Polar order parameter, $\Psi$")
    ax.set_ylim([0,1])
    # ax.set_xlim([-1,2])
    ax.legend()

    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_Rp' + str(Rp) + '_xTy' + str(xTy)
    if save_data == True:
        save_file.close()
        os.rename(os.path.join(folder, "data.txt"), os.path.join(folder, filename + '.txt'))
    plt.savefig(os.path.join(folder, filename + '.png'))

def plot_porder_alpha(mode, nPart_range, phi, noise_range, K0_range, K1_range, alpha_range, Rp_range, xTy, seed_range, save_data=False):
    """
    Plot steady state polar order parameter against alpha, for each fixed K0 & K1 value and noise value
    Averaged over a number of realizations
    """
    folder = os.path.abspath('../plots/p_order_vs_alpha/')
    if not os.path.exists(folder):
        os.makedirs(folder)
    if save_data == True:
        save_file = open(os.path.join(folder, "data.txt"), "w")

    fig, ax = plt.subplots()
    for nPart in nPart_range:
        for Rp in Rp_range:
            for noise in noise_range:
                for K1 in K1_range:
                    for K0 in K0_range:
                        p_ss = []
                        for alpha in alpha_range:
                            K = str(K0) + "_" + str(K1) + "_" + str(alpha)
                            p_ss_sum = 0
                            error_count = 0
                            for seed in seed_range:
                                sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
                                if not os.path.exists(os.path.join(sim_dir, 'stats')):
                                    print(mode, nPart, phi, noise, K0, K1, alpha, Rp, xTy, seed)
                                    error_count += 1
                                    # write_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed)
                                else:
                                    p_mean = read_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)["p_mean"]
                                    if np.isnan(p_mean):
                                        print("Nan")
                                        print(mode, nPart, phi, noise, K0, K1, alpha, Rp, xTy, seed)
                                        error_count += 1
                                    else:
                                        p_ss_sum += p_mean
                            p_ss.append(p_ss_sum/(len(seed_range)-error_count))

                        # ax.plot([float(a) for a in alpha_range], p_ss, '-o', label=r"$N=$" + str(nPart) + r"; $K_{0}=$" + str(K0) + r"; $\eta=$" + str(noise) + r"; $R_p=$" + str(Rp))
                        ax.plot([float(a) for a in alpha_range], p_ss, '-o', label=r"$N=$" + str(nPart) + r"; $K_{+}=$" + str(K0) + r"; $K_{-}=$" + str(K1))
                        if save_data == True:
                            save_file.write(str(nPart) + "\t" + str(Rp) + "\t" + str(phi) + "\t" + str(noise) + "\t" + str(K0) + "\t" + str(K1) + "\n")
                            for alpha in alpha_range:
                                save_file.write(str(alpha) + "\t")
                            save_file.write("\n")
                            for p in p_ss:
                                save_file.write(str(p) + "\t")
                            save_file.write("\n")

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"Polar order parameter, $\Psi$")
    ax.set_ylim([0,1])
    # ax.set_xlim([-1,2])
    ax.legend()

    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_Rp' + str(Rp) + '_xTy' + str(xTy)
    if save_data == True:
        save_file.close()
        os.rename(os.path.join(folder, "data.txt"), os.path.join(folder, filename + '.txt'))
    plt.savefig(os.path.join(folder, filename + '.png'))

def plot_porder_K0(mode, nPart_range, phi, noise_range, K0_range, K1_range, alpha_range, Rp_range, xTy, seed_range, save_data=False):
    """
    Plot steady state polar order parameter against K+, for each fixed alpha & K1 value and noise value
    Averaged over a number of realizations
    """
    folder = os.path.abspath('../plots/p_order_vs_K+/')
    if not os.path.exists(folder):
        os.makedirs(folder)
    if save_data == True:
        save_file = open(os.path.join(folder, "data.txt"), "w")

    fig, ax = plt.subplots()
    for nPart in nPart_range:
        for Rp in Rp_range:
            for noise in noise_range:
                for K1 in K1_range:
                    for alpha in alpha_range:
                        p_ss = []
                        for K0 in K0_range:
                            K = str(K0) + "_" + str(K1) + "_" + str(alpha)
                            p_ss_sum = 0
                            error_count = 0
                            for seed in seed_range:
                                sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
                                if not os.path.exists(os.path.join(sim_dir, 'stats')):
                                    print(mode, nPart, phi, noise, K0, K1, alpha, Rp, xTy, seed)
                                    error_count += 1
                                    # write_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed)
                                else:
                                    p_mean = read_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)["p_mean"]
                                    if np.isnan(p_mean):
                                        print("Nan")
                                        print(mode, nPart, phi, noise, K0, K1, alpha, Rp, xTy, seed)
                                        error_count += 1
                                    else:
                                        p_ss_sum += p_mean
                            p_ss.append(p_ss_sum/(len(seed_range)-error_count))

                        # ax.plot([float(a) for a in alpha_range], p_ss, '-o', label=r"$N=$" + str(nPart) + r"; $K_{0}=$" + str(K0) + r"; $\eta=$" + str(noise) + r"; $R_p=$" + str(Rp))
                        ax.plot([float(a) for a in K0_range], p_ss, '-o', label=r"$N=$" + str(nPart) + r"; $K_{-}=$" + str(K1) + r"; $\alpha=$" + str(alpha))

    ax.set_xlabel(r"$K_{+}$")
    ax.set_ylabel(r"Polar order parameter, $\Psi$")
    ax.set_ylim([0,1])
    # ax.set_xlim([-1,2])
    ax.legend()

    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_alpha' + str(alpha)
    plt.savefig(os.path.join(folder, filename + '.png'))

def plot_porder_Kstd(mode, nPart, phi, noise, K_avg_range, K_std_range, Rp, xTy, seed_range):
    """
    Plot steady state polar order parameter against Kstd, for each fixed K_avg value
    Averaged over a number of realizations
    """
    fig, ax = plt.subplots()
    for K_avg in K_avg_range:
        p_ss = []
        for K_std in K_std_range:
            K = str(K_avg) + "_" + str(K_std)
            p_ss_sum = 0
            for seed in seed_range:
                sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
                if not os.path.exists(os.path.join(sim_dir, 'stats')):
                    write_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
                p_ss_sum += read_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)["p_mean"]
            p_ss.append(p_ss_sum/len(seed_range))

        ax.plot(K_std_range, p_ss, '-o', label=r"$K_{AVG}=$" + str(K_avg))
    ax.set_xlabel(r"$K_{STD}$")
    ax.set_ylabel(r"Polar order parameter, $\Psi$")
    ax.set_ylim([0,1])
    ax.legend()

    folder = os.path.abspath('../plots/p_order_vs_Kstd/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

def plot_porder_Kratio(mode, nPart, phi, noise, K_avg_range, K_std_range, Rp, xTy, seed_range):
    """
    Plot steady state polar order parameter against Kavg/Kstd
    Averaged over a number of realizations
    """
    fig, ax = plt.subplots()
    for K_std in K_std_range:
        p_ss = []
        for K_avg in K_avg_range:
            K = str(K_avg) + "_" + str(K_std)
            p_ss_sum = 0
            for seed in seed_range:
                sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
                if not os.path.exists(os.path.join(sim_dir, 'stats')):
                    write_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
                p_ss_sum += read_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)["p_mean"]
            p_ss.append(p_ss_sum/len(seed_range))

        ax.plot([i/K_std for i in K_avg_range], p_ss, '-o', label=r"$K_{STD}=$" + str(K_std))
    ax.set_xlabel(r"$K_{AVG}/K_{STD}$")
    ax.set_ylabel(r"Polar order parameter, $\Psi$")
    ax.set_ylim([0,1])
    ax.legend()

    folder = os.path.abspath('../plots/p_order_vs_Kratio/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

def plot_porder_RI(mode, nPart_range, phi_range, noise_range, K_avg_range, K_std_range, Rp_range, xTy, seed_range, save_data=False):
    """
    Plot steady state polar order parameter against RI
    Averaged over a number of realizations
    """
    folder = os.path.abspath('../plots/p_order_vs_RI/')
    if not os.path.exists(folder):
        os.makedirs(folder)
    if save_data == True:
        save_file = open(os.path.join(folder, "data.txt"), "w")

    fig, ax = plt.subplots()
    for nPart in nPart_range:
        for phi in phi_range:
            for noise in noise_range:
                for K_avg in K_avg_range:
                    for K_std in K_std_range:
                        K = str(K_avg) + "_" + str(K_std)
                        p_ss = []
                        for Rp in Rp_range:
                            p_ss_sum = 0
                            error_count = 0
                            for seed in seed_range:
                                sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
                                if not os.path.exists(os.path.join(sim_dir, 'stats')):
                                    print(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed)
                                    error_count += 1
                                    # write_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed)
                                else:
                                    p_mean = read_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)["p_mean"]
                                    if np.isnan(p_mean):
                                        print("Nan")
                                        print(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed)
                                        error_count += 1
                                    else:
                                        p_ss_sum += p_mean
                            p_ss.append(p_ss_sum/(len(seed_range)-error_count))

                        ax.plot([float(k) for k in Rp_range], p_ss, '-o', label=r"$N=$" + str(nPart) + r"; $K_{AVG}=$" + str(K_avg) + r"; $K_{STD}=$" + str(K_std) + r"; $\rho=$" + str(phi) + r"; $\eta=$" + str(noise))
                        if save_data == True:
                            save_file.write(str(nPart) + "\t" + str(phi) + "\t" + str(noise) + "\t" + str(K_avg) + "\t" + str(K_std) + "\n")
                            for r in Rp_range:
                                save_file.write(str(r) + "\t")
                            save_file.write("\n")
                            for p in p_ss:
                                save_file.write(str(p) + "\t")
                            save_file.write("\n")

    ax.set_xlabel(r"$R_I$")
    ax.set_ylabel(r"Polar order parameter, $\Psi$")
    ax.set_ylim([0,1])
    # ax.set_xlim([-1,2])
    ax.legend()

    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_Kavg' + str(K_avg)+ '_Kstd' + str(K_std) + '_xTy' + str(xTy)
    if save_data == True:
        save_file.close()
        os.rename(os.path.join(folder, "data.txt"), os.path.join(folder, filename + '.txt'))
    plt.savefig(os.path.join(folder, filename + '.png'))


## Polar order vs L ##
def plot_porder_logL(mode, nPart_range, phi, noise, K, Rp, xTy, seed_range, y_log=False, save_data=False):
    """
    Plot steady state polar order parameter against log(L)
    Averaged over a number of realizations
    """
    folder = os.path.abspath('../plots/p_order_vs_N')
    filename = mode + '_noise' + str(noise) + '_phi' + str(phi) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy)
    if not os.path.exists(folder):
        os.makedirs(folder)
    if save_data == True:
        save_file = open(os.path.join(folder, filename + ".txt"), "w")

    fig, ax = plt.subplots()
    p_ss_mean = []
    p_ss_sd = []
    for nPart in nPart_range:
        print(nPart)
        # K = str(K_avg) + "_" + str(K_std)
        p_ss_all = []
        error_count = 0
        for seed in seed_range:
            sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
            if not os.path.exists(os.path.join(sim_dir, 'stats')):
                print(mode, nPart, phi, noise, K, Rp, xTy, seed)
                error_count += 1
                # write_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed)
            else:
                p_mean = read_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)["p_mean"]
                if np.isnan(p_mean):
                    print("Nan")
                    print(mode, nPart, phi, noise, K, Rp, xTy, seed)
                    error_count += 1
                else:
                    p_ss_all.append(p_mean)
                    print(p_mean)
        p_ss_mean.append(np.mean(p_ss_all))
        p_ss_sd.append(np.std(p_ss_all))

    # ax.plot([np.sqrt(n/phi) for n in nPart_range], p_ss_mean, '-o')
    ax.errorbar([np.sqrt(n/phi) for n in nPart_range], p_ss_mean, yerr=p_ss_sd, fmt='-o')
    ax.set_xscale("log")
    if y_log ==True:
        ax.set_yscale("log")
    if save_data == True:
        save_file.write(str(noise) + "\t" + str(Rp) + "\t" + str(phi) + "\t" + str(K) + "\n")
        for nPart in nPart_range:
            save_file.write(str(nPart) + "\t")
        save_file.write("\n")
        for p in p_ss_mean:
            save_file.write(str(p) + "\t")
        save_file.write("\n")
        for p in p_ss_sd:
            save_file.write(str(p) + "\t")
        save_file.write("\n")

    # noise_range = [float(i) for i in noise_range]
    # ax.plot(noise_range, p_ss, '-o')
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"Polar order parameter, $\Psi$")
    # ax.set_ylim([0,1])
    # ax.set_title()
    
    folder = os.path.abspath('../plots/p_order_vs_N/')
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename + '.png'), bbox_inches="tight")


## Nematic order ##
def plot_norder_Kavg(mode, nPart_range, phi, noise_range, K_avg_range, K_std_range, Rp_range, xTy, seed_range, save_data=False):
    """
    Plot steady state nematic order parameter against Kavg, for each fixed K_std value and noise value
    Averaged over a number of realizations
    """
    folder = os.path.abspath('../plots/n_order_vs_Kavg/')
    if not os.path.exists(folder):
        os.makedirs(folder)
    if save_data == True:
        save_file = open(os.path.join(folder, "data.txt"), "w")

    fig, ax = plt.subplots()
    for nPart in nPart_range:
        for Rp in Rp_range:
            for noise in noise_range:
                for K_std in K_std_range:
                    p_ss = []
                    for K_avg in K_avg_range:
                        K = str(K_avg) + "_" + str(K_std)
                        p_ss_sum = 0
                        error_count = 0
                        for seed in seed_range:
                            sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
                            if not os.path.exists(os.path.join(sim_dir, 'stats')):
                                print(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed)
                                error_count += 1
                                # write_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed)
                            else:
                                p_mean = read_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)["n_mean"]
                                if np.isnan(p_mean):
                                    print("Nan")
                                    print(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed)
                                    error_count += 1
                                else:
                                    p_ss_sum += p_mean
                        p_ss.append(p_ss_sum/(len(seed_range)-error_count))

                    ax.plot([float(k) for k in K_avg_range], p_ss, '-o', label=r"$N=$" + str(nPart) + r"; $K_{STD}=$" + str(K_std) + r"; $\eta=$" + str(noise) + r"; $R_p=$" + str(Rp))
                    if save_data == True:
                        save_file.write(str(nPart) + "\t" + str(Rp) + "\t" + str(phi) + "\t" + str(noise) + "\t" + str(K_std) + "\n")
                        for K_avg in K_avg_range:
                            save_file.write(str(K_avg) + "\t")
                        save_file.write("\n")
                        for p in p_ss:
                            save_file.write(str(p) + "\t")
                        save_file.write("\n")

    ax.set_xlabel(r"$K_{AVG}$")
    ax.set_ylabel(r"Nematic order parameter, $\Psi$")
    ax.set_ylim([0,1])
    # ax.set_xlim([-1,2])
    ax.legend()

    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_Kstd' + str(K_std) + '_Rp' + str(Rp) + '_xTy' + str(xTy)
    if save_data == True:
        save_file.close()
        os.rename(os.path.join(folder, "data.txt"), os.path.join(folder, filename + '.txt'))
    plt.savefig(os.path.join(folder, filename + '.png'))


## Susceptibility ##
def plot_psus_Kavg(mode, nPart_range, phi_range, noise_range, K_avg_range, K_std_range, Rp_range, xTy, seed_range, save_data=False):
    """
    Plot susceptibility (std of polar order parameter) against Kavg, for each fixed K_std value and noise value
    Averaged over a number of realizations
    """
    folder = os.path.abspath('../plots/p_sus_vs_Kavg/')
    if not os.path.exists(folder):
        os.makedirs(folder)
    if save_data == True:
        save_file = open(os.path.join(folder, "data.txt"), "w")

    fig, ax = plt.subplots()
    for nPart in nPart_range:
        for phi in phi_range:
            for Rp in Rp_range:
                for noise in noise_range:
                    for K_std in K_std_range:
                        p_ss = []
                        for K_avg in K_avg_range:
                            K = str(K_avg) + "_" + str(K_std)
                            p_ss_sum = 0
                            error_count = 0
                            for seed in seed_range:
                                sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
                                if not os.path.exists(os.path.join(sim_dir, 'stats')):
                                    print(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed)
                                    error_count += 1
                                    # write_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed)
                                else:
                                    p_mean = read_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)["p_sus"]
                                    if np.isnan(p_mean):
                                        print("Nan")
                                        print(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed)
                                        error_count += 1
                                    else:
                                        p_ss_sum += p_mean
                            p_ss.append(p_ss_sum/(len(seed_range)-error_count))

                        ax.plot([float(k) for k in K_avg_range], p_ss, '-o', label=r"$N=$" + str(nPart) + r"; $K_{STD}=$" + str(K_std) + r"; $\eta=$" + str(noise))
                        if save_data == True:
                            save_file.write(str(nPart) + "\t" + str(Rp) + "\t" + str(phi) + "\t" + str(noise) + "\t" + str(K_std) + "\n")
                            for K_avg in K_avg_range:
                                save_file.write(str(K_avg) + "\t")
                            save_file.write("\n")
                            for p in p_ss:
                                save_file.write(str(p) + "\t")
                            save_file.write("\n")

    ax.set_xlabel(r"$K_{AVG}$")
    ax.set_ylabel(r"Susceptibility of $\Psi$")
    # ax.set_ylim([0,1])
    # ax.set_xlim([-1,2])
    ax.legend()

    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_Kstd' + str(K_std) + '_Rp' + str(Rp) + '_xTy' + str(xTy)
    if save_data == True:
        save_file.close()
        os.rename(os.path.join(folder, "data.txt"), os.path.join(folder, filename + '.txt'))
    plt.savefig(os.path.join(folder, filename + '.png'))


## Critical value ##
def plot_kcrit_kstd(mode, nPart, phi, noise_range, K_avg_range, K_std_range, Rp, xTy, seed_range, cutoff=0.3):
    """
    Plot Kavg critical value against Kstd
    """
    fig, ax = plt.subplots()

    for noise in noise_range:
        K_crit_list = []

        for K_std in K_std_range:
            K_crit = critical_value_kavg(mode, nPart, phi, noise, K_avg_range, K_std, Rp, xTy, seed_range, cutoff)
            K_crit_list.append(K_crit)
            print(K_crit)

        ax.plot(K_std_range, K_crit_list, '-o', label=r"$\eta = $" + str(noise))
    
    ax.set_xlabel(r"$K_{STD}$")
    ax.set_ylabel(r"$K_{AVG}^C$")
    ax.legend()

    folder = os.path.abspath('../plots/Kavg_crit_vs_Kstd/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))




