import sys
sys.path.insert(1, '././analysis_functions')
from import_files import *
from stats import *

import numpy as np
import matplotlib.pyplot as plt


def centre_of_mass(x, L):
    """
    Find 1D centre of mass in periodic boundaries by mapping onto a unit circle
    https://en.wikipedia.org/wiki/Center_of_mass#Systems_with_periodic_boundary_conditions
    """
    cmap = x/L*2*np.pi
    cmean = np.arctan2(np.mean(np.cos(cmap)), np.mean(np.sin(cmap))) + np.pi
    com = L*cmean/(2*np.pi)
    return com

def mean_dist_com(file, L):
    """
    Calculate mean distance for all particles at a snapshot to the centre of mass
    """
    x, y, theta, view_time = get_pos_ex_snapshot(file)
    com_x = centre_of_mass(x,L)
    com_y = centre_of_mass(y,L)
    nPart = len(x)

    dist = 0
    for i in range(nPart):
        dist += np.sqrt(pbc_wrap_calc(x[i] - com_x, L)**2 + pbc_wrap_calc(y[i]-com_y, L)**2)
    mean_dist = dist/nPart
    return mean_dist

def mean_dist_nn(file, L):
    """
    Calculate mean distance to nearest neighbour
    """
    x, y, theta, view_time = get_pos_ex_snapshot(file)
    nPart = len(x)
    total_dist = 0

    for i in range(nPart):
        min_dist = np.infty # initialise min_dist benchmark
        for j in range(nPart):
            if i != j:
                xij = pbc_wrap_calc(x[i]-x[j],L)
                if np.abs(xij) < min_dist:
                    yij = pbc_wrap_calc(y[i]-y[j],L)
                    if np.abs(yij) < min_dist:
                        rij = np.sqrt(xij**2+yij**2)
                        if rij < min_dist:
                            min_dist = rij
        total_dist += min_dist

    mean_dist_nn = total_dist/nPart

    return mean_dist_nn

def write_cohesion_stats(mode, nPart, phi, noise, K, Rp, xTy, seed):
    """
    Write stats for cohesion (mean distance to centre of mass and mean distance to nearest neighbour) within the simulation folder
    """
    posFileExact = get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, file_name="pos_exact")
    if not os.path.exists(posFileExact):
        print(posFileExact)
        raise Exception("File does not exist")
    
    L = np.sqrt(nPart / (float(phi)*xTy))
    com = mean_dist_com(posFileExact, L)
    nn = mean_dist_nn(posFileExact, L)
    av_n = np.mean(neighbour_counts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max=Rp))

    sim_dir = get_sim_dir(mode, nPart, phi, noise, K, Rp, xTy, seed)
    statsFile = open(os.path.join(sim_dir, "stats_cohesion"), "w")
    statsFile.write(str(com) + '\n')
    statsFile.write(str(nn) + '\n')
    statsFile.write(str(av_n) + '\n')


    statsFile.close()

def read_cohesion_stats(mode, nPart, phi, noise, K, Rp, xTy, seed):
    """
    Read stats file and create dictionary with those statistics
    """
    sim_dir = get_sim_dir(mode, nPart, phi, noise, K, Rp, xTy, seed)

    with open(os.path.join(sim_dir, "stats_cohesion")) as file:
        reader = csv.reader(file, delimiter="\n")
        r = list(reader)
    stats_dict = {}
    stats_dict["com_dist"] = float(r[0][0])
    stats_dict["nn_dist"] = float(r[1][0])
    try:
        stats_dict["av_n"] = float(r[2][0])
    except Exception:
        pass

    return stats_dict

## Plot ##
def plot_com_vs_RI(mode, nPart, phi, noise, K_avg_range, K_std_range, Rp_range, xTy, seed_range, from_stats=True, save_data=False):
    """
    Plot mean distance to centre of mass against the radius of interaction
    """
    folder = os.path.abspath('../plots/com_vs_RI')
    if not os.path.exists(folder):
        os.makedirs(folder)
    if save_data == True:
        filename = mode + '_N' + str(nPart) + '_noise' + str(noise) + '_phi' + str(phi) + '_Kavg' + str(K_avg_range[-1])+ '_Kstd' + str(K_std_range[-1]) + '_xTy' + str(xTy)
        save_file = open(os.path.join(folder, filename + '.txt'), "w")
    L = np.sqrt(nPart / (phi*xTy))

    fig, ax = plt.subplots()


    for K_avg in K_avg_range:
        for K_std in K_std_range:
            K = str(K_avg) + "_" + str(K_std)
            com_arr = []
            com_arr_sd = []
            for Rp in Rp_range:
                com = []
                for seed in seed_range:
                    if from_stats == True:
                        sim_dir = get_sim_dir(mode, nPart, phi, noise, K, Rp, xTy, seed)
                        if not os.path.exists(os.path.join(sim_dir, 'stats_cohesion')):
                            print(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed)
                        else:
                            stats_dict = read_cohesion_stats(mode, nPart, phi, noise, K, Rp, xTy, seed)
                            com.append(stats_dict["com_dist"])
                    else:
                        posFileExact = get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, file_name="pos_exact")
                        if os.path.exists(posFileExact):
                            com.append(mean_dist_com(posFileExact, L))
                com_arr.append(np.mean(com))
                com_arr_sd.append(np.std(com))
            
            ax.plot(Rp_range, com_arr, '-o', label=r"$N=$" + str(nPart)+ r"; $K_{AVG}=$" + str(K_avg) + r"; $K_{STD}=$" + str(K_std) + r"; $\rho=$" + str(phi) + r"; $\eta=$" + str(noise))
            ax.errorbar(Rp_range, com_arr, yerr=com_arr_sd, fmt='none', capsize=3)
            if save_data == True:
                save_file.write(str(nPart) + "\t" + str(noise) + "\t" + str(phi) + "\t" + str(K_avg) + "\t" + str(K_std) + "\n")
                for r in Rp_range:
                    save_file.write(str(r) + "\t")
                save_file.write("\n")
                for p in com_arr:
                    save_file.write(str(p) + "\t")
                save_file.write("\n")
                for p in com_arr_sd:
                    save_file.write(str(p) + "\t")
                save_file.write("\n")
    ax.set_xlabel(r"$R_I$")
    ax.set_ylabel(r"Mean distance to centre of mass")
    ax.legend()
    # ax.set_title(r'$N=$' + str(nPart) + r'; $\rho=$' + str(phi) + r'; $\eta=$' + str(noise))

    folder = os.path.abspath('../plots/com_vs_RI')
    filename = mode + '_N' + str(nPart) + '_noise' + str(noise) + '_phi' + str(phi) + '_Kavg' + str(K_avg)+ '_Kstd' + str(K_std) + '_xTy' + str(xTy) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

def plot_com_vs_noise(mode, nPart_range, phi_range, noise_range, K_avg_range, K_std_range, Rp_range, xTy, seed_range, from_stats=True, save_data=False):
    """
    Plot mean distance to centre of mass against the noise strength
    """
    folder = os.path.abspath('../plots/com_vs_noise')
    if not os.path.exists(folder):
        os.makedirs(folder)
    if save_data == True:
        filename = mode + '_N' + str(nPart_range[-1]) + '_phi' + str(phi_range[-1]) + '_Kavg' + str(K_avg_range[-1])+ '_Kstd' + str(K_std_range[-1]) + '_Rp' + str(Rp_range[-1]) + '_xTy' + str(xTy)
        save_file = open(os.path.join(folder, filename + '.txt'), "w")

    fig, ax = plt.subplots()


    for nPart in nPart_range:
        for phi in phi_range:
            L = np.sqrt(nPart / (phi*xTy))
            for K_avg in K_avg_range:
                for K_std in K_std_range:
                    K = str(K_avg) + "_" + str(K_std)
                    for Rp in Rp_range:
                        com_arr = []
                        com_arr_sd = []
                        noise_range_plot = []
                        for noise in noise_range:
                            com = []
                            for seed in seed_range:
                                if from_stats == True:
                                    sim_dir = get_sim_dir(mode, nPart, phi, noise, K, Rp, xTy, seed)
                                    if not os.path.exists(os.path.join(sim_dir, 'stats_cohesion')):
                                        print(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed)
                                    else:
                                        stats_dict = read_cohesion_stats(mode, nPart, phi, noise, K, Rp, xTy, seed)
                                        com.append(stats_dict["com_dist"])
                                else:
                                    posFileExact = get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, file_name="pos_exact")
                                    if os.path.exists(posFileExact):
                                        com.append(mean_dist_com(posFileExact, L))
                            com_arr.append(np.mean(com))
                            com_arr_sd.append(np.std(com))
                            noise_range_plot.append(float(noise))
                        # noise_range_plot = [float(n) for n in noise_range]
                        ax.plot(noise_range_plot, com_arr, '-o', label=r"$N=$" + str(nPart)+ r"; $K_{AVG}=$" + str(K_avg) + r"; $K_{STD}=$" + str(K_std) + r"; $\rho=$" + str(phi) + r"; $R_I=$" + str(Rp))
                        ax.errorbar(noise_range_plot, com_arr, yerr=com_arr_sd, fmt='none', capsize=3)
                        if save_data == True:
                            save_file.write(str(nPart) + "\t" + str(Rp) + "\t" + str(phi) + "\t" + str(K_avg) + "\t" + str(K_std) + "\n")
                            for noise in noise_range_plot:
                                save_file.write(str(noise) + "\t")
                            save_file.write("\n")
                            for p in com_arr:
                                save_file.write(str(p) + "\t")
                            save_file.write("\n")
                            for p in com_arr_sd:
                                save_file.write(str(p) + "\t")
                        
                            save_file.write("\n")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"Mean distance to centre of mass")
    # ax.set_title(r'$N=$' + str(nPart) + r'; $\rho=$' + str(phi) + r'; $K_{AVG}=$' + str(K_avg)+ r'; $K_{STD}=$' + str(K_std) + r'; $R_I=$' + str(Rp))
    ax.legend()


    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_Kavg' + str(K_avg)+ '_Kstd' + str(K_std) + '_Rp' + str(Rp) + '_xTy' + str(xTy)
    plt.savefig(os.path.join(folder, filename + '.png'))

    if save_data == True:
        save_file.close()

def plot_av_n_vs_RI(mode, nPart, phi, noise, K_avg_range, K_std_range, Rp_range, xTy, seed_range, from_stats=True, save_data=False):
    """
    Plot mean number of neighbours per unit area in RI (basically local number density) against the radius of interaction
    """
    
    L = np.sqrt(nPart / (phi*xTy))

    fig, ax = plt.subplots()

    av_n_arr = []
    for K_avg in K_avg_range:
        for K_std in K_std_range:
            K = str(K_avg) + "_" + str(K_std)
            for Rp in Rp_range:
                # area = np.pi*Rp**2
                av_n = []
                for seed in seed_range:
                    if from_stats == True:
                        sim_dir = get_sim_dir(mode, nPart, phi, noise, K, Rp, xTy, seed)
                        if not os.path.exists(os.path.join(sim_dir, 'stats_cohesion')):
                            print(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed)
                        else:
                            stats_dict = read_cohesion_stats(mode, nPart, phi, noise, K, Rp, xTy, seed)
                            av_n.append(stats_dict["av_n"])
                    else:
                        posFileExact = get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, file_name="pos_exact")
                        if os.path.exists(posFileExact):
                            av_n.append(np.mean(neighbour_counts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max=Rp)))
                # av_n_arr.append(np.mean(av_n)/area)
                av_n_arr.append(np.mean(av_n))
            
            ax.plot(Rp_range, av_n_arr, '-o', label=r"$N=$" + str(nPart)+ r"; $K_{AVG}=$" + str(K_avg) + r"; $K_{STD}=$" + str(K_std) + r"; $\rho=$" + str(phi) + r"; $\eta=$" + str(noise) + r"; $R_I=$" + str(Rp))

    ax.set_xlabel(r"$R_I$")
    ax.set_ylabel(r"Mean number of neighbours")
    # ax.set_title()

    folder = os.path.abspath('../plots/numn_vs_RI')
    filename = mode + '_N' + str(nPart) + '_noise' + str(noise) + '_phi' + str(phi) + '_Kavg' + str(K_avg)+ '_Kstd' + str(K_std) + '_xTy' + str(xTy) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

    plt.close()
    

def plot_nn_vs_RI(mode, nPart, phi, noise, K_avg_range, K_std_range, Rp_range, xTy, seed_range, from_stats=True, save_data=False):
    """
    Plot mean distance to nearest neighbor as given in in eq. 7 in Huepe & Aldana, 2008
    https://hal.elte.hu/~vicsek/downloads/papers/aldana3.pdf
    """
    folder = os.path.abspath('../plots/nn_vs_RI')
    if not os.path.exists(folder):
        os.makedirs(folder)
    if save_data == True:
        filename = mode + '_N' + str(nPart) + '_noise' + str(noise) + '_phi' + str(phi) + '_Kavg' + str(K_avg_range[-1])+ '_Kstd' + str(K_std_range[-1]) + '_xTy' + str(xTy) + '_vp' + str(vp) + '_dt' + str(dt)
        save_file = open(os.path.join(folder, filename + '.txt'), "w")
    L = np.sqrt(nPart / (phi*xTy))

    fig, ax = plt.subplots()


    for K_avg in K_avg_range:
        for K_std in K_std_range:
            K = str(K_avg) + "_" + str(K_std)
            nn_arr = []
            nn_arr_sd = []
            for Rp in Rp_range:
                mean_nn = []
                for seed in seed_range:
                    if from_stats == True:
                        sim_dir = get_sim_dir(mode, nPart, phi, noise, K, Rp, xTy, seed)
                        if not os.path.exists(os.path.join(sim_dir, 'stats_cohesion')):
                            print(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed)
                        else:
                            stats_dict = read_cohesion_stats(mode, nPart, phi, noise, K, Rp, xTy, seed)
                            mean_nn.append(stats_dict["nn_dist"])
                    else:
                        posFileExact = get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, file_name="pos_exact")
                        if os.path.exists(posFileExact):
                            mean_nn.append(mean_dist_nn(posFileExact, L))
                try:
                    nn_arr_sd.append(np.std(mean_nn))
                    nn_arr.append(np.mean(mean_nn))
                except:
                    print('N=' + str(nPart) + '; phi=' + str(phi) + '; K=' + str(K) + '; noise=' + str(noise))
            
            inparFile = get_files(mode, nPart, phi, noise, K, Rp, xTy, seed)[0]
            params = get_params(inparFile)
            vp = params["vp"]
            dt = params["dt"]

            ax.plot(Rp_range, nn_arr, '-o', label=r"$N=$" + str(nPart) + r"; $K_{AVG}=$" + str(K_avg) + r"; $K_{STD}=$" + str(K_std) + r"; $\rho=$" + str(phi) + r"; $\eta=$" + str(noise) + r"; v_p=$" + str(vp) + r"; $\Delta t=$" + str(dt))
            if save_data == True:
                save_file.write(str(nPart) + "\t" + str(noise) + "\t" + str(phi) + "\t" + str(K_avg) + "\t" + str(K_std) + "\t" + str(vp) + "\t" + str(dt) + "\n")
                for r in Rp_range:
                    save_file.write(str(r) + "\t")
                save_file.write("\n")
                for n in nn_arr:
                    save_file.write(str(n) + "\t")
                save_file.write("\n")
                for n_sd in nn_arr_sd:
                    save_file.write(str(n_sd) + "\t")
                save_file.write("\n")
    ax.set_xlabel(r"$R_I$")
    ax.set_ylabel(r"Mean distance to nearest neighbor")
    # ax.set_title(r'$N=$' + str(nPart) + r'; $\rho=$' + str(phi) + r'; $\eta=$' + str(noise))
    ax.legend()

    folder = os.path.abspath('../plots/nn_vs_RI')
    filename = mode + '_N' + str(nPart) + '_noise' + str(noise) + '_phi' + str(phi) + '_Kavg' + str(K_avg)+ '_Kstd' + str(K_std) + '_xTy' + str(xTy) + '_vp' + str(vp) + '_dt' + str(dt) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

    if save_data == True:
        save_file.close()


def plot_nn_vs_noise(mode, nPart_range, phi_range, noise_range, K_avg_range, K_std_range, Rp_range, xTy, seed_range, from_stats=True, save_data=False):
    """
    Plot mean distance to nearest neighbor as given in in eq. 7 in Huepe & Aldana, 2008
    https://hal.elte.hu/~vicsek/downloads/papers/aldana3.pdf
    """
    folder = os.path.abspath('../plots/nn_vs_noise')
    if not os.path.exists(folder):
        os.makedirs(folder)
    if save_data == True:
        filename = mode + '_N' + str(nPart_range[-1]) + '_phi' + str(phi_range[-1]) + '_Kavg' + str(K_avg_range[-1])+ '_Kstd' + str(K_std_range[-1]) + '_Rp' + str(Rp_range[-1]) + '_xTy' + str(xTy)
        save_file = open(os.path.join(folder, filename + '.txt'), "w")

    fig, ax = plt.subplots()


    for nPart in nPart_range:
        for phi in phi_range:
            L = np.sqrt(nPart / (phi*xTy))
            for K_avg in K_avg_range:
                for K_std in K_std_range:
                    K = str(K_avg) + "_" + str(K_std)
                    for Rp in Rp_range:
                        nn_arr = []
                        nn_arr_sd = []
                        noise_range_plot = []
                        for noise in noise_range:
                            mean_nn = []
                            for seed in seed_range:
                                if from_stats == True:
                                    sim_dir = get_sim_dir(mode, nPart, phi, noise, K, Rp, xTy, seed)
                                    if not os.path.exists(os.path.join(sim_dir, 'stats_cohesion')):
                                        print(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed)
                                    else:
                                        stats_dict = read_cohesion_stats(mode, nPart, phi, noise, K, Rp, xTy, seed)
                                        mean_nn.append(stats_dict["nn_dist"])
                                else:
                                    posFileExact = get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, file_name="pos_exact")
                                    if os.path.exists(posFileExact):
                                        mean_nn.append(mean_dist_nn(posFileExact, L))
                            try:
                                nn_arr.append(np.mean(mean_nn))
                                nn_arr_sd.append(np.std(mean_nn))
                                noise_range_plot.append(float(noise))
                            except:
                                print('N=' + str(nPart) + '; phi=' + str(phi) + '; K=' + str(K) + '; noise=' + str(noise))
                        # noise_range_plot = [float(n) for n in noise_range]
                        ax.plot(noise_range_plot, nn_arr, '-o', label=r"$N=$" + str(nPart)+ r"; $K_{AVG}=$" + str(K_avg) + r"; $K_{STD}=$" + str(K_std) + r"; $\rho=$" + str(phi) + r"; $R_I=$" + str(Rp))
                        if save_data == True:
                            save_file.write(str(nPart) + "\t" + str(Rp) + "\t" + str(phi) + "\t" + str(K_avg) + "\t" + str(K_std) + "\n")
                            for noise in noise_range_plot:
                                save_file.write(str(noise) + "\t")
                            save_file.write("\n")
                            for n in nn_arr:
                                save_file.write(str(n) + "\t")
                            for n_sd in nn_arr_sd:
                                save_file.write(str(n_sd) + "\t")
                            save_file.write("\n")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"Mean distance to nearest neighbor")
    # ax.set_title(r'$N=$' + str(nPart) + r'; $\rho=$' + str(phi) + r'; $K_{AVG}=$' + str(K_avg)+ r'; $K_{STD}=$' + str(K_std) + r'; $R_I=$' + str(Rp))
    ax.legend()

    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_Kavg' + str(K_avg)+ '_Kstd' + str(K_std) + '_Rp' + str(Rp) + '_xTy' + str(xTy)
    plt.savefig(os.path.join(folder, filename + '.png'))

    if save_data == True:
        save_file.close()

def plot_nn_vs_Kavg(mode, nPart_range, phi_range, noise_range, K_avg_range, K_std_range, Rp_range, xTy, seed_range, from_stats=True, save_data=False):
    """
    Plot mean distance to nearest neighbor as given in in eq. 7 in Huepe & Aldana, 2008
    https://hal.elte.hu/~vicsek/downloads/papers/aldana3.pdf
    """
    folder = os.path.abspath('../plots/nn_vs_Kavg')
    if not os.path.exists(folder):
        os.makedirs(folder)
    if save_data == True:
        filename = mode + '_N' + str(nPart_range[-1]) + '_phi' + str(phi_range[-1]) + '_noise' + str(noise_range[-1])+ '_Kstd' + str(K_std_range[-1]) + '_Rp' + str(Rp_range[-1]) + '_xTy' + str(xTy)
        save_file = open(os.path.join(folder, filename + '.txt'), "w")

    fig, ax = plt.subplots()


    for nPart in nPart_range:
        for phi in phi_range:
            L = np.sqrt(nPart / (phi*xTy))
            for noise in noise_range:
                for K_std in K_std_range:
                    for Rp in Rp_range:
                        nn_arr = []
                        nn_arr_sd = []
                        Kavg_range_plot = []
                        for K_avg in K_avg_range:
                            K = str(K_avg) + "_" + str(K_std)
                            mean_nn = []
                            for seed in seed_range:
                                if from_stats == True:
                                    sim_dir = get_sim_dir(mode, nPart, phi, noise, K, Rp, xTy, seed)
                                    if not os.path.exists(os.path.join(sim_dir, 'stats_cohesion')):
                                        print(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed)
                                    else:
                                        stats_dict = read_cohesion_stats(mode, nPart, phi, noise, K, Rp, xTy, seed)
                                        mean_nn.append(stats_dict["nn_dist"])
                                else:
                                    posFileExact = get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, file_name="pos_exact")
                                    if os.path.exists(posFileExact):
                                        mean_nn.append(mean_dist_nn(posFileExact, L))
                            try:
                                nn_arr.append(np.mean(mean_nn))
                                nn_arr_sd.append(np.std(mean_nn))
                                Kavg_range_plot.append(float(K_avg))
                            except:
                                print('N=' + str(nPart) + '; phi=' + str(phi) + '; K=' + str(K) + '; noise=' + str(noise))
                        # noise_range_plot = [float(n) for n in noise_range]
                        ax.plot(Kavg_range_plot, nn_arr, '-o', label=r"$N=$" + str(nPart)+ r"; $K_{STD}=$" + str(K_std) + r"; $\eta=$" + str(noise) + r"; $\rho=$" + str(phi) + r"; $R_I=$" + str(Rp))
                        if save_data == True:
                            save_file.write(str(nPart) + "\t" + str(Rp) + "\t" + str(phi) + "\t" + str(noise) + "\t" + str(K_std) + "\n")
                            for k in Kavg_range_plot:
                                save_file.write(str(k) + "\t")
                            save_file.write("\n")
                            for n in nn_arr:
                                save_file.write(str(n) + "\t")
                            for n_sd in nn_arr_sd:
                                save_file.write(str(n_sd) + "\t")
                            save_file.write("\n")
    ax.set_xlabel(r"$K_{AVG}$")
    ax.set_ylabel(r"Mean distance to nearest neighbor")
    # ax.set_title(r'$N=$' + str(nPart) + r'; $\rho=$' + str(phi) + r'; $K_{AVG}=$' + str(K_avg)+ r'; $K_{STD}=$' + str(K_std) + r'; $R_I=$' + str(Rp))
    ax.legend()

    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_noise' + str(noise)+ '_Kstd' + str(K_std) + '_Rp' + str(Rp) + '_xTy' + str(xTy)
    plt.savefig(os.path.join(folder, filename + '.png'))

    if save_data == True:
        save_file.close()

