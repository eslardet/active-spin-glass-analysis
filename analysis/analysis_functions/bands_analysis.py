import sys
sys.path.insert(1, '././analysis_functions')
from import_files import *
from stats import *

import numpy as np
import matplotlib.pyplot as plt


def plot_density_profile(mode, nPart, phi, noise, K, Rp, xTy, seed, min_grid_size=2):
    """
    Plot x-directional density profile for the final snapshot
    """
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

    fig, ax = plt.subplots()
    x_vals = np.arange(0, Lx, grid_size_x)
    ax.plot(x_vals, n_density)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"Local density")

    folder = os.path.abspath('../plots/density_profile/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))
    plt.close()

def plot_density_profile_superimpose(mode, nPart, phi, noise, K_avg_range, K_std_range, Rp, xTy, seed, min_grid_size=2):
    """
    Plot x-directional density profile for the final snapshot for multiple Kavg/ Kstd values
    """
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
            ax.plot(x_vals, n_density, label = r"$\overline{K}=$" + str(K_avg) + r"; $\sigma_K=$" + str(K_std))

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"Local density")
    ax.legend()

    folder = os.path.abspath('../plots/density_profile/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))
    plt.close()



def plot_band_profiles(mode, nPart, phi, noise, K, Rp, xTy, seed, min_grid_size=2, cutoff=1.5, peak_cutoff=2):
    """
    Plot x-directional density profile for the final snapshot shifted to the origin
    """
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

    if n_density[0] > cutoff:
        in_band = 1
    else:
        in_band = 0

    band_start = []
    band_end = []
    for i, val in enumerate(n_density):
        if in_band == 0:
            if val > cutoff:
                in_band = 1
                band_start.append(i)
        elif in_band == 1:
            if val < cutoff:
                in_band = 0
                band_end.append(i)

    if len(band_start) != len(band_end):
        raise Exception("Unequal band starts/ ends")
    elif len(band_start) == 0:
        raise Exception("No bands!")
    else:
        band_number = len(band_start)

    # Handle the case where band is on boundary
    if band_end[0] < band_start[0]:
        band_start.insert(0, band_start.pop(-1))

    # Reclassify based on peak value
    band_peak = []
    for i in range(band_number):
        if band_start[i] < band_end[i]:
            band_vals = n_density[band_start[i]:band_end[i]]
        else:
            band_vals = np.concatenate((n_density[band_start[i]:], n_density[:band_end[i]]))
        peak = np.max(band_vals)
        peak_id = band_vals.argmax()
        if peak > peak_cutoff:
            band_peak.append(band_start[i]+peak_id)
        else:
            band_start.pop(i)
            band_end.pop(i)

    band_number = len(band_peak)
    if band_number == 0:
        raise Exception("No bands with large enough peak!")
    
    extra_left = int(len(x_vals) / 10)
    extra_right = int(len(x_vals) / 10)
    total_len = extra_left + extra_right

    fig, ax = plt.subplots()

    for i in range(band_number):
        if band_peak[i] + extra_right > len(x_vals):
            d_plot = np.concatenate((n_density[band_peak[i]-extra_left:], n_density[:band_peak[i]+extra_right-len(x_vals)]))
        elif band_peak[i] - extra_left < 0:
            d_plot = np.concatenate((n_density[band_peak[i]-extra_left+len(x_vals):], n_density[:band_peak[i]+extra_right]))
        else:
            d_plot = n_density[band_peak[i]-extra_left:band_peak[i]+extra_right]
        x_plot = x_vals[:total_len]
        ax.plot(x_plot, d_plot, label="band " + str(i))
    ax.legend()
    
    folder = os.path.abspath('../plots/density_profile_shifted/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))
    plt.close()


def plot_average_band_profile(mode, nPart, phi, noise, K_avg_range, K_std_range, Rp, xTy, seed_range, timestep_range=[1], pos_ex=True, min_grid_size=2, cutoff=1.5, peak_cutoff=2):
    """
    Plot x-directional density profile for the final snapshot shifted to the origin
    """

    fig, ax = plt.subplots()

    L = np.sqrt(nPart / (phi*xTy))
    Ly = L
    Lx = L*xTy

    ngrid_x = int(Lx // min_grid_size)
    grid_size_x = Lx / ngrid_x
    grid_area = grid_size_x*Ly

    x_vals = np.arange(0, Lx, grid_size_x)

    extra_left = int(200 / grid_size_x)
    extra_right = int(50 / grid_size_x)
    total_len = extra_left + extra_right

    x_plot = x_vals[:total_len]
    d_plot_av = np.zeros(total_len)

    for K_avg in K_avg_range:
        for K_std in K_std_range:
            K = str(K_avg) + "_" + str(K_std)
            no_band = 0

            for seed in seed_range:
                # posFileExact = get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed, file_name='pos_exact')
                # x, y, theta, view_time = get_pos_ex_snapshot(file=posFileExact)

                inparFile, posFile = get_files(mode, nPart, phi, noise, K, Rp, xTy, seed)

                for timestep in timestep_range:
                    if pos_ex == True:
                        posFileExact = get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, file_name='pos_exact')
                        x, y, theta, view_time = get_pos_ex_snapshot(file=posFileExact)
                    else:
                        x, y, theta = get_pos_snapshot(posFile, nPart, timestep)

                    ## Find average orientation and reverse order if left moving to conserve band symmetry with right moving bands
                    av_theta = np.arctan2(np.sum(np.sin(theta)), np.sum(np.cos(theta)))
                    if np.abs(av_theta) < np.pi/4:
                        band_dir = "right"
                    elif np.abs(av_theta) > 3*np.pi/4:
                        band_dir = "left"
                    else:
                        print("No left / right band movement for K=" + str(K) + ", s=" + str(seed))
                        break
                        # raise Exception("No left/ right band movement")

                    x = pbc_wrap(x,Lx)

                    grid_counts = np.zeros(ngrid_x)
                    for i in range(nPart):
                        gridx = int(x[i]//grid_size_x)
                        grid_counts[gridx] += 1
                    n_density = grid_counts / grid_area

                    if band_dir == "left":
                        n_density = n_density[::-1]

                    if n_density[0] > cutoff:
                        in_band = 1
                    else:
                        in_band = 0

                    band_start = []
                    band_end = []
                    for i, val in enumerate(n_density):
                        if in_band == 0:
                            if val > cutoff:
                                in_band = 1
                                band_start.append(i)
                        elif in_band == 1:
                            if val < cutoff:
                                in_band = 0
                                band_end.append(i)
                
                    if len(band_start) != len(band_end):
                        band_end.append(len(x_vals)-1)
                        # raise Exception("Unequal band starts/ ends at seed " + str(seed))
                    band_number = len(band_start)
                    if band_number == 0:
                        no_band += 1
                        print("No band at seed " + str(seed))
                        continue

                    # Handle the case where band is on boundary
                    if band_end[0] < band_start[0]:
                        band_start.insert(0, band_start.pop(-1))

                    # Reclassify based on peak value
                    band_peak = []
                    # band_start_new = band_start
                    # band_end_new = band_end
                    for i in range(band_number):
                        if band_start[i] < band_end[i]:
                            band_vals = n_density[band_start[i]:band_end[i]]
                        else: ## case where band is on boundary
                            band_vals = np.concatenate((n_density[band_start[i]:], n_density[:band_end[i]]))
                        peak = np.max(band_vals)
                        peak_id = band_vals.argmax()
                        if peak > peak_cutoff:
                            band_peak.append(band_start[i]+peak_id)
                        # else: ## if need start and end of bands
                            # band_start_new.pop(i)
                            # band_end_new.pop(i)

                    band_number = len(band_peak)
                    if band_number == 0:
                        no_band += 1
                        print("No band above peak cutoff at seed " + str(seed))
                        continue

                    d_plot = np.zeros(total_len)
                    # # Peak centered
                    # for i in range(band_number):
                    #     if band_peak[i] + extra_right > len(x_vals):
                    #         d_plot += np.concatenate((n_density[band_peak[i]-extra_left:], n_density[:band_peak[i]+extra_right-len(x_vals)]))
                    #     elif band_peak[i] - extra_left < 0:
                    #         d_plot += np.concatenate((n_density[band_peak[i]-extra_left+len(x_vals):], n_density[:band_peak[i]+extra_right]))
                    #     else:
                    #         d_plot += n_density[band_peak[i]-extra_left:band_peak[i]+extra_right]
                    # Band edge centered
                    for i in range(band_number):
                        if band_end[i] + extra_right > len(x_vals):
                            d_plot += np.concatenate((n_density[band_end[i]-extra_left:], n_density[:band_end[i]+extra_right-len(x_vals)]))
                        elif band_end[i] - extra_left < 0:
                            d_plot += np.concatenate((n_density[band_end[i]-extra_left+len(x_vals):], n_density[:band_end[i]+extra_right]))
                        else:
                            d_plot += n_density[band_end[i]-extra_left:band_end[i]+extra_right]
                    d_plot_av += d_plot/band_number
            d_plot_av = d_plot_av/(len(seed_range)*len(timestep_range)-no_band)

            ax.plot(x_plot, d_plot_av, label=r"$K_{AVG}=$" + str(K_avg) + r"; $\sigma_K=$" + str(K_std))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"Average local density")
    ax.set_title(r"Average band: $\rho=$" + str(phi) + r"$, \eta=$" + str(noise))
    ax.legend()

    folder = os.path.abspath('../plots/density_profile_shifted/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_av.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))
    plt.close()


def plot_var_density_noise(mode, nPart, phi, noise_range, K, Rp, xTy, seed_range, min_grid_size=2):
    """
    Plot the local density variance against noise
    """
    fig, ax = plt.subplots()
    vars = []
    for noise in noise_range:
        var_sum = 0
        for seed in seed_range:
            var_sum += local_density_var(mode, nPart, phi, noise, K, Rp, xTy, seed, min_grid_size)
        vars.append(var_sum/len(seed_range))

    noise_range = [float(i) for i in noise_range]
    ax.plot(noise_range, vars, 'o-')
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"Local density variance $\langle(\rho-\bar{\rho})^2\rangle$")

    folder = os.path.abspath('../plots/var_density_vs_noise/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

def plot_var_density_Kavg(mode, nPart, phi, noise, K_avg_range, K_std_range, Rp, xTy, seed_range, min_grid_size=2):
    """
    Plot the local density variance against K_AVG, for various K_STD values
    """
    fig, ax = plt.subplots()
    for K_std in K_std_range:
        vars = []
        for K_avg in K_avg_range:
            K = str(K_avg) + "_" + str(K_std)
            var_sum = 0
            error_count = 0
            for seed in seed_range:
                try:
                    var_sum += read_stats(mode, nPart, phi, noise, K, Rp, xTy, seed)["d_var"]
                except:
                    try:
                        d_var = local_density_var(mode, nPart, phi, noise, K, Rp, xTy, seed, min_grid_size)
                        var_sum += d_var
                        sim_dir = get_sim_dir(mode, nPart, phi, noise, K, Rp, xTy, seed)
                        statsFile = open(os.path.join(sim_dir, "stats"), "a")
                        statsFile.write(str(d_var) + '\n')
                        statsFile.close()
                    except:
                        print(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed)
                        error_count += 1
            vars.append(var_sum/(len(seed_range)-error_count))

        ax.plot(K_avg_range, vars, 'o-', label=r"$\sigma_K=$" + str(K_std))
    ax.set_xlabel(r"$K_{AVG}$")
    ax.set_ylabel(r"Local density variance $\langle(\rho-\bar{\rho})^2\rangle$")
    ax.legend()

    folder = os.path.abspath('../plots/var_density_vs_Kavg/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))
