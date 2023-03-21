import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm, colors
from decimal import Decimal

import csv
import os


def get_sim_dir(mode, nPart, phi, noise, K, K_new, xTy, seed):
    mode_name = "Switch_" + str(mode)

    sim_dir = os.path.abspath('../simulation_data/' + mode_name + '/N' + str(nPart) + '/phi' + str(phi) + '_n' + str(noise) + '/K' + str(K) + '/Knew' + str(K_new) + '/xTy' + str(xTy) + '/s' + str(seed))

    return sim_dir

def get_files(mode, nPart, phi, noise, K, K_new, xTy, seed):
    """
    Get file paths for the input parameters and position files
    """
    sim_dir = get_sim_dir(mode, nPart, phi, noise, K, K_new, xTy, seed)
    inparFile = os.path.join(sim_dir, "inpar")
    posFile = os.path.join(sim_dir, "pos")
    return inparFile, posFile

def get_file_path(mode, nPart, phi, noise, K, K_new, xTy, seed, file_name):
    """
    Get the file path for a certain file name in the simulation data directory
    """
    sim_dir = get_sim_dir(mode, nPart, phi, noise, K, K_new, xTy, seed)
    file_path = os.path.join(sim_dir, file_name)
    return file_path

def get_params(inparFile):
    """
    Create dictionary with parameter name and values
    """
    with open(inparFile) as f:
        reader = csv.reader(f, delimiter="\t")
        r = list(reader)
    inpar_dict = {}
    inpar_dict["nPart"] = int(r[0][0])
    inpar_dict["phi"] = float(r[1][0])
    inpar_dict["seed"] = int(r[2][0])
    inpar_dict["noise"] = r[3][0]
    inpar_dict["vp"] = float(r[4][0])
    inpar_dict["Rp"] = float(r[5][0])
    inpar_dict["xTy"] = float(r[6][0])
    inpar_dict["DT"] = float(r[14][0])
    inpar_dict["eqT"] = float(r[16][0])
    inpar_dict["switchT"] = float(r[17][0])
    inpar_dict["simulT"] = float(r[18][0])

    return inpar_dict

def pbc_wrap(x, L):
    """
    Wrap points into periodic box with length L
    """
    return x - L*np.round(x/L) + L/2


def get_pos_arr(inparFile, posFile, min_T=None, max_T=None):
    """
    Get arrays for x, y, theta positions at each save time from the positions text file
    """
    inpar_dict = get_params(inparFile)
    
    nPart = inpar_dict["nPart"]
    DT = inpar_dict["DT"]
    eqT = inpar_dict["eqT"]
    if min_T == None:
        min_T = 0
    if max_T == None:
        max_T = inpar_dict["simulT"]
    
    with open(posFile) as f:
        reader = csv.reader(f, delimiter="\t")
        r = list(reader)[6:]

    startT = float(r[0][0])

    x_all = []
    y_all = []
    theta_all = []

    for i in range(max(int((min_T-startT+eqT)/DT),0), int((max_T-startT+eqT)/DT)+1):
        x_all.append(np.array(r[(nPart+1)*i+1:(nPart+1)*i+1+nPart]).astype('float')[:,0])
        y_all.append(np.array(r[(nPart+1)*i+1:(nPart+1)*i+1+nPart]).astype('float')[:,1])
        theta_all.append(np.array(r[(nPart+1)*i+1:(nPart+1)*i+1+nPart]).astype('float')[:,2])
    return x_all, y_all, theta_all

def get_theta_arr(inparFile, posFile, min_T=None, max_T=None):
    """
    Get arrays for only the theta positions at each save time from the positions text file
    """
    inpar_dict = get_params(inparFile)
    
    nPart = inpar_dict["nPart"]
    Nx = int(np.ceil(np.sqrt(nPart)))
    nPart = Nx*Nx

    DT = inpar_dict["DT"]
    if min_T == None:
        min_T = 0
    if max_T == None:
        max_T = inpar_dict["simulT"]
    
    with open(posFile) as f:
        reader = csv.reader(f, delimiter="\t")
        r = list(reader)[6:]

    theta = []
    for i in range(int(min_T/DT), int(max_T/DT)+1):
        theta.append(np.array(r[(nPart+1)*i+1:(nPart+1)*i+1+nPart]).astype('float')[:,2])
    return theta

def get_pos_snapshot(posFile, nPart, timestep):
    """
    Get lists of x, y, theta at a single timestep
    """
    with open(posFile) as f:
        line_count = 0
        x = []
        y = []
        theta = []
        for line in f:
            line_count += 1
            if 8 + timestep*(nPart + 1) <= line_count <= 7 + timestep*(nPart + 1) + nPart:
                x.append(float(line.split('\t')[0]))
                y.append(float(line.split('\t')[1]))
                theta.append(float(line.split('\t')[2]))
            if line_count > 7 + timestep*(nPart + 1) + nPart:
                break
    return x, y, theta

def get_pos_ex_snapshot(file):
    """
    Get lists of x, y, theta from exact pos file
    """
    with open(file) as f:
        line_count = 0
        x = []
        y = []
        theta = []
        for line in f:
            line_count += 1
            if line_count == 1:
                view_time = float(line)
            else:
                x.append(float(line.split('\t')[0]))
                y.append(float(line.split('\t')[1]))
                theta.append(float(line.split('\t')[2]))
    return x, y, theta, view_time


def snapshot(mode, nPart, phi, noise, K, K_new, xTy, seed, view_time=None, pos_ex=False, show_color=True, save_in_folder=False):
    """
    Get static snapshot at specified time from the positions file
    """

    inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, K_new=K_new, xTy=xTy, seed=seed)
    if pos_ex == True:
        posFileExact = get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, K_new=K_new, xTy=xTy, seed=seed, file_name="pos_exact")
    inpar_dict = get_params(inparFile)
    
    nPart = inpar_dict["nPart"]
    phi = inpar_dict["phi"]
    noise = inpar_dict["noise"]
    DT = inpar_dict["DT"]
    eqT = inpar_dict["eqT"]
    xTy = inpar_dict["xTy"]
    simulT = inpar_dict["simulT"]

    L = np.sqrt(nPart / (phi*xTy))
    Ly = L
    Lx = L*xTy

    if pos_ex == True:
        x, y, theta, view_time = get_pos_ex_snapshot(file=posFileExact)
    else:
        if view_time == None:
            view_time = simulT
        timestep = int(view_time/DT)
        view_time = timestep*DT + eqT

        x, y, theta = get_pos_snapshot(posFile=posFile, nPart=nPart, timestep=timestep)
    
    x = pbc_wrap(x,Lx)
    y = pbc_wrap(y,Ly)
    u = np.cos(theta)
    v = np.sin(theta)

    norm = colors.Normalize(vmin=0.0, vmax=2*np.pi, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.hsv)
    cols = mapper.to_rgba(np.mod(theta, 2*np.pi))
    
    fig, ax = plt.subplots(figsize=(5*xTy,5), dpi=72)
    
    if show_color == True:
        ax.quiver(x, y, u, v, color=cols)
        plt.colorbar(mappable=mapper, ax=ax)
    else:
        ax.quiver(x, y, u, v)
    ax.set_xlim(0,Lx)
    ax.set_ylim(0,Ly)
    ax.set_aspect('equal')
    ax.set_title("t=" + str(round(view_time)))

    if save_in_folder == True:
        folder = get_sim_dir(mode, nPart, phi, noise, K, xTy, seed)
        filename = 'snapshot.png'
    else:
        folder = os.path.abspath('../snapshots_vicsek')
        filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Knew' + str(K_new) + '_xTy' + str(xTy) + '_s' + str(seed) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))


def animate(mode, nPart, phi, noise, K, K_new, xTy, seed, min_T=None, max_T=None):
    """
    Make animation from positions file
    """
    inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, K_new=K_new, xTy=xTy, seed=seed)

    x_all, y_all, theta_all = get_pos_arr(inparFile=inparFile, posFile=posFile, min_T=min_T, max_T=max_T)
    
    inpar_dict = get_params(inparFile)
    
    nPart = inpar_dict["nPart"]
    phi = inpar_dict["phi"]
    noise = inpar_dict["noise"]
    DT = inpar_dict["DT"]
    seed = inpar_dict["seed"]
    xTy = inpar_dict["xTy"]
    eqT = inpar_dict["eqT"]
    switchT = inpar_dict["switchT"]

    if min_T == None:
        min_T = 0

    with open(posFile) as f:
        reader = csv.reader(f, delimiter="\t")
        startT = float(list(reader)[6][0])

    plt.rcParams["animation.html"] = "jshtml"
    plt.ioff()
    plt.rcParams['animation.embed_limit'] = 2**128

    L = np.sqrt(nPart / (phi*xTy))
    Ly = L
    Lx = L*xTy
    
    fig, ax = plt.subplots(figsize=(5*xTy,5))

    norm = colors.Normalize(vmin=0.0, vmax=2*np.pi, clip=True)
    plt.set_cmap('hsv')

    # mapper = cm.ScalarMappable(norm=norm, cmap=cm.hsv)
    # plt.colorbar(mappable=mapper, ax=ax)

    x = pbc_wrap(x_all[0],Lx)
    y = pbc_wrap(y_all[0],Ly)
    theta = theta_all[0]
    cols = np.mod(theta, 2*np.pi)
    arrows = ax.quiver(x, y, np.cos(theta), np.sin(theta), norm(cols))

    def init():
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)
        return arrows,

    def update(n):
        x = pbc_wrap(x_all[n],Lx)
        y = pbc_wrap(y_all[n],Ly)
        theta = theta_all[n]
        cols = np.mod(theta, 2*np.pi)
        arrows.set_offsets(np.c_[x, y])
        arrows.set_UVC(np.cos(theta), np.sin(theta), norm(cols))
        if n*DT+startT+min_T < eqT+switchT:
            status = "Old couplings, K = " + str(K)
        else:
            status = "New couplings, K = " + str(K_new) 
        ax.set_title("t = " + str(round(n*DT+startT+min_T, 1)) + ";   " + status, fontsize=10, loc='left')
        
        return arrows,

    ani = FuncAnimation(fig, update, init_func=init, frames=len(theta_all), interval=10, blit=True)

    folder = os.path.abspath('../animations_vicsek')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Knew' + str(K_new) + '_xTy' + str(xTy) + '_s' + str(seed) + '.mp4'
    if not os.path.exists(folder):
        os.makedirs(folder)
    ani.save(os.path.join(folder, filename))


def plot_density_profile(mode, nPart, phi, noise, K, K_new, xTy, seed, min_grid_size=2):
    """
    Plot x-directional density profile for the final snapshot
    """
    posFileExact = get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, K_new=K_new, xTy=xTy, seed=seed, file_name='pos_exact')
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
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Knew' + str(K_new) + '_xTy' + str(xTy) + '_s' + str(seed) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))



def get_average_band_profile(mode, nPart, phi, noise, K, K_new, xTy, seed_range, timestep_range, min_grid_size=2, cutoff=1.5, peak_cutoff=2):
    """
    Get x-directional density profile averaged over timesteps and/or seeds
    """

    L = np.sqrt(nPart / (phi*xTy))
    Ly = L
    Lx = L*xTy

    ngrid_x = int(Lx // min_grid_size)
    grid_size_x = Lx / ngrid_x
    grid_area = grid_size_x*Ly

    x_vals = np.arange(0, Lx, grid_size_x)

    extra_left = int(len(x_vals) / 10)
    extra_right = int(len(x_vals) / 10)
    total_len = extra_left + extra_right

    x_plot = x_vals[:total_len]
    d_plot_av = np.zeros(total_len)

    no_band = 0

    for seed in seed_range:
        # posFileExact = get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed, file_name='pos_exact')
        # x, y, theta, view_time = get_pos_ex_snapshot(file=posFileExact)

        inparFile, posFile = get_files(mode, nPart, phi, noise, K, K_new, xTy, seed)

        for timestep in timestep_range:

            x, y, theta = get_pos_snapshot(posFile, nPart, timestep)
            x = pbc_wrap(x,Lx)

            grid_counts = np.zeros(ngrid_x)
            for i in range(nPart):
                gridx = int(x[i]//grid_size_x)
                grid_counts[gridx] += 1
            n_density = grid_counts / grid_area

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
            for i in range(band_number):
                if band_peak[i] + extra_right > len(x_vals):
                    d_plot += np.concatenate((n_density[band_peak[i]-extra_left:], n_density[:band_peak[i]+extra_right-len(x_vals)]))
                elif band_peak[i] - extra_left < 0:
                    d_plot += np.concatenate((n_density[band_peak[i]-extra_left+len(x_vals):], n_density[:band_peak[i]+extra_right]))
                else:
                    d_plot += n_density[band_peak[i]-extra_left:band_peak[i]+extra_right]
            # for i in range(band_number):
            #     if band_start[i] + extra_right > len(x_vals):
            #         d_plot += np.concatenate((n_density[band_start[i]-extra_left:], n_density[:band_start[i]+extra_right-len(x_vals)]))
            #     elif band_start[i] - extra_left < 0:
            #         d_plot += np.concatenate((n_density[band_start[i]-extra_left+len(x_vals):], n_density[:band_start[i]+extra_right]))
            #     else:
            #         d_plot += n_density[band_start[i]-extra_left:band_start[i]+extra_right]
            d_plot_av += d_plot/band_number
    d_plot_av = d_plot_av/(len(seed_range)*len(timestep_range)-no_band)

    return x_plot, d_plot_av

def plot_average_band_profile(mode, nPart, phi, noise, K, K_new, xTy, seed_range, timestep_range, min_grid_size=2, cutoff=1.5, peak_cutoff=2):
    """
    Plot x-directional density profile shifted to the origin averaged over timesteps and/or seeds
    """

    fig, ax = plt.subplots()

    x_plot, d_plot_av = get_average_band_profile(mode, nPart, phi, noise, K, K_new, xTy, seed_range, timestep_range, min_grid_size, cutoff, peak_cutoff)

    ax.plot(x_plot, d_plot_av)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"Average local density")
    ax.set_title(r"Average band: $\rho=$" + str(phi) + r"$, \eta=$" + str(noise) + r"$, K=$" + str(K))

    folder = os.path.abspath('../plots/density_profile_shifted/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Knew' + str(K_new) + '_xTy' + str(xTy) + '_av.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))