import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm, colors
from decimal import Decimal

import csv
import os


def get_sim_dir(mode, nPart, phi, noise, K, xTy, seed):
    if mode == "C":
        mode_name = "Constant"
    elif mode == "T":
        mode_name = "TwoPopulations"
    elif mode == "G":
        mode_name = "Gaussian"
    elif mode == "A":
        mode_name = "Antiferromagnetic"
    elif mode == "F":
        mode_name = "Ferromagnetic"

    sim_dir = os.path.abspath('../simulation_data/' + mode_name + '/N' + str(nPart) + '/phi' + str(phi) + '_n' + str(noise) + '/K' + str(K) + '/xTy' + str(xTy) + '/s' + str(seed))

    return sim_dir

def get_files(mode, nPart, phi, noise, K, xTy, seed):
    """
    Get file paths for the input parameters and position files
    """
    sim_dir = get_sim_dir(mode, nPart, phi, noise, K, xTy, seed)
    inparFile = os.path.join(sim_dir, "inpar")
    posFile = os.path.join(sim_dir, "pos")
    return inparFile, posFile

def get_file_path(mode, nPart, phi, noise, K, xTy, seed, file_name):
    """
    Get the file path for a certain file name in the simulation data directory
    """
    sim_dir = get_sim_dir(mode, nPart, phi, noise, K, xTy, seed)
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
    inpar_dict["noise"] = float(r[3][0])
    inpar_dict["vp"] = float(r[4][0])
    inpar_dict["Rp"] = float(r[5][0])
    inpar_dict["xTy"] = float(r[6][0])
    inpar_dict["mode"] = r[8][0]
    
    if inpar_dict["mode"] == 'C':
        inpar_dict["DT"] = float(r[11][0])
        inpar_dict["eqT"] = float(r[13][0])
        inpar_dict["simulT"] = float(r[14][0])
    elif inpar_dict["mode"] == 'T':
        inpar_dict["DT"] = float(r[13][0])
        inpar_dict["eqT"] = float(r[15][0])
        inpar_dict["simulT"] = float(r[16][0])
    else:
        inpar_dict["DT"] = float(r[12][0])
        inpar_dict["eqT"] = float(r[14][0])
        inpar_dict["simulT"] = float(r[15][0])
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


def snapshot(mode, nPart, phi, noise, K, xTy, seed, view_time=None, pos_ex=False, show_color=True, save_in_folder=False):
    """
    Get static snapshot at specified time from the positions file
    """

    inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed)
    if pos_ex == True:
        posFileExact = get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed, file_name="pos_exact")
    inpar_dict = get_params(inparFile)
    
    nPart = inpar_dict["nPart"]
    phi = inpar_dict["phi"]
    noise = inpar_dict["noise"]
    mode = inpar_dict["mode"]
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
    ax.set_title("t=" + str(view_time))

    if save_in_folder == True:
        folder = get_sim_dir(mode, nPart, phi, noise, K, xTy, seed)
        filename = 'snapshot.png'
    else:
        folder = os.path.abspath('../snapshots_vicsek')
        filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_xTy' + str(xTy) + '_s' + str(seed) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))


def animate(mode, nPart, phi, noise, K, xTy, seed, min_T=None, max_T=None):
    """
    Make animation from positions file
    """
    inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed)

    x_all, y_all, theta_all = get_pos_arr(inparFile=inparFile, posFile=posFile, min_T=min_T, max_T=max_T)
    
    inpar_dict = get_params(inparFile)
    
    nPart = inpar_dict["nPart"]
    phi = inpar_dict["phi"]
    noise = inpar_dict["noise"]
    mode = inpar_dict["mode"]
    DT = inpar_dict["DT"]
    seed = inpar_dict["seed"]
    xTy = inpar_dict["xTy"]

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

    x = pbc_wrap(x_all[0],Lx)
    y = pbc_wrap(y_all[0],Ly)
    theta = theta_all[0]
    arrows = ax.quiver(x, y, np.cos(theta), np.sin(theta))

    def init():
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)
        return arrows,

    def update(n):
        x = pbc_wrap(x_all[n],Lx)
        y = pbc_wrap(y_all[n],Ly)
        theta = theta_all[n]
        arrows.set_offsets(np.c_[x, y])
        arrows.set_UVC(np.cos(theta), np.sin(theta))
        ax.set_title("t = " + str(round(n*DT+startT, 1)), fontsize=10, loc='left')
        
        return arrows,

    ani = FuncAnimation(fig, update, init_func=init, frames=len(theta_all), interval=10, blit=True)

    folder = os.path.abspath('../animations_vicsek')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_xTy' + str(xTy) + '_s' + str(seed) + '.mp4'
    if not os.path.exists(folder):
        os.makedirs(folder)
    ani.save(os.path.join(folder, filename))


def plot_porder_time(mode, nPart, phi, noise, K, xTy, seed, min_T=None, max_T=None):
    """
    Plot polar order parameter against time for one simulation
    """
    inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed)
    inpar_dict = get_params(inparFile)
    DT = inpar_dict["DT"]
    simulT = inpar_dict["simulT"]
    if min_T == None:
        min_T = 0
    if max_T == None:
        max_T = simulT
    
    p_order = []

    with open(posFile) as f:
        line_count = 1
        timestep = int(min_T//DT)
        for line in f:
            if 8 + timestep*(nPart + 1) <= line_count <= 7 + timestep*(nPart + 1) + nPart:
                if line_count == 8 + timestep*(nPart+1):
                    cos_sum = 0
                    sin_sum = 0
                theta = float(line.split('\t')[2])
                cos_sum += np.cos(theta)
                sin_sum += np.sin(theta)
                if line_count == 7 + timestep*(nPart + 1) + nPart:
                    p_order.append(np.sqrt(cos_sum**2+sin_sum**2)/nPart)
                    timestep += 1
            line_count += 1
            if timestep*DT > max_T:
                break
    fig, ax = plt.subplots()
    t_plot = np.arange(0, max_T+DT/4, DT)
    ax.plot(t_plot, p_order)
    ax.set_xlabel("time")
    ax.set_ylabel(r"Polar order parameter, $\Psi$")

    folder = os.path.abspath('../plots/p_order_vs_time/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_xTy' + str(xTy) + '_s' + str(seed) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))


def del_pos(mode, nPart, phi, noise, K, xTy, seed):
    """
    Delete position file to save space
    """
    sim_dir = get_sim_dir(mode, nPart, phi, noise, K, xTy, seed)
    os.remove(os.path.join(sim_dir, "pos"))

def write_stats(mode, nPart, phi, noise, K, xTy, seed, min_T=None, max_T=None, remove_pos=False):
    """
    Write a file with various statistics from the simulation data (Vicsek order parameter mean, standard deviation, susceptibility)
    """
    inparFile, posFile = get_files(mode, nPart, phi, noise, K, xTy, seed)
    inpar_dict = get_params(inparFile)
    DT = inpar_dict["DT"]
    simulT = inpar_dict["simulT"]
    if min_T == None:
        min_T = 0
    if max_T == None:
        max_T = simulT
    
    p_order = []
    n_order = []
    with open(posFile) as f:
        line_count = 1
        timestep = int(min_T//DT)
        for line in f:
            if 8 + timestep*(nPart + 1) <= line_count <= 7 + timestep*(nPart + 1) + nPart:
                if line_count == 8 + timestep*(nPart+1):
                    cos_sum = 0
                    sin_sum = 0
                    cos_sq_sum = 0
                    cos_sin_sum = 0
                theta = float(line.split('\t')[2])
                cos_sum += np.cos(theta)
                sin_sum += np.sin(theta)
                cos_sq_sum += np.cos(theta)**2
                cos_sin_sum += np.sin(theta)*np.cos(theta)
                if line_count == 7 + timestep*(nPart + 1) + nPart:
                    p_order.append(np.sqrt(cos_sum**2+sin_sum**2)/nPart)
                    n_order.append(2*np.sqrt((cos_sq_sum/nPart - 1/2)**2+(cos_sin_sum/nPart)**2))
                    timestep += 1
            line_count += 1
            if timestep*DT > max_T:
                break
    p_mean = np.mean(p_order)
    p_sus = nPart*np.std(p_order)**2
    n_mean = np.mean(n_order)
    n_sus = nPart*np.std(n_order)**2


    sim_dir = get_sim_dir(mode, nPart, phi, noise, K, xTy, seed)
    statsFile = open(os.path.join(sim_dir, "stats"), "w")
    statsFile.write(str(p_mean) + '\n')
    statsFile.write(str(p_sus) + '\n')
    statsFile.write(str(n_mean) + '\n')
    statsFile.write(str(n_sus))
    statsFile.close()

    ## Write file with lower resolution than pos
    # write_pos_lowres(mode, nPart, phi, K, seed)

    if remove_pos == True:
        ## Remove position files to save space
        os.remove(os.path.join(sim_dir, "pos"))

def read_stats(mode, nPart, phi, noise, K, xTy, seed):
    """
    Read stats file and create dictionary with those statistics
    """
    sim_dir = get_sim_dir(mode, nPart, phi, noise, K, xTy, seed)

    with open(os.path.join(sim_dir, "stats")) as file:
        reader = csv.reader(file, delimiter="\n")
        r = list(reader)
    stats_dict = {}
    stats_dict["p_mean"] = float(r[0][0])
    stats_dict["p_sus"] = float(r[1][0])
    stats_dict["n_mean"] = float(r[2][0])
    stats_dict["n_sus"] = float(r[3][0])
    return stats_dict


def plot_porder_noise(mode, nPart, phi, noise_range, K, xTy, seed_range):
    """
    Plot steady state polar order parameter against noise
    Averaged over a number of realizations
    """
    fig, ax = plt.subplots()
    p_ss = []
    for noise in noise_range:
        p_ss_sum = 0
        for seed in seed_range:
            sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed)
            if not os.path.exists(os.path.join(sim_dir, 'stats')):
                write_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed)
            p_ss_sum += read_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed)["p_mean"]
        p_ss.append(p_ss_sum/len(seed_range))

    ax.plot(noise_range, p_ss, '-o')
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"Polar order parameter, $\Psi$")
    
    folder = os.path.abspath('../plots/p_order_vs_noise/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_K' + str(K) + '_xTy' + str(xTy) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))
