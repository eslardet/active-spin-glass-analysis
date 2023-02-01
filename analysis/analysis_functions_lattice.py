import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm, colors
from decimal import Decimal

import csv
import os


def get_sim_dir(mode, nPart, K, Rp, rotD, seed):
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

    sim_dir = os.path.abspath('../simulation_data_lattice/' + mode_name + '/N' + str(nPart) + '/K' + str(K) + '/Rp' + str(Rp) + '/rotD' + str(rotD) + '/s' + str(seed))
    # sim_dir = os.path.abspath('../simulation_data/' + mode_name + '/N' + str(nPart) + '/K' + str(K) + '/Rp' + str(Rp) + '/rotD' + str(rotD) + '/s' + str(seed))

    return sim_dir

def get_files(mode, nPart, K, Rp, rotD, seed):
    """
    Get file paths for the input parameters and position files
    """
    sim_dir = get_sim_dir(mode, nPart, K, Rp, rotD, seed)
    inparFile = os.path.join(sim_dir, "inpar")
    posFile = os.path.join(sim_dir, "pos")
    return inparFile, posFile

def get_file_path(mode, nPart, K, Rp, rotD, seed, file_name):
    """
    Get the file path for a certain file name in the simulation data directory
    """
    sim_dir = get_sim_dir(mode, nPart, K, Rp, rotD, seed)
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
    inpar_dict["seed"] = int(r[1][0])
    inpar_dict["rotD"] = float(r[2][0])
    inpar_dict["Rp"] = float(r[3][0])
    inpar_dict["mode"] = r[5][0]
    
    if inpar_dict["mode"] == 'C':
        inpar_dict["DT"] = float(r[8][0])
        inpar_dict["simulT"] = float(r[11][0])
    elif inpar_dict["mode"] == 'T':
        inpar_dict["DT"] = float(r[10][0])
        inpar_dict["simulT"] = float(r[13][0])
    else:
        inpar_dict["DT"] = float(r[9][0])
        inpar_dict["simulT"] = float(r[12][0])
    return inpar_dict

def pbc_wrap(x, L):
    """
    Wrap points into periodic box with length L
    """
    return x - L*np.round(x/L) + L/2


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
        theta.append(np.array(r[(nPart+1)*i+1:(nPart+1)*i+1+nPart]).astype('float'))
    return theta

def get_xy_lattice(nPart):
    beta = 1
    Nx = int(np.ceil(np.sqrt(nPart)))
    Ny = Nx

    Lx = Nx*beta
    Ly = np.sqrt(3)/2*Ny*beta

    Ntot = int(Nx*Ny)

    y = np.zeros(Ntot)
    x = np.zeros(Ntot)

    for i in range(Ny):
        for j in range(Nx):
            y[i*Nx+j] = i*np.sqrt(3)/2*beta + beta/2
            x[i*Nx+j] = j*beta
            if i % 2 == 1:
                x[i*Nx+j] += beta/2
    return x, y


def get_initpos_xy(mode, nPart, K, Rp, rotD, seed):
    initposFile = get_file_path(mode=mode, nPart=nPart, K=K, Rp=Rp, rotD=rotD, seed=seed, file_name="initpos")

    Nx = int(np.ceil(np.sqrt(nPart)))
    nPart = Nx*Nx

    with open(initposFile) as f:
        reader = csv.reader(f, delimiter="\t")
        r = list(reader)[5:]

    x = np.array(r[1:nPart+1]).astype('float')[:,0]
    y = np.array(r[1:nPart+1]).astype('float')[:,1]

    return x, y

def snapshot(mode, nPart, K, Rp, rotD, seed, view_time):
    """
    Get static snapshot at specified time from the positions file
    """

    inparFile, posFile = get_files(mode=mode, nPart=nPart, K=K, Rp=Rp, rotD=rotD, seed=seed)
    inpar_dict = get_params(inparFile)
    
    nPart = inpar_dict["nPart"]
    mode = inpar_dict["mode"]
    DT = inpar_dict["DT"]
    Rp = inpar_dict["Rp"]

    beta = 1
    
    Nx = int(np.ceil(np.sqrt(nPart)))
    Lx = Nx*beta
    Ly = np.sqrt(3)/2*Nx*beta
    xTy = Lx/Ly
    nPart = Nx*Nx
    
    with open(posFile) as f:
        reader = csv.reader(f, delimiter="\t")
        r = list(reader)[6:]
    
    i = int(view_time/DT)
    view_time = i*DT

    x, y = get_initpos_xy(mode=mode, nPart=nPart, K=K, Rp=Rp, rotD=rotD, seed=seed)

    theta = np.array(r[(nPart+1)*i+1:(nPart+1)*i+1+nPart]).astype('float')
    
    fig, ax = plt.subplots(figsize=(5*xTy,5), dpi=72)

    norm = colors.Normalize(vmin=0.0, vmax=2*np.pi, clip=True)
    # norm = colors.Normalize(vmin=-1.0, vmax=1.0, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.hsv)
    
    if mode == "T":
        nA = nPart//2
        ax.plot(x[:nA], y[:nA], 'o', zorder=1)
        ax.plot(x[nA:], y[nA:], 'o', zorder=1)
    else:
        ax.plot(x, y, 'o', zorder=1)
    ax.quiver(x, y, np.cos(theta), np.sin(theta), zorder=2)
    ax.set_xlim(0,Lx)
    ax.set_ylim(0,Ly)
    ax.set_aspect('equal')
    ax.set_title("t=" + str(view_time))
    # cbar.ax.set_ylabel(r'$\cos(\theta_i)$', rotation=270)
    
    folder = os.path.abspath('../snapshots_lattice')
    filename = mode + '_N' + str(nPart) + '_K' + str(K) + '_s' + str(seed) + '_Rp' + str(Rp) + '_rotD' + str(rotD) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))


def animate(mode, nPart, K, Rp, rotD, seed, min_T=None, max_T=None):
    """
    Make animation from positions file
    """
    inparFile, posFile = get_files(mode=mode, nPart=nPart, K=K, Rp=Rp, rotD=rotD, seed=seed)

    # x, y = get_xy_lattice(nPart) # could get from initpos file instead
    x, y = get_initpos_xy(mode=mode, nPart=nPart, K=K, Rp=Rp, rotD=rotD, seed=seed)
    
    theta_all = get_theta_arr(inparFile, posFile, min_T, max_T)
    
    inpar_dict = get_params(inparFile)
    
    nPart = inpar_dict["nPart"]
    mode = inpar_dict["mode"]
    DT = inpar_dict["DT"]
    seed = inpar_dict["seed"]
    Rp = inpar_dict["Rp"]
    rotD = inpar_dict["rotD"]

    with open(posFile) as f:
        reader = csv.reader(f, delimiter="\t")
        startT = float(list(reader)[6][0])

    plt.rcParams["animation.html"] = "jshtml"
    plt.ioff()
    plt.rcParams['animation.embed_limit'] = 2**128
    
    beta = 1.0

    Nx = np.ceil(np.sqrt(nPart))
    Lx = Nx*beta
    Ly = np.sqrt(3)/2*Nx*beta
    xTy = Lx/Ly
    
    fig, ax = plt.subplots(figsize=(5*xTy,5))

    points_A, = plt.plot([], [], 'o', zorder=1)
    points_B, = plt.plot([], [], 'o', zorder=2)

    theta = theta_all[0]
    arrows = plt.quiver(x, y, np.cos(theta), np.sin(theta), zorder=3)

    def init():
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)
        if mode == "T":
            return arrows, points_A, points_B,
        else:
            return arrows, points_A

    def update(n):
        theta = theta_all[n]
        if mode == "T":
            nA = nPart//2
            points_A.set_data(x[:nA], y[:nA])
            points_B.set_data(x[nA:], y[nA:])
        else: 
            points_A.set_data(x, y)
        arrows.set_offsets(np.c_[x, y])
        arrows.set_UVC(np.cos(theta), np.sin(theta))
        ax.set_title("t = " + str(round(n*DT+startT, 1)), fontsize=10, loc='left')
        
        if mode == "T":
            return arrows, points_A, points_B
        else: 
            return arrows, points_A

    ani = FuncAnimation(fig, update, init_func=init, frames=len(theta_all), interval=10, blit=True)

    folder = os.path.abspath('../animations_lattice')
    filename = mode + '_N' + str(nPart) + '_K' + str(K) + '_s' + str(seed) + '_Rp' + str(Rp) + '_rotD' + str(rotD) + '.mp4'
    if not os.path.exists(folder):
        os.makedirs(folder)
    ani.save(os.path.join(folder, filename))


def plot_porder_time(mode, nPart, K, Rp, rotD, seed, min_T=None, max_T=None):
    """
    Plot polar order parameter against time for one simulation
    """
    inparFile, posFile = get_files(mode=mode, nPart=nPart, K=K, Rp=Rp, rotD=rotD, seed=seed)
    inpar_dict = get_params(inparFile)
    DT = inpar_dict["DT"]
    print(DT)
    simulT = inpar_dict["simulT"]
    if min_T == None:
        min_T = 0
    if max_T == None:
        max_T = simulT
    
    Nx = int(np.ceil(np.sqrt(nPart)))
    nPart = Nx*Nx
    p_order = []
    with open(posFile) as f:
        line_count = 1
        timestep = int(min_T//DT)
        for line in f:
            if 8 + timestep*(nPart + 1) <= line_count <= 7 + timestep*(nPart + 1) + nPart:
                if line_count == 8 + timestep*(nPart+1):
                    cos_sum = 0
                    sin_sum = 0
                cos_sum += np.cos(float(line))
                sin_sum += np.sin(float(line))
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
    filename = mode + '_N' + str(nPart) + '_K' + str(K) + '_s' + str(seed) + '_Rp' + str(Rp) + '_rotD' + str(rotD) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))


def plot_norder_time(mode, nPart, K, Rp, rotD, seed, min_T=None, max_T=None):
    """
    Plot nematic order parameter against time for one simulation
    """
    inparFile, posFile = get_files(mode=mode, nPart=nPart, K=K, Rp=Rp, rotD=rotD, seed=seed)
    inpar_dict = get_params(inparFile)
    DT = inpar_dict["DT"]
    simulT = inpar_dict["simulT"]
    if min_T == None:
        min_T = 0
    if max_T == None:
        max_T = simulT

    Nx = int(np.ceil(np.sqrt(nPart)))
    nPart = Nx*Nx

    n_order = []
    with open(posFile) as f:
        line_count = 1
        timestep = int(min_T//DT)
        for line in f:
            if 8 + timestep*(nPart + 1) <= line_count <= 7 + timestep*(nPart + 1) + nPart:
                if line_count == 8 + timestep*(nPart+1):
                    cos_sq_sum = 0
                    cos_sin_sum = 0
                cos_sq_sum += np.cos(float(line))**2
                cos_sin_sum += np.sin(float(line))*np.cos(float(line))
                if line_count == 7 + timestep*(nPart + 1) + nPart:
                    n_order.append(2*np.sqrt((cos_sq_sum/nPart - 1/2)**2+(cos_sin_sum/nPart)**2))
                    timestep += 1
            line_count += 1
            if timestep*DT > max_T:
                break

    fig, ax = plt.subplots()
    t_plot = np.arange(0, max_T+DT/4, DT)
    ax.plot(t_plot, n_order)
    ax.set_xlabel("time")
    ax.set_ylabel(r"Nematic order parameter, $\Psi$")
    
    folder = os.path.abspath('../plots/n_order_vs_time/')
    filename = mode + '_N' + str(nPart) + '_K' + str(K) + '_s' + str(seed) + '_Rp' + str(Rp) + '_rotD' + str(rotD) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

def write_stats(mode, nPart, K, Rp, rotD, seed, min_T=None, max_T=None, remove_pos=False):
    """
    Write a file with various statistics from the simulation data (Vicsek order parameter mean, standard deviation, susceptibility)
    """
    inparFile, posFile = get_files(mode=mode, nPart=nPart, K=K, Rp=Rp, rotD=rotD, seed=seed)
    inpar_dict = get_params(inparFile)
    sim_dir = get_sim_dir(mode=mode, nPart=nPart, K=K, Rp=Rp, rotD=rotD, seed=seed)
    
    theta_all = get_theta_arr(inparFile, posFile, min_T, max_T)
    Nx = int(np.ceil(np.sqrt(nPart)))
    nPart = Nx*Nx
    v_order = []
    for theta_t in theta_all:
        cos_sum = 0
        sin_sum = 0
        for i in theta_t:
            cos_sum += np.cos(i)
            sin_sum += np.sin(i)
        v_order.append(np.sqrt(cos_sum**2+sin_sum**2)/nPart)
    v_mean = np.mean(v_order)
    v_sd = np.std(v_order)
    v_sus = nPart*v_sd**2


    statsFile = open(os.path.join(sim_dir, "stats"), "w")
    statsFile.write(str(v_mean) + '\n')
    statsFile.write(str(v_sus))
    statsFile.close()

    ## Write file with lower resolution than pos

    if remove_pos == True:
        ## Remove position files to save space
        os.remove(os.path.join(sim_dir, "pos"))

def read_stats(mode, nPart, K, Rp, rotD, seed):
    """
    Read stats file and create dictionary with those statistics
    """
    sim_dir = get_sim_dir(mode=mode, nPart=nPart, K=K, Rp=Rp, rotD=rotD, seed=seed)

    with open(os.path.join(sim_dir, "stats")) as file:
        reader = csv.reader(file, delimiter="\n")
        r = list(reader)
    stats_dict = {}
    stats_dict["v_mean"] = float(r[0][0])
    stats_dict["v_sus"] = float(r[1][0])
    return stats_dict

def plot_vorder_k(mode, nPart_range, K_range, Rp, rotD, seed_range):
    """
    Plot steady state Vicsek order parameter against K
    Averaged over a number of realizations
    Superimposed plots for various N and KSTD
    """
    fig, ax = plt.subplots()
    for nPart in nPart_range:
        for K in K_range:
            v_ss = []
            v_ss_sum = 0
            for seed in seed_range:
                sim_dir = get_sim_dir(mode=mode, nPart=nPart, K=str(K), Rp=Rp, rotD=rotD, seed=seed)
                if not os.path.exists(os.path.join(sim_dir, 'stats')):
                    write_stats(mode=mode, nPart=nPart, K=str(K), Rp=Rp, rotD=rotD, seed=seed, min_T=10.0)
                v_ss_sum += read_stats(mode=mode, nPart=nPart,  K=str(K), Rp=Rp, rotD=rotD, seed=seed)["v_mean"]
            v_ss.append(v_ss_sum/(len(seed_range)))
        ax.plot(K_range, v_ss, 'o-', label="N=" + str(nPart) + ", K=" + str(K))
    ax.set_xlabel("KAVG")
    ax.set_ylabel(r"Vicsek order parameter, $\Psi$")
    ax.legend()

    folder = os.path.abspath('../plots/v_order_vs_K/')
    filename = mode + '_Rp' + str(Rp) + 'rotD' + str(rotD) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))