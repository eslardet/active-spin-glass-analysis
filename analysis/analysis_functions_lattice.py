import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm, colors
from decimal import Decimal

import csv
import os


def get_sim_dir(mode, nPart, K, Rp, seed):
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

    sim_dir = os.path.abspath('../simulation_data_lattice/' + mode_name + '/N' + str(nPart) + '/K' + str(K) + '/Rp' + str(Rp) + '/s' + str(seed))

    return sim_dir

def get_files(mode, nPart, K, Rp, seed):
    """
    Get file paths for the input parameters and position files
    """
    sim_dir = get_sim_dir(mode, nPart, K, Rp, seed)
    inparFile = os.path.join(sim_dir, "inpar")
    posFile = os.path.join(sim_dir, "pos")
    return inparFile, posFile

def get_file_path(mode, nPart, K, Rp, seed, file_name):
    """
    Get the file path for a certain file name in the simulation data directory
    """
    sim_dir = get_sim_dir(mode, nPart, K, Rp, seed)
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

def get_pos_arr(inparFile, posFile, min_T=None, max_T=None):
    """
    Get arrays for x, y, theta positions at each save time from the positions text file
    """
    inpar_dict = get_params(inparFile)
    
    nPart = inpar_dict["nPart"]
    DT = inpar_dict["DT"]
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

    for i in range(max(int((min_T-startT)/DT),0), int((max_T-startT)/DT)+1):
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
    xTy = Lx/Ly

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


def get_initpos_xy(mode, nPart, K, Rp, seed):
    initposFile = get_file_path(mode=mode, nPart=nPart, K=K, Rp=Rp, seed=seed, file_name="initpos")

    Nx = int(np.ceil(np.sqrt(nPart)))
    nPart = Nx*Nx

    with open(initposFile) as f:
        reader = csv.reader(f, delimiter="\t")
        r = list(reader)[5:]

    x = np.array(r[1:nPart+1]).astype('float')[:,0]
    y = np.array(r[1:nPart+1]).astype('float')[:,1]

    return x, y

def snapshot(mode, nPart, K, Rp, seed, view_time):
    """
    Get static snapshot at specified time from the positions file
    """

    inparFile, posFile = get_files(mode=mode, nPart=nPart, K=K, Rp=Rp, seed=seed)
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

    x, y = get_initpos_xy(mode=mode, nPart=nPart, K=K, Rp=Rp, seed=seed)

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
    filename = mode + '_N' + str(nPart) + '_K' + str(K) + '_s' + str(seed) + '_Rp' + str(Rp) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))


def animate(mode, nPart, K, Rp, seed, min_T=None, max_T=None):
    """
    Make animation from positions file
    """
    inparFile, posFile = get_files(mode=mode, nPart=nPart, K=K, Rp=Rp, seed=seed)

    # x, y = get_xy_lattice(nPart) # could get from initpos file instead
    x, y = get_initpos_xy(mode=mode, nPart=nPart, K=K, Rp=Rp, seed=seed)
    
    theta_all = get_theta_arr(inparFile, posFile, min_T, max_T)
    
    inpar_dict = get_params(inparFile)
    
    nPart = inpar_dict["nPart"]
    mode = inpar_dict["mode"]
    DT = inpar_dict["DT"]
    seed = inpar_dict["seed"]
    Rp = inpar_dict["Rp"]
    xTy = inpar_dict["xTy"]

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
    filename = mode + '_N' + str(nPart) + '_K' + str(K) + '_s' + str(seed) + '_Rp' + str(Rp) + '.mp4'
    if not os.path.exists(folder):
        os.makedirs(folder)
    ani.save(os.path.join(folder, filename))


