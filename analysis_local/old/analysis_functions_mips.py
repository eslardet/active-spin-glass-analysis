import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm, colors
from decimal import Decimal

import csv
import os


def get_sim_dir(mode, nPart, phi, K, seed, Pe=None):
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
    
    if Pe == None:
        sim_dir = os.path.abspath('../simulation_data/' + mode_name + '/N' + str(nPart) + '/phi' + str(phi) + '/K' + str(K) + '/s' + str(seed))
    else:
        sim_dir = os.path.abspath('../simulation_data_mips/' + mode_name + '/N' + str(nPart) + '/phi' + str(phi) + '_Pe' + str(Pe) + '/K' + str(K) + '/s' + str(seed))

    return sim_dir

def get_files(mode, nPart, phi, K, seed, Pe=None):
    """
    Get file paths for the input parameters and position files
    """
    sim_dir = get_sim_dir(mode, nPart, phi, K, seed, Pe)
    inparFile = os.path.join(sim_dir, "inpar")
    posFile = os.path.join(sim_dir, "pos")
    return inparFile, posFile

def get_file_path(mode, nPart, phi, K, seed, file_name, Pe=None):
    """
    Get the file path for a certain file name in the simulation data directory
    """
    sim_dir = get_sim_dir(mode, nPart, phi, K, seed, Pe)
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
    inpar_dict["Pe"] = float(r[4][0])
    # inpar_dict["Rp"] = float(r[6][0])
    # inpar_dict["xTy"] = float(r[7][0])
    # inpar_dict["mode"] = r[9][0]
    # inpar_dict["repulsion"] = r[-1][0]
    
    # if inpar_dict["mode"] == 'C':
    #     inpar_dict["DT"] = float(r[12][0])
    #     inpar_dict["simulT"] = float(r[15][0])
    # elif inpar_dict["mode"] == 'T':
    #     inpar_dict["DT"] = float(r[14][0])
    #     inpar_dict["simulT"] = float(r[17][0])
    # else:
    #     inpar_dict["DT"] = float(r[13][0])
    #     inpar_dict["simulT"] = float(r[16][0])

    inpar_dict["xTy"] = float(r[6][0])
    inpar_dict["repulsion"] = r[-1][0]
    
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

def get_theta_snapshot(posFile, nPart, timestep):
    """
    Get list of theta orientations at a single timestep
    """
    with open(posFile) as f:
        line_count = 0
        theta = []
        for line in f:
            line_count += 1
            if 8 + timestep*(nPart + 1) <= line_count <= 7 + timestep*(nPart + 1) + nPart:
                theta.append(float(line.split('\t')[2]))
            if line_count > 7 + timestep*(nPart + 1) + nPart:
                break
    return theta

def snapshot(mode, nPart, phi, Pe, K, seed, view_time, show_quiver=False, show_color=True):
    """
    Get static snapshot at specified time from the positions file
    """

    inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, seed=seed)
    inpar_dict = get_params(inparFile)
    
    nPart = inpar_dict["nPart"]
    phi = inpar_dict["phi"]
    Pe = inpar_dict["Pe"]
    # mode = inpar_dict["mode"]
    DT = inpar_dict["DT"]
    # Rp = inpar_dict["Rp"]
    xTy = inpar_dict["xTy"]
    repulsion = inpar_dict["repulsion"]

    if repulsion == 'W':
        beta = 2**(1/6)
    if repulsion == 'H':
        beta = 2
    if repulsion == 'C':
        beta = 1
    else:
        beta = 2**(1/6)
    
    L = np.sqrt(nPart*np.pi*beta**2 / (4*phi*xTy))
    Ly = L
    Lx = L*xTy

    timestep = int(view_time/DT)
    view_time = timestep*DT

    x, y, theta = get_pos_snapshot(posFile=posFile, nPart=nPart, timestep=timestep)
    x = pbc_wrap(x,Lx)
    y = pbc_wrap(y,Ly)
    
    fig, ax = plt.subplots(figsize=(5*xTy,5), dpi=72)
    diameter = (ax.get_window_extent().height * 72/fig.dpi) /L *beta
    if repulsion == 'C':
        diameter = diameter * 1.6

    norm = colors.Normalize(vmin=0.0, vmax=2*np.pi, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.hsv)
    
    if mode == "T":
        nA = nPart//2
        ax.plot(x[:nA], y[:nA], 'o', ms=diameter, zorder=1)
        ax.plot(x[nA:], y[nA:], 'o', ms=diameter, zorder=1)
    else:
        if show_color == True:
            for i in range(nPart):
                color = mapper.to_rgba(theta[i]%(2*np.pi))
                ax.plot(x[i], y[i], 'o', ms=diameter, color=color, zorder=1)
        else:
            ax.plot(x, y, 'o', ms=diameter, zorder=1)
    if show_quiver == True:
        ax.quiver(x, y, np.cos(theta), np.sin(theta), zorder=2)
    ax.set_xlim(0,Lx)
    ax.set_ylim(0,Ly)
    ax.set_aspect('equal')
    ax.set_title("t=" + str(view_time))
    cbar = plt.colorbar(mappable=mapper, ax=ax)
    
    folder = os.path.abspath('../snapshots')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_Pe' + str(Pe) + '_K' + str(K) + '_s' + str(seed) + '_' + repulsion + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))


def snapshot_pos_ex(mode, nPart, phi, Pe, K, seed, show_quiver=False, show_color=True):
    """
    Get static snapshot at specified time from the exact positions file
    The file structure must have the time on the first line and the x,y,theta coordinates for the N particles on each
    of the following lines
    """
    inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, seed=seed)
    posFileExact = get_file_path(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, seed=seed, file_name="pos_exact")
    inpar_dict = get_params(inparFile)
    
    nPart = inpar_dict["nPart"]
    phi = inpar_dict["phi"]
    Pe = inpar_dict["Pe"]
    mode = inpar_dict["mode"]
    DT = inpar_dict["DT"]
    Rp = inpar_dict["Rp"]
    xTy = inpar_dict["xTy"]
    repulsion = inpar_dict["repulsion"]

    if repulsion == 'W':
        beta = 2**(1/6)
    if repulsion == 'H':
        beta = 2
    if repulsion == 'C':
        beta = 1
    else:
        beta = 2**(1/6)
    
    L = np.sqrt(nPart*np.pi*beta**2 / (4*phi*xTy))
    Ly = L
    Lx = L*xTy
    
    with open(posFileExact) as f:
        reader = csv.reader(f, delimiter="\t")
        r = list(reader)
    
    view_time = round(float(r[0][0]), 2)
    x = pbc_wrap(np.array(r[1:nPart+1]).astype('float')[:,0], Lx)
    y = pbc_wrap(np.array(r[1:nPart+1]).astype('float')[:,1], Ly)
    theta = np.array(r[1:nPart+1]).astype('float')[:,2]
    
    fig, ax = plt.subplots(figsize=(5*xTy,5), dpi=72)
    diameter = (ax.get_window_extent().height * 72/fig.dpi) /L *beta
    if repulsion == 'C':
        diameter = diameter * 1.6

    norm = colors.Normalize(vmin=0.0, vmax=2*np.pi, clip=True)
    # norm = colors.Normalize(vmin=-1.0, vmax=1.0, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.hsv)
    
    if mode == "T":
        nA = nPart//2
        ax.plot(x[:nA], y[:nA], 'o', ms=diameter, zorder=1)
        ax.plot(x[nA:], y[nA:], 'o', ms=diameter, zorder=1)
    else:
        if show_color == True:
            for i in range(nPart):
                color = mapper.to_rgba(theta[i]%(2*np.pi))
                # color = mapper.to_rgba(np.cos(theta[i]))
                ax.plot(x[i], y[i], 'o', ms=diameter, color=color, zorder=1)
            plt.colorbar(mappable=mapper, ax=ax)
        else:
            ax.plot(x, y, 'o', ms=diameter, zorder=1)
    if show_quiver == True:
        ax.quiver(x, y, np.cos(theta), np.sin(theta), zorder=2)
    ax.set_xlim(0,Lx)
    ax.set_ylim(0,Ly)
    ax.set_aspect('equal')
    ax.set_title("t=" + str(view_time))
    
    folder = os.path.abspath('../snapshots')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_Pe' + str(Pe) + '_K' + str(K) + '_s' + str(seed) + '_Rp' + str(Rp) + '_' + repulsion + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

def animate(mode, nPart, phi, Pe, K, seed, min_T=None, max_T=None):
    """
    Make animation from positions file
    """
    inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, seed=seed)
    
    x_all, y_all, theta_all = get_pos_arr(inparFile, posFile, min_T, max_T)
    
    inpar_dict = get_params(inparFile)
    
    nPart = inpar_dict["nPart"]
    phi = inpar_dict["phi"]
    Pe = inpar_dict['Pe']
    # mode = inpar_dict["mode"]
    DT = inpar_dict["DT"]
    seed = inpar_dict["seed"]
    # Rp = inpar_dict["Rp"]
    xTy = inpar_dict["xTy"]
    repulsion = inpar_dict["repulsion"]

    with open(posFile) as f:
        reader = csv.reader(f, delimiter="\t")
        startT = float(list(reader)[6][0])

    plt.rcParams["animation.html"] = "jshtml"
    plt.ioff()
    plt.rcParams['animation.embed_limit'] = 2**128

    fig, ax = plt.subplots(figsize=(3*xTy,3))
    
    if repulsion == 'W':
        beta = 2**(1/6)
    if repulsion == 'H':
        beta = 2
    if repulsion == 'C':
        beta = 1
    else:
        beta = 2**(1/6)
    L = np.sqrt(nPart*np.pi*beta**2 / (4*phi*xTy))
    Ly = L
    Lx = L*xTy
    
    diameter = (ax.get_window_extent().height * 72/fig.dpi) /L * beta
    if repulsion == 'C':
        diameter = diameter * 1.6

    points_A, = plt.plot([], [], 'o', ms=diameter, zorder=1)
    points_B, = plt.plot([], [], 'o', ms=diameter, zorder=2)

    x = pbc_wrap(x_all[0], Lx)
    y = pbc_wrap(y_all[0], Ly)
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
        x = pbc_wrap(x_all[n], Lx)
        y = pbc_wrap(y_all[n], Ly)
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

    ani = FuncAnimation(fig, update, init_func=init, frames=len(x_all), interval=10, blit=True)

    # ani.save("ani.mp4")
    folder = os.path.abspath('../animations')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_Pe' + str(Pe) + '_K' + str(K) + '_s' + str(seed) + '_' + repulsion + '.mp4'
    if not os.path.exists(folder):
        os.makedirs(folder)
    ani.save(os.path.join(folder, filename))


def plot_porder_time(mode, nPart, phi, K, seed, min_T=None, max_T=None):
    """
    Plot Vicsek order parameter against time for one simulation
    """
    inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, K=K, seed=seed)
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
    ax.set_ylabel(r"Vicsek order parameter, $\Psi$")
    
    plt.savefig(os.path.abspath('../plots/p_order_vs_time/' + mode + '_N' + str(nPart) + '_phi' + str(phi) + '_K' + str(K) + '_s' + str(seed) + '.png'))


def pos_lowres(mode, nPart, phi, Pe, K, seed, DT_new=1.0, delete=True):
    """
    Write a file with lower resolution, delete old position file and modify inpar file with new resolution
    """
    sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, seed=seed)
    posFile = os.path.join(sim_dir, "pos")
    lowresFile = os.path.join(sim_dir, "pos_low_res")
    lrf = open(lowresFile, "w")
    with open(posFile) as pf:
        line_count = 1
        for line in pf:
            if (line_count - 7) % (nPart+1) == 0:
                time = float(line)
            if line_count < 7:
                lrf.write(line)
            elif Decimal(str(time)) % Decimal(str(DT_new)) == 0: # Need to use Decimal because of Python floating point arithmetic being silly
                lrf.write(line)
            line_count += 1
    lrf.close()

    inparFile = os.path.join(sim_dir, "inpar")
    ifile = open(inparFile, 'r')
    inpar_data = ifile.readlines()
    
    if mode == 'C':
        inpar_data[12] = str(DT_new) + '\n'
    elif mode == 'T':
        inpar_data[14] = str(DT_new) + '\n'
    else:
        inpar_data[13] = str(DT_new) + '\n'
    ifile = open(inparFile, 'w')
    ifile.writelines(inpar_data)
    ifile.close()
    
    if delete == True:
        os.remove(posFile)
        os.rename(lowresFile, posFile)

def del_pos(mode, nPart, phi, Pe, K, seed):
    """
    Delete position file to save space
    """
    sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, seed=seed)
    os.remove(os.path.join(sim_dir, "pos"))

def write_stats(mode, nPart, phi, K, seed, min_T=None, max_T=None, remove_pos=False):
    """
    Write a file with various statistics from the simulation data (Vicsek order parameter mean, standard deviation, susceptibility)
    """
    inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, K=K, seed=seed)
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


    sim_dir = get_sim_dir(mode, nPart, phi, K, seed)
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

def read_stats(mode, nPart, phi, K, seed):
    """
    Read stats file and create dictionary with those statistics
    """
    sim_dir = get_sim_dir(mode, nPart, phi, K, seed)

    with open(os.path.join(sim_dir, "stats")) as file:
        reader = csv.reader(file, delimiter="\n")
        r = list(reader)
    stats_dict = {}
    stats_dict["p_mean"] = float(r[0][0])
    stats_dict["p_sus"] = float(r[1][0])
    stats_dict["n_mean"] = float(r[2][0])
    stats_dict["n_sus"] = float(r[3][0])
    return stats_dict


def plot_vorder_ksd(mode, nPart_range, phi, KAVG, KSTD_range, seed_range, log=False):
    """
    Plot steady state Vicsek order parameter against K_std for a fixed K_avg (Gaussian distributed couplings)
    Averaged over a number of realizations
    Superimposed plots for various N
    """
    fig, ax = plt.subplots()
    for nPart in nPart_range:
        v_ss = []
        for KSTD in KSTD_range:
            v_ss_sum = 0
            count_err = 0
            for seed in seed_range:
                sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed)
                if not os.path.exists(os.path.join(sim_dir, 'stats')):
                    try:
                        write_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed, avg_over=1000)
                    except:
                        print(nPart, KAVG, KSTD, seed)
                try:
                    v_ss_sum += read_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG) + "_" + str(KSTD), seed=seed)["v_mean"]
                except:
                    print("error")
                    count_err += 1
            v_ss.append(v_ss_sum/(len(seed_range) - count_err))
        ax.plot(KSTD_range, v_ss, 'o-', label="N=" + str(nPart))
    ax.set_xlabel("KSTD")
    ax.set_ylabel(r"Vicsek order parameter, $\Psi$")
    ax.legend()
    if log == True:
        ax.set_xscale("log")
    
    folder = os.path.abspath('../plots/v_order_vs_K/')
    filename = mode + '_phi' + str(phi) + '_KAVG' + str(KAVG) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))



def plot_vorder_kavg(mode, nPart_range, phi, KAVG_range, KSTD_range, seed_range):
    """
    Plot steady state Vicsek order parameter against K_avg for a fixed K_std (Gaussian distributed couplings)
    Averaged over a number of realizations
    Superimposed plots for various N and KSTD
    """
    fig, ax = plt.subplots()
    for nPart in nPart_range:
        for KSTD in KSTD_range:
            v_ss = []
            for KAVG in KAVG_range:
                v_ss_sum = 0
                count_err = 0
                for seed in seed_range:
                    sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed)
                    if not os.path.exists(os.path.join(sim_dir, 'stats')):
                        try:
                            write_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed, avg_over=1000)
                        except:
                            print(nPart, KAVG, KSTD, seed)
                            count_err += 1
                    try:
                        v_ss_sum += read_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG) + "_" + str(KSTD), seed=seed)["v_mean"]
                    except:
                        print("error")
                v_ss.append(v_ss_sum/(len(seed_range) - count_err))
            ax.plot(KAVG_range, v_ss, 'o-', label="N=" + str(nPart) + ", K_STD=" + str(KSTD))
    ax.set_xlabel("KAVG")
    ax.set_ylabel(r"Vicsek order parameter, $\Psi$")
    ax.legend()

    folder = os.path.abspath('../plots/v_order_vs_K/')
    filename = mode + '_phi' + str(phi) + '_KSTD' + str(KSTD) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

def plot_q_kavg(mode, nPart_range, phi, KAVG_range, KSTD_range, seed_range):
    """
    Plot steady state EA order parameter against K_avg for a fixed K_std (Gaussian distributed couplings)
    Averaged over a number of realizations
    Superimposed plots for various N and KSTD
    """
    fig, ax = plt.subplots()
    for nPart in nPart_range:
        for KSTD in KSTD_range:
            v_ss = []
            for KAVG in KAVG_range:
                v_ss_sum = 0
                count_err = 0
                for seed in seed_range:
                    sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed)
                    if not os.path.exists(os.path.join(sim_dir, 'stats')):
                        try:
                            write_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed, avg_over=1000)
                        except:
                            print(nPart, KAVG, KSTD, seed)
                            count_err += 1
                    try:
                        v_ss_sum += read_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG) + "_" + str(KSTD), seed=seed)["q"]
                    except:
                        print("error")
                v_ss.append(v_ss_sum/(len(seed_range) - count_err))
            ax.plot(KAVG_range, v_ss, 'o-', label="N=" + str(nPart) + ", K_STD=" + str(KSTD))
    ax.set_xlabel("KAVG")
    ax.set_ylabel(r"q order parameter")
    ax.legend()

    folder = os.path.abspath('../plots/q_vs_K/')
    filename = mode + '_phi' + str(phi) + '_KSTD' + str(KSTD) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))


        
def plot_vorder_sus_ksd(mode, nPart_range, phi, KAVG, KSTD_range, seed_range):
    """
    Plot the susceptibility of the Vicsek order parameter (over the final 1000 saved timesteps) against K_std, with a fixed K_std
    Averaged over a number of realizations
    Superimposed plots for various N
    """
    fig, ax = plt.subplots()
    for nPart in nPart_range:
        v_sus = []
        for KSTD in KSTD_range:
            v_sus_sum = 0
            count_err = 0
            for seed in seed_range:
                sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed)
                if not os.path.exists(os.path.join(sim_dir, 'stats')):
                    try:
                        write_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed, avg_over=1000)
                    except:
                        print(nPart, KAVG, KSTD, seed)
                        count_err += 1
                try: 
                    v_sus_sum += read_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG) + "_" + str(KSTD), seed=seed)["v_sus"]
                except:
                    print("error")
            v_sus.append(v_sus_sum/len(seed_range))
        
        ax.plot(KSTD_range, v_sus, 'o-', label="N=" + str(nPart))
    ax.set_xlabel("KSTD")
    ax.set_ylabel(r"Vicsek order parameter susceptibility")
    ax.legend()
    
    folder = os.path.abspath('../plots/v_order_sus_vs_K/')
    filename = mode + '_phi' + str(phi) + '_KSTD' + str(KSTD) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))


def plot_vorder_sus_kavg(mode, nPart_range, phi, KAVG_range, KSTD_range, seed_range):
    """
    Plot the susceptibility of the Vicsek order parameter (over the final 1000 saved timesteps) against K_avg, with a fixed K_std
    Averaged over a number of realizations
    Superimposed plots for various N
    """
    fig, ax = plt.subplots()
    for nPart in nPart_range:
        for KSTD in KSTD_range:
            v_sus = []
            for KAVG in KAVG_range:
                v_sus_sum = 0
                count_err = 0
                for seed in seed_range:
                    sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed)
                    if not os.path.exists(os.path.join(sim_dir, 'stats')):
                        try:
                            write_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed, avg_over=1000)
                        except:
                            print(nPart, KAVG, KSTD, seed)
                            count_err += 1
                    try:
                        v_sus_sum += read_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG) + "_" + str(KSTD), seed=seed)["v_sus"]
                    except:
                        print("error")
                v_sus.append(v_sus_sum/len(seed_range))
            
            ax.plot(KAVG_range, v_sus, 'o-', label="N=" + str(nPart) + ", KSTD=" + str(KSTD))
    ax.set_xlabel("KAVG")
    ax.set_ylabel(r"Vicsek order parameter susceptibility")
    ax.legend()
    
    folder = os.path.abspath('../plots/v_order_sus_vs_K/')
    filename = mode + '_phi' + str(phi) + '_KSTD' + str(KSTD) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

def plot_vorder_N(mode, nPart_range, phi, K, seed, log=False):
    """
    Plot the mean Vicsek order parameter against the number of particles in the system N, 
    for a given K (or "K_avg_K_std")
    """
    v_ss = []
    for nPart in nPart_range:
        sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, K=K, seed=seed)
        if not os.path.exists(os.path.join(sim_dir, 'stats')):
            write_stats(mode=mode, nPart=nPart, phi=phi, K=K, seed=seed, avg_over=1000)
        v_ss.append(read_stats(mode=mode, nPart=nPart, phi=phi, K=K, seed=seed)["v_mean"])
    fig, ax = plt.subplots()
    ax.plot(nPart_range, v_ss, 'o-')
    ax.set_xlabel("N")
    ax.set_ylabel(r"Vicsek order parameter, $\Psi$")

    if log == True:
        ax.set_xscale("log")
    
    folder = os.path.abspath('../plots/v_order_vs_N/')
    filename = mode + '_phi' + str(phi) + '_K' + str(K) + '_s' + str(seed) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))


def critical_value_kavg(mode, nPart, phi, KAVG_range, KSTD, seed_range):
    """
    Find where the plot crosses the horizontal 0.5 line to extract the critical value of KAVG
    """
    v_ss = []
    for KAVG in KAVG_range:
        v_ss_sum = 0
        count_err = 0
        for seed in seed_range:
            sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed)
            if not os.path.exists(os.path.join(sim_dir, 'stats')):
                try:
                    write_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed, avg_over=1000)
                except:
                    print(nPart, KAVG, KSTD, seed)
                    count_err += 1
            try:
                v_ss_sum += read_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG) + "_" + str(KSTD), seed=seed)["v_mean"]
            except:
                print("error")
        v_ss.append(v_ss_sum/(len(seed_range) - count_err))
    for i in range(len(v_ss)):
        if v_ss[i] > 0.5: # For a strictly increasing function
            break

    # Midpoint method
    # KAVG_crit = (KAVG_range[i] + KAVG_range[i-1])/2

    # Equation of line method (more accurate)
    grad = (v_ss[i]-v_ss[i-1])/(KAVG_range[i]-KAVG_range[i-1])
    intercept = v_ss[i] - grad*KAVG_range[i]

    KAVG_crit = (0.5-intercept)/grad
    
    return KAVG_crit



def plot_vorder_kavg_ax(mode, nPart_range, phi, KAVG_range, KSTD_range, seed_range, fig, ax):
    """
    Plot steady state Vicsek order parameter against K_avg for a fixed K_std (Gaussian distributed couplings)
    Averaged over a number of realizations
    Superimposed plots for various N and KSTD
    """
    # fig, ax = plt.subplots()
    for nPart in nPart_range:
        for KSTD in KSTD_range:
            v_ss = []
            for KAVG in KAVG_range:
                v_ss_sum = 0
                count_err = 0
                for seed in seed_range:
                    sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed)
                    if not os.path.exists(os.path.join(sim_dir, 'stats')):
                        try:
                            write_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed, avg_over=1000)
                        except:
                            print(nPart, KAVG, KSTD, seed)
                            count_err += 1
                    try:
                        v_ss_sum += read_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG) + "_" + str(KSTD), seed=seed)["v_mean"]
                    except:
                        print("error")
                v_ss.append(v_ss_sum/(len(seed_range) - count_err))
            ax.plot(KAVG_range, v_ss, 'o-', label="N=" + str(nPart) + ", K_STD=" + str(KSTD))
    ax.set_xlabel("KAVG")
    ax.set_ylabel(r"Vicsek order parameter, $\Psi$")
    ax.legend()

    return fig, ax

def plot_vorder_kavg_sus_ax(mode, nPart_range, phi, KAVG_range, KSTD_range, seed_range, fig, ax):
    """
    Plot susceptibility of steady state Vicsek order parameter against K_avg for a fixed K_std (Gaussian distributed couplings)
    Averaged over a number of realizations
    Superimposed plots for various N and KSTD
    """
    # fig, ax = plt.subplots()
    for nPart in nPart_range:
        for KSTD in KSTD_range:
            v_ss = []
            for KAVG in KAVG_range:
                v_ss_sum = 0
                count_err = 0
                for seed in seed_range:
                    sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed)
                    if not os.path.exists(os.path.join(sim_dir, 'stats')):
                        try:
                            write_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed, avg_over=1000)
                        except:
                            print(nPart, KAVG, KSTD, seed)
                            count_err += 1
                    try:
                        v_ss_sum += read_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG) + "_" + str(KSTD), seed=seed)["v_sus"]
                    except:
                        print("error")
                v_ss.append(v_ss_sum/(len(seed_range) - count_err))
            ax.plot(KAVG_range, v_ss, 'o-', label="N=" + str(nPart) + ", K_STD=" + str(KSTD))
    ax.set_xlabel("KAVG")
    ax.set_ylabel(r"Vicsek order parameter susceptibility")
    ax.legend()

    return fig, ax

def plot_vorder_kratio_ax(mode, nPart_range, phi, KAVG_range, KSTD_range, seed_range, intercept, power, fig, ax):
    """
    Plot steady state Vicsek order parameter against K_avg/K_STD (Gaussian distributed couplings)
    Averaged over a number of realizations
    Superimposed plots for various N and KSTD
    """
    # fig, ax = plt.subplots()
    for nPart in nPart_range:
        for KSTD in KSTD_range:
            v_ss = []
            for KAVG in KAVG_range:
                v_ss_sum = 0
                count_err = 0
                for seed in seed_range:
                    sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed)
                    if not os.path.exists(os.path.join(sim_dir, 'stats')):
                        try:
                            write_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed, avg_over=1000)
                        except:
                            print(nPart, KAVG, KSTD, seed)
                            count_err += 1
                    try:
                        v_ss_sum += read_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG) + "_" + str(KSTD), seed=seed)["v_mean"]
                    except:
                        print("error")
                v_ss.append(v_ss_sum/(len(seed_range) - count_err))
            ax.plot([(k-intercept)/KSTD**power for k in KAVG_range], v_ss, 'o-', label="N=" + str(nPart) + ", K_STD=" + str(KSTD))
    ax.set_xlabel("KAVG/KSTD^" + str(power))
    ax.set_ylabel(r"Vicsek order parameter, $\Psi$")
    ax.legend()

    return fig, ax

def read_couplings(mode, nPart, phi, Pe, K, seed):
    sim_dir = get_sim_dir(mode, nPart, phi, K, seed, Pe)
    couplingFile = os.path.join(sim_dir, "coupling")

    with open(couplingFile) as f:
        reader = csv.reader(f, delimiter="\t")
        r = list(reader)

    couplings = np.array(r).astype('float')

    return couplings


def rij_avg(mode, nPart, phi, Pe, K, seed, avg_over):
    inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, seed=seed)
    inpar_dict = get_params(inparFile)
    DT = inpar_dict["DT"]
    simulT = inpar_dict["simulT"]
    max_T = simulT
    min_T = simulT - avg_over*DT

    x_all, y_all, theta_all = get_pos_arr(inparFile, posFile, min_T, max_T)
    avg_over = len(x_all)

    beta = 2
    xTy = inpar_dict["xTy"]
    L = np.sqrt(nPart*np.pi*beta**2 / (4*phi*xTy))
    Ly = L
    Lx = L*xTy

    rij_sum = np.zeros(int(nPart*(nPart-1)/2))

    for t in range(avg_over):
        x = x_all[t]
        y = y_all[t]
        rij_t = []
        for i in range(nPart):
            for j in range(i+1, nPart):
                xij = x[i] - x[j]
                xij = xij - Lx*round(xij/Lx)
                yij = y[i] - y[j]
                yij = yij - Ly*round(yij/Ly)
                rij = np.sqrt(xij**2 + yij**2)
                rij_t.append(rij)
        rij_sum += np.asarray(rij_t)

    return rij_sum/avg_over

def plot_dist_coupling(mode, nPart, phi, Pe, KAVG, KSTD, seed, avg_over):
    K = str(KAVG) + '_' + str(KSTD)
    couplings = read_couplings(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, seed=seed)
    rij = rij_avg(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, seed=seed, avg_over=avg_over)

    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(couplings, rij, 'o', alpha=0.2, ms=1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$K_{ij}$")
    ax.set_ylabel(r"$\langle r_{ij}\rangle_t$")

    folder = os.path.abspath('../plots/dist_coupling/')
    filename = mode + '_phi' + str(phi) + '_Pe' + str(Pe) + '_K' + str(K) + '_s' + str(seed) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))