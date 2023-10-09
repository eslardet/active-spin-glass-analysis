import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm, colors
from decimal import Decimal
# import freud
import scipy.stats as sps
import bisect

import csv
import os


def get_sim_dir(mode, nPart, phi, noise, K, Rp, xTy, seed):
    if mode == "C":
        mode_name = "Constant"
    elif mode == "T":
        mode_name = "TwoPopulations"
    elif mode == "G":
        mode_name = "Gaussian"
    elif mode == "A":
        mode_name = "Asymmetric"
    elif mode == "F":
        mode_name = "Fraction"

    sim_dir = os.path.abspath('../simulation_data/' + mode_name + '/N' + str(nPart) + '/phi' + str(phi) + '_n' + str(noise) + '/K' + str(K) + '/Rp' + str(Rp) + '/xTy' + str(xTy) + '/s' + str(seed))

    return sim_dir

def get_files(mode, nPart, phi, noise, K, Rp, xTy, seed):
    """
    Get file paths for the input parameters and position files
    """
    sim_dir = get_sim_dir(mode, nPart, phi, noise, K, Rp, xTy, seed)
    inparFile = os.path.join(sim_dir, "inpar")
    posFile = os.path.join(sim_dir, "pos")
    return inparFile, posFile

def get_file_path(mode, nPart, phi, noise, K, Rp, xTy, seed, file_name):
    """
    Get the file path for a certain file name in the simulation data directory
    """
    sim_dir = get_sim_dir(mode, nPart, phi, noise, K, Rp, xTy, seed)
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
    try:
        inpar_dict["Rp"] = float(r[5][0])
    except:
        inpar_dict["Rp"] = str(r[5][0])
    inpar_dict["xTy"] = float(r[6][0])
    inpar_dict["start_mode"] = r[7][0]
    inpar_dict["mode"] = r[8][0]
    
    if inpar_dict["mode"] == 'C':
        inpar_dict["dt"] = float(r[10][0])
        inpar_dict["DT"] = float(r[11][0])
        inpar_dict["eqT"] = float(r[13][0])
        inpar_dict["simulT"] = float(r[14][0])
    elif inpar_dict["mode"] == 'T':
        inpar_dict["dt"] = float(r[12][0])
        inpar_dict["DT"] = float(r[13][0])
        inpar_dict["eqT"] = float(r[15][0])
        inpar_dict["simulT"] = float(r[16][0])
    else:
        inpar_dict["dt"] = float(r[11][0])
        inpar_dict["DT"] = float(r[12][0])
        inpar_dict["eqT"] = float(r[14][0])
        inpar_dict["simulT"] = float(r[15][0])
    return inpar_dict

def pbc_wrap(x, L):
    """
    Wrap points into periodic box with length L (from 0 to L) for display
    """
    return x - L*np.round(x/L) + L/2

def pbc_wrap_calc(x, L):
    """
    Wrap points into periodic box with length L (from -L/2 to L/2) for calculations
    """
    return x - L*np.round(x/L)


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


def snapshot(mode, nPart, phi, noise, K, Rp, xTy, seed, view_time=None, pos_ex=True, show_color=True, show_plot=False, save_plot=True, save_in_folder=False, timestep=None, neigh_col=False, r_max=None):
    """
    Get static snapshot at specified time from the positions file
    """

    inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
    if pos_ex == True:
        posFileExact = get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, file_name="pos_exact")
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
        if timestep == None:
            if view_time == None:
                view_time = simulT
            timestep = int(view_time/DT)
            view_time = timestep*DT + eqT

        x, y, theta = get_pos_snapshot(posFile=posFile, nPart=nPart, timestep=timestep)
    
    x = pbc_wrap(x,Lx)
    y = pbc_wrap(y,Ly)
    u = np.cos(theta)
    v = np.sin(theta)
    
    fig, ax = plt.subplots(figsize=(10*xTy,10), dpi=72)
    if view_time == None:
        view_time = timestep
    ax.set_title("t=" + str(round(view_time)))
    if neigh_col == True:
        if r_max == None:
            r_max = Rp
        num_nei = neighbour_counts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max)
        norm = colors.Normalize(vmin=0.0, vmax=np.max(num_nei), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.plasma)
        cols = mapper.to_rgba(num_nei)
        ax.quiver(x, y, u, v, color=cols)
        # cbar = plt.colorbar(mappable=mapper, ax=ax)
        # cbar.set_label('# neighbours', rotation=270)
        ax.set_title("t=" + str(round(view_time)) + r", $r_{max}$=" + str(r_max))

    elif show_color == True:
        norm = colors.Normalize(vmin=0.0, vmax=2*np.pi, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.hsv)
        cols = mapper.to_rgba(np.mod(theta, 2*np.pi))
        ax.quiver(x, y, u, v, color=cols)
        # plt.colorbar(mappable=mapper, ax=ax)
    else:
        ax.quiver(x, y, u, v)
    ax.set_xlim(0,Lx)
    ax.set_ylim(0,Ly)
    ax.set_aspect('equal')

    if save_plot == True:
        if save_in_folder == True:
            folder = get_sim_dir(mode, nPart, phi, noise, K, Rp, xTy, seed)
            filename = 'snapshot.png'
        else:
            folder = os.path.abspath('../snapshots_vicsek')
            filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed) + '.png'
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(os.path.join(folder, filename))
    if show_plot == True:
        plt.show()
    plt.close()


def animate(mode, nPart, phi, noise, K, Rp, xTy, seed, min_T=None, max_T=None):
    """
    Make animation from positions file
    """
    inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)

    x_all, y_all, theta_all = get_pos_arr(inparFile=inparFile, posFile=posFile, min_T=min_T, max_T=max_T)
    
    inpar_dict = get_params(inparFile)
    
    nPart = inpar_dict["nPart"]
    phi = inpar_dict["phi"]
    noise = inpar_dict["noise"]
    mode = inpar_dict["mode"]
    DT = inpar_dict["DT"]
    seed = inpar_dict["seed"]
    xTy = inpar_dict["xTy"]

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

    mapper = cm.ScalarMappable(norm=norm, cmap=cm.hsv)
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
        arrows.set_offsets(np.c_[x, y])
        cols = np.mod(theta, 2*np.pi)
        arrows.set_UVC(np.cos(theta), np.sin(theta), norm(cols))
        ax.set_title("t = " + str(round(n*DT+startT, 1)), fontsize=10, loc='left')
        
        return arrows,

    ani = FuncAnimation(fig, update, init_func=init, frames=len(theta_all), interval=10, blit=True)

    folder = os.path.abspath('../animations_vicsek')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed) + '.mp4'
    if not os.path.exists(folder):
        os.makedirs(folder)
    ani.save(os.path.join(folder, filename))

def animate_highlight(mode, nPart, phi, noise, K, Rp, xTy, seed, h, min_T=None, max_T=None):
    """
    Make animation from positions file
    """
    inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)

    x_all, y_all, theta_all = get_pos_arr(inparFile=inparFile, posFile=posFile, min_T=min_T, max_T=max_T)
    
    inpar_dict = get_params(inparFile)
    
    nPart = inpar_dict["nPart"]
    phi = inpar_dict["phi"]
    noise = inpar_dict["noise"]
    mode = inpar_dict["mode"]
    DT = inpar_dict["DT"]
    seed = inpar_dict["seed"]
    xTy = inpar_dict["xTy"]

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

    nh = list(set(list(range(nPart))) - set(h))

    # mapper = cm.ScalarMappable(norm=norm, cmap=cm.hsv)
    # plt.colorbar(mappable=mapper, ax=ax)

    x = pbc_wrap(x_all[0],Lx)
    y = pbc_wrap(y_all[0],Ly)
    theta = theta_all[0]
    col = ['k'] * nPart
    for i in h:
        col[i] = 'r'
    arrows = ax.quiver(x, y, np.cos(theta), np.sin(theta), color=col)

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
        ax.set_title("t = " + str(round(n*DT+startT+min_T, 1)), fontsize=10, loc='left')
        
        return arrows,

    ani = FuncAnimation(fig, update, init_func=init, frames=len(theta_all), interval=10, blit=True)

    folder = os.path.abspath('../animations_vicsek')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed) + '_highlights.mp4'
    if not os.path.exists(folder):
        os.makedirs(folder)
    ani.save(os.path.join(folder, filename))


def plot_porder_time(mode, nPart, phi, noise, K, Rp, xTy, seed, min_T=None, max_T=None):
    """
    Plot polar order parameter against time for one simulation
    """
    inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
    inpar_dict = get_params(inparFile)
    DT = inpar_dict["DT"]
    simulT = inpar_dict["simulT"]
    eqT = inpar_dict["eqT"]
    start_mode = inpar_dict["start_mode"]
    if min_T == None:
        min_T = 0
    if max_T == None:
        max_T = simulT
    
    p_order = []

    with open(posFile) as f:
        line_count = 1
        timestep = int(min_T//DT)
        startT = 0
        for line in f:
            if line_count == 7:
                startT = float(line)
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
            if timestep*DT > max_T-startT:
                break
                
    fig, ax = plt.subplots()
    t_plot = np.arange(startT, max_T+DT/4, DT)
    ax.plot(t_plot, p_order)
    ax.set_ylim([0,1])
    ax.set_xlabel("time")
    ax.set_ylabel(r"Polar order parameter, $\Psi$")

    folder = os.path.abspath('../plots/p_order_vs_time/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))


def del_pos(mode, nPart, phi, noise, K, Rp, xTy, seed):
    """
    Delete position file to save space
    """
    sim_dir = get_sim_dir(mode, nPart, phi, noise, K, Rp, xTy, seed)
    if os.path.exists(os.path.join(sim_dir, "pos")):
        os.remove(os.path.join(sim_dir, "pos"))
    else:
        print("No position file to delete:" + sim_dir)

def del_files(mode, nPart, phi, noise, K, Rp, xTy, seed, files):
    """
    Delete position file to save space
    """
    sim_dir = get_sim_dir(mode, nPart, phi, noise, K, Rp, xTy, seed)
    for file in files:
        path = os.path.join(sim_dir, file)
        if os.path.exists(path):
            os.remove(path)
        else:
            print("No file with name '" + file + "' to delete")

def write_stats(mode, nPart, phi, noise, K, Rp, xTy, seed, min_T=None, max_T=None, remove_pos=False, density_var=False, moments=False):
    """
    Write a file with various statistics from the simulation data (Vicsek order parameter mean, standard deviation, susceptibility)
    """
    inparFile, posFile = get_files(mode, nPart, phi, noise, K, Rp, xTy, seed)
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
    p_second = np.mean([p**2 for p in p_order])
    p_third = np.mean([p**3 for p in p_order])
    p_fourth = np.mean([p**4 for p in p_order])


    sim_dir = get_sim_dir(mode, nPart, phi, noise, K, Rp, xTy, seed)
    statsFile = open(os.path.join(sim_dir, "stats"), "w")
    statsFile.write(str(p_mean) + '\n')
    statsFile.write(str(p_sus) + '\n')
    statsFile.write(str(n_mean) + '\n')
    statsFile.write(str(n_sus) + '\n')

    if moments == True:
        statsFile.write(str(p_second) + '\n')
        statsFile.write(str(p_third) + '\n')
        statsFile.write(str(p_fourth) + '\n')

    if density_var == True:
        d_var = local_density_var(mode, nPart, phi, noise, K, Rp, xTy, seed)
        statsFile.write(str(d_var + '\n'))

    statsFile.close()

    ## Write file with lower resolution than pos
    # write_pos_lowres(mode, nPart, phi, K, seed)

    if remove_pos == True:
        ## Remove position files to save space
        os.remove(os.path.join(sim_dir, "pos"))

def read_stats(mode, nPart, phi, noise, K, Rp, xTy, seed):
    """
    Read stats file and create dictionary with those statistics
    """
    sim_dir = get_sim_dir(mode, nPart, phi, noise, K, Rp, xTy, seed)

    with open(os.path.join(sim_dir, "stats")) as file:
        reader = csv.reader(file, delimiter="\n")
        r = list(reader)
    stats_dict = {}
    stats_dict["p_mean"] = float(r[0][0])
    stats_dict["p_sus"] = float(r[1][0])
    stats_dict["n_mean"] = float(r[2][0])
    stats_dict["n_sus"] = float(r[3][0])

    try:
        stats_dict["p_2"] = float(r[4][0])
        stats_dict["p_3"] = float(r[5][0])
        stats_dict["p_4"] = float(r[6][0])
    except Exception:
        pass
    try:
        stats_dict["d_var"] = float(r[7][0])
    except Exception:
        pass
        # print("No d_var for rho=" + str(phi) + ", noise=" + str(noise) + ", K=" + str(K) + ", s=" + str(seed))
    return stats_dict


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

# def plot_porder_noise(mode, nPart, phi, noise_range, K, Rp, xTy, seed_range):
#     """
#     Plot steady state polar order parameter against noise
#     Averaged over a number of realizations
#     """
#     fig, ax = plt.subplots()
#     p_ss = []
#     print("hi, we started")
#     for noise in noise_range:
#         print("noise!" + str(noise))
#         p_ss_sum = 0
#         for seed in seed_range:
#             print("seed!" + str(seed))
#             sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
#             if not os.path.exists(os.path.join(sim_dir, 'stats')):
#                 write_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
#             p_mean = read_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)["p_mean"]
#             print(noise, p_mean)
#             p_ss_sum += p_mean
#         p_ss.append(p_ss_sum/len(seed_range))

#     noise_range = [float(i) for i in noise_range]
#     ax.plot(noise_range, p_ss, '-o')
#     ax.set_xlabel(r"$\eta$")
#     ax.set_ylabel(r"Polar order parameter, $\Psi$")
#     # ax.set_ylim([0,1])
    
#     folder = os.path.abspath('../plots/p_order_vs_noise/')
#     filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_K' + str(K) + '_Rp' + str(Rp) +  '_xTy' + str(xTy) + '.png'
#     if not os.path.exists(folder):
#         os.makedirs(folder)
#     plt.savefig(os.path.join(folder, filename))

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

##########################
#### Banding analysis ####
##########################

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

    extra_left = int(100 / grid_size_x)
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

            ax.plot(x_plot, d_plot_av, label=r"$K_{AVG}=$" + str(K_avg) + r"; $K_{STD}=$" + str(K_std))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"Average local density")
    ax.set_title(r"Average band: $\rho=$" + str(phi) + r"$, \eta=$" + str(noise))
    ax.legend()

    folder = os.path.abspath('../plots/density_profile_shifted/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_av.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))
    plt.show()

def local_density_var(mode, nPart, phi, noise, K, Rp, xTy, seed, min_grid_size=2, pos_ex=True, timestep=None):
    """
    Calculate the variance in the local densities of smaller grids from the final snapshot
    """
    if pos_ex == True:
        posFileExact = get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, file_name='pos_exact')
        x, y, theta, view_time = get_pos_ex_snapshot(file=posFileExact)
    else: 
        inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
        inpar_dict = get_params(inparFile)
        DT = inpar_dict["DT"]
        simulT = inpar_dict["simulT"]
        eqT = inpar_dict["eqT"]
        if timestep == None:
            timestep = int((simulT-eqT)/DT) 
        x, y, theta = get_pos_snapshot(posFile, nPart, timestep)

    L = np.sqrt(nPart / (phi*xTy))
    Ly = L
    Lx = L*xTy

    x = pbc_wrap(x,Lx)
    y = pbc_wrap(y,Ly)

    ngrid_x = int(Lx // min_grid_size)
    grid_size_x = Lx / ngrid_x
    ngrid_y = int(Ly // min_grid_size)
    grid_size_y = Ly / ngrid_y

    grid_area = grid_size_x*grid_size_y

    grid_counts = np.zeros((ngrid_x, ngrid_y))

    for i in range(nPart):
        gridx = int(x[i]//grid_size_x)
        gridy = int(y[i]//grid_size_y)
        grid_counts[gridx,gridy] += 1
    n_density = grid_counts / grid_area

    var_density = np.std(n_density)**2

    return var_density

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

        ax.plot(K_avg_range, vars, 'o-', label=r"$K_{STD}=$" + str(K_std))
    ax.set_xlabel(r"$K_{AVG}$")
    ax.set_ylabel(r"Local density variance $\langle(\rho-\bar{\rho})^2\rangle$")
    ax.legend()

    folder = os.path.abspath('../plots/var_density_vs_Kavg/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))



def critical_value_kavg(mode, nPart, phi, noise, K_avg_range, K_std, Rp, xTy, seed_range, cutoff):
    """
    Find where the plot crosses the horizontal line at a cutoff value to extract the critical value of KAVG
    """
    p_ss = []
    for K_avg in K_avg_range:
        p_ss_sum = 0
        count_err = 0
        for seed in seed_range:
            sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, noise=noise, K=str(K_avg)+'_'+str(K_std), Rp=Rp, xTy=xTy, seed=seed)
            if not os.path.exists(os.path.join(sim_dir, 'stats')):
                try:
                    write_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=str(K_avg)+'_'+str(K_std), Rp=Rp, xTy=xTy, seed=seed)
                except:
                    print(nPart, K_avg, K_std, seed)
                    count_err += 1
            try:
                p_ss_sum += read_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=str(K_avg) + "_" + str(K_std), Rp=Rp, xTy=xTy, seed=seed)["p_mean"]
            except:
                print("error")
        p_ss.append(p_ss_sum/(len(seed_range) - count_err))
    for i in range(len(p_ss)):
        if p_ss[i] > cutoff: # For a strictly increasing function
            break

    # Midpoint method
    # KAVG_crit = (KAVG_range[i] + KAVG_range[i-1])/2

    # Equation of line method (more accurate)
    grad = (p_ss[i]-p_ss[i-1])/(K_avg_range[i]-K_avg_range[i-1])
    intercept = p_ss[i] - grad*K_avg_range[i]

    KAVG_crit = (cutoff-intercept)/grad
    
    return KAVG_crit


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

###########################
## Correlation Functions ##
###########################

def plot_correlation(mode, nPart, phi, noise, K_avg_range, K_std_range, Rp, xTy, seed_range, timestep_range, pos_ex=False):
    """
    Plot equal time 2-point correlation function, averaged over time and seeds
    """
    import freud
    L = np.sqrt(nPart / (phi*xTy))
    Ly = L
    Lx = L*xTy

    r_max = Ly / 2.01

    fig, ax = plt.subplots()

    for K_avg in K_avg_range:
        for K_std in K_std_range:
            cf = freud.density.CorrelationFunction(bins=25, r_max=r_max)
            K = str(K_avg) + "_" + str(K_std)
            for seed in seed_range:

                if pos_ex == True:
                    posFileExact = get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, file_name="pos_exact")
                else:
                    inparFile, posFile = get_files(mode, nPart, phi, noise, K, Rp, xTy, seed)

                for t in timestep_range:
                    if pos_ex == True:
                        x, y, theta, view_time = get_pos_ex_snapshot(file=posFileExact)
                    else:
                        x, y, theta = get_pos_snapshot(posFile=posFile, nPart=nPart, timestep=t)

                    points = np.zeros((nPart, 3))
                    points[:,0] = x
                    points[:,1] = y
                    box = freud.Box.from_box([Lx, Ly])
                    points = box.wrap(points)

                    theta = np.array(theta)
                    values = np.array(np.exp(theta * 1j))

                    cf.compute(system=(box, points), values=values, query_points=points, query_values=values, reset=False)

            ax.plot(cf.bin_centers, cf.correlation, label=r"$K_{AVG}=$" + str(K_avg) + r"$; K_{STD}=$" + str(K_std))

    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$C(r)$")
    ax.hlines(y=0, xmin=0, xmax=r_max, color="grey", linestyle="dashed")
    # ax.set_ylim([0,1])
    ax.legend()

    folder = os.path.abspath('../plots/correlation/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))


def scatter_corr_vel_fluc(mode, nPart, phi, noise, K, Rp, xTy, seed, pos_ex=True, timestep=None):
    """
    Plot correlation function for the velocity fluctations perpendicular to the mean heading angle as scatterplot
    """
    if pos_ex == True:
        posFileExact = get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, file_name='pos_exact')
        x, y, theta, view_time = get_pos_ex_snapshot(file=posFileExact)
    else: 
        inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
        inpar_dict = get_params(inparFile)
        DT = inpar_dict["DT"]
        simulT = inpar_dict["simulT"]
        eqT = inpar_dict["eqT"]
        if timestep == None:
            timestep = int((simulT-eqT)/DT) 
        x, y, theta = get_pos_snapshot(posFile, nPart, timestep)

    L = np.sqrt(nPart / (phi*xTy))
    Ly = L
    Lx = L*xTy

    velocity = [np.array([np.cos(p), np.sin(p)]) for p in theta]
    av_vel = np.mean(velocity, axis=0)

    dv = [v - av_vel for v in velocity]

    # av_unit = av_vel / np.linalg.norm(av_vel)
    av_norm = np.array([-av_vel[1], av_vel[0]])

    # fluc_par = [np.dot(f, av_unit) * av_unit for f in fluc_vel]
    dv_perp = [np.dot(f, av_norm) * av_norm for f in dv]


    ## Plot!
    fig, ax = plt.subplots()

    for i in range(nPart):
        for j in range(i+1, nPart):
            ## Can add time average here later
            corr = np.dot(dv_perp[i],dv_perp[j])
    
            xij = x[i] - x[j]
            xij = xij - Lx*round(xij/Lx)
            yij = y[i] - y[j]
            yij = yij - Ly*round(yij/Ly)
            rij = np.sqrt(xij**2 + yij**2)
            # Discount if rij is about a certain distance??
            
            ax.plot(rij, corr, '+', color='tab:blue', alpha=0.2)
    
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$C_{\perp}(r)$")

    ## Plot on lin-lin scale
    folder = os.path.abspath('../plots/correlation/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_xTy' + str(xTy) + '_s' + str(seed) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

    ## Plot on log-log
    ax.set_xscale('log')
    ax.set_yscale('log')

    folder = os.path.abspath('../plots/correlation/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_xTy' + str(xTy) + '_s' + str(seed) + '_loglog.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

    ## Plot on log-lin scale
    ax.set_xscale('linear')
    ax.set_yscale('log')

    folder = os.path.abspath('../plots/correlation/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_xTy' + str(xTy) + '_s' + str(seed) + '_loglin.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))


def write_corr_vel(mode, nPart, phi, noise, K, Rp, xTy, seed_range, r_scale, timestep_range=[0], d_type='v', corr_r_min=0, corr_r_max=10, r_bin_num=120):
    """
    Write to file correlation function for the density fluctuations
    """
    rij_all = []
    corr_all = []
    corr_r_max_sq = corr_r_max**2

    L = np.sqrt(nPart / (phi*xTy))
    Ly = L
    Lx = L*xTy

    folder = os.path.abspath('../plot_data/correlation_velocity/')
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = d_type + '_' + mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed_range[-1]) + '_' + r_scale + '.txt'
    corrFile = open(os.path.join(folder, filename), "w")

    corrFile.write(str(corr_r_max) + "\n")
    corrFile.write(str(r_bin_num) + "\n")
    corrFile.write(str(timestep_range[0]) + "\t" + str(timestep_range[-1]) + "\n")

    for seed in seed_range:
        for timestep in timestep_range:
            sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
            if not os.path.exists(os.path.join(sim_dir, 'pos')):
                print(mode, nPart, phi, noise, K, Rp, xTy, seed)
            else:
                inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
                x, y, theta = get_pos_snapshot(posFile, nPart, timestep)
            

                velocity = [np.array([np.cos(p), np.sin(p)]) for p in theta]
                av_vel = np.mean(velocity, axis=0)

                dv = [v - av_vel for v in velocity]

                if d_type == 'v':
                    corr_dot = velocity
                elif d_type == 'dv':
                    corr_dot = dv
                elif d_type == 'dv_par':
                    av_unit = av_vel / np.linalg.norm(av_vel)
                    corr_dot = [np.dot(f, av_unit) * av_unit for f in dv]
                elif d_type == 'dv_perp':
                    av_norm = np.array([-av_vel[1], av_vel[0]])
                    corr_dot = [np.dot(f, av_norm) * av_norm for f in dv]
                else:
                    raise Exception("Type not valid. Must be 'v', 'dv', 'dv_par', or 'dv_perp'")

                # normalization
                c0 = 0
                for i in range(nPart):
                    c0 += np.dot(corr_dot[i], corr_dot[i])
                c0 = c0/nPart

                for i in range(nPart):
                    for j in range(i+1, nPart):
                        xij = x[i] - x[j]
                        xij = xij - Lx*round(xij/Lx)
                        if corr_r_min < xij < corr_r_max:
                            yij = y[i] - y[j]
                            yij = yij - Ly*round(yij/Ly)
                            rij_sq = xij**2 + yij**2
                            if rij_sq < corr_r_max_sq:
                                rij = np.sqrt(rij_sq)
                                rij_all.append(rij)
                                corr_all.append(np.dot(corr_dot[i],corr_dot[j])/c0)

    corr_all = np.array(corr_all)
    rij_all = np.array(rij_all)

    if r_scale == 'lin':
        bin_size = (corr_r_max-corr_r_min) / r_bin_num
        r_plot = np.linspace(corr_r_min, corr_r_max, num=r_bin_num, endpoint=False) + bin_size/2
    elif r_scale == 'log':
        if corr_r_min != 0:
            r_plot = np.logspace(np.log10(corr_r_min), np.log10(corr_r_max), num=r_bin_num, endpoint=True)
        else:
            r_plot = np.logspace(np.log10(np.min(rij_all)), np.log10(corr_r_max), num=r_bin_num, endpoint=True)
    else:
        raise Exception("Not a valid scale for r; should be 'lin' or 'log")

    for i in range(r_bin_num):
        lower = r_plot[i]
        try:
            upper = r_plot[i+1]
        except:
            upper = corr_r_max+1
        idx = np.where((rij_all>lower)&(rij_all<upper))[0]
        if len(idx) != 0:
            corr = np.mean(corr_all[idx])
            corrFile.write(str(r_plot[i]) + "\t" + str(corr) + "\n")

    corrFile.close()

def read_corr_vel(mode, nPart, phi, noise, K, Rp, xTy, seed_range, r_scale, d_type, bin_ratio=1):
    r_plot = []
    corr_bin_av = []

    folder = os.path.abspath('../plot_data/correlation_velocity/')
    filename = d_type + '_' + mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed_range[-1]) + '_' + r_scale + ".txt"
    corrFile = os.path.join(folder, filename)
    with open(corrFile) as f:
        line_count = 1
        for line in f:
            if line_count == 2:
                r_bin_num = float(line)
            if line_count > 3:
                r_plot.append(float(line.split('\t')[0]))
                corr_bin_av.append(float(line.split('\t')[1]))
            line_count += 1

    ## To reduce number of bins
    if bin_ratio>1:
        bin_ratio = int(bin_ratio)
        r_plot_new = []
        corr_new = []
        for i in np.arange(0, r_bin_num, bin_ratio):
            i = int(i)
            if i+bin_ratio+1>len(r_plot):
                r_plot_new.append(np.mean(r_plot[i:]))
                corr_new.append(np.mean(corr_bin_av[i:]))
            else:
                r_plot_new.append(np.mean(r_plot[i:i+bin_ratio]))
                corr_new.append(np.mean(corr_bin_av[i:i+bin_ratio]))

        r_plot = r_plot_new
        corr_bin_av = corr_new

    return r_plot, corr_bin_av

def plot_corr_vel_file(mode, nPart, phi, noise, K, Rp, xTy, seed_range, d_type, x_scale, y_scale, bin_ratio=1):
    fig, ax = plt.subplots()
    
    r_plot, corr_bin_av = read_corr_vel(mode, nPart, phi, noise, K, Rp, xTy, seed_range, x_scale, d_type, bin_ratio)
    ax.plot(r_plot, np.abs(corr_bin_av), '-', label="K=" + str(K))

    if x_scale == 'log':
        ax.set_xscale('log')
    if y_scale == 'log':
        ax.set_yscale('log')
    else:
        ax.set_ylim(bottom=0)

    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$C(r)$")

    # plt.show()

    folder = os.path.abspath('../plots/correlation_velocity/')
    filename = d_type + '_' + mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_' + y_scale + x_scale + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

def plot_corr_vel_file_superimpose(mode, nPart, phi, noise, K_avg_range, K_std_range, Rp, xTy, seed_range, d_type, x_scale, y_scale, bin_ratio=1):
    fig, ax = plt.subplots()
    
    for K_avg in K_avg_range:
        for K_std in K_std_range:
            K = str(K_avg) + "_" + str(K_std)
            r_plot, corr_bin_av = read_corr_vel(mode, nPart, phi, noise, K, Rp, xTy, seed_range, x_scale, d_type, bin_ratio)
            ax.plot(r_plot, np.abs(corr_bin_av), '-', label="K=" + str(K))

    if x_scale == 'log':
        ax.set_xscale('log')
    if y_scale == 'log':
        ax.set_yscale('log')
        ax.set_xlim(left=1)
        # ax.set_ylim(bottom=10**(-3))
    else:
        ax.set_ylim(bottom=0)
    ax.set_ylim(top=1)
    
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$C(r)$")
    ax.legend()
    ax.set_title(str(d_type) + r"; $N=$" + str(nPart) + r"; $\rho=$" + str(phi) + r"; $\eta=$" + str(noise) + r"; $R_I=$" + str(Rp))
    # plt.show()

    folder = os.path.abspath('../plots/correlation_velocity/')
    filename = d_type + '_' + mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_' + y_scale + x_scale + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

def plot_corr_vel_file_superimpose_N(mode, nPart_range, phi, noise, K_avg_range, K_std_range, Rp, xTy, seed_range, d_type, x_scale, y_scale, bin_ratio=1):
    fig, ax = plt.subplots()
    
    for nPart in nPart_range:
        for K_avg in K_avg_range:
            for K_std in K_std_range:
                K = str(K_avg) + "_" + str(K_std)
                r_plot, corr_bin_av = read_corr_vel(mode, nPart, phi, noise, K, Rp, xTy, seed_range, x_scale, d_type, bin_ratio)
                ax.plot(r_plot, np.abs(corr_bin_av), '-', label= "N=" + str(nPart) + ", K=" + str(K))

    if x_scale == 'log':
        ax.set_xscale('log')
    if y_scale == 'log':
        ax.set_yscale('log')
        ax.set_xlim(left=1)
        # ax.set_ylim(bottom=10**(-3))
    else:
        ax.set_ylim(bottom=0)
    ax.set_ylim(top=1)
    
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$C(r)$")
    ax.legend()
    ax.set_title(str(d_type) + r"; $N=$" + str(nPart) + r"; $\rho=$" + str(phi) + r"; $\eta=$" + str(noise) + r"; $R_I=$" + str(Rp))
    # plt.show()

    folder = os.path.abspath('../plots/correlation_velocity/')
    filename = d_type + '_' + mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_' + y_scale + x_scale + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))


def get_exponent_corr_vel(mode, nPart, phi, noise, K, Rp, xTy, seed_range, d_type, min_r=2, max_r=10):
    r_plot, corr_bin_av = read_corr_vel(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed_range=seed_range, r_scale="log", d_type=d_type, bin_ratio=1)
    r_plot = np.array(r_plot)
    corr_bin_av = np.array(corr_bin_av)
    idx1 = np.where(r_plot<max_r)[0]
    idx2 = np.where(r_plot>min_r)[0]
    idx = list(set(idx1) & set(idx2))
    # print(corr_bin_av[idx])

    exponent = np.polyfit(x=np.log10(r_plot[idx]), y=np.log10(np.abs(corr_bin_av[idx])), deg=1)[0]

    return exponent

def plot_exponents_Kavg_corr_vel(mode, nPart_range, phi, noise, K_avg_range, K_std_range, Rp, xTy, seed_range, d_type, min_r=2, max_r=10):
    fig, ax = plt.subplots()
    for nPart in nPart_range:
        for K_std in K_std_range:
            exponents = []
            for K_avg in K_avg_range:
                K = str(K_avg) + "_" + str(K_std)
                exponents.append(get_exponent_corr_vel(mode, nPart, phi, noise, K, Rp, xTy, seed_range, d_type, min_r, max_r))
            ax.plot(K_avg_range, exponents, '-o', label="N=" + str(nPart) + r"; $K_{STD}=$" + str(K_std))

    ax.set_title(str(d_type) + r"; $N=$" + str(nPart) + r"; $\rho=$" + str(phi) + r"; $\eta=$" + str(noise) + r"; $R_I=$" + str(Rp))
    ax.set_xlabel(r"$K_{AVG}$")
    ax.set_ylabel(r"$\lambda$")
    ax.legend()

    folder = os.path.abspath('../plots/correlation_velocity_exp/')
    filename = d_type + '_' + mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))


def plot_corr_vel(mode, nPart, phi, noise, K, Rp, xTy, seed_range, pos_ex=True, timestep=None, linlin=True, loglin=True, loglog=True, d_type='v', r_max=10, r_bin_num=20):
    """
    Plot correlation function for the velocity fluctations perpendicular to the mean heading angle with line from scatterplot

    Type can be: v (usual velocity correlation), dv (fluctuation from mean heading angle), dv_par (flucation parallel to mean heading angle),
    or dv_perp (fluctuation perpendicular to mean heading angle)
    """
    rij_all = []
    corr_all = []
    r_max_sq = r_max**2

    for seed in seed_range:
        if pos_ex == True:
            posFileExact = get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, file_name='pos_exact')
            x, y, theta, view_time = get_pos_ex_snapshot(file=posFileExact)
        else: 
            inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
            inpar_dict = get_params(inparFile)
            DT = inpar_dict["DT"]
            simulT = inpar_dict["simulT"]
            eqT = inpar_dict["eqT"]
            if timestep == None:
                timestep = int((simulT-eqT)/DT) 
            x, y, theta = get_pos_snapshot(posFile, nPart, timestep)

        L = np.sqrt(nPart / (phi*xTy))
        Ly = L
        Lx = L*xTy

        velocity = [np.array([np.cos(p), np.sin(p)]) for p in theta]
        av_vel = np.mean(velocity, axis=0)

        dv = [v - av_vel for v in velocity]

        if d_type == 'v':
            corr_dot = velocity
        elif d_type == 'dv':
            corr_dot = dv
        elif d_type == 'dv_par':
            av_unit = av_vel / np.linalg.norm(av_vel)
            corr_dot = [np.dot(f, av_unit) * av_unit for f in dv]
        elif d_type == 'dv_perp':
            av_norm = np.array([-av_vel[1], av_vel[0]])
            corr_dot = [np.dot(f, av_norm) * av_norm for f in dv]
        else:
            raise Exception("Type not valid. Must be 'v', 'dv', 'dv_par', or 'dv_perp'")

        # Normalization factor
        c0 = 0
        for i in range(nPart):
            c0 += np.dot(corr_dot[i], corr_dot[i])
        c0 = c0/nPart

        for i in range(nPart):
            for j in range(i+1, nPart):
                xij = x[i] - x[j]
                xij = xij - Lx*round(xij/Lx)
                if xij < r_max:
                    yij = y[i] - y[j]
                    yij = yij - Ly*round(yij/Ly)
                    rij_sq = xij**2 + yij**2
                    if rij_sq < r_max_sq:
                        rij = np.sqrt(rij_sq)
                        rij_all.append(rij)
                        corr_all.append(np.dot(corr_dot[i],corr_dot[j])/c0)
                
                # ax.plot(rij, corr, '+', color='tab:blue', alpha=0.2)

    corr_all = np.array(corr_all)
    rij_all = np.array(rij_all)
    bin_size = r_max / r_bin_num

    xscale_all = []
    yscale_all = []
    if linlin == True:
        xscale_all.append("lin")
        yscale_all.append("lin")
    if loglin == True:
        xscale_all.append("lin")
        yscale_all.append("log")
    if loglog == True:
        xscale_all.append("log")
        yscale_all.append("log")

    for xscale, yscale in zip(xscale_all, yscale_all):
        if xscale == 'lin':
            r_plot = np.linspace(0, r_max, num=r_bin_num, endpoint=False) + bin_size/2
        elif xscale == 'log':
            r_plot = np.logspace(np.log10(np.min(rij_all)), np.log10(r_max), num=r_bin_num, endpoint=True)
        else:
            raise Exception("xscale type not valid")
        
        corr_bin_av = []
        r_plot_new = []
        for i in range(r_bin_num):
            lower = r_plot[i]
            try:
                upper = r_plot[i+1]
            except:
                upper = r_max+1
            idx = np.where((rij_all>lower)&(rij_all<upper))[0]
            if len(idx) != 0:
                corr = np.mean(corr_all[idx])
                corr_bin_av.append(corr)
                r_plot_new.append(r_plot[i])

        fig, ax = plt.subplots()
        ax.plot(r_plot_new, np.abs(corr_bin_av), '-')

        if xscale == 'log':
            ax.set_xscale('log')
        if yscale == 'log':
            ax.set_yscale('log')
        else:
            ax.set_ylim(bottom=0)

        ax.set_xlabel(r"$r$")
        ax.set_ylabel(r"$C(r)$ for " + d_type)

        # plt.show()

        folder = os.path.abspath('../plots/correlation_velocity/')
        filename = str(d_type) + '_' + mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed) + '_' + yscale + xscale + '.png'
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(os.path.join(folder, filename))


### Density correlations ###

def write_corr_density_points(mode, nPart, phi, noise, K, Rp, xTy, seed_range, timestep_range=[0], rho_r_max=1, samples=None, corr_r_max=10, r_bin_num=120, r_scale='lin', corr_r_min=0):
    """
    Write to file correlation function for the density fluctuations
    """
    import freud
    rij_all = []
    corr_all = []
    corr_r_max_sq = corr_r_max**2

    L = np.sqrt(nPart / (phi*xTy))
    Ly = L
    Lx = L*xTy
    box = freud.Box.from_box([Lx, Ly])
    ld = freud.density.LocalDensity(r_max=rho_r_max, diameter=0)

    rng = np.random.default_rng(seed=1)
    if samples == None:
        samples = nPart
    rand_points = np.zeros((samples, 3))
    rand_points[:,0] = rng.uniform(-Lx/2,Lx/2,samples)
    rand_points[:,1] = rng.uniform(-Ly/2,Ly/2,samples)

    folder = os.path.abspath('../plot_data/correlation_density/')
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed_range[-1]) + '_' + r_scale + ".txt"
    corrFile = open(os.path.join(folder, filename), "w")

    corrFile.write(str(rho_r_max) + "\n")
    corrFile.write(str(corr_r_max) + "\n")
    corrFile.write(str(r_bin_num) + "\n")
    corrFile.write(str(timestep_range[0]) + "\t" + str(timestep_range[-1]) + "\n")

    for seed in seed_range:
        for timestep in timestep_range:
            sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
            if not os.path.exists(os.path.join(sim_dir, 'pos')):
                print(mode, nPart, phi, noise, K, Rp, xTy, seed)
            else:
                inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
                x, y, theta = get_pos_snapshot(posFile, nPart, timestep)
            

                points = np.zeros((nPart, 3))
                points[:,0] = x
                points[:,1] = y
                points = box.wrap(points)

                # get local densities
                rho_all = ld.compute(system=(box, points), query_points=rand_points).density

                rho_mean = np.mean(rho_all)
                d_fluc = [rho - rho_mean for rho in rho_all]

                corr_dot = d_fluc

                # normalization
                c0 = 0
                for i in range(samples):
                    c0 += corr_dot[i] * corr_dot[i]
                c0 = c0/samples

                for i in range(samples):
                    for j in range(i+1, samples):
                        xij = rand_points[i,0] - rand_points[j,0]
                        xij = xij - Lx*round(xij/Lx)
                        if xij < corr_r_max:
                            yij = rand_points[i,1] - rand_points[j,1]
                            yij = yij - Ly*round(yij/Ly)
                            rij_sq = xij**2 + yij**2
                            if rij_sq < corr_r_max_sq:
                                rij = np.sqrt(rij_sq)
                                rij_all.append(rij)
                                corr_all.append(corr_dot[i]*corr_dot[j]/c0)

    corr_all = np.array(corr_all)
    rij_all = np.array(rij_all)
    bin_size = corr_r_max / r_bin_num

    if r_scale == 'lin':
        bin_size = (corr_r_max-corr_r_min) / r_bin_num
        r_plot = np.linspace(corr_r_min, corr_r_max, num=r_bin_num, endpoint=False) + bin_size/2
    elif r_scale == 'log':
        if corr_r_min != 0:
            r_plot = np.logspace(np.log10(corr_r_min), np.log10(corr_r_max), num=r_bin_num, endpoint=True)
        else:
            r_plot = np.logspace(np.log10(np.min(rij_all)), np.log10(corr_r_max), num=r_bin_num, endpoint=True)
    else:
        raise Exception("Not a valid scale for r; should be 'lin' or 'log")

    for i in range(r_bin_num):
        lower = r_plot[i]
        try:
            upper = r_plot[i+1]
        except:
            upper = corr_r_max+1
        idx = np.where((rij_all>lower)&(rij_all<upper))[0]
        if len(idx) != 0:
            corr = np.mean(corr_all[idx])
            corrFile.write(str(r_plot[i]) + "\t" + str(corr) + "\n")

    corrFile.close()

def read_corr_density_points(mode, nPart, phi, noise, K, Rp, xTy, seed_range, r_scale, bin_ratio=1):
    r_plot = []
    corr_bin_av = []

    folder = os.path.abspath('../plot_data/correlation_density/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed_range[-1]) + '_' + r_scale + ".txt"
    corrFile = os.path.join(folder, filename)
    with open(corrFile) as f:
        line_count = 1
        for line in f:
            if line_count == 3:
                r_bin_num = float(line)
            if line_count > 4:
                r_plot.append(float(line.split('\t')[0]))
                corr_bin_av.append(float(line.split('\t')[1]))
            line_count += 1

    ## To reduce number of bins
    if bin_ratio>1:
        bin_ratio = int(bin_ratio)
        r_plot_new = []
        corr_new = []
        for i in np.arange(0, r_bin_num, bin_ratio):
            i = int(i)
            if i+bin_ratio+1>len(r_plot):
                r_plot_new.append(np.mean(r_plot[i:]))
                corr_new.append(np.mean(corr_bin_av[i:]))
            else:
                r_plot_new.append(np.mean(r_plot[i:i+bin_ratio]))
                corr_new.append(np.mean(corr_bin_av[i:i+bin_ratio]))

        r_plot = r_plot_new
        corr_bin_av = corr_new

    return r_plot, corr_bin_av


def get_distance_matrix(ngridx, ngridy, min_grid_size):
    """
    Output matrix is distance shift matrix in terms of x, y distance wrapped by number of grid points
    """
    x = pbc_wrap_calc(np.tile(np.arange(0,ngridy), (ngridx,1)),ngridy)*min_grid_size
    y = pbc_wrap_calc(np.tile(np.arange(0,ngridx), (ngridy,1)),ngridx).T*min_grid_size
    dist = np.sqrt(x**2+y**2)
    return dist

def get_r_corr(x,y,inparFile, min_grid_size=1):

    params = get_params(inparFile)
    nPart = params['nPart']
    phi = params['phi']
    xTy = params['xTy']

    L = np.sqrt(nPart / (phi*xTy))
    Ly = L
    Lx = L*xTy

    ngridx = int(Lx // min_grid_size)
    ngridy = int(Ly // min_grid_size)

    grid_size_x = Lx / ngridx
    grid_size_y = Ly / ngridy

    count_arr = np.zeros((ngridx, ngridy))
    for i in range(nPart):
        ix = int(pbc_wrap(x[i],Lx) // grid_size_x)
        iy = int(pbc_wrap(y[i],Ly) // grid_size_y)
        count_arr[ix, iy] += 1

    density_arr = count_arr / (grid_size_x * grid_size_y)
    density_fluc_arr = density_arr - np.mean(density_arr) # for fluctuations

    a=np.fft.fft2(density_fluc_arr)
    b=np.fft.fft2(density_fluc_arr[::-1,::-1])
    corr=np.round(np.real(np.fft.ifft2(a*b))[::-1,::-1],0)
    corr = np.abs(corr/corr[0,0]) # normalization
    dist = get_distance_matrix(ngridx, ngridy, min_grid_size)
    return dist.flatten(), corr.flatten()

def get_r_corr_all(mode, nPart, phi, noise, K, Rp, xTy, seed_range, pos_ex=False, timestep_range=[0], min_grid_size=1):
    r_all = []
    corr_all = []
    for seed in seed_range:
        for timestep in timestep_range:
            try:
                inparFile, posFile = get_files(mode, nPart, phi, noise, K, Rp, xTy, seed)
                if pos_ex:
                    posFileExact = get_file_path(mode, nPart, phi, noise, K, Rp, xTy, seed, file_name="pos_exact")
                    x, y, theta, view_time = get_pos_ex_snapshot(file=posFileExact)
                else:
                    x, y, theta = get_pos_snapshot(posFile, nPart, timestep)
                dist, corr = get_r_corr(x,y,inparFile, min_grid_size)
                r_all += list(dist)
                corr_all += list(corr)
            except:
                print(str(mode), str(nPart), str(phi), str(noise), str(K), str(Rp), str(xTy), str(seed))
                print("Error in seed " + str(seed) + " timestep " + str(timestep))
    return np.array(r_all), np.array(corr_all)

def write_corr_density_grid(mode, nPart, phi, noise, K, Rp, xTy, seed_range, pos_ex=True, timestep_range=[0], min_grid_size=1):
    folder = os.path.abspath('../plot_data/correlation_density_grid/')
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed_range[-1]) + '_g' + str(min_grid_size) + ".txt"
    
    corrFile = open(os.path.join(folder, filename), "w")

    corrFile.write(str(min_grid_size) + "\n")
    corrFile.write(str(timestep_range[0]) + "\t" + str(timestep_range[-1]) + "\n")

    r_all, corr_all = get_r_corr_all(mode, nPart, phi, noise, K, Rp, xTy, seed_range, pos_ex, timestep_range, min_grid_size)

    for i in range(len(r_all)):
        corrFile.write(str(r_all[i]) + "\t" + str(corr_all[i]) + "\n")
    corrFile.close()


def read_corr_density(mode, nPart, phi, noise, K, Rp, xTy, seed_range, min_grid_size=1):
    r_all = []
    corr_all = []

    folder = os.path.abspath('../plot_data/correlation_density_grid/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed_range[-1]) + '_g' + str(min_grid_size) + ".txt"
    corrFile = os.path.join(folder, filename)
    with open(corrFile) as f:
        line_count = 1
        for line in f:
            if line_count > 2:
                r_all.append(float(line.split('\t')[0]))
                corr_all.append(float(line.split('\t')[1]))
            line_count += 1

    return r_all, corr_all

def get_corr_binned_bins(dist, corr, bin_size=1, min_r=0, max_r=10):
    corr = np.array(corr)
    r_plot = np.linspace(min_r, max_r, num=int(max_r/bin_size))
    corr_plot = []
    r_plot_2 = []
    for i in range(len(r_plot)):
        lower = r_plot[i]
        try:
            upper = r_plot[i+1]
        except:
            upper = np.max(dist)
        idx = np.where((dist>=lower) & (dist<upper))[0].astype(int)
        if len(idx)>0:
            c = np.mean(corr[idx])
            corr_plot.append(c)
            r_plot_2.append(r_plot[i]+bin_size/2)
    return r_plot_2, corr_plot

def get_corr_binned(dist, corr, min_r=0, max_r=10):
    r_plot = np.unique(dist)
    r_plot2 = r_plot[np.where((r_plot>=min_r) & (r_plot<=max_r))[0]]
    corr = np.array(corr)
    corr_plot = []
    for i in range(len(r_plot2)):
        idx = np.where(dist == r_plot2[i])[0]
        if len(idx)>0:
            c = np.mean(corr[idx])
            corr_plot.append(c)
    return r_plot2, corr_plot

def plot_corr_density(mode, nPart, phi, noise, K, Rp, xTy, seed_range, pos_ex=True, timestep_range=[0], log_y=True, min_grid_size=1, min_r=0, max_r=10):
    dist, corr = get_r_corr_all(mode, nPart, phi, noise, K, Rp, xTy, seed_range, pos_ex, timestep_range, min_grid_size)
    r_plot, corr_plot = get_corr_binned(dist, corr, min_r=min_r, max_r=max_r)
    fig, ax = plt.subplots()
    ax.plot(r_plot, np.abs(corr_plot), '-')
    if log_y == True:
        ax.set_yscale('log')
    else:
        ax.set_ylim(bottom=0)

    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$C(r)$")

    folder = os.path.abspath('../plots/correlation_density_grid/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed_range[-1]) + '_g' + str(min_grid_size)
    if log_y == True:
        filename += "_log"
    else:
        filename += "_lin"
    filename += "lin"
    filename += '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))
    plt.close()

def plot_corr_density_file(mode, nPart, phi, noise, K, Rp, xTy, seed_range, log_y=True, min_grid_size=1, min_r=0, max_r=10):
    # r_plot, corr_bin_av = read_corr_density(mode, nPart, phi, noise, K, Rp, xTy, seed_range, r_scale, bin_ratio)
    dist, corr = read_corr_density(mode, nPart, phi, noise, K, Rp, xTy, seed_range, min_grid_size)

    r_plot, corr_plot = get_corr_binned(dist, corr, min_r=min_r, max_r=max_r)

    fig, ax = plt.subplots()
    ax.plot(r_plot, np.abs(corr_plot), '-')

    if log_y == True:
        ax.set_yscale('log')
    else:
        ax.set_ylim(bottom=0)

    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$C(r)$")

    folder = os.path.abspath('../plots/correlation_density_grid/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed_range[-1]) + '_g' + str(min_grid_size)
    if log_y == True:
        filename += "_log"
    else:
        filename += "_lin"
    filename += "lin"
    filename += '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))
    plt.close()

def plot_corr_density_superimpose(mode, nPart, phi, noise, K_avg_range, K_std_range, Rp, xTy, seed_range, 
                                  pos_ex=True, timestep_range=[0], log_x=False, log_y=True, min_grid_size=1, min_r=0, max_r=10):
    
    colors = plt.cm.GnBu(np.linspace(0.2, 1, len(K_avg_range)*len(K_std_range)))

    fig, ax = plt.subplots()

    i = 0
    for K_avg in K_avg_range:
        for K_std in K_std_range:
            K = str(K_avg) + "_" + str(K_std)
            dist, corr = get_r_corr_all(mode, nPart, phi, noise, K, Rp, xTy, seed_range, pos_ex, timestep_range, min_grid_size)
            r_plot, corr_plot = get_corr_binned(dist, corr, min_r=min_r, max_r=max_r)
            ax.plot(r_plot, np.abs(corr_plot), '-', label=r"$\overline{K}=$" + str(K_avg) + r"; $\sigma_K=$" + str(K_std), color=colors[i])
            i += 1
    if log_y == True:
        ax.set_yscale('log')
    if log_x == True:
        ax.set_xscale('log')


    ax.set_xlabel(r"$r$", fontsize=12)
    ax.set_ylabel(r"$C(r)$", fontsize=12)
    ax.set_title(r"Density fluctions correlation; $N=$" + str(nPart) + r"; $\rho=$" + str(phi) + r"; $\eta=$" + str(noise) + r"; $R_I=$" + str(Rp))
    ax.legend()

    folder = os.path.abspath('../plots/correlation_density_grid/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed_range[-1]) + '_g' + str(min_grid_size)
    if log_y == True:
        filename += "_log"
    else:
        filename += "_lin"
    if log_x == True:
        filename += "log"
    else:
        filename += "lin"
    filename += '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))
    plt.close()

def plot_corr_density_file_superimpose(mode, nPart, phi, noise, K_avg_range, K_std_range, Rp, xTy, seed_range, log_y=True, min_grid_size=1, min_r=0, max_r=10):
    
    colors = plt.cm.GnBu(np.linspace(0.2, 1, len(K_avg_range)*len(K_std_range)))

    fig, ax = plt.subplots()

    i = 0
    for K_avg in K_avg_range:
        for K_std in K_std_range:
            K = str(K_avg) + "_" + str(K_std)
            dist, corr = read_corr_density(mode, nPart, phi, noise, K, Rp, xTy, seed_range, min_grid_size)
            r_plot, corr_plot = get_corr_binned(dist, corr, min_r=min_r, max_r=max_r)
            ax.plot(r_plot, np.abs(corr_plot), '-', label=r"$\overline{K}=$" + str(K_avg) + r"; $\sigma_K=$" + str(K_std), color=colors[i])
            i += 1
    if log_y == True:
        ax.set_yscale('log')
    else:
        ax.set_ylim(bottom=0)

    ax.set_xlabel(r"$r$", fontsize=12)
    ax.set_ylabel(r"$C(r)$", fontsize=12)
    ax.set_title(r"Density fluctions correlation; $N=$" + str(nPart) + r"; $\rho=$" + str(phi) + r"; $\eta=$" + str(noise) + r"; $R_I=$" + str(Rp))
    ax.legend()
    # plt.show()

    folder = os.path.abspath('../plots/correlation_density_grid/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed_range[-1]) + '_g' + str(min_grid_size)
    if log_y == True:
        filename += "_log"
    else:
        filename += "_lin"
    filename += "lin"
    filename += '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))
    plt.close()


def get_exponent_corr_density(mode, nPart, phi, noise, K, Rp, xTy, seed_range, min_r, max_r):
    r_plot, corr_bin_av = read_corr_density(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed_range=seed_range, min_grid_size=1)
    r_plot = np.array(r_plot)
    corr_bin_av = np.array(corr_bin_av)
    idx1 = np.where(r_plot<max_r)[0]
    idx2 = np.where(r_plot>min_r)[0]
    idx = list(set(idx1) & set(idx2))
    # print(corr_bin_av[idx])

    exponent = np.polyfit(x=np.log10(r_plot[idx]), y=np.log10(np.abs(corr_bin_av[idx])), deg=1)[0]

    return exponent

def plot_exponents_Kavg_corr_density(mode, nPart_range, phi, noise, K_avg_range, K_std_range, Rp, xTy, seed_range, min_r, max_r):
    fig, ax = plt.subplots()
    for nPart in nPart_range:
        for K_std in K_std_range:
            exponents = []
            for K_avg in K_avg_range:
                K = str(K_avg) + "_" + str(K_std)
                exponents.append(get_exponent_corr_density(mode, nPart, phi, noise, K, Rp, xTy, seed_range, min_r, max_r))
            ax.plot(K_avg_range, exponents, '-o', label="N=" + str(nPart) + r"; $K_{STD}=$" + str(K_std))

    ax.set_title(r"Density correlation exponents; $N=$" + str(nPart) + r"; $\rho=$" + str(phi) + r"; $\eta=$" + str(noise) + r"; $R_I=$" + str(Rp))
    ax.set_xlabel(r"$K_{AVG}$")
    ax.set_ylabel(r"$\lambda$")
    ax.legend()

    folder = os.path.abspath('../plots/correlation_density_exp/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

###############################
## Average neighbour numbers ##
###############################

def neighbour_counts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, pos_ex=True, timestep_range=[1]):
    import freud
    L = np.sqrt(nPart / (phi*xTy))
    Ly = L
    Lx = L*xTy

    inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
    inpar_dict = get_params(inparFile)
    
    if pos_ex == True:
        posFileExact = get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, file_name="pos_exact")
        x, y, theta, view_time = get_pos_ex_snapshot(file=posFileExact)

    n_nei = np.array([])
    for t in timestep_range:
        if pos_ex == False:
            x, y, theta = get_pos_snapshot(posFile=posFile, nPart=nPart, timestep=t)
        points = np.zeros((nPart, 3))
        points[:,0] = x
        points[:,1] = y
        box = freud.Box.from_box([Lx, Ly])
        points = box.wrap(points)
        ld = freud.density.LocalDensity(r_max=r_max, diameter=0)
        n_nei = np.append(n_nei, ld.compute(system=(box, points)).num_neighbors)
    return n_nei

def neighbour_stats(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, pos_ex=True, timestep_range=[1]):
    
    av_nei_i = neighbour_counts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, pos_ex, timestep_range)

    nei_av = np.mean(av_nei_i)
    nei_median = np.median(av_nei_i)
    nei_std = np.std(av_nei_i)
    nei_max = np.max(av_nei_i)

    return nei_av, nei_median, nei_std, nei_max

def neighbour_hist(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, pos_ex=True, timestep_range=[1], print_stats=True, n_max=None, c_max=None):
    
    av_nei_i = neighbour_counts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, pos_ex, timestep_range)

    if print_stats == True:
        print(np.mean(av_nei_i), np.median(av_nei_i), np.std(av_nei_i), np.max(av_nei_i))

    fig, ax = plt.subplots(figsize=(7,5))

    # ax.hist(av_nei_i, bins=np.arange(0, np.max(av_nei_i)+1))
    unique, counts = np.unique(av_nei_i, return_counts=True)
    ax.bar(unique, counts)
    ax.set_xlabel(r"$\langle N_i\rangle$", fontsize=16)
    ax.set_ylabel("Count", fontsize=16)
    # ax.set_title(r"$N=$" + str(nPart) + r"; $\rho=$" + str(phi) + r"; $\eta=$" + str(noise) + r"; $K=$" + str(K) + r"; $r_{max}=$" + str(r_max))

    if n_max != None:
        ax.set_xlim([0,n_max])
    if c_max != None:
        ax.set_ylim([0,c_max])

    folder = os.path.abspath('../plots/neighbour_hist/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))


#########################
## Steady state checks ##
#########################

def local_density_distribution(mode, nPart, phi, noise, K, Rp, xTy, seed, timestep_range, min_grid_size=2):
    """
    Plot local density distribution for various snapshots over time
    """

    inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
    inpar_dict = get_params(inparFile)
    DT = inpar_dict["DT"]
    eqT = inpar_dict["eqT"]

    L = np.sqrt(nPart / (phi*xTy))
    Ly = L
    Lx = L*xTy

    fig, ax = plt.subplots()

    for timestep in timestep_range:
        x, y, theta = get_pos_snapshot(posFile, nPart, timestep)

        x = pbc_wrap(x,Lx)
        y = pbc_wrap(y,Ly)

        ngrid_x = int(Lx // min_grid_size)
        grid_size_x = Lx / ngrid_x
        ngrid_y = int(Ly // min_grid_size)
        grid_size_y = Ly / ngrid_y

        grid_area = grid_size_x*grid_size_y

        grid_counts = np.zeros((ngrid_x, ngrid_y))

        for i in range(nPart):
            gridx = int(x[i]//grid_size_x)
            gridy = int(y[i]//grid_size_y)
            grid_counts[gridx,gridy] += 1
        n_density = grid_counts / grid_area

        view_time = eqT + timestep*DT
        n,x = np.histogram(n_density, bins=100)
        bin_centers = 0.5*(x[1:]+x[:-1])
        ax.plot(bin_centers, n, label="t=" + str(int(view_time)))


    ax.set_xlabel("Number density")
    ax.set_ylabel("Probability density")
    ax.legend()

    folder = os.path.abspath('../plots/density_distribution/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed) + '_box.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

def local_density_distribution_freud(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, timestep_range, time_av=[0], random_sample=False, samples=None, density_cap=10, bins=100):
    """
    Plot local density distribution for various snapshots over time
    """
    import freud
    inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
    inpar_dict = get_params(inparFile)
    DT = inpar_dict["DT"]
    eqT = inpar_dict["eqT"]

    L = np.sqrt(nPart / (phi*xTy))
    Ly = L
    Lx = L*xTy
    box = freud.Box.from_box([Lx, Ly])
    ld = freud.density.LocalDensity(r_max=r_max, diameter=0)

    if random_sample:
        rng = np.random.default_rng(seed=1)
        if samples == None:
            samples = nPart

        rand_points = np.zeros((samples, 3))
        rand_points[:,0] = rng.uniform(-Lx/2,Lx/2,samples)
        rand_points[:,1] = rng.uniform(-Ly/2,Ly/2,samples)

    fig, ax = plt.subplots()

    for timestep in timestep_range:
        n_density = []
        for time_shift in time_av:
            
            x, y, theta = get_pos_snapshot(posFile, nPart, timestep+time_shift)

            points = np.zeros((nPart, 3))
            points[:,0] = x
            points[:,1] = y
            points = box.wrap(points)

            if random_sample == True:
                query_points = rand_points
            else:
                query_points = None

            n_density_t = ld.compute(system=(box, points), query_points=query_points).density
            n_density.append(n_density_t)

        # unique, counts = np.unique(n_density, return_counts=True)
        # counts = counts/len(time_av)

        # index_cap = bisect.bisect_left(unique, density_cap)
        # unique = unique[:index_cap]
        # counts = counts[:index_cap]

        view_time = eqT + timestep*DT

        # ax.plot(unique, counts, label="t=" + str(int(view_time)))

        # ax.hist(n_density, bins=bins)

        n,x = np.histogram(n_density, bins=bins)
        bin_centers = 0.5*(x[1:]+x[:-1])
        ax.plot(bin_centers, n, label="t=" + str(int(view_time)))
        # ax.hist(x, n, label="t=" + str(int(view_time)))

    ax.set_title(r"Number densities for $r_{max}=$" + str(r_max))
    ax.set_xlabel("Number density")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_xlim(right=density_cap)

    folder = os.path.abspath('../plots/density_distribution/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

def local_density_distribution_diff_freud(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, timestep_range, time_av=[0], density_cap=10):
    """
    Plot local density distribution for various snapshots over time by taking their differences
    """
    import freud
    inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
    inpar_dict = get_params(inparFile)
    DT = inpar_dict["DT"]
    eqT = inpar_dict["eqT"]

    L = np.sqrt(nPart / (phi*xTy))
    Ly = L
    Lx = L*xTy
    box = freud.Box.from_box([Lx, Ly])
    ld = freud.density.LocalDensity(r_max=r_max, diameter=0)

    fig, ax = plt.subplots()

    for i, timestep in enumerate(timestep_range):
        n_density = []
        for time_shift in time_av:
            
            x, y, theta = get_pos_snapshot(posFile, nPart, timestep+time_shift)

            points = np.zeros((nPart, 3))
            points[:,0] = x
            points[:,1] = y
            points = box.wrap(points)

            n_density_t = ld.compute(system=(box, points)).density
            n_density.append(n_density_t)


        unique, counts = np.unique(n_density, return_counts=True)
        counts = counts/len(time_av)

        index_cap = bisect.bisect_left(unique, density_cap)
        unique = unique[:index_cap]
        counts = counts[:index_cap]

        view_time = eqT + timestep*DT
        if i == 0:
            counts_old = counts
            view_time_old = view_time
        else:
            ax.plot(unique, counts-counts_old, label="t=" + str(int(view_time_old)) + " vs t=" + str(int(view_time)))
            counts_old = counts
            view_time_old = view_time

    ax.set_title(r"Number densities for $r_{max}=$" + str(r_max))
    ax.set_xlabel("Number density")
    ax.set_ylabel("Difference")
    ax.legend()

    folder = os.path.abspath('../plots/density_distribution/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed) + '_diff.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

def local_density_distribution_voronoi(mode, nPart, phi, noise, K, Rp, xTy, seed, timestep_range, time_av=[0], bins=50, density_cap=10):
    """
    Plot local density distribution for various snapshots over time using Voronoi method
    """
    import freud
    inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
    inpar_dict = get_params(inparFile)
    DT = inpar_dict["DT"]
    eqT = inpar_dict["eqT"]

    L = np.sqrt(nPart / (phi*xTy))
    Ly = L
    Lx = L*xTy
    box = freud.Box.from_box([Lx, Ly])
    voro = freud.locality.Voronoi()


    fig, ax = plt.subplots()

    for timestep in timestep_range:
        n_density = []
        for time_shift in time_av:
            
            x, y, theta = get_pos_snapshot(posFile, nPart, timestep+time_shift)

            points = np.zeros((nPart, 3))
            points[:,0] = x
            points[:,1] = y
            points = box.wrap(points)

            voro.compute((box,points))
            n_density_t = 1/voro.volumes
            n_density.append(n_density_t)

        view_time = eqT + timestep*DT
        # ax.hist(n_density, bins=50, label="t=" + str(view_time))
        n,x = np.histogram(n_density, bins=bins, range=(0,density_cap))
        bin_centers = 0.5*(x[1:]+x[:-1])
        ax.plot(bin_centers, n, label="t=" + str(int(view_time)))
        
    ax.set_title(r"Number densities from Voronoi")
    ax.set_xlabel("Number density")
    ax.set_ylabel("Density")
    ax.legend()

    folder = os.path.abspath('../plots/density_distribution_voronoi/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

#####################
## Binder Cumulant ##
#####################

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

                    ax.plot([float(k) for k in K_avg_range], binder, '-o', label=r"$N=$" + str(nPart) + r"; $K_{STD}=$" + str(K_std) + r"; $\eta=$" + str(noise) + r"; $R_p=$" + str(Rp))
                    if save_data == True:
                        save_file.write(str(nPart) + "\t" + str(Rp) + "\t" + str(phi) + "\t" + str(K_avg) + "\t" + str(K_std) + "\n")
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

#######################
## Coupling Analysis ##
#######################
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

#############################
## Neighbour contact times ##
#############################

def get_nlist(posFile, nPart, box, timestep, r_max):
    import freud
    x, y, theta = get_pos_snapshot(posFile=posFile, nPart=nPart, timestep=timestep)

    points = np.zeros((nPart, 3))
    points[:,0] = x
    points[:,1] = y
    points = box.wrap(points)

    aq = freud.locality.AABBQuery(box, points)

    query_points = points

    query_result = aq.query(query_points, dict(r_max=r_max, exclude_ii=True))
    nlist = query_result.toNeighborList()
    return nlist

def write_contacts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, tape_time):
    import freud
    # Initialize stuff
    inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
    sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
    L = np.sqrt(nPart / (phi*xTy))
    Ly = L
    Lx = L*xTy

    box = freud.Box.from_box([Lx, Ly])

    contactsFile = open(os.path.join(sim_dir, "contacts_r" + str(r_max)), "w")
    contactsFile.write(str(r_max) + '\t' + str(tape_time) + '\n')

    nlist = get_nlist(posFile=posFile, nPart=nPart, box=box, timestep=0, r_max=r_max)

    in_contact_t0 = []
    for (i,j) in nlist:
        if j>i:
            index = int((nPart*(nPart-1)/2) - (nPart-i)*((nPart-i)-1)/2 + j - i - 1)
            in_contact_t0.append(index)

    in_contact_t_old = in_contact_t0
    start_contact = np.zeros(int(nPart*(nPart-1)/2))
    # contact_duration = []

    # Start going through timesteps
    for t in range(1,tape_time+1):
        in_contact_t = []
        nlist = get_nlist(posFile=posFile, nPart=nPart, box=box, timestep=t, r_max=r_max)
        for (i,j) in nlist:
            if j>i:
                index = int((nPart*(nPart-1)/2) - (nPart-i)*((nPart-i)-1)/2 + j - i - 1)
                in_contact_t.append(index)
                if index not in in_contact_t_old:
                    start_contact[index] = t

        # check if contact has ended
        for index in in_contact_t_old:
            if index not in in_contact_t:
                if start_contact[index] != 0:
                    # contact_duration.append(t-start_contact[index])
                    i = int(nPart - 2 - np.floor(np.sqrt(-8*index + 4*nPart*(nPart-1)-7)/2.0 - 0.5))
                    j = int(index + i + 1 - nPart*(nPart-1)/2 + (nPart-i)*((nPart-i)-1)/2)
                    contactsFile.write(str(i) + '\t' + str(j) + '\t' + str(t-start_contact[index]) + '\n')

        # update for next time step
        in_contact_t_old = in_contact_t

    # Full simulation tape contacts
    for index in in_contact_t:
        if index in in_contact_t0:
            if start_contact[index] == 0:
                # contact_duration.append(tape_time)
                i = int(nPart - 2 - np.floor(np.sqrt(-8*index + 4*nPart*(nPart-1)-7)/2.0 - 0.5))
                j = int(index + i + 1 - nPart*(nPart-1)/2 + (nPart-i)*((nPart-i)-1)/2)
                contactsFile.write(str(i) + '\t' + str(j) + '\t' + str(tape_time) + '\n')
    
    contactsFile.close()


def read_contacts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max):
    sim_dir = get_sim_dir(mode, nPart, phi, noise, K, Rp, xTy, seed)
    if not os.path.exists(os.path.join(sim_dir, 'contacts_r' + str(r_max))):
        raise Exception("Contacts file does not exist")
    contactsFile = os.path.join(sim_dir, "contacts_r" + str(r_max))
    i = []
    j = []
    duration = []
    with open(contactsFile) as f:
        line_count = 1
        for line in f:
            if line_count == 1:
                r_max = float(line.split('\t')[0])
                tape_time = int(line.split('\t')[1])
            else:
                i.append(float(line.split('\t')[0]))
                j.append(float(line.split('\t')[1]))
                duration.append(float(line.split('\t')[2]))
            line_count += 1
    return i, j, duration, r_max, tape_time

## Add seed range?
def plot_contacts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, log=False):
    i, j, contact_duration, r_max, tape_time = read_contacts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max)

    fig, ax = plt.subplots()

    ax.hist(contact_duration, bins=np.arange(1,tape_time+1), density=True)
    if log == True:
        ax.set_yscale("log")
    # ax.set_title(r"$r_{max}=$" + str(r_max) + r"; $T=$" + str(tape_time))
    ax.set_title(r"$N=$" + str(nPart) + r"; $\rho=$" + str(phi) + r"; $\eta=$" + str(noise) + r"; $K=$" + str(K) + r"; $r_{max}=$" + str(r_max))
    ax.set_xlabel("Pairwise contact time")
    ax.set_ylabel("Density")
    folder = os.path.abspath('../plots/contact_duration/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_rmax' + str(r_max)
    if log == True:
        filename += "_log"
    filename += '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

def plot_K_vs_contact_time(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, log_x=False, log_y=False, n_max=None):
    ix, jx, contact_duration, r_max, tape_time = read_contacts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max)

    Kij = get_couplings(mode, nPart, phi, noise, K, Rp, xTy, seed)

    K_t = []

    for t in np.unique(contact_duration):
        contact_t = [i for i,v in enumerate(contact_duration) if v==t]
        ixs = [int(ix[k]) for k in contact_t]
        jxs = [int(jx[k]) for k in contact_t]

        K_contact = []
        for i,j in zip(ixs,jxs):
            index = int((nPart*(nPart-1)/2) - (nPart-i)*((nPart-i)-1)/2 + j - i - 1)
            K_contact.append(Kij[index])


        K_t.append(np.mean(K_contact))

    fig, ax = plt.subplots()

    ax.plot(np.unique(contact_duration)[:100], K_t[:100], '-o')
    ax.plot(np.unique(contact_duration)[:100], np.abs(np.unique(contact_duration)[:100])**(1/2), '--', label=r"$y=x^{1/2}$")
    ax.legend()
    ax.set_title(r"$N=$" + str(nPart) + r"; $\rho=$" + str(phi) + r"; $\eta=$" + str(noise) + r"; $K=$" + str(K) + r"; $r_{max}=$" + str(r_max))
    # ax.set_title(r"$r_{max}=$" + str(r_max))
    ax.set_xlabel("Neighbour duration")
    ax.set_ylabel(r"Mean $K_{ij}$")
    if log_x == True:
        ax.set_xscale("log")
    if log_y == True:
        ax.set_yscale("log")

    if n_max != None:
        ax.set_xlim(right=n_max)

    folder = os.path.abspath('../plots/K_vs_contact_time/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_rmax' + str(r_max)
    if log_x == True:
        filename += "_log"
    if log_y == True:
        filename += "log"
    filename += '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))



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


### Centre of Mass calculations ###

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