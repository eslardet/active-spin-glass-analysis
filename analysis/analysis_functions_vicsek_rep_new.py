import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm, colors
from decimal import Decimal
import freud

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
        mode_name = "Antiferromagnetic"
    elif mode == "F":
        mode_name = "Ferromagnetic"

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
    inpar_dict["vp"] = float(r[5][0])
    try:
        inpar_dict["Rp"] = float(r[5][0])
    except:
        inpar_dict["Rp"] = str(r[5][0])
    inpar_dict["xTy"] = float(r[8][0])
    inpar_dict["mode"] = r[10][0]
    inpar_dict["repulsion"] = r[-1][0]
    
    if inpar_dict["mode"] == 'C':
        inpar_dict["DT"] = float(r[13][0])
        inpar_dict["eqT"] = float(r[15][0])
        inpar_dict["simulT"] = float(r[16][0])
    elif inpar_dict["mode"] == 'T':
        inpar_dict["DT"] = float(r[15][0])
        inpar_dict["eqT"] = float(r[17][0])
        inpar_dict["simulT"] = float(r[18][0])
    else:
        inpar_dict["DT"] = float(r[14][0])
        inpar_dict["eqT"] = float(r[16][0])
        inpar_dict["simulT"] = float(r[17][0])
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


def snapshot(mode, nPart, phi, noise, K, Rp, xTy, seed, view_time=None, pos_ex=False, show_color=True, save_in_folder=False):
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

    beta = 2**(1/6)

    L = np.sqrt(nPart*np.pi*beta**2 / (4*phi*xTy))
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
    
    fig, ax = plt.subplots(figsize=(10*xTy,10), dpi=72)

    diameter = (ax.get_window_extent().height * 72/fig.dpi) /L * beta
    
    if show_color == True:
        for i in range(nPart):
            color = mapper.to_rgba(theta[i]%(2*np.pi))
            ax.plot(x[i], y[i], 'o', ms=diameter, color=color, zorder=1)
        plt.colorbar(mappable=mapper, ax=ax)
    else:
        ax.plot(x, y, 'o', ms=diameter)
        ax.quiver(x, y, u, v)
    ax.set_xlim(0,Lx)
    ax.set_ylim(0,Ly)
    ax.set_aspect('equal')
    ax.set_title("t=" + str(round(view_time)))

    if save_in_folder == True:
        folder = get_sim_dir(mode, nPart, phi, noise, K, Rp, xTy, seed)
        filename = 'snapshot.png'
    else:
        folder = os.path.abspath('../snapshots')
        filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) +  '_xTy' + str(xTy) + '_s' + str(seed) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))


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
    repulsion = inpar_dict["repulsion"]

    if min_T == None:
        min_T = 0

    with open(posFile) as f:
        reader = csv.reader(f, delimiter="\t")
        startT = float(list(reader)[6][0])

    plt.rcParams["animation.html"] = "jshtml"
    plt.ioff()
    plt.rcParams['animation.embed_limit'] = 2**128
    
    fig, ax = plt.subplots(figsize=(5*xTy,5))

    if repulsion == 'W':
        beta = 2**(1/6)
    elif repulsion == 'H':
        beta = 1
    elif repulsion == 'C':
        beta = 1
    else:
        beta = 2**(1/6)
    L = np.sqrt(nPart*np.pi*beta**2 / (4*phi*xTy))
    Ly = L
    Lx = L*xTy
    print(repulsion == "H")
    print(beta, repulsion, Lx,Ly)
    diameter = (ax.get_window_extent().height * 72/fig.dpi) /L * beta

    # norm = colors.Normalize(vmin=0.0, vmax=2*np.pi, clip=True)
    # plt.set_cmap('hsv')

    # mapper = cm.ScalarMappable(norm=norm, cmap=cm.hsv)
    # plt.colorbar(mappable=mapper, ax=ax)

    x = pbc_wrap(x_all[0],Lx)
    y = pbc_wrap(y_all[0],Ly)
    theta = theta_all[0]
    # cols = np.mod(theta, 2*np.pi)
    arrows = ax.quiver(x, y, np.cos(theta), np.sin(theta), zorder=2)
    points, = plt.plot([], [], 'o', ms=diameter, zorder=1)

    def init():
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)
        return arrows, points

    def update(n):
        x = pbc_wrap(x_all[n],Lx)
        y = pbc_wrap(y_all[n],Ly)
        theta = theta_all[n]
        # cols = np.mod(theta, 2*np.pi)
        points.set_data(x, y)
        arrows.set_offsets(np.c_[x, y])
        arrows.set_UVC(np.cos(theta), np.sin(theta))
        ax.set_title("t = " + str(round(n*DT+startT+min_T, 1)), fontsize=10, loc='left')
        
        return arrows, points

    ani = FuncAnimation(fig, update, init_func=init, frames=len(theta_all), interval=10, blit=True)

    folder = os.path.abspath('../animations')
    filename = repulsion + '_' + mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) +  '_xTy' + str(xTy) + '_s' + str(seed) + '.mp4'
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

    # with open(posFile) as f:
    #     reader = csv.reader(f, delimiter="\t")
    #     startT = float(list(reader)[6][0])
    startT = 0

    if min_T == None:
        min_T = 0
    if max_T == None:
        max_T = simulT-startT
    
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
    t_plot = np.arange(0+startT, max_T+startT+DT/4, DT)
    ax.plot(t_plot, p_order)
    ax.set_xlabel("time")
    ax.set_ylabel(r"Polar order parameter, $\Psi$")
    ax.set_ylim([0,1])

    folder = os.path.abspath('../plots/p_order_vs_time/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) +  '_xTy' + str(xTy) + '_s' + str(seed) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))


def del_pos(mode, nPart, phi, noise, K, Rp, xTy, seed):
    """
    Delete position file to save space
    """
    sim_dir = get_sim_dir(mode, nPart, phi, noise, K, Rp, xTy, seed)
    os.remove(os.path.join(sim_dir, "pos"))

def write_stats(mode, nPart, phi, noise, K, Rp, xTy, seed, min_T=None, max_T=None, remove_pos=False, density_var=False):
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


    sim_dir = get_sim_dir(mode, nPart, phi, noise, K, Rp, xTy, seed)
    statsFile = open(os.path.join(sim_dir, "stats"), "w")
    statsFile.write(str(p_mean) + '\n')
    statsFile.write(str(p_sus) + '\n')
    statsFile.write(str(n_mean) + '\n')
    statsFile.write(str(n_sus) + '\n')

    if density_var == True:
        # d_var_list = []
        # timestep_min = int(min_T//DT)
        # timestep_max = int(max_T//DT)
        # for timestep in range(timestep_min, timestep_max, 10):
        #     d_var_list.append(local_density_var(mode, nPart, phi, noise, K, Rp, xTy, seed, pos_ex=False, timestep=timestep))
        # d_var_mean = d_var_list/(len(d_var_list))
        # statsFile.write(str(d_var_mean) + '\n')

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
        stats_dict["d_var"] = float(r[4][0])
    except Exception:
        pass
        # print("No d_var for rho=" + str(phi) + ", noise=" + str(noise) + ", K=" + str(K) + ", s=" + str(seed))
    return stats_dict


def plot_porder_noise(mode, nPart, phi, noise_range, K, Rp, xTy, seed_range):
    """
    Plot steady state polar order parameter against noise
    Averaged over a number of realizations
    """
    fig, ax = plt.subplots()
    p_ss = []
    for noise in noise_range:
        p_ss_sum = 0
        for seed in seed_range:
            sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
            if not os.path.exists(os.path.join(sim_dir, 'stats')):
                write_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
            p_ss_sum += read_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)["p_mean"]
        p_ss.append(p_ss_sum/len(seed_range))

    noise_range = [float(i) for i in noise_range]
    ax.plot(noise_range, p_ss, '-o')
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"Polar order parameter, $\Psi$")
    ax.set_ylim([0,1])
    
    folder = os.path.abspath('../plots/p_order_vs_noise/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_K' + str(K) + '_Rp' + str(Rp) +  '_xTy' + str(xTy) + '.png'
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
    filename = mode + '_N' + str(nPart) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) +  '_xTy' + str(xTy) + '.png'
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
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_Kstd0.0' + '_xTy' + str(xTy) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

def plot_porder_Kavg(mode, nPart, phi, noise_range, K_avg_range, K_std_range, Rp_range, xTy, seed_range, save_data=False):
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
    for Rp in Rp_range:
        for noise in noise_range:
            for K_std in K_std_range:
                p_ss = []
                for K_avg in K_avg_range:
                    K = str(K_avg) + "_" + str(K_std)
                    p_ss_sum = 0
                    error_count = 0
                    for seed in seed_range:
                        # sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
                        # if not os.path.exists(os.path.join(sim_dir, 'stats')):
                            # print(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed)
                            # error_count += 1
                            # write_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
                        try:
                            p_mean = read_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)["p_mean"]
                            if p_mean > 1.0:
                                print("p_mean is bigger than 1!!")
                                print(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed)
                                error_count += 1
                            elif p_mean < 0.0:
                                print("p_mean is smaller than 0!!")
                                print(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed)
                                error_count += 1                            
                            else:
                                p_ss_sum += p_mean
                        except:
                            print("No stats file to read")
                            print(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed)
                            error_count += 1
                    p_ss_av = p_ss_sum/(len(seed_range)-error_count)
                    if p_ss_av > 1.0:
                        print("Average is greater than 1!")
                        print(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed)
                        print(p_ss_sum, len(seed_range), error_count)
                    p_ss.append(p_ss_sum/(len(seed_range)-error_count))

                ax.plot(K_avg_range, p_ss, '-o', label=r"$K_{STD}=$" + str(K_std) + r"; $\eta=$" + str(noise) + r"; $\phi=$" + str(phi) + r"; $R_p=$" + str(Rp))
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
            error_count = 0
            for K_avg in K_avg_range:
                K = str(K_avg) + "_" + str(K_std)
                p_ss_sum = 0
                for seed in seed_range:
                    sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
                    if not os.path.exists(os.path.join(sim_dir, 'stats')):
                        print(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed)
                        error_count += 1
                        # write_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
                    p_ss_sum += read_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)["p_mean"]
                p_ss.append(p_ss_sum/(len(seed_range)-error_count))

            ax.plot(K_avg_range, p_ss, '-o', label=r"$K_{STD}=$" + str(K_std) + r"; $\eta=$" + str(noise))
    ax.set_xlabel(r"$K_{AVG}$")
    ax.set_ylabel(r"Polar order parameter, $\Psi$")
    ax.set_ylim([0,1])
    ax.legend()

    return ax

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
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_xTy' + str(xTy) + '.png'
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

        ax.plot([(i-0.7)/K_std for i in K_avg_range], p_ss, '-o', label=r"$K_{STD}=$" + str(K_std))
    ax.set_xlabel(r"$(K_{AVG}-0.7)/K_{STD}$")
    ax.set_ylabel(r"Polar order parameter, $\Psi$")
    ax.set_ylim([0,1])
    ax.legend()

    folder = os.path.abspath('../plots/p_order_vs_Kratio/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_xTy' + str(xTy) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

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
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) +  '_xTy' + str(xTy) + '_s' + str(seed) + '.png'
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
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) +  '_xTy' + str(xTy) + '_s' + str(seed) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

## TO DO: Make this be a time average
## Would need to perform before deleting pos file
def plot_average_band_profile(mode, nPart, phi, noise, K, Rp, xTy, seed_range, timestep_range, min_grid_size=2, cutoff=1.5, peak_cutoff=2):
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

    extra_left = int(len(x_vals) / 10)
    extra_right = int(len(x_vals) / 10)
    total_len = extra_left + extra_right

    x_plot = x_vals[:total_len]
    d_plot_av = np.zeros(total_len)

    no_band = 0

    for seed in seed_range:
        # posFileExact = get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, file_name='pos_exact')
        # x, y, theta, view_time = get_pos_ex_snapshot(file=posFileExact)

        inparFile, posFile = get_files(mode, nPart, phi, noise, K, Rp, xTy, seed)

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
            d_plot_av += d_plot/band_number
    d_plot_av = d_plot_av/(len(seed_range)*len(timestep_range)-no_band)

    ax.plot(x_plot, d_plot_av)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"Average local density")
    ax.set_title(r"Average band: $\rho=$" + str(phi) + r"$, \eta=$" + str(noise) + r"$, K=$" + str(K))

    folder = os.path.abspath('../plots/density_profile_shifted/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) +  '_xTy' + str(xTy) + '_av.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

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
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_K' + str(K) + '_Rp' + str(Rp) +  '_xTy' + str(xTy) + '.png'
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
        error_count = 0
        for K_avg in K_avg_range:
            K = str(K_avg) + "_" + str(K_std)
            var_sum = 0
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
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_xTy' + str(xTy) + '.png'
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
                print("No stats!")
                print(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed)
                count_err += 1
            else:
            # try:
                p_mean = read_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=str(K_avg) + "_" + str(K_std), Rp=Rp, xTy=xTy, seed=seed)["p_mean"]
                if np.isnan(p_mean):
                    count_err += 1
                else:
                    p_ss_sum += p_mean
            # except:
            #     print("No stats!")
            #     print(nPart, K_avg, K_std, seed)
            #     count_err += 1
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

    if np.isnan(KAVG_crit):
        print(p_ss)
        print(grad, intercept)
    
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
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_xTy' + str(xTy) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))


###############################
## Average neighbour numbers ##
###############################
def neighbour_counts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, pos_ex=True, timestep_range=[1]):
    
    L = np.sqrt(nPart / (phi*xTy))
    Ly = L
    Lx = L*xTy

    inparFile, posFile = get_files(mode, nPart, phi, noise, K, Rp, xTy, seed)
    inpar_dict = get_params(inparFile)

    
    if pos_ex == True:
        posFileExact = get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, file_name="pos_exact")
        x, y, theta, view_time = get_pos_ex_snapshot(file=posFileExact)


    for t in timestep_range:
        if pos_ex == False:
            x, y, theta = get_pos_snapshot(posFile=posFile, nPart=nPart, timestep=t)
        points = np.zeros((nPart, 3))
        points[:,0] = x
        points[:,1] = y
        box = freud.Box.from_box([Lx, Ly])
        points = box.wrap(points)
        ld = freud.density.LocalDensity(r_max=r_max, diameter=0)
        n_nei = ld.compute(system=(box, points)).num_neighbors
    av_nei_i = n_nei / (len(timestep_range))

    return av_nei_i

def neighbour_stats(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, pos_ex=True, timestep_range=[1]):
    
    av_nei_i = neighbour_counts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, pos_ex, timestep_range)

    nei_av = np.mean(av_nei_i)
    nei_std = np.std(av_nei_i)
    nei_max = np.max(av_nei_i)

    return nei_av, nei_std, nei_max

def neighbour_hist(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, pos_ex=True, timestep_range=[1], print_stats=True):
    
    av_nei_i = neighbour_counts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, pos_ex, timestep_range)

    if print_stats == True:
        print(np.mean(av_nei_i), np.median(av_nei_i), np.std(av_nei_i), np.max(av_nei_i))

    fig, ax = plt.subplots()

    # ax.hist(av_nei_i, bins=np.arange(0, np.max(av_nei_i)+1))
    unique, counts = np.unique(av_nei_i, return_counts=True)
    ax.bar(unique, counts)
    ax.set_xlabel(r"$\langle N_i\rangle$")
    ax.set_ylabel("Count")
    ax.set_title(r"$N=$" + str(nPart) + r"; $\phi=$" + str(phi) + r"; $\eta=$" + str(noise) + r"; $K=$" + str(K) + r"; $r_{max}=$" + str(r_max))

    folder = os.path.abspath('../plots/neighbour_hist/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) +  '_xTy' + str(xTy) + '_s' + str(seed) + '_rmax' + str(r_max) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))



#######################
## Coupling Analysis ##
#######################
def get_coupling_rij(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max=None, pos_ex=True, timestep_range=[0]):
    if pos_ex == True:
        posFileExact = get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, file_name='pos_exact')
        x, y = get_pos_ex_snapshot(file=posFileExact)[:2]
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

def plot_dist_coupling_hist(mode, nPart, phi, noise, K, Rp, xTy, seed, pos_ex=True, timestep_range=[0], bin_size=100, bin_ratio=1, r_max=None, K_max=None):

    K_list, rij_list = get_coupling_rij(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, r_max=r_max, pos_ex=pos_ex, timestep_range=timestep_range)
    
    fig, ax = plt.subplots(figsize=(10,10/bin_ratio)) 
    # plt.tight_layout()
    
    if K_max != None:
        ax.hist2d(K_list, rij_list, bins=(bin_size, int(bin_size/bin_ratio)), range=[[-K_max,K_max], [0,r_max]], cmap=cm.jet)
    else:
        ax.hist2d(K_list, rij_list, bins=(bin_size, int(bin_size/bin_ratio)), cmap=cm.jet)
        
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$K_{ij}$")
    ax.set_ylabel(r"$r_{ij}$")

    folder = os.path.abspath('../plots/dist_coupling_rep/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed) + '_hist.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

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
    folder = os.path.abspath('../plots/dist_coupling_rep/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed) + '_histdiff.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))