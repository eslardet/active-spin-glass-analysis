import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import csv
import os


def get_sim_dir(mode, nPart, phi, K, seed):
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
    
    sim_dir = os.path.abspath('../simulation_data/' + mode_name + '/N' + str(nPart) + '/phi' + str(phi) + '/K' + str(K) + '/s' + str(seed))

    return sim_dir

def get_files(mode, nPart, phi, K, seed):
    sim_dir = get_sim_dir(mode, nPart, phi, K, seed)
    inparFile = os.path.join(sim_dir, "inpar")
    posFile = os.path.join(sim_dir, "pos")
    return inparFile, posFile

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
    inpar_dict["mode"] = r[9][0]
    
    if inpar_dict["mode"] == 'C':
        inpar_dict["DT"] = float(r[12][0])
        inpar_dict["simulT"] = float(r[14][0])
    elif inpar_dict["mode"] == 'T':
        inpar_dict["DT"] = float(r[14][0])
        inpar_dict["simulT"] = float(r[16][0])
    else:
        inpar_dict["DT"] = float(r[13][0])
        inpar_dict["simulT"] = float(r[15][0])
    return inpar_dict

def pbc_wrap(x, L):
    """
    Wrap points into periodic box with length L
    """
    return x - L*np.round(x/L)

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

    x_all = []
    y_all = []
    theta_all = []

    for i in range(int(max_T/DT)+1):
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

def snapshot(mode, nPart, phi, K, seed, view_time):
    """
    Get static snapshot at specified time from the positions file
    """

    inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, K=K, seed=seed)
    inpar_dict = get_params(inparFile)
    
    nPart = inpar_dict["nPart"]
    phi = inpar_dict["phi"]
    mode = inpar_dict["mode"]
    DT = inpar_dict["DT"]
    
    beta = 2**(1/6)
    L = np.sqrt(nPart*np.pi*beta**2 / (4*phi))
    
    with open(posFile) as f:
        reader = csv.reader(f, delimiter="\t")
        r = list(reader)[6:]
    
    i = int(view_time/DT)
    view_time = i*DT
    x = pbc_wrap(np.array(r[(nPart+1)*i+1:(nPart+1)*i+1+nPart]).astype('float')[:,0], L)
    y = pbc_wrap(np.array(r[(nPart+1)*i+1:(nPart+1)*i+1+nPart]).astype('float')[:,1], L)
    theta = np.array(r[(nPart+1)*i+1:(nPart+1)*i+1+nPart]).astype('float')[:,2]
    
    fig, ax = plt.subplots(figsize=(5,5), dpi=72)
    diameter = (ax.get_window_extent().width * 72/fig.dpi) /L *beta
    
    if mode == "T":
        nA = nPart//2
        ax.plot(x[:nA], y[:nA], 'o', ms=diameter, zorder=1)
        ax.plot(x[nA:], y[nA:], 'o', ms=diameter, zorder=1)
    else:
        ax.plot(x, y, 'o', ms=diameter, zorder=1)
    ax.quiver(x, y, np.cos(theta), np.sin(theta), zorder=2)
    ax.set_xlim(-L/2,L/2)
    ax.set_ylim(-L/2,L/2)
    ax.set_aspect('equal')
    ax.set_title("t=" + str(view_time))
    plt.show()

def animate(mode, nPart, phi, K, seed, min_T=None, max_T=None):
    """
    Make animation from positions file
    """
    inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, K=K, seed=seed)
    
    x_all, y_all, theta_all = get_pos_arr(inparFile, posFile, min_T, max_T)
    
    inpar_dict = get_params(inparFile)
    
    nPart = inpar_dict["nPart"]
    phi = inpar_dict["phi"]
    mode = inpar_dict["mode"]
    DT = inpar_dict["DT"]
    seed = inpar_dict["seed"]

    plt.rcParams["animation.html"] = "jshtml"
    plt.ioff()
    plt.rcParams['animation.embed_limit'] = 2**128

    fig, ax = plt.subplots(figsize=(3,3))
    
    beta = 2**(1/6)
    L = np.sqrt(nPart*np.pi*beta**2 / (4*phi))
    
    diameter = (ax.get_window_extent().width * 72/fig.dpi) /L * beta

    points_A, = plt.plot([], [], 'o', ms=diameter, zorder=1)
    points_B, = plt.plot([], [], 'o', ms=diameter, zorder=2)

    x = pbc_wrap(x_all[0], L)
    y = pbc_wrap(y_all[0], L)
    theta = theta_all[0]
    arrows = plt.quiver(x, y, np.cos(theta), np.sin(theta), zorder=3)

    def init():
        ax.set_xlim(-L/2, L/2)
        ax.set_ylim(-L/2, L/2)
        if mode == "T":
            return arrows, points_A, points_B,
        else:
            return arrows, points_A

    def update(n):
        x = pbc_wrap(x_all[n], L)
        y = pbc_wrap(y_all[n], L)
        theta = theta_all[n]
        if mode == "T":
            nA = nPart//2
            points_A.set_data(x[:nA], y[:nA])
            points_B.set_data(x[nA:], y[nA:])
        else: 
            points_A.set_data(x, y)
        arrows.set_offsets(np.c_[x, y])
        arrows.set_UVC(np.cos(theta), np.sin(theta))
        ax.set_title("t = " + str(round(n*DT, 1)), fontsize=10, loc='left')
        
        if mode == "T":
            return arrows, points_A, points_B
        else: 
            return arrows, points_A

    ani = FuncAnimation(fig, update, init_func=init, frames=len(x_all), interval=10, blit=True)
    ani.save(os.path.abspath('../animations/' + mode + '_N' + str(nPart) + '_phi' + str(phi) + '_K' + str(K) + '_s' + str(seed) + '.mp4'))


def plot_vorder_time(mode, nPart, phi, K, seed, min_T=None, max_T=None):
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
    
    theta_all = get_theta_arr(inparFile=inparFile, posFile=posFile, min_T=min_T, max_T=max_T)
    v_order = []
    for theta_t in theta_all:
        cos_sum = 0
        sin_sum = 0
        for i in theta_t:
            cos_sum += np.cos(i)
            sin_sum += np.sin(i)
        v_order.append(np.sqrt(cos_sum**2+sin_sum**2)/nPart)
    fig, ax = plt.subplots()
    t_plot = np.arange(0, max_T+DT/4, DT)
    ax.plot(t_plot, v_order)
    ax.set_xlabel("time")
    ax.set_ylabel(r"Vicsek order parameter, $\Psi$")
    
    plt.savefig(os.path.abspath('../plots/v_order_vs_time/' + mode + '_N' + str(nPart) + '_phi' + str(phi) + '_K' + str(K) + '_s' + str(seed) + '.png'))


def v_order_ss(mode, nPart, phi, K, seed, avg_over):
    """
    Find steady state Vicsek order parameter for simulation, averaged over a certain number of final timesteps
    """
    inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, K=K, seed=seed)
    inpar_dict = get_params(inparFile)
    DT = inpar_dict["DT"]
    simulT = inpar_dict["simulT"]
    max_T = simulT
    min_T = simulT - avg_over*DT
    
    theta_all = get_theta_arr(inparFile, posFile, min_T, max_T)
    v_ss = 0
    for theta_t in theta_all:
        cos_sum = 0
        sin_sum = 0
        for i in theta_t:
            cos_sum += np.cos(i)
            sin_sum += np.sin(i)
        v_ss += np.sqrt(cos_sum**2+sin_sum**2)/nPart
    v_ss = v_ss/len(theta_all)
    return v_ss

def v_order_sd(mode, nPart, phi, K, seed, avg_over):
    """
    Find standard deviation of Vicsek order parameter over the last avg_over timesteps saved to file
    """
    inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, K=K, seed=seed)
    
    inpar_dict = get_params(inparFile)
    DT = inpar_dict["DT"]
    simulT = inpar_dict["simulT"]
    max_T = simulT
    min_T = simulT - avg_over*DT

    theta_all = get_theta_arr(inparFile, posFile, min_T, max_T)
    v_order = []
    for theta_t in theta_all:
        cos_sum = 0
        sin_sum = 0
        for i in theta_t:
            cos_sum += np.cos(i)
            sin_sum += np.sin(i)
        v_order.append(np.sqrt(cos_sum**2+sin_sum**2)/nPart)
    v_sd = np.std(v_order)
    return v_sd

def write_stats(mode, nPart, phi, K, seed, avg_over):
    """
    Write a file with various statistics from the simulation data (Vicsek order parameter mean and the standard deviation)
    """
    v_mean = v_order_ss(mode, nPart, phi, K, seed, avg_over)
    v_sd = v_order_sd(mode, nPart, phi, K, seed, avg_over)
    v_sus = nPart*v_sd**2
    
    sim_dir = get_sim_dir(mode, nPart, phi, K, seed)
    statsFile = open(os.path.join(sim_dir, "stats"), "w")
    statsFile.write(str(v_mean) + '\n')
    statsFile.write(str(v_sd) + '\n')
    statsFile.write(str(v_sus))
    statsFile.close()

def read_stats(mode, nPart, phi, K, seed):
    """
    Read stats file and create dictionary with those statistics
    """
    sim_dir = get_sim_dir(mode, nPart, phi, K, seed)

    with open(os.path.join(sim_dir, "stats")) as file:
        reader = csv.reader(file, delimiter="\n")
        r = list(reader)
    stats_dict = {}
    stats_dict["v_mean"] = float(r[0][0])
    stats_dict["v_sd"] = float(r[1][0])
    stats_dict["v_sus"] = float(r[2][0])
    return stats_dict

def plot_vorder_ksd(mode, nPart, phi, KAVG, KSTD_range, seed_range, log=False):
    """
    Plot steady state Vicsek order parameter against K_std for a fixed K_avg (Gaussian distributed couplings)
    Averaged over a number of realizations
    """
    v_ss = []
    for KSTD in KSTD_range:
        v_ss_sum = 0
        for seed in seed_range:
            sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed)
            if not os.path.exists(os.path.join(sim_dir, 'stats')):
                write_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed, avg_over=1000)
            v_ss_sum += read_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG) + "_" + str(KSTD), seed=seed)["v_mean"]
        v_ss.append(v_ss_sum/len(seed_range))
    fig, ax = plt.subplots()
    ax.plot(KSTD_range, v_ss, 'o-')
    ax.set_xlabel("KSTD")
    ax.set_ylabel(r"Vicsek order parameter, $\Psi$")
    if log == True:
        ax.set_xscale("log")
    
    folder = os.path.abspath('../plots/v_order_vs_K/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_KAVG' + str(KAVG) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

def plot_vorder_ksd_superimpose(mode, nPart_range, phi, KAVG, KSTD_range, seed_range, log=False):
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
            for seed in seed_range:
                sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed)
                if not os.path.exists(os.path.join(sim_dir, 'stats')):
                    write_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed, avg_over=1000)
                v_ss_sum += read_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG) + "_" + str(KSTD), seed=seed)["v_mean"]
            v_ss.append(v_ss_sum/len(seed_range))
        ax.plot(KSTD_range, v_ss, 'o-', label="N=" + str(nPart))
    ax.set_xlabel("KSTD")
    ax.set_ylabel(r"Vicsek order parameter, $\Psi$")
    ax.legend()
    if log == True:
        ax.set_xscale("log")
    
    folder = os.path.abspath('../plots/v_order_vs_K/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_KAVG' + str(KAVG) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

def plot_vorder_kavg(mode, nPart, phi, KAVG_range, KSTD, seed_range):
    """
    Plot steady state Vicsek order parameter against K_avg for a fixed K_std (Gaussian distributed couplings)
    Averaged over a number of realizations
    """
    v_ss = []
    for KAVG in KAVG_range:
        v_ss_sum = 0
        for seed in seed_range:
            sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed)
            if not os.path.exists(os.path.join(sim_dir, 'stats')):
                write_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed, avg_over=1000)
            v_ss_sum += read_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG) + "_" + str(KSTD), seed=seed)["v_mean"]
        v_ss.append(v_ss_sum/len(seed_range))
    fig, ax = plt.subplots()
    ax.plot(KAVG_range, v_ss, 'o-')
    ax.set_xlabel("KAVG")
    ax.set_ylabel(r"Vicsek order parameter, $\Psi$")

    folder = os.path.abspath('../plots/v_order_vs_K/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_KSTD' + str(KSTD) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))


def plot_vorder_kavg_superimpose(mode, nPart_range, phi, KAVG_range, KSTD, seed_range):
    """
    Plot steady state Vicsek order parameter against K_avg for a fixed K_std (Gaussian distributed couplings)
    Averaged over a number of realizations
    Superimposed plots for various N
    """
    fig, ax = plt.subplots()
    for nPart in nPart_range:
        v_ss = []
        for KAVG in KAVG_range:
            v_ss_sum = 0
            for seed in seed_range:
                sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed)
                if not os.path.exists(os.path.join(sim_dir, 'stats')):
                    write_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed, avg_over=1000)
                v_ss_sum += read_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG) + "_" + str(KSTD), seed=seed)["v_mean"]
            v_ss.append(v_ss_sum/len(seed_range))
        ax.plot(KAVG_range, v_ss, 'o-', label="N=" + str(nPart))
    ax.set_xlabel("KAVG")
    ax.set_ylabel(r"Vicsek order parameter, $\Psi$")
    ax.legend()

    folder = os.path.abspath('../plots/v_order_vs_K/')
    filename = mode + '_phi' + str(phi) + '_KSTD' + str(KSTD) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))


def plot_vorder_sus_ksd(mode, nPart, phi, KAVG, KSTD_range, seed_range):
    """
    Plot susceptibility of Vicsek order parameter against K_std for a given K_avg
    Averaged over a number of realizations
    """
    v_sus = []
    for KSTD in KSTD_range:
        v_sus_sum = 0
        for seed in seed_range:
            sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed)
            if not os.path.exists(os.path.join(sim_dir, 'stats')):
                write_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed, avg_over=1000)
            v_sus_sum += read_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG) + "_" + str(KSTD), seed=seed)["v_sus"]
        v_sus.append(v_sus_sum/len(seed_range))
    fig, ax = plt.subplots()
    ax.plot(KSTD_range, v_sus, 'o-')
    ax.set_xlabel("KSTD")
    ax.set_ylabel(r"Vicsek order parameter susceptibility")
    ax.set_xscale("log")


    folder = os.path.abspath('../plots/v_order_sus_vs_K/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_KAVG' + str(KAVG) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))
        
def plot_vorder_sus_ksd_superimpose(mode, nPart_range, phi, KAVG, KSTD_range, seed_range):
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
            for seed in seed_range:
                sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed)
                if not os.path.exists(os.path.join(sim_dir, 'stats')):
                    write_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed, avg_over=1000)
                v_sus_sum += read_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG) + "_" + str(KSTD), seed=seed)["v_sus"]
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


def plot_vorder_sus_kavg(mode, nPart, phi, KAVG_range, KSTD, seed_range):
    """
    Plot the susceptibility of the Vicsek order parameter (over the final 1000 saved timesteps) against K_avg, with a fixed K_std
    Averaged over a number of realizations
    """
    v_sus = []
    for KAVG in KAVG_range:
        v_sus_sum = 0
        for seed in seed_range:
            sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed)
            if not os.path.exists(os.path.join(sim_dir, 'stats')):
                write_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed, avg_over=1000)
            v_sus_sum += read_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG) + "_" + str(KSTD), seed=seed)["v_sus"]
        v_sus.append(v_sus_sum/len(seed_range))
    fig, ax = plt.subplots()
    ax.plot(KAVG_range, v_sus, 'o-')
    ax.set_xlabel("KAVG")
    ax.set_ylabel(r"Vicsek order parameter susceptibility")
    
    folder = os.path.abspath('../plots/v_order_sus_vs_K/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_KSTD' + str(KSTD) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

def plot_vorder_sus_kavg_superimpose(mode, nPart_range, phi, KAVG_range, KSTD, seed_range):
    """
    Plot the susceptibility of the Vicsek order parameter (over the final 1000 saved timesteps) against K_avg, with a fixed K_std
    Averaged over a number of realizations
    Superimposed plots for various N
    """
    fig, ax = plt.subplots()
    for nPart in nPart_range:
        v_sus = []
        for KAVG in KAVG_range:
            v_sus_sum = 0
            for seed in seed_range:
                sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed)
                if not os.path.exists(os.path.join(sim_dir, 'stats')):
                    write_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG)+'_'+str(KSTD), seed=seed, avg_over=1000)
                v_sus_sum += read_stats(mode=mode, nPart=nPart, phi=phi, K=str(KAVG) + "_" + str(KSTD), seed=seed)["v_sus"]
            v_sus.append(v_sus_sum/len(seed_range))
        
        ax.plot(KAVG_range, v_sus, 'o-', label="N=" + str(nPart))
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