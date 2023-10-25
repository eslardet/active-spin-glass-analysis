import sys
sys.path.insert(1, '././analysis_functions')
from import_files import *
from stats import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm, colors

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
    
    fig, ax = plt.subplots(figsize=(10*xTy,10))

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

    ani = FuncAnimation(fig, update, init_func=init, frames=len(theta_all), interval=50, blit=True)

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