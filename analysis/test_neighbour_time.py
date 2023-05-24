import numpy as np
import analysis_functions_vicsek_new as fun
import matplotlib.pyplot as plt
import time
import scipy.stats as sps
import os
import freud
import analysis_functions_vicsek_new as fun
from matplotlib import cm, colors

mode = "G"
nPart = 1000
phi = 1.0
# noise_range = [format(i, '.3f') for i in np.arange(0.80,0.811,0.01)]
noise = "0.20"
# noise_range = np.arange(0.80,0.01,0.82)
Rp = 1.0
K = "0.0_8.0"
xTy = 1.0
seed = 1
rho_r_max = 1
r_max = 2
samples = 100
tape_time = 5



# fun.animate(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
# fun.plot_porder_time(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)

# posFileExact = fun.get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, file_name='pos_exact')
# x, y, theta, view_time = fun.get_pos_ex_snapshot(file=posFileExact)

def get_nlist(posFile, nPart, box, timestep, r_max):
    x, y, theta = fun.get_pos_snapshot(posFile=posFile, nPart=nPart, timestep=timestep)

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

    # Initialize stuff
    inparFile, posFile = fun.get_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
    sim_dir = fun.get_sim_dir(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
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
                    i = nPart - 2 - np.floor(np.sqrt(-8*index + 4*nPart*(nPart-1)-7)/2.0 - 0.5)
                    j = index + i + 1 - nPart*(nPart-1)/2 + (nPart-i)*((nPart-i)-1)/2
                    contactsFile.write(str(i) + '\t' + str(j) + '\t' + str(t-start_contact[index]) + '\n')

        # update for next time step
        in_contact_t_old = in_contact_t

    # Full simulation tape contacts
    for index in in_contact_t:
        if index in in_contact_t0:
            if start_contact[index] == 0:
                # contact_duration.append(tape_time)
                i = nPart - 2 - np.floor(np.sqrt(-8*index + 4*nPart*(nPart-1)-7)/2.0 - 0.5)
                j = index + i + 1 - nPart*(nPart-1)/2 + (nPart-i)*((nPart-i)-1)/2
                contactsFile.write(str(i) + '\t' + str(j) + '\t' + str(tape_time) + '\n')
    
    contactsFile.close()


def read_contacts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max):
    sim_dir = fun.get_sim_dir(mode, nPart, phi, noise, K, Rp, xTy, seed)
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
def plot_contacts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max):
    i, j, contact_duration, r_max, tape_time = read_contacts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max)

    fig, ax = plt.subplots()
    ax.hist(contact_duration, bins=np.arange(1,tape_time+1), density=True)
    ax.set_title(r"$r_{max}=$" + str(r_max) + r"; $T=$" + str(tape_time))
    ax.set_xlabel("Pairwise contact time")
    ax.set_ylabel("Density")
    folder = os.path.abspath('../plots/contact_duration/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_rmax' + str(r_max) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

r_max = 1
tape_time = 100
# write_contacts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, tape_time)
# i, j, contact_duration, r_max, tape_time = read_contacts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max)

# # unique, counts = np.unique(contact_duration, return_counts=True)
# # print(np.unique(contact_duration, return_counts=True))

# fig, ax = plt.subplots()
# ax.hist(contact_duration, bins=np.arange(1,tape_time+1), density=True)
# ax.set_title(r"$r_{max}=$" + str(r_max) + r"; $T=$" + str(tape_time))
# ax.set_xlabel("Pairwise contact time")
# ax.set_ylabel("Density")
# # plt.hist(contact_duration, bins=np.logspace(np.log10(1.0),np.log10(100.0), 50))
# # plt.gca().set_xscale("log")
# plt.show()



plot_contacts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max)