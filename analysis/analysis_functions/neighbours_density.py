import sys
sys.path.insert(1, '././analysis_functions')
from import_files import *
from stats import *
from coupling import *

import numpy as np
import matplotlib.pyplot as plt
import bisect


def neighbour_hist(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, pos_ex=True, timestep_range=[1], print_stats=True, n_max=None, c_max=None):
    """
    Plot histogram of number of neighbours inside radius of r_max for each particle
    """
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


def get_nlist(posFile, nPart, box, timestep, r_max):
    """
    Get freud neighbour list for given snapshot and radius
    """
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
    """
    Write to file 'contact' time between particles defined as within r_max of each other
    Using freud neighbour list
    """
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
    """
    Read contacts file

    Returns:
    i, j: particle indices of contacts
    duration: duration time of contact
    r_max: contact radius
    tape_time: time interval of interest in simulation
    """
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
    """
    Plot histogram of contact duration
    """
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
    """
    Plot coupling value Kij vs contact duration of i,j
    """
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