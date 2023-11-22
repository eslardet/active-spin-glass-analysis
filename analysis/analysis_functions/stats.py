
import sys
sys.path.insert(1, '././analysis_functions')
from import_files import *

import numpy as np
import matplotlib.pyplot as plt

## General tatistics ##
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

## Polar order vs time ##
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
    ax.set_xlim(left=startT)
    ax.set_xlabel("time")
    ax.set_ylabel(r"Polar order parameter, $\Psi$")

    folder = os.path.abspath('../plots/p_order_vs_time/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

## Density ##
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

## Neighbours ##
def neighbour_counts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, pos_ex=True, timestep_range=[1]):
    """
    Count number of neighbours within radius r_max for each particle
    """
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
    """
    Statistics of the number of neighbours within radius r_max for each particle
    """
    av_nei_i = neighbour_counts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, pos_ex, timestep_range)

    nei_av = np.mean(av_nei_i)
    nei_median = np.median(av_nei_i)
    nei_std = np.std(av_nei_i)
    nei_max = np.max(av_nei_i)

    return nei_av, nei_median, nei_std, nei_max

## Critical value ##
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
