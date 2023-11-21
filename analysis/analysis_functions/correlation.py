import sys
sys.path.insert(1, '././analysis_functions')
from import_files import *
from stats import *

import numpy as np
import matplotlib.pyplot as plt


## Velocity correlations ##
def plot_correlation(mode, nPart, phi, noise, K_avg_range, K_std_range, Rp, xTy, seed_range, timestep_range, pos_ex=False):
    """
    Plot equal time 2-point correlation function, averaged over time and seeds using Freud
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

            ax.plot(cf.bin_centers, cf.correlation, label=r"$K_{AVG}=$" + str(K_avg) + r"$; \sigma_K=$" + str(K_std))

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
    """
    Read velocity fluctuation correlation function from saved data file and return lists of r and C(r)
    """
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
    """
    Plot velocity fluctuation correlation function from saved data file
    """
    fig, ax = plt.subplots()
    
    r_plot, corr_bin_av = read_corr_vel(mode, nPart, phi, noise, K, Rp, xTy, seed_range, x_scale, d_type, bin_ratio)
    ax.plot(r_plot, corr_bin_av, '-', label="K=" + str(K))

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
    """
    Plot velocity fluctuation correlation function from saved data file for different Kavg and Kstd
    """
    fig, ax = plt.subplots()
    
    for K_avg in K_avg_range:
        for K_std in K_std_range:
            K = str(K_avg) + "_" + str(K_std)
            r_plot, corr_bin_av = read_corr_vel(mode, nPart, phi, noise, K, Rp, xTy, seed_range, x_scale, d_type, bin_ratio)
            ax.plot(r_plot, corr_bin_av, '-', label="K=" + str(K))

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
    """
    Plot velocity fluctuation correlation function from saved data file for different N
    """
    fig, ax = plt.subplots()
    
    for nPart in nPart_range:
        for K_avg in K_avg_range:
            for K_std in K_std_range:
                K = str(K_avg) + "_" + str(K_std)
                r_plot, corr_bin_av = read_corr_vel(mode, nPart, phi, noise, K, Rp, xTy, seed_range, x_scale, d_type, bin_ratio)
                ax.plot(r_plot, corr_bin_av, '-', label= "N=" + str(nPart) + ", K=" + str(K))

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
    """
    Get correlation decay exponent for velocity fluctuations from file
    """
    r_plot, corr_bin_av = read_corr_vel(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed_range=seed_range, r_scale="log", d_type=d_type, bin_ratio=1)
    r_plot = np.array(r_plot)
    corr_bin_av = np.array(corr_bin_av)
    idx1 = np.where(r_plot<max_r)[0]
    idx2 = np.where(r_plot>min_r)[0]
    idx = list(set(idx1) & set(idx2))
    # print(corr_bin_av[idx])

    exponent = np.polyfit(x=np.log10(r_plot[idx]), y=np.log10(corr_bin_av[idx]), deg=1)[0]

    return exponent

def plot_exponents_Kavg_corr_vel(mode, nPart_range, phi, noise, K_avg_range, K_std_range, Rp, xTy, seed_range, d_type, min_r=2, max_r=10):
    """
    Plot correlation decay exponent for velocity fluctuations against Kavg
    """
    fig, ax = plt.subplots()
    for nPart in nPart_range:
        for K_std in K_std_range:
            exponents = []
            for K_avg in K_avg_range:
                K = str(K_avg) + "_" + str(K_std)
                exponents.append(get_exponent_corr_vel(mode, nPart, phi, noise, K, Rp, xTy, seed_range, d_type, min_r, max_r))
            ax.plot(K_avg_range, exponents, '-o', label="N=" + str(nPart) + r"; $\sigma_K=$" + str(K_std))

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
    (not from saved file)

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
        ax.plot(r_plot_new, corr_bin_av, '-')

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
    """
    Read density fluctuation correlation function from saved data file and return lists of r and C(r)
    """
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
    """
    Find C(r) and r for density fluctuations using grid method and FFT from x,y data at single timestep
    """
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
    corr = corr/corr[0,0]
    dist = get_distance_matrix(ngridx, ngridy, min_grid_size)
    return dist.flatten(), corr.flatten()

def get_r_corr_all(mode, nPart, phi, noise, K, Rp, xTy, seed_range, pos_ex=False, timestep_range=[0], min_grid_size=1):
    """
    Find r and C(r) for density fluctuations using grid method and FFT for multiple time steps
    Return as 1d arrays of r and C(r)
    """
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

def write_corr_density_grid(mode, nPart, phi, noise, K, Rp, xTy, seed_range, max_r=100, pos_ex=True, timestep_range=[0], min_grid_size=1):
    """
    Write grid density fluctuation correlation function to file
    """
    folder = os.path.abspath('../plot_data/correlation_density_grid/')
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed_range[-1]) + '_g' + str(min_grid_size) + ".txt"
    
    corrFile = open(os.path.join(folder, filename), "w")

    corrFile.write(str(min_grid_size) + "\n")
    corrFile.write(str(timestep_range[0]) + "\t" + str(timestep_range[-1]) + "\n")

    r_all, corr_all = get_r_corr_all(mode, nPart, phi, noise, K, Rp, xTy, seed_range, pos_ex, timestep_range, min_grid_size)

    for i in range(len(r_all)):
        if r_all[i] < max_r:
            corrFile.write(str(r_all[i]) + "\t" + str(corr_all[i]) + "\n")
    corrFile.close()

def read_corr_density(mode, nPart, phi, noise, K, Rp, xTy, seed_range, min_grid_size=1):
    """
    Read density fluctuation correlation function from saved data file and return lists of r and C(r)
    """
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
    """
    Bin correlation function from r and C(r) lists in terms of r by averaging over bins of size bin_size
    Restrict to range min_r to max_r
    """
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
    """
    Bin correlation function from r and C(r) lists in terms of r by averaging over bins exactly equal to r values
    (Used for grid method)
    Restrict to range min_r to max_r
    """
    r_plot = np.unique(dist)
    r_plot2 = r_plot[np.where((r_plot>=min_r) & (r_plot<=max_r))[0]]
    corr = np.array(corr)
    corr_plot = []
    for i in range(len(r_plot2)):
        idx = np.where(dist == r_plot2[i])[0]
        if len(idx)>0:
            c = np.mean(corr[idx])
            corr_plot.append(c)

    r_plot2 = np.array(r_plot2)
    corr_plot = np.array(corr_plot)
    return r_plot2, corr_plot

def plot_corr_density(mode, nPart, phi, noise, K, Rp, xTy, seed_range, pos_ex=True, timestep_range=[0], log_y=True, min_grid_size=1, min_r=0, max_r=10):
    """
    Plot density fluctuation correlation function directly using FFT method
    """
    dist, corr = get_r_corr_all(mode, nPart, phi, noise, K, Rp, xTy, seed_range, pos_ex, timestep_range, min_grid_size)
    r_plot, corr_plot = get_corr_binned(dist, corr, min_r=min_r, max_r=max_r)
    fig, ax = plt.subplots()
    ax.plot(r_plot, corr_plot, '-')
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
    """
    Plot density fluctuation correlation function from saved data file
    """
    # r_plot, corr_bin_av = read_corr_density(mode, nPart, phi, noise, K, Rp, xTy, seed_range, r_scale, bin_ratio)
    dist, corr = read_corr_density(mode, nPart, phi, noise, K, Rp, xTy, seed_range, min_grid_size)

    r_plot, corr_plot = get_corr_binned(dist, corr, min_r=min_r, max_r=max_r)

    fig, ax = plt.subplots()
    ax.plot(r_plot, corr_plot, '-')

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
    """
    Plot density fluctuation correlation function for different Kavg/Kstd values superimposed
    Calculated directly using FFT method
    """
    
    colors = plt.cm.GnBu(np.linspace(0.2, 1, len(K_avg_range)*len(K_std_range)))

    fig, ax = plt.subplots()

    i = 0
    for K_avg in K_avg_range:
        for K_std in K_std_range:
            K = str(K_avg) + "_" + str(K_std)
            dist, corr = get_r_corr_all(mode, nPart, phi, noise, K, Rp, xTy, seed_range, pos_ex, timestep_range, min_grid_size)
            r_plot, corr_plot = get_corr_binned(dist, corr, min_r=min_r, max_r=max_r)
            ax.plot(r_plot, corr_plot, '-', label=r"$\overline{K}=$" + str(K_avg) + r"; $\sigma_K=$" + str(K_std), color=colors[i])
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
    """
    Plot density fluctuation correlation function from saved date files for different Kavg/Kstd values superimposed
    """

    colors = plt.cm.GnBu(np.linspace(0.2, 1, len(K_avg_range)*len(K_std_range)))

    fig, ax = plt.subplots()

    i = 0
    for K_avg in K_avg_range:
        for K_std in K_std_range:
            K = str(K_avg) + "_" + str(K_std)
            dist, corr = read_corr_density(mode, nPart, phi, noise, K, Rp, xTy, seed_range, min_grid_size)
            r_plot, corr_plot = get_corr_binned(dist, corr, min_r=min_r, max_r=max_r)
            ax.plot(r_plot, corr_plot, '-', label=r"$\overline{K}=$" + str(K_avg) + r"; $\sigma_K=$" + str(K_std), color=colors[i])
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

def get_exponent_corr_density_points(mode, nPart, phi, noise, K, Rp, xTy, seed_range, min_r, max_r):
    """
    Calculate correlation decay exponent for density fluctuations from saved data files using points method
    """
    r_plot, corr_bin_av = read_corr_density(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed_range=seed_range, min_grid_size=1)
    r_plot = np.array(r_plot)
    corr_bin_av = np.array(corr_bin_av)
    idx1 = np.where(r_plot<max_r)[0]
    idx2 = np.where(r_plot>min_r)[0]
    idx = list(set(idx1) & set(idx2))
    # print(corr_bin_av[idx])

    exponent = np.polyfit(x=np.log10(r_plot[idx]), y=np.log10(corr_bin_av[idx]), deg=1)[0]

    return exponent

def get_exponent_corr_density_grid(mode, nPart, phi, noise, K, Rp, xTy, seed_range, pos_ex, timestep_range, min_grid_size, min_r, max_r):
    """
    Calculate correlation decay exponent for density fluctuations directly using grid and FFT method
    """
    dist, corr = get_r_corr_all(mode, nPart, phi, noise, K, Rp, xTy, seed_range, pos_ex, timestep_range, min_grid_size)
    r_plot, corr_plot = get_corr_binned(dist, corr, min_r=min_r, max_r=max_r)
    r_plot = np.array(r_plot)
    corr_plot= np.array(corr_plot)
    idx1 = np.where(r_plot<=max_r)[0]
    idx2 = np.where(r_plot>=min_r)[0]
    idx = list(set(idx1) & set(idx2))
    # print(corr_bin_av[idx])

    # Exponential fit
    exponent = np.polyfit(x=r_plot[idx], y=np.log(corr_plot[idx]), deg=1)[0]

    return exponent

def plot_exponents_Kavg_corr_density_points(mode, nPart_range, phi, noise, K_avg_range, K_std_range, Rp, xTy, seed_range, min_r, max_r):
    """
    Plot correlation decay exponent for density fluctuations vs Kavg for different Kstd values using saved file and points method
    """
    fig, ax = plt.subplots()
    for nPart in nPart_range:
        for K_std in K_std_range:
            exponents = []
            for K_avg in K_avg_range:
                K = str(K_avg) + "_" + str(K_std)
                exponents.append(get_exponent_corr_density_points(mode, nPart, phi, noise, K, Rp, xTy, seed_range, min_r, max_r))
            ax.plot(K_avg_range, exponents, '-o', label="N=" + str(nPart) + r"; $\sigma_K=$" + str(K_std))

    ax.set_title(r"Density correlation exponents; $N=$" + str(nPart) + r"; $\rho=$" + str(phi) + r"; $\eta=$" + str(noise) + r"; $R_I=$" + str(Rp))
    ax.set_xlabel(r"$K_{AVG}$")
    ax.set_ylabel(r"$\lambda$")
    ax.legend()

    folder = os.path.abspath('../plots/correlation_density_exp/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))

def plot_exponents_Kstd_corr_density_grid(mode, nPart, phi, noise, K_avg, K_std_range, Rp, xTy, seed_range, pos_ex, timestep_range, min_grid_size, min_r, max_r):
    """
    Plot correlation decay exponent for density fluctuations vs Kstd using grid and FFT method
    """
    fig, ax = plt.subplots()
    exponents = []
    for K_std in K_std_range:
        K = str(K_avg) + "_" + str(K_std)
        exponents.append(get_exponent_corr_density_grid(mode, nPart, phi, noise, K, Rp, xTy, seed_range, pos_ex, timestep_range, min_grid_size, min_r, max_r))
    ax.plot(K_std_range, exponents, '-o')

    ax.set_title(r"Density correlation exponents; $N=$" + str(nPart) + r"; $\rho=$" + str(phi) + r"; $\eta=$" + str(noise) + r"; $R_I=$" + str(Rp))
    ax.set_xlabel(r"$\sigma_K$")
    ax.set_ylabel(r"$\alpha$")
    # ax.legend()

    folder = os.path.abspath('../plots/correlation_density_exp/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_Kavg' + str(K_avg) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '.png'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))
    plt.close()