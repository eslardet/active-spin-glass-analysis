import numpy as np
import analysis_functions_vicsek as fun
import matplotlib.pyplot as plt


# v = np.array([1, 3])

# mean_head = np.array([2,0])

# unit_vec = mean_head / np.linalg.norm(mean_head)

# unit_norm = np.array([-unit_vec[1], unit_vec[0]])

# v_perp = np.dot(v, unit_norm) * unit_norm
# v_par = np.dot(v, unit_vec) * unit_vec

# print(v_perp)

# print(v_par)


mode = "C"
nPart = 100
phi = 0.1
noise = "0.60"
K = "1.0"
xTy=5.0
seed=1


def plot_corr_vel_fluc(mode, nPart, phi, noise, K, xTy, seed, pos_ex=True, timestep=None):
    """
    Get velocity fluctations for each particle
    """
    if pos_ex == True:
        posFileExact = fun.get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed, file_name='pos_exact')
        x, y, theta, view_time = fun.get_pos_ex_snapshot(file=posFileExact)
    else: 
        inparFile, posFile = fun.get_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed)
        inpar_dict = fun.get_params(inparFile)
        DT = inpar_dict["DT"]
        simulT = inpar_dict["simulT"]
        eqT = inpar_dict["eqT"]
        if timestep == None:
            timestep = int((simulT-eqT)/DT) 
        x, y, theta = fun.get_pos_snapshot(posFile, nPart, timestep)

    L = np.sqrt(nPart / (phi*xTy))
    Ly = L
    Lx = L*xTy

    x = fun.pbc_wrap(x,Lx)
    y = fun.pbc_wrap(y,Ly)

    velocity = [np.array([np.cos(p), np.sin(p)]) for p in theta]
    av_vel = np.mean(velocity, axis=0)

    dv = [v - av_vel for v in velocity]

    # av_unit = av_vel / np.linalg.norm(av_vel)
    av_norm = np.array([-av_vel[1], av_vel[0]])

    # fluc_par = [np.dot(f, av_unit) * av_unit for f in fluc_vel]
    dv_perp = [np.dot(f, av_norm) * av_norm for f in dv]

    ## Next: 
    # take Fourier transform
    # compute equal-time spatial correlation function in Fourier space
    # (Could just do in real space first as Fourier transform stuff has some inconsistencies in the Zhao et al. paper)
    # Plot!

    fig, ax = plt.subplots()
    
    L = np.sqrt(nPart / (phi*xTy))
    Ly = L
    Lx = L*xTy

    for i in range(nPart):
        for j in range(i+1, nPart):
            ## Can add time average here
            corr = np.dot(dv_perp[i],dv_perp[j])
    
            xij = x[i] - x[j]
            xij = xij - Lx*round(xij/Lx)
            yij = y[i] - y[j]
            yij = yij - Ly*round(yij/Ly)
            rij = np.sqrt(xij**2 + yij**2)
            
            ax.plot(rij, corr, '+')
    
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$C_{\perp}(r)$")

    ## Plot on log-log/ lin-log scale

    plt.show()
    # folder = os.path.abspath('../plots/dist_coupling/')
    # filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_xTy' + str(xTy) + '_s' + str(seed) + '.png'
    # if not os.path.exists(folder):
    #     os.makedirs(folder)
    # plt.savefig(os.path.join(folder, filename))



plot_corr_vel_fluc(mode, nPart, phi, noise, K, xTy, seed)

def get_velocity_fields(mode, nPart, phi, noise, K, xTy, seed, min_grid_size=2, pos_ex=True, timestep=None):
    """
    Get velocity fields from local averages in small cells
    """
    if pos_ex == True:
        posFileExact = fun.get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed, file_name='pos_exact')
        x, y, theta, view_time = fun.get_pos_ex_snapshot(file=posFileExact)
    else: 
        inparFile, posFile = fun.get_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed)
        inpar_dict = fun.get_params(inparFile)
        DT = inpar_dict["DT"]
        simulT = inpar_dict["simulT"]
        eqT = inpar_dict["eqT"]
        if timestep == None:
            timestep = int((simulT-eqT)/DT) 
        x, y, theta = fun.get_pos_snapshot(posFile, nPart, timestep)

    L = np.sqrt(nPart / (phi*xTy))
    Ly = L
    Lx = L*xTy

    x = fun.pbc_wrap(x,Lx)
    y = fun.pbc_wrap(y,Ly)

    ngrid_x = int(Lx // min_grid_size)
    grid_size_x = Lx / ngrid_x
    ngrid_y = int(Ly // min_grid_size)
    grid_size_y = Ly / ngrid_y

    grid_area = grid_size_x*grid_size_y

    grid_counts = np.zeros((ngrid_x, ngrid_y))

    grid_velocities_sum = np.zeros((ngrid_x, ngrid_y))

    for i in range(nPart):
        gridx = int(x[i]//grid_size_x)
        gridy = int(y[i]//grid_size_y)
        grid_counts[gridx,gridy] += 1
    #     velocity = ??
    #     grid_velocities_sum[gridx,gridy] += 2
    # n_density = grid_counts / grid_area

    # var_density = np.std(n_density)**2

    # return var_density


