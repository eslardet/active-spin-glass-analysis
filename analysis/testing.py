import numpy as np
import analysis_functions_vicsek_nd as fun

def local_density_var(mode, nPart, phi, Pe, K, xTy, seed, min_grid_size=2):
    posFileExact = fun.get_file_path(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, xTy=xTy, seed=seed, file_name='pos_exact')
    x, y, theta, view_time = fun.get_pos_ex_snapshot(file=posFileExact)

    L = np.sqrt(nPart / (phi*xTy))
    Ly = L
    Lx = L*xTy

    x = fun.pbc_wrap(x,Lx)
    y = fun.pbc_wrap(y,Ly)

    ngrid_x = int(Lx // min_grid_size)
    grid_size_x = Lx / ngrid_x
    ngrid_y = int(Ly // min_grid_size)
    grid_size_y = Ly / ngrid_y

    print(ngrid_x, ngrid_y)

    grid_area = grid_size_x*grid_size_y

    grid_counts = np.zeros((ngrid_x, ngrid_y))

    for i in range(nPart):
        gridx = int(x[i]//grid_size_x)
        gridy = int(y[i]//grid_size_y)
        grid_counts[gridx,gridy] += 1
    n_density = grid_counts / grid_area

    var_density = np.std(n_density)**2

    return var_density

# def plot_var_density_pe(mode, nPart, phi, Pe, K, xTy, seed):


var = local_density_var(mode="C", nPart=1000, phi=1.0, Pe=3.0, K=1.0, xTy=5.0, seed=1)

print(var)
