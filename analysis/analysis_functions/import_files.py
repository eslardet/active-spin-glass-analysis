import numpy as np
import csv
import os

## Find files and import data

def get_sim_dir(mode, nPart, phi, noise, K, Rp, xTy, seed):
    """
    Get path as string to simulation data directory
    """
    if mode == "C":
        mode_name = "Constant"
    elif mode == "T":
        mode_name = "ThreePopulations_NR"
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
    inpar_dict["seed"] = str(r[2][0])
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
        inpar_dict["dt"] = float(r[18][0])
        inpar_dict["DT"] = float(r[19][0])
        inpar_dict["eqT"] = float(r[21][0])
        inpar_dict["simulT"] = float(r[22][0])
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

    for i in range(max(int((min_T-startT+eqT)/DT),0), int((max_T-startT)/DT)+1):
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