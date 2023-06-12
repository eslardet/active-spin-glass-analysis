import numpy as np
import os

def read_corr_density(mode, nPart, phi, noise, K, Rp, xTy, seed_range, bin_ratio=1):
    r_plot = []
    corr_bin_av = []

    folder = os.path.abspath('../plot_data/correlation_density/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed_range[-1]) + ".txt"
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