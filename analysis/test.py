import numpy as np
import analysis_functions_vicsek_nd as fun
import matplotlib.pyplot as plt
import gzip
import shutil


with open('coupling', 'rb') as f_in, gzip.open('coupling.gz', 'wb') as f_out:
    shutil.copyfileobj(f_in, f_out)

# Lx = 2*np.pi

# cutoff = 1
# x_vals = np.linspace(0,Lx,1000)
# n_density = 2*np.sin(x_vals)+2*np.sin(4*x_vals)

# # plt.plot(x_vals, n_density)
# # plt.show()

# if n_density[0] > 1:
#     in_band = 1
# else:
#     in_band = 0

# band_start = []
# band_end = []
# for i, val in enumerate(n_density):
#     if in_band == 0:
#         if val > cutoff:
#             in_band = 1
#             band_start.append(i)
#     if in_band == 1:
#         if val < cutoff:
#             in_band = 0
#             band_end.append(i)

# if len(band_start) != len(band_start):
#     raise Exception("Unequal band starts/ ends")
# else:
#     band_number = len(band_start)

# if band_number == 0:
#     raise Exception("No bands!")

# # Handle the case where band is on boundary
# if band_end[0] < band_start[0]:
#     band_start.insert(0, band_start.pop(-1))

# # Reclassify based on peak value
# band_centre = []
# for i in range(band_number):
#     if band_start[i] < band_end[i]:
#         band_vals = n_density[band_start[i]:band_end[i]]
#     else:
#         band_vals = np.concatenate((n_density[band_start[i]:], n_density[:band_end[i]]))
#     peak = np.max(band_vals)
#     peak_id = band_vals.argmax()
#     if peak > 2:
#         band_centre.append(band_start[i]+peak_id)
#     else:
#         band_start.pop(i)
#         band_end.pop(i)

# band_number = len(band_centre)
# if band_number == 0:
#     raise Exception("No bands with large enough peak!")

# extra_left = int(len(x_vals) / 10)
# extra_right = int(len(x_vals) / 10)
# total_len = extra_left + extra_right

# fig, ax = plt.subplots()

# for i in range(band_number):
#     if band_centre[i] + extra_right > len(x_vals):
#         d_plot = np.concatenate((n_density[band_centre[i]-extra_left:], n_density[:band_centre[i]+extra_right-len(x_vals)]))
#     elif band_centre[i] - extra_left < 0:
#         d_plot = np.concatenate((n_density[band_centre[i]-extra_left+len(x_vals):], n_density[:band_centre[i]+extra_right]))
#     else:
#         d_plot = n_density[band_centre[i]-extra_left:band_centre[i]+extra_right]
#     x_plot = x_vals[:total_len]
#     ax.plot(x_plot, d_plot, label="band " + str(i))
# ax.legend()
# plt.show()