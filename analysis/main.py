import numpy as np
import analysis_functions as fun
import os
import matplotlib.pyplot as plt

# f.plot_vorder_ksd(mode="G", nPart=500, phi=0.2, KAVG=1.0, KSTD_range=np.arange(0, 41, 1.0), seed=2, save=True, view=False)

# f.snapshot(mode="C", nPart=1000, phi=0.4, K=1.0, seed=2, view_time=10)

# fun.snapshot(mode="G", nPart=800, phi=0.2, K="1.0_0.0", seed=1, view_time=10)



intercept = 0.361
power = 0.5
fig, ax = plt.subplots()
fig, ax = fun.plot_vorder_kratio_ax(mode="G", nPart_range=[1600], phi=0.2, KAVG_range=[-2.0,-1.5,-1.0,-0.5,-0.25,0.0,0.1,0.2,0.22,0.24,0.25,0.26,0.28,0.3,0.4,0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0], KSTD_range=[1.0,2.0], 
                                    seed_range=np.arange(1,11), intercept=intercept, power=power, fig=fig, ax=ax)

fig, ax = fun.plot_vorder_kratio_ax(mode="G", nPart_range=[1600], phi=0.2, KAVG_range=[-2.0,-1.5,-1.0,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0], KSTD_range=[4.0,6.0,8.0,10.0], 
                                    seed_range=np.arange(1,11), intercept=intercept, power=power, fig=fig, ax=ax)

folder = os.path.abspath('../plots/ratio_plots/')
filename = 'KAVG_KSTD^05'
plt.savefig(os.path.join(folder, filename))

# crit_vals = []
# crit = fun.critical_value_kavg(mode="G", nPart=1600, phi=0.2, KAVG_range=[0.2,0.3,0.35,0.4,0.5], KSTD=0.0, seed_range=np.arange(1,11))
# crit_vals.append(crit)
# for KSTD in [1.0,2.0]:
#     crit = fun.critical_value_kavg(mode="G", nPart=1600, phi=0.2, KAVG_range=[-2.0,-1.5,-1.0,-0.5,-0.25,0.0,0.1,0.2,0.22,0.24,0.25,0.26,0.28,0.3,0.35,0.4,0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0], KSTD=KSTD, seed_range=np.arange(1,11))
#     crit_vals.append(crit)

# for KSTD in [4.0,6.0,8.0,10.0]:
#     crit = fun.critical_value_kavg(mode="G", nPart=1600, phi=0.2, KAVG_range=[-2.0,-1.5,-1.0,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0], KSTD=KSTD, seed_range=np.arange(1,11))
#     crit_vals.append(crit)

# for KSTD in [12.0,14.0]:
#     crit = fun.critical_value_kavg(mode="G", nPart=1600, phi=0.2, KAVG_range=[0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0], KSTD=KSTD, seed_range=np.arange(1,11))
#     crit_vals.append(crit)

# print(crit_vals)

# KSTD_range = [0.0,1.0,2.0,4.0,6.0,8.0,10.0]
# ax.plot(KSTD_range, crit_vals, 'o-')
# ax.set_xlim(0,12)
# ax.set_ylim(0,1.2)

# ax.set_xlabel("K_STD")
# ax.set_ylabel("Critical K_AVG value")
# folder = os.path.abspath('../plots/')
# filename = 'my_plot'
# plt.savefig(os.path.join(folder, filename))
