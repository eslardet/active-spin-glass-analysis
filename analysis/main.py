import numpy as np
import analysis_functions as fun
import os
import matplotlib.pyplot as plt

# f.plot_vorder_ksd(mode="G", nPart=500, phi=0.2, KAVG=1.0, KSTD_range=np.arange(0, 41, 1.0), seed=2, save=True, view=False)

# f.snapshot(mode="C", nPart=1000, phi=0.4, K=1.0, seed=2, view_time=10)

# fun.snapshot(mode="G", nPart=800, phi=0.2, K="1.0_0.0", seed=1, view_time=10)


# fig, ax = plt.subplots()
# fig, ax = fun.plot_vorder_kavg_ax(mode="G", nPart_range=[1600], phi=0.2, KAVG_range=[-2.0,-1.5,-1.0,-0.5,-0.25,0.0,0.1,0.2,0.22,0.24,0.25,0.26,0.28,0.3,0.4,0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0], KSTD_range=[1.0,2.0], seed_range=np.arange(1,11), fig=fig, ax=ax)
# fig, ax = fun.plot_vorder_kavg_ax(mode="G", nPart_range=[1600], phi=0.2, KAVG_range=[-2.0,-1.5,-1.0,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0], KSTD_range=[4.0,6.0,8.0,10.0], seed_range=np.arange(1,11), fig=fig, ax=ax)
# folder = os.path.abspath('../plots/v_order_vs_K/')
# filename = 'my_plot'
# plt.savefig(os.path.join(folder, filename))
for KSTD in [1.0,2.0]:
    v, crit = fun.critical_value_kavg(mode="G", nPart=1600, phi=0.2, KAVG_range=[-2.0,-1.5,-1.0,-0.5,-0.25,0.0,0.1,0.2,0.22,0.24,0.25,0.26,0.28,0.3,0.4,0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0], KSTD=KSTD, seed_range=np.arange(1,11))
    print(KSTD, crit)

for KSTD in [4.0,6.0,8.0,10.0]:
    v, crit = fun.critical_value_kavg(mode="G", nPart=1600, phi=0.2, KAVG_range=[-2.0,-1.5,-1.0,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0], KSTD=KSTD, seed_range=np.arange(1,11))
    print(KSTD, crit)