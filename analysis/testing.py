import numpy as np
import analysis_functions_vicsek as fun
import matplotlib.pyplot as plt
from matplotlib import cm, colors

K_crit = []

# 0.20
K_crit.append([0.04011276036171672, -0.024627434167354943, -0.06034634664703252, -0.14589330308458925,
                -0.2324053673171384, -0.3160787177130014, -0.3858028596116728, -0.4544133863597353])

# 0.40
K_crit.append([0.20824695786078262,0.16386085052011543,0.11039837457478775, 0.05210467854263792, 
               -0.027217767253488182, -0.09072967871466574, -0.15386959784638107, -0.22730718902117258])

# 0.60 
K_crit.append([0.51747787627541, 0.48204703428778123, 0.44709777838820564, 0.38430423876538755,
                0.33215852575542254, 0.2740370748833158, 0.22053303218349699, 0.167170987252005])

# 0.80
K_crit.append([0.9994535518783407, 0.9862795263865451 ,0.947777394097244, 0.8905316377938783, 
               0.8486886905367337, 0.7878966972594424, 0.7368779073729201, 0.6871807518955116])

K_std_range = np.arange(1,9,1)
noise_range = np.round(np.arange(0.2, 0.85, 0.2),2)


fig, ax = plt.subplots()

for i, noise in enumerate(noise_range):
    ax.plot(K_std_range, K_crit[i], '-o', label=r"$\eta = $" + str(noise))

# ax.plot(K_std_range, [-i/14 + 0.1 for i in K_std_range])
# ax.plot(K_std_range, [-i/14 + 0.3 for i in K_std_range])
# ax.plot(K_std_range, [-i/18 + 0.6 for i in K_std_range])
# ax.plot(K_std_range, [-i/20 + 1.1 for i in K_std_range])

ax.set_xlabel(r"$K_{STD}$")
ax.set_ylabel(r"$K_{AVG}^C$")
ax.legend()

plt.show()