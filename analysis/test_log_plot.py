import numpy as np
import matplotlib.pyplot as plt

# x_vals = np.arange(0,20,1)

# y_vals = -x_vals/18 + 0.6

# fig, ax = plt.subplots()

# # ax.plot(x_vals, y_vals, '-o')
# # ax.plot(x_vals, -y_vals+0.6, 'o-')
# ax.plot(np.log(x_vals), np.log(18*(-y_vals+0.6)), 'o-')
# ax.plot(np.log(x_vals), np.log(x_vals), '--')
# # ax.set_xscale('log')
# # ax.set_yscale('log')
# plt.show()


# theta = [0, np.pi, 2*np.pi]

# sum_c = 0
# sum_s = 0
# for p in theta:
#     sum_c += np.cos(p)
#     sum_s += np.sin(p)

# av_c = sum_c/len(theta)
# av_s = sum_s/len(theta)

# print(av_c)
# print(av_s)

# vel_x = np.cos(theta)

# print(vel_x - av_c)

theta = [np.pi/4,np.pi]
vel = [np.array([np.cos(t),np.sin(t)]) for t in theta]
print(vel)

av_vel = np.mean(vel, axis=0)
print(av_vel)

av_vel = np.array([np.cos(np.pi/2),np.sin(np.pi/2)])

fluc_vel = [v - av_vel for v in vel]
# print(fluc_vel)

av_unit = av_vel / np.linalg.norm(av_vel)
av_norm = np.array([-av_vel[1], av_vel[0]])

fluc_par = [np.dot(f, av_unit) * av_unit for f in fluc_vel]
fluc_perp = [np.dot(f, av_norm) * av_norm for f in fluc_vel]

# print([np.dot(f, av_norm) for f in fluc_par])
# print(fluc_par)
# print(fluc_perp)

print([1]+[2])