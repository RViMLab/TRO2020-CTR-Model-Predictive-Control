import numpy as np
from Tube import Tube
from CTR_MPC import CTR_MPC
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from matplotlib import cm
import matplotlib.pyplot as plt
import time

# Defining parameters of each tube, numbering starts with the most inner tube
# length, length_curved, diameter_inner, diameter_outer, stiffness, torsional_stiffness, x_curvature, y_curvature
tube1 = Tube(400e-3, 200e-3, 2 * 0.35e-3, 2 * 0.55e-3, 70.0e+9, 10.0e+9, 12, 0)
tube2 = Tube(300e-3, 150e-3, 2 * 0.7e-3, 2 * 0.9e-3, 70.0e+9, 10.0e+9, 6, 0)
# Initial guess for Joint variables
q = np.array([0.0, 0.0, 0, 0.0])
# Initial position of joints
q_0 = np.array([-250e-3, -200e-3, 0, 0])
# initial twist (for ivp solver)
uz_0 = np.array([0.0, 0.0, 0.0])
u1_xy_0 = np.array([[0.0], [0.0]])
# time
t = np.linspace(0, 1, num=10)
# velocity in m/sec
v = 1e-3

# initializing the mpc solver
s = 2 * np.ones((4, 1))
w = 0.1 * np.ones((4, 1))
mu = 1

q_array = np.zeros((1,4))
x_d_array = np.array([[0e-2], [-5e-02], [1.1e-01]])
for i in t:
    x_d = np.array([[0], [-5e-02-v*i], [1.1e-01 + v*i]])
    CTR = CTR_MPC(tube1, tube2, q, q_0, 0.01)
    Q, Cost = CTR.solve_mpc(q, q_0, x_d, w, s, mu)
    q = Q[np.argmin(Cost), :]
    print(q)
    x_d_array = np.concatenate((x_d_array, x_d), axis=0)
    q_array = np.concatenate((q_array, q.reshape(1, 4)), axis=0)

# plot the robot shape
fig = plt.figure()
ax = plt.axes(projection='3d')
counter = 1
for i in t:
    x_d = np.array([[v*i], [-5e-02], [1.1e-01 + v*i]])
    q = q_array[counter,:].reshape(4, )
    CTR.ode_solver(q, x_d)
    ax.scatter(x_d[0, 0], x_d[1, 0], x_d[2, 0], c='r', marker='o')
    ax.plot(CTR.r[:, 0], CTR.r[:, 1], CTR.r[:, 2], '-b')
    ax.auto_scale_xyz([np.amin(CTR.r[:, 0]), np.amax(CTR.r[:, 0]) + 0.01],
                      [np.amin(CTR.r[:, 1]), np.amax(CTR.r[:, 1]) + 0.01],
                      [np.amin(CTR.r[:, 2]), np.amax(CTR.r[:, 2]) + 0.01])
    counter+=1

ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_zlabel('Z [mm]')
plt.grid(True)
plt.show()

