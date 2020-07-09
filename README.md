# Model Predictive Control of Concentric Tube Robots
Repo for control a concentric tube robot with 2 tubes. It implements the methodology presented in the following publication:

M. Khadem, J. O’Neill, Z. Mitros, L. da Cruz and C. Bergeles, "Autonomous Steering of Concentric Tube Robots via Nonlinear Model Predictive Control," in IEEE Transactions on Robotics, 
doi: 10.1109/TRO.2020.2991651.

URL: https://ieeexplore.ieee.org/document/9097934

If you enjoy this repository and use it, please cite our paper
```
@ARTICLE{khadem_tro,
  author={M. {Khadem} and J. {O’Neill} and Z. {Mitros} and L. {da Cruz} and C. {Bergeles}},
  journal={IEEE Transactions on Robotics}, 
  title={Autonomous Steering of Concentric Tube Robots via Nonlinear Model Predictive Control}, 
  year={2020},
  volume={},
  number={},
  pages={1-8},}
```

Dependencies: numpy, scipy, mpl_toolkits, matplotlib, numdifftools.

The CTR_MPC module includes functions for modeling a  CTR. It accepts tubes parameters, joint variables q, initial value of joints q_0, input force f, tolerance for solver Tol, and a desired method for solving equations (1 for using BVP and 2 for IVP). The Class depends on two other modules Segment.py and Tube.py. CTR_MPC module includes several functions including:

ode-solver(q, x_d): This function accepts robots' joint variables and dsired position as inputs.

solve_mpc(q, q0, x_d, w, s, mu): This function solves the control problem to find desired joint angles q to reach desired tip position x_d. The inputs are: initial guess for robot 
joints q, initial position of robots joints q0, desired pos x_d, lagrange multipliers w, slack variables s, and perturbation parameter mu.

Example.py shows how the module can be used to control robot trajectory.

