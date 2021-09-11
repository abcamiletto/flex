#!/usr/bin/env python3
from urdf2optcontrol import optimizer
from matplotlib import pyplot as plt 
import numpy as np

# URDF options
urdf_path = './urdf/rrbot.urdf'
root = 'link1'
end = 'link3'

def trajectory_target(t):
    q = [3.14, 0]
    return q
in_cond = [0]*4


def my_cost_func(q, qd, u, t):
    return u.T@u / 2000
def my_final_term_cost(q_f, qd_f, u_f):
    return (-np.sin(q_f[0])*qd_f[0] -np.sin(q_f[1])*qd_f[1])

def my_constraint1(q, q_dot, u, ee_pos, q_ddot):
    return [-30, -30], u, [30, 30]
def my_constraint2(q, q_dot, u, ee_pos, q_ddot):
    return [-4, -4], q_dot, [4, 4]
my_constraints = [my_constraint1, my_constraint2]


def my_final_constraint1(q, q_dot, u, ee_pos, q_ddot):
    return [3.14/2, 0], q, [3.14/2, 0]

def my_final_constraint2(q, q_dot, u, ee_pos, q_ddot):
    return [0, 0], q_dot, [0, 0]


my_final_constraints = [my_final_constraint1]

time_horizon = 2
steps = 20

# Load the urdf and calculate the differential equations
optimizer.load_robot(urdf_path, root, end)

# Loading the problem conditions
optimizer.load_problem(
    my_cost_func,
    steps,
    in_cond,
    time_horizon=time_horizon,
    constraints=my_constraints,
    final_constraints=my_final_constraints,
    max_iter=500
    )

# Solving the non linear problem
optimizer.solve()

# Print the results
fig = optimizer.plot_result()
plt.show()
