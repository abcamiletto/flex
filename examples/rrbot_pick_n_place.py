#!/usr/bin/env python3
from urdf2optcontrol import optimizer
from matplotlib import pyplot as plt 
import pathlib

# URDF options
urdf_path = pathlib.Path(__file__).parent.joinpath('urdf', 'rrbot.urdf').absolute()
root = "link1"
end = "ee"

in_cond = [0] * 4

def my_cost_func(q, qd, qdd, ee_pos, u, t):
    return u.T @ u / 10**8

def my_final_cost_func(q, qd, qdd, ee_pos, u):
    return (ee_pos[0] + 1.0)**2 + (ee_pos[2] - 2.0)**2


def my_constraint1(q, qd, qdd, ee_pos, u):
    return [-30, -30], u, [30, 30]


def my_constraint2(q, qd, qdd, ee_pos, u):
    return [-4, -4], qd, [4, 4]


my_constraints = [my_constraint1, my_constraint2]

time_horizon = 2.0
steps = 40

# Load the urdf and calculate the differential equations
optimizer.load_robot(urdf_path, root, end)

# Loading the problem conditions
optimizer.load_problem(
    my_cost_func,
    steps,
    in_cond,
    time_horizon=time_horizon,
    final_term_cost = my_final_cost_func,
    constraints=my_constraints,
    max_iter=500
)

# Solving the non linear problem
res = optimizer.solve()
print('u = ', res['u'][0])
print('q = ', res['q'][0])
print('ee = ', res['ee_pos'])

# Print the results!
fig = optimizer.plot_result(show=True)

