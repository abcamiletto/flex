#!/usr/bin/env python3
from urdf2optcontrol import optimizer
from matplotlib import pyplot as plt 
import numpy as np
import pathlib

# URDF options
urdf_path = pathlib.Path(__file__).parent.joinpath('urdf', 'rrbot.urdf').absolute()
root = "link1"
end = "link3"

in_cond = [0] * 4


def my_cost_func(q, qd, qdd, ee_pos, u, t):
    return 0 # u.T @ u / 10**8


def my_final_term_cost(qf, qdf, qddf, ee_posf, uf):
    return (-np.sin(qf[0]) * 1 * qdf[0] -np.sin(qf[0]+qf[1]) * 1 * (qdf[0]+qdf[1]))

def my_constraint1(q, qd, qdd, ee_pos, u):
    return [-30, -30], u, [30, 30]


def my_constraint2(q, qd, qdd, ee_pos, u):
    return [-4, -4], qd, [4, 4]


my_constraints = [my_constraint1, my_constraint2]


def my_final_constraint1(q, qd, qdd, ee_pos, u):
    return [3.14 / 2, 0], q, [3.14 / 2, 0]


my_final_constraints = []

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
    final_term_cost=my_final_term_cost,
    constraints=my_constraints,
    final_constraints=my_final_constraints,
    max_iter=500
)

# Solving the non linear problem
res = optimizer.solve()

# Print the results!
fig = optimizer.plot_result(show=True)
