#!/usr/bin/env python3
from urdf2optcontrol import optimizer
from matplotlib import pyplot as plt 

# URDF options
urdf_path = './urdf/rrbot.urdf'
root = "link1" 
end = "link3"

# point to point low energy
in_cond = [0]*4

def my_cost_func(q, qd, qdd, ee_pos, u, t):
    return u.T@u


def my_constraint1(q, qd, qdd, ee_pos, u):
    return [-30, -30], u, [30, 30]
def my_constraint2(q, qd, qdd, ee_pos, u):
    return [-4, -4], qdd, [4, 4]
my_constraints = [my_constraint1, my_constraint2]


def my_final_constraint1(q, qd, qdd, ee_pos, u):
    return [3.14/2, 0], q, [3.14/2, 0]
def my_final_constraint2(q, qd, qdd, ee_pos, u):
    return [0, 0], qdd, [0, 0]
my_final_constraints = [my_final_constraint1, my_final_constraint2]

time_horizon = 2
steps = 50

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

# Print the results!
fig = optimizer.plot_result()
plt.show()
