#!/usr/bin/env python3
from urdf_optcontrol import optimizer
from matplotlib import pyplot as plt

# URDF options
urdf_path = '../urdf/rrbot.urdf'
root = 'link1'
end = 'link3'


# The trajectory in respect we want to minimize the cost function
# If qdot isn't given, it will be obtained with differentiation from q
def trajectory_target_(t):
    q = [t] * 2
    qdot = [0] * 2
    return (q, qdot)

# Our cost function
def my_cost_func(q, qd, u, t):
    return 10*q.T@q + u.T@u/10

# Our final term to be added at the end to our cost function
def my_final_term_cost(q_f, qd_f, u_f):
    return 10*q_f.T@q_f

# Additional Constraints I may want to set
def my_constraint1(q, q_dot, u, ee_pos):
    return [-10, -10], u, [10, 10]
def my_constraint2(q, q_dot, u, ee_pos):
    return [-4, -4], q_dot, [4, 4]
def my_constraint3(q, q_dot, u, ee_pos):
    return 0, ee_pos[0]**2 + ee_pos[1]**2 + ee_pos[2]**2, 20
my_constraints=[my_constraint1, my_constraint2, my_constraint3]

# Constraints to be imposed on the final instant of the simulation
# e.g. impose a value of 1 radiant for both the joints
def my_final_constraint1(q, q_dot, u, ee_pos):
    return [1, 1], q, [1, 1]
my_final_constraints = [my_final_constraint1]    # if not set, it is free (and optimized)

# Initial Condition in terms of q, qdot
in_cond = [0,0] + [0,0]

# Optimization parameters
steps = 50
time_horizon = 1    # if not set, it is free (and optimized)

# Load the urdf and calculate the differential equations
optimizer.load_robot(urdf_path, root, end)

# Solve an optimal problem with above parameters
# Results will be a dictionary with q, qd, u as keys
optimizer.load_problem(
    my_cost_func,
    steps,
    in_cond,
    trajectory_target_,
    time_horizon=time_horizon,
    final_term_cost=my_final_term_cost, 
    my_constraint=my_constraints, 
    my_final_constraint=my_final_constraints,
    max_iter=70
    )

# Print the results!
fig = optimizer.show_result()
plt.show()
