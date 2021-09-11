#!/usr/bin/env python3
from urdf2optcontrol import optimizer
from matplotlib import pyplot as plt 

# URDF options
urdf_path = './urdf/rrbot.urdf'
root = 'link1'
end = 'link3'


# The trajectory in respect we want to minimize the cost function
# If qd isn't given, it will be obtained with differentiation from q
def trajectory_target_(t):
    q = [t] * 2
    return q

# Our cost function
def my_cost_func(q, qd, u, t):
    return 100*q.T@q + u.T@u/10

# Additional Constraints I may want to set
def my_constraint1(q, q_d, q_dd, u, ee_pos):
    return [-30, -30], u, [30, 30]
def my_constraint2(q, q_d, q_dd, u, ee_pos):
    return [-15, -15], q_dd, [15, 15]
my_constraints=[my_constraint1, my_constraint2]

# Constraints to be imposed on the final instant of the simulation
# e.g. impose a value of 1 radiant for both the joints
def my_final_constraint1(q, q_d, q_dd, u, ee_pos):
    return [1, 1], q, [1, 1]
my_final_constraints = [my_final_constraint1] 

# Initial Condition in terms of q, qd
in_cond = [0,0] + [0,0]

# Optimization parameters
steps = 35
time_horizon = 1    # if not set, it is free (and optimized)

# Load the urdf and calculate the differential equations
optimizer.load_robot(urdf_path, root, end)

# Loading the problem conditions
optimizer.load_problem(
    my_cost_func,
    steps,
    in_cond,
    trajectory_target = trajectory_target_,
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
